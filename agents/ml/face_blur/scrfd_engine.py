"""SCRFD ONNX inference (InsightFace), CPU-friendly and keypoint-aware."""
from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None  # type: ignore


def _distance2bbox(points: np.ndarray, distance: np.ndarray, max_shape: tuple[int, int] | None = None) -> np.ndarray:
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = np.clip(x1, 0, max_shape[1])
        y1 = np.clip(y1, 0, max_shape[0])
        x2 = np.clip(x2, 0, max_shape[1])
        y2 = np.clip(y2, 0, max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)


def _distance2kps(points: np.ndarray, distance: np.ndarray, max_shape: tuple[int, int] | None = None) -> np.ndarray:
    preds: List[np.ndarray] = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        if max_shape is not None:
            px = np.clip(px, 0, max_shape[1])
            py = np.clip(py, 0, max_shape[0])
        preds.extend([px, py])
    return np.stack(preds, axis=-1)


class ScrfdOnnxEngine:
    """Minimal SCRFD runner for Open Redaction face detection."""

    def __init__(self, model_path: str):
        if ort is None:
            raise ImportError("onnxruntime is required for SCRFD")
        self.model_path = model_path
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.center_cache: dict = {}
        self.nms_thresh = 0.4
        self._init_vars()

    def _init_vars(self) -> None:
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        if isinstance(input_shape[2], str):
            self.input_size: tuple[int, int] | None = None
        else:
            self.input_size = tuple(input_shape[2:4][::-1])  # type: ignore[arg-type]
        self.input_name = input_cfg.name
        outputs = self.session.get_outputs()
        self.batched = len(outputs[0].shape) == 3
        self.output_names = [o.name for o in outputs]
        self.use_kps = len(outputs) in {9, 15}
        self._num_anchors = 2 if len(outputs) in {6, 9} else 1
        self.fmc = 3 if len(outputs) in {6, 9} else 5
        self._feat_stride_fpn = [8, 16, 32] if self.fmc == 3 else [8, 16, 32, 64, 128]

    def _forward(self, img: np.ndarray, thresh: float):
        scores_list: List[np.ndarray] = []
        bboxes_list: List[np.ndarray] = []
        kpss_list: List[np.ndarray] = []
        input_size = tuple(img.shape[0:2][::-1])
        blob = cv2.dnn.blobFromImage(img, 1.0 / 128, input_size, (127.5, 127.5, 127.5), swapRB=True)
        net_outs = self.session.run(self.output_names, {self.input_name: blob})
        input_height = blob.shape[2]
        input_width = blob.shape[3]
        for idx, stride in enumerate(self._feat_stride_fpn):
            if self.batched:
                scores = net_outs[idx][0]
                bbox_preds = net_outs[idx + self.fmc][0] * stride
                kps_preds = net_outs[idx + self.fmc * 2][0] * stride if self.use_kps else None
            else:
                scores = net_outs[idx]
                bbox_preds = net_outs[idx + self.fmc] * stride
                kps_preds = net_outs[idx + self.fmc * 2] * stride if self.use_kps else None

            height = input_height // stride
            width = input_width // stride
            key = (height, width, stride)
            anchor_centers = self.center_cache.get(key)
            if anchor_centers is None:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

            pos_inds = np.where(scores >= thresh)[0]
            bboxes = _distance2bbox(anchor_centers, bbox_preds)
            scores_list.append(scores[pos_inds])
            bboxes_list.append(bboxes[pos_inds])
            if self.use_kps and kps_preds is not None:
                kpss = _distance2kps(anchor_centers, kps_preds)
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                kpss_list.append(kpss[pos_inds])
        return scores_list, bboxes_list, kpss_list

    def _nms(self, dets: np.ndarray) -> List[int]:
        x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep: List[int] = []
        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]
        return keep

    def detect_raw(
        self,
        img: np.ndarray,
        *,
        thresh: float,
        input_size: tuple[int, int] = (640, 640),
    ) -> Tuple[np.ndarray, np.ndarray | None]:
        """Return Nx5 [x1,y1,x2,y2,score] and optional Nx5x2 keypoints in image coords."""
        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / img.shape[0]
        resized = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized

        scores_list, bboxes_list, kpss_list = self._forward(det_img, thresh)
        if not scores_list:
            return np.zeros((0, 5), dtype=np.float32), None

        scores = np.vstack(scores_list)
        bboxes = np.vstack(bboxes_list) / det_scale
        order = scores.ravel().argsort()[::-1]
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)[order]
        keep = self._nms(pre_det)
        det = pre_det[keep]
        kpss = None
        if self.use_kps and kpss_list:
            kpss = np.vstack(kpss_list) / det_scale
            kpss = kpss[order][keep]
        return det, kpss
