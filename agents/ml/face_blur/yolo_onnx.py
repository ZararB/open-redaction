"""Shared YOLOv8/v11 ONNX decode (Ultralytics export, single- or multi-class)."""
from __future__ import annotations

from typing import List, Sequence, Tuple

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None  # type: ignore

Box = Tuple[int, int, int, int]
ScoredBox = Tuple[Box, float]


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
    if boxes.size == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


class YoloOnnxDetector:
    """Generic YOLO ONNX detector with optional class filter."""

    def __init__(
        self,
        model_path: str,
        *,
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        input_size: int = 640,
        class_ids: Sequence[int] | None = None,
        name: str = "YoloOnnxDetector",
    ):
        if ort is None:
            raise ImportError("onnxruntime is required for YOLO ONNX detectors")
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        self.class_ids = set(class_ids) if class_ids is not None else None
        self._name = name
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        shape = model_input.shape
        if len(shape) != 4 or shape[1] != 3:
            raise ValueError(f"unsupported_plate_model_input:{shape}; expected NCHW RGB")
        if isinstance(shape[2], int) and shape[2] != input_size:
            raise ValueError(f"plate_model_input_size_mismatch:{shape[2]} != {input_size}")
        if isinstance(shape[3], int) and shape[3] != input_size:
            raise ValueError(f"plate_model_input_size_mismatch:{shape[3]} != {input_size}")

    @property
    def name(self) -> str:
        return self._name

    def detect_scored(self, frame: np.ndarray) -> List[ScoredBox]:
        h, w = frame.shape[:2]
        blob, ratio, pad = self._letterbox(frame)
        outputs = self.session.run(None, {self.input_name: blob})
        preds = np.asarray(outputs[0])
        if preds.ndim == 3 and preds.shape[0] == 1:
            preds = preds[0]
        if preds.ndim != 2 or min(preds.shape) < 5:
            raise ValueError(f"unsupported_plate_model_output:{preds.shape}")
        if preds.shape[0] < preds.shape[1]:
            preds = preds.T
        if preds.shape[1] < 5 or preds.shape[1] > 256:
            raise ValueError(f"unsupported_plate_model_output:{preds.shape}")
        if not np.isfinite(preds).all():
            raise ValueError("plate_model_output_nonfinite")
        boxes_xywh = preds[:, :4]
        scores_all = preds[:, 4:]
        if scores_all.shape[1] == 1:
            class_ids = np.zeros(scores_all.shape[0], dtype=np.int32)
            confidences = scores_all[:, 0]
        else:
            class_ids = np.argmax(scores_all, axis=1)
            confidences = scores_all[np.arange(scores_all.shape[0]), class_ids]

        mask = (confidences >= self.confidence_threshold) & (confidences <= 1.0)
        mask &= (boxes_xywh[:, 2] > 0) & (boxes_xywh[:, 3] > 0)
        if self.class_ids is not None:
            mask &= np.isin(class_ids, list(self.class_ids))
        boxes_xywh = boxes_xywh[mask]
        confidences = confidences[mask]
        if boxes_xywh.size == 0:
            return []

        boxes = self._scale_boxes(boxes_xywh, ratio, pad, w, h)
        keep = _nms(boxes, confidences, self.iou_threshold)
        out: List[ScoredBox] = []
        for idx in keep:
            x1, y1, x2, y2 = boxes[idx]
            bw = int(x2 - x1)
            bh = int(y2 - y1)
            if bw <= 0 or bh <= 0:
                continue
            out.append(((int(x1), int(y1), bw, bh), float(confidences[idx])))
        return out

    def detect(self, frame: np.ndarray) -> List[Box]:
        return [box for box, _ in self.detect_scored(frame)]

    def _letterbox(self, img: np.ndarray) -> tuple[np.ndarray, float, tuple[float, float]]:
        shape = img.shape[:2]
        r = min(self.input_size / shape[0], self.input_size / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw = self.input_size - new_unpad[0]
        dh = self.input_size - new_unpad[1]
        dw /= 2
        dh /= 2
        resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        blob = rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return blob, r, (dw, dh)

    def _scale_boxes(
        self,
        boxes_xywh: np.ndarray,
        ratio: float,
        pad: tuple[float, float],
        orig_w: int,
        orig_h: int,
    ) -> np.ndarray:
        dw, dh = pad
        xywh = boxes_xywh.copy()
        if xywh[:, 2].max() <= 1.5 and xywh[:, 3].max() <= 1.5:
            xywh[:, 0] *= self.input_size
            xywh[:, 1] *= self.input_size
            xywh[:, 2] *= self.input_size
            xywh[:, 3] *= self.input_size
        cx, cy, bw, bh = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2
        x1 = (x1 - dw) / ratio
        y1 = (y1 - dh) / ratio
        x2 = (x2 - dw) / ratio
        y2 = (y2 - dh) / ratio
        x1 = np.clip(x1, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        x2 = np.clip(x2, 0, orig_w)
        y2 = np.clip(y2, 0, orig_h)
        return np.stack([x1, y1, x2, y2], axis=-1)
