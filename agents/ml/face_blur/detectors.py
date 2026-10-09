"""Face box detectors used by UnifiedFaceBlurService."""
from __future__ import annotations

import logging
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from .boxes import box_iou, expand_face_to_head_box, nms_boxes

logger = logging.getLogger(__name__)

ScoredBox = Tuple[Tuple[int, int, int, int], float]


class FaceDetector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Return face boxes as (x, y, w, h) in pixel coordinates."""

    def detect_scored(self, frame: np.ndarray) -> List[ScoredBox]:
        """Optional scored detections for ByteTrack-style association."""
        return [(box, 1.0) for box in self.detect(frame)]

    @property
    def name(self) -> str:
        return self.__class__.__name__


def _postprocess_ssd_detections(
    detections: np.ndarray,
    frame_w: int,
    frame_h: int,
    confidence_threshold: float,
) -> List[Tuple[int, int, int, int]]:
    """Parse OpenCV SSD / compatible ONNX output; tolerates [1,1,N,7], [1,N,7], or [N,7]."""
    faces: List[Tuple[int, int, int, int]] = []
    if detections is None or detections.size == 0:
        return faces
    d = np.squeeze(np.array(detections))
    if d.ndim == 1:
        d = d.reshape(1, -1)
    if d.ndim != 2 or d.shape[1] < 7:
        logger.warning("Unexpected SSD detection tensor shape %s", getattr(d, "shape", None))
        return faces
    for i in range(d.shape[0]):
        confidence = float(d[i, 2])
        if confidence <= confidence_threshold:
            continue
        box = d[i, 3:7] * np.array([frame_w, frame_h, frame_w, frame_h])
        x, y, x2, y2 = box.astype("int")
        x = max(0, x)
        y = max(0, y)
        w_box = min(frame_w, x2) - x
        h_box = min(frame_h, y2) - y
        if w_box > 0 and h_box > 0:
            faces.append((x, y, w_box, h_box))
    return faces


class OpenCvCaffeFaceDetector(FaceDetector):
    """Res10 300x300 SSD from OpenCV samples + Haar fallback."""

    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self._net: cv2.dnn_Net | None = None
        self._cascade: cv2.CascadeClassifier | None = None
        self._init()

    def _init(self) -> None:
        try:
            from . import config as fb_config

            baked = fb_config.models_dir() / "opencv_ssd"
            if (baked / "deploy.prototxt").is_file() and (baked / "res10_300x300_ssd_iter_140000.caffemodel").is_file():
                prototxt_path = baked / "deploy.prototxt"
                model_path = baked / "res10_300x300_ssd_iter_140000.caffemodel"
            else:
                model_dir = Path(tempfile.gettempdir()) / "opencv_face_models"
                model_dir.mkdir(exist_ok=True)
                prototxt_path = model_dir / "deploy.prototxt"
                model_path = model_dir / "res10_300x300_ssd_iter_140000.caffemodel"
            prototxt_url = (
                "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/"
                "face_detector/deploy.prototxt"
            )
            model_url = (
                "https://raw.githubusercontent.com/opencv/opencv_3rdparty/"
                "dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
            )
            if not prototxt_path.is_file() or not model_path.is_file():
                logger.info("Downloading OpenCV DNN face detection model files...")
                import urllib.request

                prototxt_path.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(prototxt_url, prototxt_path)
                urllib.request.urlretrieve(model_url, model_path)
                logger.info("Face detection Caffe models ready")
            self._net = cv2.dnn.readNetFromCaffe(str(prototxt_path), str(model_path))
            logger.info("Initialized OpenCV Caffe SSD face detector")
        except Exception as e:
            logger.warning("OpenCV Caffe SSD init failed (%s); using Haar cascade", e)
            self._net = None
            self._init_haar()

    def _init_haar(self) -> None:
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            cascade = cv2.CascadeClassifier(cascade_path)
            if cascade.empty():
                raise RuntimeError("Haar cascade file missing")
            self._cascade = cascade
            logger.info("Initialized Haar cascade face detector")
        except Exception as e:
            logger.error("Haar cascade init failed: %s", e)
            self._cascade = None

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        if self._net is not None:
            return self._detect_dnn(frame)
        if self._cascade is not None:
            return self._detect_haar(frame)
        return []

    def _detect_dnn(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), [104, 117, 123])
        assert self._net is not None
        self._net.setInput(blob)
        detections = self._net.forward()
        raw = _postprocess_ssd_detections(detections, w, h, self.confidence_threshold)
        from . import config as fb_config

        if not fb_config.head_expand_enabled():
            return raw
        return [
            expand_face_to_head_box(
                x=float(x),
                y=float(y),
                w=float(bw),
                h=float(bh),
                landmarks=None,
                frame_w=w,
                frame_h=h,
            )
            for (x, y, bw, bh) in raw
        ]

    def _detect_haar(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        assert self._cascade is not None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected = self._cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in detected]


class YuNetFaceDetector(FaceDetector):
    """OpenCV Zoo YuNet with landmark head expansion and dual-confidence recall."""

    def __init__(
        self,
        onnx_path: str,
        confidence_threshold: float = 0.30,
        low_confidence_threshold: float = 0.18,
        input_size: tuple[int, int] = (320, 320),
        *,
        expand_to_head: bool = True,
    ):
        if not hasattr(cv2, "FaceDetectorYN"):
            raise RuntimeError("cv2.FaceDetectorYN unavailable (opencv-python >= 4.5.4 required)")
        self.confidence_threshold = confidence_threshold
        self.low_confidence_threshold = min(low_confidence_threshold, confidence_threshold)
        self.expand_to_head = expand_to_head
        self._input_size = input_size
        self._onnx_path = onnx_path
        self._detector = cv2.FaceDetectorYN.create(onnx_path, "", input_size)
        logger.info("Initialized YuNet face detector from %s", onnx_path)

    def _scored_at_size(self, frame: np.ndarray, *, frame_w: int, frame_h: int) -> List[ScoredBox]:
        self._detector.setInputSize((frame_w, frame_h))
        _count, faces = self._detector.detect(frame)
        if faces is None or len(faces) == 0:
            return []
        scored: List[ScoredBox] = []
        for row in faces:
            score = float(row[14]) if len(row) > 14 else float(row[-1])
            if score < self.low_confidence_threshold:
                continue
            x, y, bw, bh = [float(v) for v in row[0:4]]
            if bw <= 1 or bh <= 1:
                continue
            landmarks = np.array(row[4:14], dtype=np.float32).reshape(5, 2) if len(row) >= 14 else None
            if self.expand_to_head:
                box = expand_face_to_head_box(
                    x=x, y=y, w=bw, h=bh, landmarks=landmarks, frame_w=frame_w, frame_h=frame_h
                )
            else:
                box = (int(x), int(y), int(bw), int(bh))
            scored.append((box, score))
        return scored

    def detect_scored(self, frame: np.ndarray) -> List[ScoredBox]:
        h, w = frame.shape[:2]
        scored = self._scored_at_size(frame, frame_w=w, frame_h=h)
        if max(h, w) > 720:
            scale = 720.0 / max(h, w)
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            resized = cv2.resize(frame, (nw, nh))
            for box, score in self._scored_at_size(resized, frame_w=nw, frame_h=nh):
                x, y, bw, bh = box
                scored.append(
                    (
                        (int(x / scale), int(y / scale), max(1, int(bw / scale)), max(1, int(bh / scale))),
                        score,
                    )
                )
        if not scored:
            return []
        scored.sort(key=lambda item: item[1], reverse=True)
        kept: List[ScoredBox] = []
        for box, score in scored:
            if all(box_iou(box, k[0]) < 0.42 for k in kept):
                kept.append((box, score))
        return kept

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        return [box for box, score in self.detect_scored(frame) if score >= self.confidence_threshold]

    @property
    def name(self) -> str:
        return f"YuNetFaceDetector({Path(self._onnx_path).name})"


class EnsembleFaceDetector(FaceDetector):
    """Merge multiple detectors (recall boost for occlusion / profile faces)."""

    def __init__(self, detectors: List[FaceDetector]):
        if not detectors:
            raise ValueError("ensemble_requires_at_least_one_detector")
        self._detectors = detectors

    def detect_scored(self, frame: np.ndarray) -> List[ScoredBox]:
        merged: List[ScoredBox] = []
        for detector in self._detectors:
            merged.extend(detector.detect_scored(frame))
        if not merged:
            return []
        merged.sort(key=lambda item: item[1], reverse=True)
        kept: List[ScoredBox] = []
        for box, score in merged:
            if all(box_iou(box, k[0]) < 0.48 for k in kept):
                kept.append((box, score))
        return kept

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        return [box for box, _score in self.detect_scored(frame)]

    @property
    def name(self) -> str:
        names = "+".join(d.name for d in self._detectors)
        return f"Ensemble({names})"


class ScrfdFaceDetector(FaceDetector):
    """SCRFD-2.5G with keypoints — strong recall on Hard/WIDER and occlusions."""

    def __init__(
        self,
        onnx_path: str,
        confidence_threshold: float = 0.30,
        low_confidence_threshold: float = 0.12,
        *,
        expand_to_head: bool = True,
    ):
        from .scrfd_engine import ScrfdOnnxEngine

        self.confidence_threshold = confidence_threshold
        self.low_confidence_threshold = min(low_confidence_threshold, confidence_threshold)
        self.expand_to_head = expand_to_head
        self._input_size = (640, 640)
        self._onnx_path = onnx_path
        self._engine = ScrfdOnnxEngine(onnx_path)
        logger.info("Initialized SCRFD face detector from %s", onnx_path)

    def _to_head_boxes(
        self,
        det: np.ndarray,
        kpss: np.ndarray | None,
        *,
        frame_w: int,
        frame_h: int,
    ) -> List[ScoredBox]:
        out: List[ScoredBox] = []
        for i in range(det.shape[0]):
            x1, y1, x2, y2, score = det[i]
            if float(score) < self.low_confidence_threshold:
                continue
            x, y, w, h = float(x1), float(y1), float(x2 - x1), float(y2 - y1)
            landmarks = kpss[i] if kpss is not None and i < len(kpss) else None
            if self.expand_to_head:
                box = expand_face_to_head_box(
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    landmarks=landmarks,
                    frame_w=frame_w,
                    frame_h=frame_h,
                )
            else:
                box = (int(x), int(y), max(1, int(w)), max(1, int(h)))
            out.append((box, float(score)))
        return out

    def _detect_scaled(
        self,
        frame: np.ndarray,
        *,
        frame_w: int,
        frame_h: int,
        scale: float = 1.0,
    ) -> List[ScoredBox]:
        if scale != 1.0:
            nw, nh = max(1, int(frame_w * scale)), max(1, int(frame_h * scale))
            img = cv2.resize(frame, (nw, nh))
            det, kpss = self._engine.detect_raw(
                img,
                thresh=self.low_confidence_threshold,
                input_size=self._input_size,
            )
            out: List[ScoredBox] = []
            for i in range(det.shape[0]):
                x1, y1, x2, y2, score = det[i]
                if float(score) < self.low_confidence_threshold:
                    continue
                x, y, bw, bh = float(x1 / scale), float(y1 / scale), float((x2 - x1) / scale), float((y2 - y1) / scale)
                landmarks = kpss[i] / scale if kpss is not None and i < len(kpss) else None
                if self.expand_to_head:
                    box = expand_face_to_head_box(
                        x=x, y=y, w=bw, h=bh, landmarks=landmarks, frame_w=frame_w, frame_h=frame_h
                    )
                else:
                    box = (int(x), int(y), max(1, int(bw)), max(1, int(bh)))
                out.append((box, float(score)))
            return out

        det, kpss = self._engine.detect_raw(
            frame,
            thresh=self.low_confidence_threshold,
            input_size=self._input_size,
        )
        return self._to_head_boxes(det, kpss, frame_w=frame_w, frame_h=frame_h)

    def detect_scored(self, frame: np.ndarray) -> List[ScoredBox]:
        h, w = frame.shape[:2]
        merged = self._detect_scaled(frame, frame_w=w, frame_h=h, scale=1.0)
        if max(h, w) > 960:
            merged.extend(self._detect_scaled(frame, frame_w=w, frame_h=h, scale=960.0 / max(h, w)))
        if not merged:
            return []
        merged.sort(key=lambda item: item[1], reverse=True)
        kept: List[ScoredBox] = []
        for box, score in merged:
            if all(box_iou(box, k[0]) < 0.42 for k in kept):
                kept.append((box, score))
        return kept

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        return [box for box, score in self.detect_scored(frame) if score >= self.confidence_threshold]

    @property
    def name(self) -> str:
        return f"ScrfdFaceDetector({Path(self._onnx_path).name})"


class OnnxSsdFaceDetector(FaceDetector):
    """
    ONNX Runtime inference for SSD-style face outputs [1, 1, N, 7].

    Expects the same logical mapping as OpenCV's face_detector sample
    (class, score, xmin, ymin, xmax, ymax in normalized 0..1 for corners).
    """

    def __init__(self, onnx_path: str, confidence_threshold: float = 0.5, input_size: int = 300):
        import onnxruntime as ort  # type: ignore

        self.confidence_threshold = confidence_threshold
        self.input_size = input_size
        self._onnx_path = onnx_path
        avail = ort.get_available_providers()
        prov = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in avail:
            prov = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._session = ort.InferenceSession(onnx_path, providers=prov)
        self._input_name = self._session.get_inputs()[0].name
        outs = self._session.get_outputs()
        self._output_names = [o.name for o in outs]
        logger.info("Initialized ONNX face detector from %s", onnx_path)

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (self.input_size, self.input_size)),
            1.0,
            (self.input_size, self.input_size),
            [104, 117, 123],
        )
        outputs = self._session.run(self._output_names, {self._input_name: blob})
        detections = outputs[0]
        if detections.ndim < 2 or detections.shape[-1] < 7:
            logger.warning(
                "Unexpected ONNX output shape %s from %s; expected SSD [...,7]",
                getattr(detections, "shape", None),
                self._onnx_path,
            )
            return []
        return _postprocess_ssd_detections(detections, w, h, self.confidence_threshold)

    @property
    def name(self) -> str:
        return f"OnnxSsdFaceDetector({Path(self._onnx_path).name})"
