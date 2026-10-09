"""Privacy redaction facade (faces + license plates) for Django/Celery pipelines."""
from __future__ import annotations

import logging

from agents.ml.face_blur.unified_privacy_redaction import UnifiedPrivacyRedactionService

logger = logging.getLogger(__name__)


class PrivacyRedactionService:
    def __init__(
        self,
        blur_strength: int = 51,
        confidence_threshold: float = 0.5,
        *,
        blur_faces: bool = True,
        blur_plates: bool = True,
    ):
        self._inner = UnifiedPrivacyRedactionService.from_env(
            blur_strength=blur_strength,
            confidence_threshold=confidence_threshold,
            blur_faces=blur_faces,
            blur_plates=blur_plates,
        )

    def redact_video(self, input_path: str, output_path: str) -> dict:
        return self._inner.redact_video_fast(input_path, output_path)
