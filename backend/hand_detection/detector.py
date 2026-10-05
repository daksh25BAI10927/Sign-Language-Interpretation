"""
Hand detection module.

Wraps MediaPipe's HandLandmarker (the current Tasks API — the legacy
``mediapipe.solutions.hands`` API has been removed from recent MediaPipe
distributions) to detect one or more hands in a camera frame and return
structured landmark data (see ``landmarks.py``). This module has no
knowledge of the camera or the ML sign-classification model — it purely
converts an image frame into ``DetectionResult`` objects.

The HandLandmarker requires a small model-bundle file (``hand_landmarker.task``).
If it isn't found at ``model_asset_path``, this module will try to download
it automatically from Google's public model store on first use. If that
download fails (e.g. no internet access in a sandboxed/offline environment),
a clear ``HandDetectorError`` is raised explaining how to obtain the file
manually.
"""

from __future__ import annotations

import logging
import time
import urllib.request
from pathlib import Path
from typing import Optional

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

from backend.hand_detection.landmarks import DetectedHand, DetectionResult, Landmark

logger = logging.getLogger(__name__)

# Official Google-hosted MediaPipe HandLandmarker model bundle.
_DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)


class HandDetectorError(RuntimeError):
    """Raised when the MediaPipe HandLandmarker fails to initialize."""


def _ensure_model_available(model_asset_path: str, download_url: str = _DEFAULT_MODEL_URL) -> str:
    """Ensure the HandLandmarker model bundle exists on disk, downloading
    it if necessary. Returns the resolved path to the model file.

    Raises:
        HandDetectorError: If the model is missing and cannot be
            downloaded (e.g. no network access).
    """
    path = Path(model_asset_path)
    if path.exists():
        return str(path)

    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("HandLandmarker model not found at %s; downloading from %s", path, download_url)

    try:
        urllib.request.urlretrieve(download_url, str(path))
    except Exception as exc:
        raise HandDetectorError(
            f"HandLandmarker model file not found at '{model_asset_path}' and "
            f"automatic download failed ({exc}). Download it manually from "
            f"{download_url} and place it at that path, or set "
            "HAND_LANDMARKER_MODEL_PATH to an existing model file."
        ) from exc

    logger.info("Downloaded HandLandmarker model to %s", path)
    return str(path)


class HandDetector:
    """Detects hands and hand landmarks in BGR image frames using MediaPipe
    Tasks' HandLandmarker."""

    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
        model_asset_path: str = "backend/hand_detection/models/hand_landmarker.task",
    ) -> None:
        # model_complexity is accepted for interface/config compatibility
        # with the legacy API but is not used by HandLandmarker, which
        # picks its own model variant via the task bundle.
        self._max_num_hands = max_num_hands
        self._min_detection_confidence = min_detection_confidence
        self._min_tracking_confidence = min_tracking_confidence
        self._landmarker: Optional[HandLandmarker] = None

        try:
            resolved_model_path = _ensure_model_available(model_asset_path)

            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=resolved_model_path),
                running_mode=RunningMode.IMAGE,
                num_hands=self._max_num_hands,
                min_hand_detection_confidence=self._min_detection_confidence,
                min_tracking_confidence=self._min_tracking_confidence,
            )
            self._landmarker = HandLandmarker.create_from_options(options)
        except HandDetectorError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise HandDetectorError(f"Failed to initialize MediaPipe HandLandmarker: {exc}") from exc

        logger.info(
            "HandDetector initialized (max_hands=%s, detection_conf=%s, tracking_conf=%s)",
            max_num_hands,
            min_detection_confidence,
            min_tracking_confidence,
        )

    def detect(self, frame_bgr: Optional[np.ndarray]) -> DetectionResult:
        """Detect hands in a single BGR frame.

        Args:
            frame_bgr: The camera frame in BGR color order (OpenCV default).

        Returns:
            A ``DetectionResult`` — empty (``hand_detected`` False) if no
            hand was found. Never raises on "no hand"; that is a normal,
            expected outcome, not an error.
        """
        if frame_bgr is None or self._landmarker is None:
            return DetectionResult()

        height, width = frame_bgr.shape[:2]

        # MediaPipe Tasks expects RGB input wrapped in an mp.Image.
        frame_rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        try:
            result = self._landmarker.detect(mp_image)
        except Exception:
            logger.exception("MediaPipe hand processing failed for this frame")
            return DetectionResult(image_width=width, image_height=height)

        detection = DetectionResult(image_width=width, image_height=height)

        if not result.hand_landmarks:
            return detection

        handedness_list = result.handedness or []

        for index, hand_landmark_list in enumerate(result.hand_landmarks):
            landmarks = [
                Landmark(x=lm.x, y=lm.y, z=lm.z) for lm in hand_landmark_list
            ]

            handedness_label = "Unknown"
            handedness_confidence = 0.0
            if index < len(handedness_list) and handedness_list[index]:
                category = handedness_list[index][0]
                handedness_label = category.category_name or "Unknown"
                handedness_confidence = float(category.score)

            detection.hands.append(
                DetectedHand(
                    landmarks=landmarks,
                    handedness=handedness_label,
                    handedness_confidence=handedness_confidence,
                )
            )

        return detection

    def close(self) -> None:
        """Release MediaPipe resources. Safe to call multiple times."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
            logger.info("HandDetector closed")

    def __enter__(self) -> "HandDetector":
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()
