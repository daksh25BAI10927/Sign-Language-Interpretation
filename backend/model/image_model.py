"""Keras image model adapter for the trained sign classifier."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from backend.hand_detection.landmarks import DetectedHand
from backend.model.base_model import (
    LABEL_LOW_CONFIDENCE,
    LABEL_NO_HAND,
    LABEL_UNKNOWN,
    Prediction,
    SignLanguageModel,
)

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_PATH = Path(__file__).with_name("modelnet_model.h5")
_DEFAULT_LABELS_PATH = Path(__file__).with_name("model_labels.txt")
_IMAGE_SIZE = 224


class KerasImageSignLanguageModel(SignLanguageModel):
    """Load the trained CNN and classify cropped hand images.

    The model was trained on RGB images resized to 224x224 and normalized to
    the [0, 1] range. ``predict_frame`` is used by the live pipeline, while
    ``predict`` remains compatible with the generic model interface.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        labels_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
    ) -> None:
        resolved_model_path = Path(model_path or _DEFAULT_MODEL_PATH)
        resolved_labels_path = Path(labels_path or _DEFAULT_LABELS_PATH)

        if not resolved_model_path.exists():
            raise FileNotFoundError(f"Trained model not found: {resolved_model_path}")
        if not resolved_labels_path.exists():
            raise FileNotFoundError(f"Model labels not found: {resolved_labels_path}")

        try:
            from tensorflow.keras.models import load_model
        except ImportError as exc:
            raise RuntimeError(
                "TensorFlow is required to load the trained sign model. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc

        self._model = load_model(str(resolved_model_path), compile=False)
        self._labels = [
            label.strip()
            for label in resolved_labels_path.read_text(encoding="utf-8").splitlines()
            if label.strip()
        ]
        output_count = int(self._model.output_shape[-1])
        if len(self._labels) != output_count:
            raise ValueError(
                f"Model outputs {output_count} classes but labels file contains "
                f"{len(self._labels)} labels"
            )
        self._confidence_threshold = confidence_threshold
        logger.info("Loaded Keras sign model with %s labels", len(self._labels))

    def predict(self, features: Optional[np.ndarray]) -> Prediction:
        """Predict from a BGR image or a single-image batch."""
        if features is None:
            return Prediction(label=LABEL_NO_HAND, confidence=0.0)

        image = np.asarray(features)
        if image.ndim == 4 and image.shape[0] == 1:
            image = image[0]
        if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
            return Prediction(label=LABEL_UNKNOWN, confidence=0.0)

        batch = self._prepare_image(image)
        probabilities = np.asarray(self._model.predict(batch, verbose=0))[0]
        class_index = int(np.argmax(probabilities))
        confidence = float(probabilities[class_index])
        if confidence < self._confidence_threshold:
            return Prediction(label=LABEL_LOW_CONFIDENCE, confidence=confidence)
        return Prediction(label=self._labels[class_index], confidence=confidence)

    def predict_frame(
        self,
        frame_bgr: np.ndarray,
        hand: Optional[DetectedHand],
    ) -> Prediction:
        """Crop the detected hand from a camera frame and classify it."""
        if hand is None or not hand.is_valid():
            return Prediction(label=LABEL_NO_HAND, confidence=0.0)

        height, width = frame_bgr.shape[:2]
        x_values = [landmark.x * width for landmark in hand.landmarks]
        y_values = [landmark.y * height for landmark in hand.landmarks]
        padding = 20
        x_min = max(0, int(min(x_values)) - padding)
        y_min = max(0, int(min(y_values)) - padding)
        x_max = min(width, int(max(x_values)) + padding)
        y_max = min(height, int(max(y_values)) + padding)
        if x_max <= x_min or y_max <= y_min:
            return Prediction(label=LABEL_UNKNOWN, confidence=0.0)

        return self.predict(frame_bgr[y_min:y_max, x_min:x_max])

    @staticmethod
    def _prepare_image(image_bgr: np.ndarray) -> np.ndarray:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image_rgb, (_IMAGE_SIZE, _IMAGE_SIZE))
        normalized = resized.astype(np.float32) / 255.0
        return np.expand_dims(normalized, axis=0)

    def close(self) -> None:
        self._model = None
