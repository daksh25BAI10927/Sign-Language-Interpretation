"""
real_model.py
-------------
Real sign-language model implementation using an ONNX model exported
from the training pipeline.

Uses onnxruntime for inference — no TensorFlow/Keras dependency at
runtime, making this compatible with Python 3.14 and any future
Python version.

To activate this model, in backend/main.py replace:
    model = MockSignLanguageModel(...)
with:
    model = RealSignLanguageModel()

Nothing else needs to change.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import onnxruntime as ort

from backend.model.base_model import (
    LABEL_LOW_CONFIDENCE,
    LABEL_NO_HAND,
    LABEL_UNKNOWN,
    Prediction,
    SignLanguageModel,
)

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_PATH  = Path(__file__).parent / "sign_model.onnx"
_DEFAULT_LABELS_PATH = Path(__file__).parent / "labels.json"


class RealSignLanguageModel(SignLanguageModel):
    """
    Production sign-language classifier backed by an ONNX model.

    The model expects a (63,) float32 feature vector: the wrist-centred,
    scale-normalised concatenation of 21 MediaPipe hand landmarks
    (x, y, z) — identical to what backend.preprocessing.Preprocessor
    already produces.

    Args:
        model_path:           Path to sign_model.onnx
        labels_path:          Path to labels.json
        confidence_threshold: Predictions below this are returned as
                              LABEL_LOW_CONFIDENCE instead of the label.
    """

    def __init__(
        self,
        model_path: str | Path = _DEFAULT_MODEL_PATH,
        labels_path: str | Path = _DEFAULT_LABELS_PATH,
        confidence_threshold: float = 0.6,
    ) -> None:
        model_path  = Path(model_path)
        labels_path = Path(labels_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found at '{model_path}'. "
                "Run scripts/train_model.py first to generate it."
            )
        if not labels_path.exists():
            raise FileNotFoundError(
                f"Labels file not found at '{labels_path}'. "
                "Run scripts/train_model.py first to generate it."
            )

        logger.info("Loading ONNX model from %s", model_path)
        self._session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name

        with open(labels_path, "r", encoding="utf-8") as f:
            self._labels: List[str] = json.load(f)

        self._confidence_threshold = confidence_threshold

        logger.info(
            "RealSignLanguageModel ready | %d classes | threshold=%.2f",
            len(self._labels),
            confidence_threshold,
        )

    def predict(self, features: Optional[np.ndarray]) -> Prediction:
        """
        Predict a sign label from a (63,) landmark feature vector.

        Args:
            features: 1-D numpy array of shape (63,), or None if no hand
                      was detected in the current frame.

        Returns:
            Prediction(label, confidence)
        """
        if features is None:
            return Prediction(label=LABEL_NO_HAND, confidence=0.0)

        if features.size == 0 or not np.isfinite(features).all():
            logger.debug("Invalid feature vector received; returning Unknown")
            return Prediction(label=LABEL_UNKNOWN, confidence=0.0)

        # ONNX Runtime expects a 2-D batch: (1, 63)
        x = features.astype(np.float32).reshape(1, -1)

        try:
            outputs = self._session.run(None, {self._input_name: x})
        except Exception:
            logger.exception("ONNX inference failed")
            return Prediction(label=LABEL_UNKNOWN, confidence=0.0)

        probabilities = outputs[0][0]          # shape: (num_classes,)
        class_index   = int(np.argmax(probabilities))
        confidence    = float(probabilities[class_index])

        if class_index >= len(self._labels):
            logger.warning("Model returned unexpected class index %d", class_index)
            return Prediction(label=LABEL_UNKNOWN, confidence=confidence)

        label = self._labels[class_index]

        if confidence < self._confidence_threshold:
            return Prediction(label=LABEL_LOW_CONFIDENCE, confidence=confidence)

        return Prediction(label=label, confidence=confidence)

    def warmup(self) -> None:
        """Run a dummy inference to warm up the ONNX Runtime session."""
        dummy = np.zeros((1, 63), dtype=np.float32)
        self._session.run(None, {self._input_name: dummy})
        logger.debug("ONNX session warmed up")

    def close(self) -> None:
        """No-op: onnxruntime sessions are garbage-collected automatically."""
        logger.debug("RealSignLanguageModel.close() called (no-op)")
