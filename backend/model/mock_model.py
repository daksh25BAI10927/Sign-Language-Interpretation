"""
Mock sign-language model.

A drop-in stand-in for a real trained model, used so the entire backend
pipeline can be built, run, and tested before the actual ML model exists.
It returns plausible-looking predictions derived deterministically (but
pseudo-randomly) from the input features, so behavior is stable and
reproducible across runs for the same input.

Replace with a real implementation later by subclassing
``backend.model.base_model.SignLanguageModel`` and swapping the
instantiation in ``main.py``.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from backend.model.base_model import (
    LABEL_LOW_CONFIDENCE,
    LABEL_NO_HAND,
    LABEL_UNKNOWN,
    Prediction,
    SignLanguageModel,
)

logger = logging.getLogger(__name__)

DEFAULT_LABELS: List[str] = ["A", "B", "C", "Hello", "Thank You", "Yes", "No"]


class MockSignLanguageModel(SignLanguageModel):
    """Deterministic mock predictor for development and testing.

    Args:
        labels: The pool of candidate sign labels to "predict" from.
        confidence_threshold: Predictions below this confidence are
            reported as ``LABEL_LOW_CONFIDENCE`` instead of the raw label,
            mirroring how a real model's low-confidence outputs would be
            handled.
        seed: Random seed for reproducibility across runs.
    """

    def __init__(
        self,
        labels: Optional[List[str]] = None,
        confidence_threshold: float = 0.5,
        seed: int = 42,
    ) -> None:
        self._labels = labels or DEFAULT_LABELS
        self._confidence_threshold = confidence_threshold
        self._rng = np.random.default_rng(seed)
        logger.info(
            "MockSignLanguageModel initialized with %s labels (threshold=%s)",
            len(self._labels),
            confidence_threshold,
        )

    def predict(self, features: Optional[np.ndarray]) -> Prediction:
        if features is None:
            return Prediction(label=LABEL_NO_HAND, confidence=0.0)

        if features.size == 0 or not np.isfinite(features).all():
            logger.debug("Received invalid feature vector; returning Unknown")
            return Prediction(label=LABEL_UNKNOWN, confidence=0.0)

        # Derive a deterministic-but-varying pseudo prediction from the
        # feature values themselves, so the same hand pose tends to
        # produce the same mock label (useful for demos/tests) while
        # different poses produce different labels.
        feature_hash = float(np.sum(np.abs(features)) * 1000) % 1.0
        label_index = int(feature_hash * len(self._labels)) % len(self._labels)
        confidence = float(0.5 + 0.5 * self._rng.random())

        label = self._labels[label_index]

        if confidence < self._confidence_threshold:
            return Prediction(label=LABEL_LOW_CONFIDENCE, confidence=confidence)

        return Prediction(label=label, confidence=confidence)
