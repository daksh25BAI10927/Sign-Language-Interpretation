"""
Abstract ML model interface.

Defines the contract that any sign-recognition model must satisfy so it
can be plugged into the pipeline without touching camera, hand-detection,
preprocessing, API, or frontend code.

To add a real model later:

    1. Create e.g. ``backend/model/real_model.py``.
    2. Subclass ``SignLanguageModel`` and implement ``predict``.
    3. Load your TensorFlow/PyTorch/ONNX/scikit-learn model in ``__init__``.
    4. Swap the instantiation in ``main.py`` (or wherever the pipeline is
       built) from ``MockSignLanguageModel()`` to your new class.

Nothing else in the codebase needs to change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

# Sentinel labels used across the system for non-standard outcomes.
LABEL_NO_HAND = "No sign detected"
LABEL_UNKNOWN = "Unknown"
LABEL_LOW_CONFIDENCE = "Low confidence"


@dataclass(frozen=True)
class Prediction:
    """The result of a single model inference."""

    label: str
    confidence: float

    def to_dict(self) -> dict:
        return {"label": self.label, "confidence": round(self.confidence, 4)}


class SignLanguageModel(ABC):
    """Abstract base class for any sign-recognition model implementation."""

    @abstractmethod
    def predict(self, features: Optional[np.ndarray]) -> Prediction:
        """Predict a sign label from a preprocessed feature vector.

        Args:
            features: A 1D feature vector produced by
                ``backend.preprocessing.preprocessor.Preprocessor``, or
                ``None`` if no hand was detected in the current frame.

        Returns:
            A ``Prediction`` containing a label and confidence score.
            Implementations should return ``Prediction(LABEL_NO_HAND, 0.0)``
            when ``features`` is ``None`` rather than raising.
        """
        raise NotImplementedError

    def warmup(self) -> None:
        """Optional hook for models that benefit from a warm-up pass
        (e.g. first-inference JIT compilation). Default is a no-op."""
        return None

    def close(self) -> None:
        """Optional hook to release model resources. Default is a no-op."""
        return None
