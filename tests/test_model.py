"""Tests for the mock ML model implementation."""

from __future__ import annotations

import numpy as np
import pytest

from backend.model.base_model import LABEL_NO_HAND, LABEL_UNKNOWN
from backend.model.mock_model import MockSignLanguageModel


def test_predict_returns_no_hand_label_for_none_features():
    model = MockSignLanguageModel()
    prediction = model.predict(None)
    assert prediction.label == LABEL_NO_HAND
    assert prediction.confidence == 0.0


def test_predict_returns_unknown_for_empty_features():
    model = MockSignLanguageModel()
    prediction = model.predict(np.array([]))
    assert prediction.label == LABEL_UNKNOWN


def test_predict_returns_unknown_for_non_finite_features():
    model = MockSignLanguageModel()
    features = np.array([1.0, np.nan, 3.0])
    prediction = model.predict(features)
    assert prediction.label == LABEL_UNKNOWN


def test_predict_returns_label_from_pool_for_valid_features():
    model = MockSignLanguageModel(labels=["A", "B", "Hello"], confidence_threshold=0.0)
    features = np.random.default_rng(0).random(63).astype(np.float32)
    prediction = model.predict(features)
    assert prediction.label in ["A", "B", "Hello"]
    assert 0.0 <= prediction.confidence <= 1.0


def test_predict_is_deterministic_for_same_features():
    model = MockSignLanguageModel(seed=1)
    features = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    first = model.predict(features)
    second = model.predict(features)
    assert first.label == second.label


def test_prediction_to_dict():
    model = MockSignLanguageModel()
    prediction = model.predict(None)
    d = prediction.to_dict()
    assert d["label"] == LABEL_NO_HAND
    assert "confidence" in d
