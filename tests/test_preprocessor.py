"""Tests for the Preprocessor module."""

from __future__ import annotations

import numpy as np
import pytest

from backend.hand_detection.landmarks import DetectedHand, Landmark
from backend.preprocessing.preprocessor import Preprocessor, PreprocessConfig


def _make_hand(offset: float = 0.0) -> DetectedHand:
    landmarks = [
        Landmark(x=0.1 * i + offset, y=0.05 * i + offset, z=0.01 * i)
        for i in range(21)
    ]
    return DetectedHand(landmarks=landmarks, handedness="Right")


def test_process_returns_none_for_none_hand():
    preprocessor = Preprocessor()
    assert preprocessor.process(None) is None


def test_process_returns_none_for_invalid_landmark_count():
    hand = DetectedHand(landmarks=[Landmark(0.0, 0.0, 0.0)] * 5)
    preprocessor = Preprocessor()
    assert preprocessor.process(hand) is None


def test_process_returns_flattened_vector_by_default():
    hand = _make_hand()
    preprocessor = Preprocessor()
    features = preprocessor.process(hand)

    assert features is not None
    assert features.ndim == 1
    assert features.shape[0] == 21 * 3  # x, y, z per landmark


def test_process_centers_on_wrist():
    hand = _make_hand(offset=1.0)
    preprocessor = Preprocessor(PreprocessConfig(center_on_wrist=True, scale_normalize=False))
    features = preprocessor.process(hand).reshape(21, 3)

    # The wrist (landmark 0) should be at the origin after centering.
    np.testing.assert_allclose(features[0], np.zeros(3), atol=1e-6)


def test_process_without_z_produces_2d_features():
    hand = _make_hand()
    preprocessor = Preprocessor(PreprocessConfig(include_z=False))
    features = preprocessor.process(hand)

    assert features.shape[0] == 21 * 2


def test_process_is_deterministic():
    hand = _make_hand()
    preprocessor = Preprocessor()
    first = preprocessor.process(hand)
    second = preprocessor.process(hand)
    np.testing.assert_array_equal(first, second)


def test_process_many_skips_invalid_hands():
    valid_hand = _make_hand()
    invalid_hand = DetectedHand(landmarks=[Landmark(0.0, 0.0, 0.0)] * 3)
    preprocessor = Preprocessor()

    features = preprocessor.process_many([valid_hand, invalid_hand])
    assert len(features) == 1
