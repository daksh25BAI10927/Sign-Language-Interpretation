"""Tests for the HandDetector module and landmark data structures."""

from __future__ import annotations

import numpy as np
import pytest

from backend.hand_detection.landmarks import DetectedHand, DetectionResult, Landmark


def test_detected_hand_is_valid_with_21_landmarks():
    landmarks = [Landmark(x=0.1 * i, y=0.1 * i, z=0.0) for i in range(21)]
    hand = DetectedHand(landmarks=landmarks, handedness="Right")
    assert hand.is_valid() is True


def test_detected_hand_is_invalid_with_wrong_landmark_count():
    landmarks = [Landmark(x=0.0, y=0.0, z=0.0) for _ in range(10)]
    hand = DetectedHand(landmarks=landmarks)
    assert hand.is_valid() is False


def test_detection_result_no_hand():
    result = DetectionResult()
    assert result.hand_detected is False
    assert result.hand_count == 0
    assert result.primary_hand() is None


def test_detection_result_with_hands():
    landmarks = [Landmark(x=0.0, y=0.0, z=0.0) for _ in range(21)]
    hand = DetectedHand(landmarks=landmarks, handedness="Left")
    result = DetectionResult(hands=[hand], image_width=640, image_height=480)

    assert result.hand_detected is True
    assert result.hand_count == 1
    assert result.primary_hand() is hand


def _build_real_detector():
    """Build a real HandDetector, skipping the test if the MediaPipe
    HandLandmarker model bundle can't be obtained (e.g. no internet
    access in a sandboxed/offline CI environment). On a normal
    developer machine with internet access this will succeed and the
    model will be cached under backend/hand_detection/models/ for
    subsequent runs.
    """
    from backend.hand_detection.detector import HandDetector, HandDetectorError

    try:
        return HandDetector(max_num_hands=1)
    except HandDetectorError as exc:
        pytest.skip(f"HandLandmarker model unavailable in this environment: {exc}")


def test_hand_detector_returns_empty_result_for_blank_frame():
    """Smoke test using the real MediaPipe detector on a blank frame.

    A blank (all-black) frame should not contain a hand, exercising the
    "no hand detected" path end-to-end.
    """
    detector = _build_real_detector()
    try:
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(blank_frame)
        assert result.hand_detected is False
        assert result.image_width == 640
        assert result.image_height == 480
    finally:
        detector.close()


def test_hand_detector_handles_none_frame_gracefully():
    detector = _build_real_detector()
    try:
        result = detector.detect(None)
        assert result.hand_detected is False
    finally:
        detector.close()
