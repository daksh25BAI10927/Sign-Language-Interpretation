"""Tests for the Camera module.

These tests avoid requiring real camera hardware by patching
``cv2.VideoCapture`` with an in-memory fake.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backend.camera.camera import Camera, CameraError


def _fake_capture(opened: bool = True, frame: "np.ndarray | None" = None):
    cap = MagicMock()
    cap.isOpened.return_value = opened
    if frame is None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cap.read.return_value = (True, frame)
    return cap


def test_start_raises_camera_error_when_device_unavailable():
    with patch("cv2.VideoCapture", return_value=_fake_capture(opened=False)):
        camera = Camera(camera_index=0)
        with pytest.raises(CameraError):
            camera.start()


def test_start_sets_running_state():
    with patch("cv2.VideoCapture", return_value=_fake_capture(opened=True)):
        camera = Camera(camera_index=0)
        camera.start()
        try:
            assert camera.is_running() is True
        finally:
            camera.stop()


def test_read_returns_frame_after_start():
    with patch("cv2.VideoCapture", return_value=_fake_capture(opened=True)):
        camera = Camera(camera_index=0)
        camera.start()
        try:
            frame = camera.read(timeout=1.0)
            assert frame is not None
            assert frame.shape == (480, 640, 3)
        finally:
            camera.stop()


def test_read_returns_none_when_not_started():
    camera = Camera(camera_index=0)
    assert camera.read(timeout=0.1) is None


def test_stop_releases_and_clears_state():
    fake_cap = _fake_capture(opened=True)
    with patch("cv2.VideoCapture", return_value=fake_cap):
        camera = Camera(camera_index=0)
        camera.start()
        camera.stop()
        assert camera.is_running() is False
        fake_cap.release.assert_called()
