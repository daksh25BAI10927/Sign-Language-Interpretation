"""
Camera module.

This module is responsible ONLY for camera-related functionality:
opening the device, continuously capturing frames, and releasing the
device safely. It contains no hand-detection or ML logic whatsoever,
so it can be reused independently in any other project.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraError(RuntimeError):
    """Raised when the camera cannot be opened or read from."""


class Camera:
    """Thin, thread-safe wrapper around an OpenCV ``VideoCapture`` device.

    Frames are captured on a dedicated background thread so that
    consumers (e.g. a processing pipeline) can call :meth:`read` at any
    time to get the most recent frame without blocking on I/O.
    """

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        mirror: bool = True,
    ) -> None:
        self._camera_index = camera_index
        self._width = width
        self._height = height
        self._fps = fps
        self._mirror = mirror

        self._capture: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()

        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_ready = threading.Event()
        self._last_error: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Open the camera device and begin capturing frames in the background.

        Raises:
            CameraError: If the camera device cannot be opened.
        """
        if self.is_running():
            logger.warning("Camera.start() called but camera is already running")
            return

        import sys
        if sys.platform.startswith("win"):
            capture = cv2.VideoCapture(self._camera_index, cv2.CAP_DSHOW)
        else:
            capture = cv2.VideoCapture(self._camera_index)
        if not capture.isOpened():
            capture.release()
            raise CameraError(
                f"Could not open camera at index {self._camera_index}. "
                "Check that the device exists and is not in use by another app."
            )

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        capture.set(cv2.CAP_PROP_FPS, self._fps)

        self._capture = capture
        self._last_error = None
        self._running.set()
        self._thread = threading.Thread(
            target=self._capture_loop, name="camera-capture-loop", daemon=True
        )
        self._thread.start()
        logger.info(
            "Camera started (index=%s, requested=%sx%s@%sfps)",
            self._camera_index,
            self._width,
            self._height,
            self._fps,
        )

    def stop(self) -> None:
        """Stop capturing and release the camera device."""
        if not self.is_running():
            return

        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        if self._capture is not None:
            self._capture.release()
            self._capture = None

        with self._lock:
            self._latest_frame = None
        self._frame_ready.clear()
        logger.info("Camera stopped and released")

    def is_running(self) -> bool:
        """Return whether the capture loop is currently active."""
        return self._running.is_set()

    def read(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Return the most recently captured frame.

        Args:
            timeout: Max seconds to wait for the first frame to become
                available if none has been captured yet.

        Returns:
            A BGR ``numpy.ndarray`` frame, or ``None`` if unavailable.
        """
        if not self.is_running():
            return None

        if self._latest_frame is None:
            self._frame_ready.wait(timeout=timeout)

        with self._lock:
            if self._latest_frame is None:
                return None
            # Return a copy so callers can safely mutate/draw on it
            # without racing the capture thread.
            return self._latest_frame.copy()

    @property
    def last_error(self) -> Optional[str]:
        """Return the last error message encountered by the capture loop, if any."""
        return self._last_error

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _capture_loop(self) -> None:
        assert self._capture is not None
        consecutive_failures = 0
        max_consecutive_failures = 30  # ~1 second at 30fps before giving up

        while self._running.is_set():
            ok, frame = self._capture.read()
            if not ok or frame is None:
                consecutive_failures += 1
                self._last_error = "Failed to read frame from camera"
                logger.warning(
                    "Camera read failure %s/%s", consecutive_failures, max_consecutive_failures
                )
                if consecutive_failures >= max_consecutive_failures:
                    logger.error("Camera appears disconnected; stopping capture loop")
                    self._running.clear()
                    break
                time.sleep(0.05)
                continue

            consecutive_failures = 0
            if self._mirror:
                frame = cv2.flip(frame, 1)

            with self._lock:
                self._latest_frame = frame
            self._frame_ready.set()

        # Ensure the device is released even if the loop exits due to
        # repeated read failures rather than an explicit stop() call.
        if self._capture is not None:
            self._capture.release()
