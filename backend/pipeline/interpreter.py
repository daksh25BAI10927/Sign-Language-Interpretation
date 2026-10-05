"""
Central real-time processing pipeline.

``Interpreter`` wires together the camera, hand detector, preprocessor,
model, and visualizer, and runs them in a background thread so the
FastAPI server (or a simple ``main.py`` script) never blocks on the
camera loop. It exposes a small, clean surface (``start``, ``stop``,
``get_status``, ``get_latest_frame``) that the API layer and the
manual/dev entry point both consume identically.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np

from backend.camera.camera import Camera, CameraError
from backend.hand_detection.detector import HandDetector, HandDetectorError
from backend.model.base_model import Prediction, SignLanguageModel
from backend.preprocessing.preprocessor import Preprocessor
from backend.visualization.visualizer import Visualizer

logger = logging.getLogger(__name__)

# --- ML Fix 2: Temporal smoothing config ---
_SMOOTH_WINDOW = 7       # frames to majority-vote over — eliminates single-frame flicker

# --- FPS Fix 3: JPEG pre-encode quality ---
_JPEG_QUALITY = 70       # 70 is visually fine and ~50% smaller than default (~95)


class InterpreterError(RuntimeError):
    """Raised for interpreter-level failures (e.g. duplicate start)."""


@dataclass
class InterpreterStatus:
    """A snapshot of the interpreter's current live state."""

    running: bool = False
    camera_connected: bool = False
    hand_detected: bool = False
    hand_count: int = 0
    handedness: str = "Unknown"
    prediction: Optional[str] = None
    confidence: float = 0.0
    error: Optional[str] = None
    fps: float = 0.0

    def to_dict(self) -> dict:
        return {
            "running": self.running,
            "camera_connected": self.camera_connected,
            "hand_detected": self.hand_detected,
            "hand_count": self.hand_count,
            "handedness": self.handedness,
            "prediction": self.prediction,
            "confidence": round(self.confidence, 4),
            "error": self.error,
            "fps": round(self.fps, 1),
        }


class Interpreter:
    """Orchestrates the camera -> detector -> preprocessor -> model ->
    visualizer pipeline in real time on a background thread.

    Each dependency is injected so every component remains independently
    testable and swappable (e.g. a real ML model can replace the mock
    model without touching this class).
    """

    def __init__(
        self,
        camera: Camera,
        detector: HandDetector,
        preprocessor: Preprocessor,
        model: SignLanguageModel,
        visualizer: Optional[Visualizer] = None,
        target_fps: int = 30,
        on_frame: Optional[Callable[[np.ndarray], None]] = None,
    ) -> None:
        self._camera = camera
        self._detector = detector
        self._preprocessor = preprocessor
        self._model = model
        self._visualizer = visualizer or Visualizer()
        self._target_fps = max(1, target_fps)
        self._on_frame = on_frame

        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._stop_requested = threading.Event()

        self._status_lock = threading.Lock()
        self._status = InterpreterStatus()

        self._frame_lock = threading.Lock()
        self._latest_annotated_frame: Optional[np.ndarray] = None
        # FPS Fix 3: pre-encoded JPEG bytes stored alongside the raw frame
        self._latest_jpeg: Optional[bytes] = None

        # ML Fix 2: rolling buffer for temporal smoothing
        self._prediction_buffer: deque[str] = deque(maxlen=_SMOOTH_WINDOW)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Start the camera and begin the real-time processing loop.

        Raises:
            InterpreterError: If the interpreter is already running.
            CameraError: If the camera fails to open.
        """
        if self.is_running():
            raise InterpreterError("Interpreter is already running")

        self._stop_requested.clear()

        try:
            self._camera.start()
        except CameraError as exc:
            logger.error("Failed to start camera: %s", exc)
            with self._status_lock:
                self._status = InterpreterStatus(running=False, error=str(exc))
            raise

        self._running.set()
        with self._status_lock:
            self._status = InterpreterStatus(running=True, camera_connected=True)

        self._thread = threading.Thread(
            target=self._run_loop, name="interpreter-loop", daemon=True
        )
        self._thread.start()
        logger.info("Interpreter started")

    def stop(self) -> None:
        """Stop the processing loop and release the camera."""
        if not self.is_running():
            logger.info("Interpreter.stop() called but interpreter was not running")
            return

        self._stop_requested.set()
        self._running.clear()

        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

        self._camera.stop()

        with self._status_lock:
            self._status = InterpreterStatus(running=False)
        with self._frame_lock:
            self._latest_annotated_frame = None
            self._latest_jpeg = None

        self._prediction_buffer.clear()
        logger.info("Interpreter stopped")

    def is_running(self) -> bool:
        return self._running.is_set()

    def get_status(self) -> InterpreterStatus:
        """Return a thread-safe copy of the current interpreter status."""
        with self._status_lock:
            return InterpreterStatus(**self._status.to_dict())

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Return the most recent annotated (skeleton + text overlay) frame."""
        with self._frame_lock:
            if self._latest_annotated_frame is None:
                return None
            return self._latest_annotated_frame.copy()

    def get_latest_jpeg(self) -> Optional[bytes]:
        """Return the most recent annotated frame pre-encoded as JPEG bytes.

        Avoids re-encoding on every HTTP request — the encoding is done
        once per frame inside the interpreter loop (FPS Fix 3).
        """
        with self._frame_lock:
            return self._latest_jpeg

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _run_loop(self) -> None:
        frame_interval = 1.0 / self._target_fps
        fps_window_start = time.monotonic()
        frames_in_window = 0
        current_fps = 0.0

        while self._running.is_set() and not self._stop_requested.is_set():
            loop_start = time.monotonic()

            frame = self._camera.read(timeout=0.5)
            if frame is None:
                if not self._camera.is_running():
                    logger.error("Camera disconnected during interpreter run")
                    with self._status_lock:
                        self._status = InterpreterStatus(
                            running=False,
                            camera_connected=False,
                            error=self._camera.last_error or "Camera disconnected",
                        )
                    self._running.clear()
                    break
                continue

            detection = self._detector.detect(frame)
            primary_hand = detection.primary_hand()

            features = self._preprocessor.process(primary_hand)
            prediction: Prediction = self._model.predict(features)

            # --- ML Fix 2: Temporal smoothing via majority vote ---
            self._prediction_buffer.append(prediction.label)
            smoothed_label = Counter(self._prediction_buffer).most_common(1)[0][0]
            smoothed_prediction = Prediction(
                label=smoothed_label,
                confidence=prediction.confidence,
            )

            annotated = self._visualizer.draw(
                frame,
                detection=detection,
                prediction=smoothed_prediction,
                extra_status_lines=["Camera Running"],
            )

            # --- FPS Fix 3: Pre-encode JPEG once per frame at quality=70 ---
            ret, jpeg_buf = cv2.imencode(
                ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY]
            )
            jpeg_bytes = jpeg_buf.tobytes() if ret else None

            with self._frame_lock:
                self._latest_annotated_frame = annotated
                self._latest_jpeg = jpeg_bytes

            if self._on_frame is not None:
                try:
                    self._on_frame(annotated)
                except Exception:
                    logger.exception("on_frame callback raised an exception")

            frames_in_window += 1
            elapsed_window = time.monotonic() - fps_window_start
            if elapsed_window >= 1.0:
                current_fps = frames_in_window / elapsed_window
                frames_in_window = 0
                fps_window_start = time.monotonic()

            with self._status_lock:
                self._status = InterpreterStatus(
                    running=True,
                    camera_connected=self._camera.is_running(),
                    hand_detected=detection.hand_detected,
                    hand_count=detection.hand_count,
                    handedness=primary_hand.handedness if primary_hand else "Unknown",
                    prediction=smoothed_prediction.label,
                    confidence=smoothed_prediction.confidence,
                    fps=current_fps,
                )

            if detection.hand_detected:
                logger.debug(
                    "Hand detected -> Prediction: %s (%.2f)",
                    smoothed_prediction.label,
                    smoothed_prediction.confidence,
                )

            # Simple frame-rate pacing so the loop doesn't spin faster
            # than the target FPS and burn CPU unnecessarily.
            elapsed = time.monotonic() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
