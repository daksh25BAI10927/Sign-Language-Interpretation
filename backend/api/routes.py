"""
FastAPI REST routes for controlling the interpreter.

Designed so the frontend never needs to know anything about OpenCV,
MediaPipe, or the ML model — it only talks to these HTTP endpoints and
receives plain JSON.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from backend.camera.camera import CameraError
from backend.pipeline.interpreter import Interpreter, InterpreterError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/interpreter", tags=["interpreter"])


class StatusResponse(BaseModel):
    running: bool
    camera_connected: bool
    hand_detected: bool
    hand_count: int
    handedness: str
    prediction: str | None
    confidence: float
    error: str | None
    fps: float


class ActionResponse(BaseModel):
    success: bool
    message: str


def _get_interpreter(request: Request) -> Interpreter:
    """Retrieve the shared ``Interpreter`` instance from app state.

    The interpreter (and its camera/detector/model dependencies) is
    constructed once at application startup and stored on
    ``app.state.interpreter`` — see ``backend/main.py``.
    """
    interpreter: Interpreter | None = getattr(request.app.state, "interpreter", None)
    if interpreter is None:
        raise HTTPException(status_code=500, detail="Interpreter is not initialized")
    return interpreter


@router.post("/start", response_model=ActionResponse)
def start_interpreter(request: Request) -> ActionResponse:
    """Start the camera and real-time sign interpretation pipeline."""
    interpreter = _get_interpreter(request)

    if interpreter.is_running():
        return ActionResponse(success=False, message="Interpreter is already running")

    try:
        interpreter.start()
    except CameraError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except InterpreterError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return ActionResponse(success=True, message="Interpreter started")


@router.post("/stop", response_model=ActionResponse)
def stop_interpreter(request: Request) -> ActionResponse:
    """Stop the camera and real-time sign interpretation pipeline."""
    interpreter = _get_interpreter(request)

    if not interpreter.is_running():
        return ActionResponse(success=False, message="Interpreter is not running")

    interpreter.stop()
    return ActionResponse(success=True, message="Interpreter stopped")


import time
from fastapi.responses import StreamingResponse


@router.get("/video_feed")
def video_feed(request: Request):
    """Stream live annotated video frames (MJPEG) from the backend interpreter.

    FPS Fix 1: The generator no longer has a hardcoded time.sleep(0.03)
    (which was capping the stream at ~33 fps regardless of actual processing
    speed). Instead it polls until a *new* frame arrives using object-id
    comparison, then yields immediately.

    FPS Fix 2: Frames are consumed as pre-encoded JPEG bytes produced by the
    interpreter loop (quality=70) — no redundant cv2.imencode() call here.
    """
    interpreter = _get_interpreter(request)

    def frame_generator():
        last_frame_id: int | None = None
        while True:
            if not interpreter.is_running():
                time.sleep(0.05)
                continue

            jpeg = interpreter.get_latest_jpeg()
            if jpeg is None:
                time.sleep(0.01)
                continue

            frame_id = id(jpeg)
            if frame_id == last_frame_id:
                # No new frame yet — tiny sleep to avoid busy-spin
                time.sleep(0.005)
                continue

            last_frame_id = frame_id
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            )

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )



@router.get("/status", response_model=StatusResponse)
def get_status(request: Request) -> StatusResponse:
    """Return the current live status of the interpreter."""
    interpreter = _get_interpreter(request)
    status = interpreter.get_status()
    return StatusResponse(**status.to_dict())

