"""
Main application entry point.

``main.py`` is intentionally thin: it only builds the dependency graph
(camera, detector, preprocessor, model, visualizer -> Interpreter) and
wires it either into a FastAPI app or a simple OpenCV preview window.
All real logic lives in the dedicated modules under ``backend/``.

Usage
-----
Manual / development camera-preview mode (opens an OpenCV window):

    python main.py

Manual mode without a preview window (headless, runs until Ctrl+C):

    python main.py --no-preview

FastAPI server mode (frontend can POST /interpreter/start etc.):

    uvicorn backend.main:app --reload
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from contextlib import asynccontextmanager

import cv2
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router as interpreter_router
from backend.api.websocket import router as websocket_router
from backend.camera.camera import Camera, CameraError
from backend.config.settings import get_settings
from backend.hand_detection.detector import HandDetector, HandDetectorError
from backend.model.image_model import KerasImageSignLanguageModel
from backend.pipeline.interpreter import Interpreter
from backend.preprocessing.preprocessor import Preprocessor
from backend.utils.logging_config import configure_logging
from backend.visualization.visualizer import Visualizer

logger = logging.getLogger(__name__)


def build_interpreter() -> Interpreter:
    """Construct the full component graph and return a ready ``Interpreter``.

    This is the single place where concrete implementations are chosen.
    The trained image model receives the raw frame and detected hand crop;
    the interpreter still supports landmark-based models for tests and
    future alternatives.
    """
    settings = get_settings()

    camera = Camera(
        camera_index=settings.camera_index,
        width=settings.camera_width,
        height=settings.camera_height,
        fps=settings.camera_fps,
        mirror=settings.camera_mirror,
    )
    detector = HandDetector(
        max_num_hands=settings.mp_max_num_hands,
        min_detection_confidence=settings.mp_min_detection_confidence,
        min_tracking_confidence=settings.mp_min_tracking_confidence,
        model_complexity=settings.mp_model_complexity,
        model_asset_path=settings.hand_landmarker_model_path,
    )
    preprocessor = Preprocessor()
    model = KerasImageSignLanguageModel(
        model_path=settings.model_path,
        labels_path=settings.model_labels_path,
        confidence_threshold=settings.model_confidence_threshold,
    )
    visualizer = Visualizer()

    return Interpreter(
        camera=camera,
        detector=detector,
        preprocessor=preprocessor,
        model=model,
        visualizer=visualizer,
        target_fps=settings.pipeline_target_fps,
    )


# ----------------------------------------------------------------------
# FastAPI application (Mode 2: frontend-triggered)
# ----------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    logger.info("Starting Sign Language Interpreter API")

    app.state.interpreter = build_interpreter()
    try:
        yield
    finally:
        if app.state.interpreter.is_running():
            app.state.interpreter.stop()
        logger.info("Sign Language Interpreter API shut down")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Sign Language Interpreter API",
        description=(
            "Backend API for controlling a real-time sign-language "
            "interpreter pipeline (camera -> hand detection -> "
            "preprocessing -> ML model -> live feedback)."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # Permissive CORS so a locally-developed frontend (any origin/port)
    # can call the API during development. Tighten this for production.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(interpreter_router)
    app.include_router(websocket_router)

    @app.get("/", tags=["health"])
    def root() -> dict:
        return {"status": "ok", "service": "sign-language-interpreter-backend"}

    return app


# Module-level ASGI app, used by: uvicorn backend.main:app
app = create_app()


# ----------------------------------------------------------------------
# Manual mode (Mode 1: run directly with `python main.py`)
# ----------------------------------------------------------------------
def run_manual(show_preview: bool = True) -> None:
    """Run the interpreter directly from the command line.

    If ``show_preview`` is True, an OpenCV window displays the live
    camera feed with the hand skeleton and prediction overlay; press
    'Q' to stop safely. Otherwise the interpreter runs headlessly until
    interrupted with Ctrl+C.
    """
    settings = get_settings()
    configure_logging(settings.log_level)
    logger.info("Starting Sign Language Interpreter (manual mode)")

    try:
        interpreter = build_interpreter()
    except HandDetectorError as exc:
        logger.error("Failed to initialize hand detector: %s", exc)
        sys.exit(1)

    try:
        interpreter.start()
    except CameraError as exc:
        logger.error("Failed to start camera: %s", exc)
        sys.exit(1)

    window_name = "Sign Language Interpreter (dev preview)"
    try:
        while interpreter.is_running():
            if show_preview:
                frame = interpreter.get_latest_frame()
                if frame is not None:
                    cv2.imshow(window_name, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == ord("Q"):
                    logger.info("Quit key pressed; stopping interpreter")
                    break
            else:
                time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user; stopping interpreter")
    finally:
        interpreter.stop()
        if show_preview:
            cv2.destroyAllWindows()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sign Language Interpreter backend")
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Run headlessly without opening an OpenCV preview window.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_manual(show_preview=not args.no_preview)
