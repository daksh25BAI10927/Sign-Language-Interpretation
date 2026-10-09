"""Tests for the Interpreter pipeline (start/stop state) and the FastAPI
REST endpoints. Camera and detector dependencies are mocked so these
tests run without real hardware or MediaPipe overhead.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.hand_detection.landmarks import DetectionResult
from backend.model.base_model import Prediction
from backend.model.mock_model import MockSignLanguageModel
from backend.pipeline.interpreter import Interpreter, InterpreterError
from backend.preprocessing.preprocessor import Preprocessor
from backend.visualization.visualizer import Visualizer


def _build_test_interpreter() -> Interpreter:
    """Build an Interpreter with a fully mocked Camera and HandDetector
    so tests don't require real hardware."""
    fake_camera = MagicMock()
    fake_camera.is_running.return_value = True
    fake_camera.read.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
    fake_camera.last_error = None

    fake_detector = MagicMock()
    fake_detector.detect.return_value = DetectionResult()

    return Interpreter(
        camera=fake_camera,
        detector=fake_detector,
        preprocessor=Preprocessor(),
        model=MockSignLanguageModel(),
        visualizer=Visualizer(),
        target_fps=30,
    )


def test_interpreter_starts_and_stops():
    interpreter = _build_test_interpreter()
    assert interpreter.is_running() is False

    interpreter.start()
    assert interpreter.is_running() is True

    interpreter.stop()
    assert interpreter.is_running() is False


def test_interpreter_double_start_raises():
    interpreter = _build_test_interpreter()
    interpreter.start()
    try:
        with pytest.raises(InterpreterError):
            interpreter.start()
    finally:
        interpreter.stop()


def test_interpreter_stop_without_start_is_safe():
    interpreter = _build_test_interpreter()
    interpreter.stop()  # should not raise
    assert interpreter.is_running() is False


def test_interpreter_status_reflects_running_state():
    interpreter = _build_test_interpreter()
    assert interpreter.get_status().running is False

    interpreter.start()
    try:
        assert interpreter.get_status().running is True
    finally:
        interpreter.stop()


# ----------------------------------------------------------------------
# API endpoint tests
# ----------------------------------------------------------------------
@pytest.fixture
def api_client():
    """A TestClient wired to a FastAPI app whose interpreter dependencies
    are mocked, avoiding any real camera/MediaPipe usage in CI."""
    from backend.api.routes import router as interpreter_router
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(interpreter_router)
    app.state.interpreter = _build_test_interpreter()

    with TestClient(app) as client:
        yield client, app.state.interpreter


def test_status_endpoint_when_not_running(api_client):
    client, _ = api_client
    response = client.get("/interpreter/status")
    assert response.status_code == 200
    assert response.json()["running"] is False


def test_start_and_stop_endpoints(api_client):
    client, interpreter = api_client

    start_response = client.post("/interpreter/start")
    assert start_response.status_code == 200
    assert start_response.json()["success"] is True
    assert interpreter.is_running() is True

    stop_response = client.post("/interpreter/stop")
    assert stop_response.status_code == 200
    assert stop_response.json()["success"] is True
    assert interpreter.is_running() is False


def test_duplicate_start_returns_unsuccessful_response(api_client):
    client, interpreter = api_client
    client.post("/interpreter/start")
    try:
        response = client.post("/interpreter/start")
        assert response.json()["success"] is False
    finally:
        interpreter.stop()


def test_status_endpoint_when_interpreter_none():
    from backend.api.routes import router as interpreter_router
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(interpreter_router)
    app.state.interpreter = None
    app.state.init_error = "Missing library libGLESv2.so.2"

    with TestClient(app) as client:
        response = client.get("/interpreter/status")
        assert response.status_code == 200
        data = response.json()
        assert data["running"] is False
        assert "Missing library" in data["error"]

        start_resp = client.post("/interpreter/start")
        assert start_resp.status_code == 503


def test_main_app_health_and_root():
    from backend.main import create_app

    app = create_app()
    with TestClient(app) as client:
        health_resp = client.get("/health")
        assert health_resp.status_code == 200
        assert health_resp.json()["status"] == "ok"

        root_resp = client.get("/", headers={"accept": "application/json"})
        assert root_resp.status_code == 200
        assert root_resp.json()["status"] == "ok"

