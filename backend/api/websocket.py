"""
WebSocket endpoint for live interpreter feedback.

Rather than polling ``GET /interpreter/status`` repeatedly, a frontend
can open a WebSocket connection to ``/interpreter/ws`` and receive a
JSON status update automatically several times per second while the
interpreter is running. This is the simplest mechanism that meets the
"live feedback" requirement without overcomplicating the architecture
(no message broker, no pub/sub system — just a periodic push per
connection).
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.pipeline.interpreter import Interpreter

logger = logging.getLogger(__name__)

router = APIRouter()

# How often to push a status update to connected clients.
# FPS Fix 4: Doubled from 0.1s (10 Hz) to 0.05s (20 Hz) for smoother
# confidence bar and FPS counter updates on the frontend.
_PUSH_INTERVAL_SECONDS = 0.05  # 20 updates/sec


@router.websocket("/interpreter/ws")
async def interpreter_status_ws(websocket: WebSocket) -> None:
    """Push live interpreter status JSON to the client periodically.

    The connection stays open for as long as the client keeps it open;
    it works whether or not the interpreter is currently running (it
    will simply report ``running: false`` until ``/interpreter/start``
    is called).
    """
    await websocket.accept()
    interpreter: Interpreter | None = getattr(
        websocket.app.state, "interpreter", None
    )

    if interpreter is None:
        await websocket.send_json({"error": "Interpreter is not initialized"})
        await websocket.close()
        return

    logger.info("WebSocket client connected for live status updates")
    try:
        while True:
            status = interpreter.get_status()
            await websocket.send_json(status.to_dict())
            await asyncio.sleep(_PUSH_INTERVAL_SECONDS)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception:
        logger.exception("Unexpected error in status WebSocket loop")
        try:
            await websocket.close()
        except Exception:
            pass
