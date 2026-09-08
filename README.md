# Sign Language Interpreter — Backend

A modular, real-time computer-vision **backend** for a sign-language
interpreter application. It captures webcam video, detects a hand and its
21 landmark points with MediaPipe, draws a live skeleton overlay,
preprocesses the landmarks into a feature vector, and feeds that vector
into a pluggable ML model interface to produce a predicted sign.

This repository includes:

- a frontend (React/HTML/etc.)
- a trained 36-class sign-recognition image model
- a runtime adapter for the trained Keras model

It also retains the clean `SignLanguageModel` interface and mock
implementation for tests and alternative landmark-based models.

---

## 1. Project Purpose

The backend implements this pipeline:

```
Camera → Hand Detection → Landmark Extraction → Skeleton Visualization
   → Preprocessing → ML Model Interface → Predicted Sign → Live Feedback
```

It can run in two modes:

- **Manual mode** — `python main.py` opens an OpenCV preview window for
  local development/testing.
- **API mode** — `uvicorn backend.main:app` exposes REST + WebSocket
  endpoints so a future frontend can start/stop the interpreter and
  receive live predictions, without knowing anything about OpenCV,
  MediaPipe, or the ML model.

---

## 2. Architecture

```
sign-language-interpreter/
│
├── backend/
│   ├── main.py                  # Orchestrator: builds components, exposes FastAPI app + manual mode
│   │
│   ├── api/
│   │   ├── routes.py            # POST /interpreter/start, /stop, GET /status
│   │   └── websocket.py         # WS /interpreter/ws — live status push
│   │
│   ├── camera/
│   │   └── camera.py            # Camera I/O only (open, read, release)
│   │
│   ├── hand_detection/
│   │   ├── detector.py          # MediaPipe HandLandmarker wrapper
│   │   └── landmarks.py         # Plain dataclasses: Landmark, DetectedHand, DetectionResult
│   │
│   ├── preprocessing/
│   │   └── preprocessor.py      # Landmarks -> normalized feature vector
│   │
│   ├── model/
│   │   ├── base_model.py        # Abstract SignLanguageModel interface + Prediction type
│   │   ├── image_model.py       # Trained Keras image-model adapter
│   │   ├── modelnet_model.h5    # Trained 36-class model
│   │   ├── model_labels.txt     # Output label order
│   │   └── mock_model.py        # Deterministic test model
│   │
│   ├── visualization/
│   │   └── visualizer.py        # Draws skeleton + status/prediction text onto frames
│   │
│   ├── pipeline/
│   │   └── interpreter.py       # Interpreter: runs the real-time loop on a background thread
│   │
│   ├── config/
│   │   └── settings.py          # Centralized settings, loaded from .env
│   │
│   └── utils/
│       └── logging_config.py    # Shared logging setup
│
├── tests/                       # Unit tests for every module (mocked camera/hardware)
├── requirements.txt
├── .env.example
└── README.md
```

**Design principle:** one responsibility per module. The camera knows
nothing about hands. The detector knows nothing about the camera or the
model. The model knows nothing about MediaPipe. `main.py` only wires
these pieces together via the `Interpreter` orchestrator — it contains
no business logic itself.

---

## 3. Installation

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> **Note:** `requirements.txt` lists `opencv-python`. If you're running
> in a headless/server environment (no display), you can use
> `opencv-python-headless` instead — the API is identical.

Copy the environment example and adjust as needed:

```bash
cp .env.example .env
```

### About the hand-tracking model

This project uses MediaPipe's current **Tasks API** (`HandLandmarker`),
since the older `mediapipe.solutions.hands` API has been removed from
recent MediaPipe releases. `HandLandmarker` needs a small model bundle
file (`hand_landmarker.task`, a few MB). On first run, `HandDetector`
will automatically download it to
`backend/hand_detection/models/hand_landmarker.task`. If your machine
has no internet access at runtime, download it manually from:

```
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

and place it at the path above (or point `HAND_LANDMARKER_MODEL_PATH` in
`.env` at wherever you put it).

---

## 4. Dependencies

| Library | Purpose |
|---|---|
| `opencv-python` | Camera capture, frame drawing |
| `mediapipe` | Hand detection & 21-point landmark extraction |
| `numpy` | Numerical processing of landmarks/features |
| `tensorflow` | Loads and runs the trained Keras sign model |
| `fastapi` | REST + WebSocket API |
| `uvicorn` | ASGI server for FastAPI |
| `pydantic` / `pydantic-settings` | Data validation, settings management |
| `python-dotenv` | Loads `.env` files |
| `websockets` | WebSocket support (via FastAPI/Starlette) |
| `pytest`, `httpx` | Testing |

---

## 5. Running: Manual Mode

Opens an OpenCV preview window with the live camera feed, hand skeleton
overlay, and trained-model prediction:

```bash
python main.py
```

Press **Q** in the preview window to stop safely.

Run headlessly (no window, runs until Ctrl+C) — useful on servers or
inside containers:

```bash
python main.py --no-preview
```

> Run this from the project root (`sign-language-interpreter/`) so the
> `backend` package resolves correctly. If you prefer running from
> inside `backend/`, adjust `sys.path`/imports or run as a module:
> `python -m backend.main`.

---

## 6. Running: FastAPI Server

```bash
uvicorn backend.main:app --reload
```

By default this serves on `http://127.0.0.1:8000`. Interactive API docs
are available at `http://127.0.0.1:8000/docs`.

---

## 7. API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET`  | `/` | Health check |
| `POST` | `/interpreter/start` | Starts the camera + processing pipeline |
| `POST` | `/interpreter/stop` | Stops the camera + processing pipeline |
| `GET`  | `/interpreter/status` | Returns current live status (JSON) |
| `WS`   | `/interpreter/ws` | Streams live status updates (~10/sec) |

### Example requests

```bash
curl -X POST http://127.0.0.1:8000/interpreter/start
curl -X POST http://127.0.0.1:8000/interpreter/stop
curl http://127.0.0.1:8000/interpreter/status
```

Example `/interpreter/status` response:

```json
{
  "running": true,
  "camera_connected": true,
  "hand_detected": true,
  "hand_count": 1,
  "handedness": "Right",
  "prediction": "Hello",
  "confidence": 0.93,
  "error": null,
  "fps": 28.4
}
```

### WebSocket example (JavaScript, for the future frontend)

```javascript
const ws = new WebSocket("ws://127.0.0.1:8000/interpreter/ws");
ws.onmessage = (event) => {
  const status = JSON.parse(event.data);
  console.log(status.prediction, status.confidence);
};
```

The frontend never touches OpenCV, MediaPipe, or the model directly — it
only calls these three REST endpoints and/or opens the WebSocket.

---

## 8. How the Camera Pipeline Works

1. `Camera.start()` opens the configured device index (`cv2.VideoCapture`)
   and spawns a background thread that continuously reads frames,
   mirrors them horizontally, and stores the latest one.
2. `Camera.read()` returns a copy of the most recent frame without
   blocking on I/O — safe to call from another thread (the pipeline loop).
3. `Camera.stop()` signals the capture thread to exit and releases the
   device.

Camera failures (device busy, disconnected mid-stream, etc.) are caught
and surfaced via `CameraError` on start, or reflected in
`Interpreter` status (`camera_connected: false`, `error: "..."`) if the
camera drops out mid-run.

---

## 9. How Hand Detection Works

`HandDetector` wraps MediaPipe's `HandLandmarker` (Tasks API). For each
frame:

1. The BGR frame is converted to RGB and wrapped in an `mp.Image`.
2. `HandLandmarker.detect()` returns up to `MP_MAX_NUM_HANDS` hands, each
   with 21 `(x, y, z)` landmarks (normalized to the image) and a
   handedness classification (`Left`/`Right`) with a confidence score.
3. These are converted into simple `DetectedHand` / `DetectionResult`
   dataclasses (defined in `landmarks.py`) that carry no MediaPipe types,
   so the rest of the codebase never imports `mediapipe` directly.

If no hand is present, `DetectionResult.hand_detected` is `False` — this
is treated as a normal outcome, not an error, throughout the pipeline.

---

## 10. Trained Model

The application loads `backend/model/modelnet_model.h5` and its output
labels from `backend/model/model_labels.txt` by default. The model was
trained for 36 classes (`0`-`9` and `a`-`z`) using 224x224 RGB hand images.

The hand detector supplies the landmarks used to crop the hand from each
camera frame. `backend/model/image_model.py` then converts that crop to
the model's expected RGB tensor and returns the predicted label and
confidence. The configured paths can be overridden with `MODEL_PATH` and
`MODEL_LABELS_PATH` in `.env`.

The mock model remains available for tests and for environments where
TensorFlow or the trained artifact is not installed.

---

## 11. How the Frontend Should Communicate with the Backend

The frontend should treat this backend purely as an HTTP/WebSocket
service:

1. User clicks "Start" → frontend sends `POST /interpreter/start`.
2. Frontend opens a WebSocket to `/interpreter/ws` (or polls
   `GET /interpreter/status`) to receive live `hand_detected`,
   `prediction`, and `confidence` values to render in its own UI.
3. User clicks "Stop" → frontend sends `POST /interpreter/stop`.

The backend's own OpenCV preview window (Section 5) is for **development
only** — the production frontend renders its own UI from the JSON/WebSocket
data; it does not need the raw annotated video frames unless you choose
to add a video-streaming endpoint later (e.g. MJPEG or WebRTC), which
can be layered on top of `Interpreter.get_latest_frame()` without
changing any other module.

---

## 12. Troubleshooting Common Camera Issues

| Symptom | Likely cause / fix |
|---|---|
| `CameraError: Could not open camera at index 0` | Another app is using the camera, wrong `CAMERA_INDEX`, or no camera present. Try a different index (1, 2, …) in `.env`. |
| Preview window is black | Some drivers need a short warm-up; check `camera.last_error` / logs. Try lowering `CAMERA_WIDTH`/`CAMERA_HEIGHT`. |
| `HandDetectorError: ... automatic download failed` | No internet access to download `hand_landmarker.task`. Download it manually (see Section 3) and set `HAND_LANDMARKER_MODEL_PATH`. |
| Low FPS / laggy skeleton | Lower `CAMERA_WIDTH`/`CAMERA_HEIGHT`, or reduce `PIPELINE_TARGET_FPS`. |
| `/interpreter/start` returns `success: false` | The interpreter is already running — call `/interpreter/stop` first, or check `/interpreter/status`. |
| Permission denied opening camera (Linux) | Ensure your user is in the `video` group, or run with appropriate permissions. |

---

## 13. Running Tests

```bash
pytest
```

Camera and MediaPipe hardware dependencies are mocked in the test suite
so tests run without a physical webcam. Two detector tests that exercise
the *real* `HandLandmarker` will automatically skip themselves if the
model bundle can't be downloaded (e.g. offline CI) — they'll run
normally on a machine with internet access.

---

## 14. Future Compatibility

This architecture is designed to support, without rewriting existing
modules:

- Real ML model (see Section 10)
- Two-hand recognition (`MP_MAX_NUM_HANDS=2`; `Preprocessor.process_many`
  already supports multiple hands)
- Gesture sequences / temporal models (buffer `Interpreter` predictions
  over time in a new module)
- Sentence formation, confidence smoothing, prediction history
- Sign-to-word translation, text-to-speech (new modules consuming
  `Interpreter.get_status()` / the WebSocket stream)
