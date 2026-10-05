"""
Centralized application configuration.

All tunable values (camera parameters, MediaPipe thresholds, model paths,
API host/port, debug flags, etc.) live here so that no other module needs
to hardcode constants. Values are loaded from environment variables / a
local ``.env`` file, with sensible defaults so the project runs out of
the box.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings, populated from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Camera settings ---
    camera_index: int = 0
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    camera_mirror: bool = True

    # --- MediaPipe Hands settings ---
    mp_max_num_hands: int = 1
    mp_min_detection_confidence: float = 0.6
    mp_min_tracking_confidence: float = 0.5
    mp_model_complexity: int = 1

    # Path to the MediaPipe HandLandmarker task-bundle model file
    # (hand_landmarker.task). If the file is missing, HandDetector will
    # attempt to download it automatically from Google's model store.
    hand_landmarker_model_path: str = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "hand_detection",
        "models",
        "hand_landmarker.task",
    )

    # --- ML model settings ---
    # Path to the trained ONNX model file produced by scripts/train_model.py
    model_onnx_path: str = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "model",
        "sign_model.onnx",
    )
    # Path to the JSON label list produced by scripts/train_model.py
    model_labels_path: str = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "model",
        "labels.json",
    )
    model_confidence_threshold: float = 0.6
    # Set to False to keep using MockSignLanguageModel even if the ONNX file exists
    use_real_model: bool = True

    # --- Application / debug settings ---
    debug: bool = False
    log_level: str = "INFO"

    # --- API settings ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # --- Pipeline settings ---
    pipeline_target_fps: int = 30


@lru_cache
def get_settings() -> Settings:
    """Return a cached, process-wide ``Settings`` instance.

    Using ``lru_cache`` means the ``.env`` file / environment is only
    parsed once, and every module that calls ``get_settings()`` shares
    the same configuration object.
    """
    return Settings()
