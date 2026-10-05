"""
Preprocessing module.

Converts raw MediaPipe-style hand landmarks (see ``hand_detection.landmarks``)
into a deterministic, normalized feature vector suitable for feeding into
an ML model. This module knows nothing about MediaPipe, OpenCV, or the
model itself — it only transforms already-extracted landmark data.

    Raw Landmarks -> Preprocessor -> Normalized Feature Vector -> ML Model
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from backend.hand_detection.landmarks import DetectedHand

logger = logging.getLogger(__name__)

WRIST_INDEX = 0
MIDDLE_FINGER_MCP_INDEX = 9  # used as a stable scale reference


@dataclass
class PreprocessConfig:
    """Configuration controlling how landmarks are turned into features."""

    center_on_wrist: bool = True
    scale_normalize: bool = True
    include_z: bool = True
    flatten: bool = True
    mirror_left_hand: bool = True  # flip x for Left hands → right-hand model works for both


class Preprocessor:
    """Transforms ``DetectedHand`` landmark data into ML-ready feature vectors."""

    def __init__(self, config: Optional[PreprocessConfig] = None) -> None:
        self._config = config or PreprocessConfig()

    def process(self, hand: Optional[DetectedHand]) -> Optional[np.ndarray]:
        """Produce a feature vector for a single detected hand.

        Args:
            hand: The detected hand to process, or ``None`` if no hand
                was detected in the current frame.

        Returns:
            A 1D ``numpy.ndarray`` of features, or ``None`` if ``hand``
            is ``None`` or malformed (missing landmarks). Returning
            ``None`` lets downstream code cleanly represent "no hand"
            instead of feeding zeros into the model.

        Notes:
            When ``mirror_left_hand`` is enabled (the default), left-hand
            landmarks are mirrored along the x-axis *before* wrist-centring
            and scale normalisation. This lets a model trained exclusively
            on right-hand data classify signs made with either hand.
        """
        if hand is None:
            return None

        if not hand.is_valid():
            logger.warning(
                "Received hand with unexpected landmark count (%s); expected %s",
                len(hand.landmarks),
                hand.NUM_LANDMARKS,
            )
            return None

        coords = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand.landmarks],
            dtype=np.float32,
        )  # shape: (21, 3)

        # --- ML Fix 1: Mirror left-hand x so the right-hand model works for both ---
        if self._config.mirror_left_hand and hand.handedness == "Left":
            coords[:, 0] = 1.0 - coords[:, 0]
            logger.debug("Left hand detected — x-axis mirrored for right-hand model")

        if self._config.center_on_wrist:
            wrist = coords[WRIST_INDEX].copy()
            coords = coords - wrist

        if self._config.scale_normalize:
            scale_ref = np.linalg.norm(coords[MIDDLE_FINGER_MCP_INDEX])
            if scale_ref > 1e-6:
                coords = coords / scale_ref

        if not self._config.include_z:
            coords = coords[:, :2]

        if self._config.flatten:
            return coords.flatten()
        return coords

    def process_many(self, hands: List[DetectedHand]) -> List[np.ndarray]:
        """Process every hand in a list, skipping any invalid ones."""
        features = []
        for hand in hands:
            feature_vector = self.process(hand)
            if feature_vector is not None:
                features.append(feature_vector)
        return features
