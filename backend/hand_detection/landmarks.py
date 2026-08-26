"""
Structured data types describing detected hands and their landmarks.

These are plain dataclasses (no MediaPipe or OpenCV types leak outside of
the ``hand_detection`` package) so that downstream modules (preprocessing,
visualization, API layer) depend only on this simple, stable schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional

Handedness = Literal["Left", "Right", "Unknown"]


@dataclass(frozen=True)
class Landmark:
    """A single hand landmark point in normalized image coordinates.

    ``x`` and ``y`` are normalized to [0, 1] relative to the image width
    and height. ``z`` is MediaPipe's relative depth (roughly, distance
    from the wrist, with smaller values closer to the camera).
    """

    x: float
    y: float
    z: float = 0.0


@dataclass(frozen=True)
class DetectedHand:
    """A single detected hand: its 21 landmarks plus metadata."""

    landmarks: List[Landmark]
    handedness: Handedness = "Unknown"
    handedness_confidence: float = 0.0

    NUM_LANDMARKS: int = field(default=21, init=False, repr=False)

    def is_valid(self) -> bool:
        """Return True if this hand has the expected number of landmarks."""
        return len(self.landmarks) == self.NUM_LANDMARKS


@dataclass
class DetectionResult:
    """The full result of running hand detection on a single frame."""

    hands: List[DetectedHand] = field(default_factory=list)
    image_width: int = 0
    image_height: int = 0

    @property
    def hand_detected(self) -> bool:
        return len(self.hands) > 0

    @property
    def hand_count(self) -> int:
        return len(self.hands)

    def primary_hand(self) -> Optional[DetectedHand]:
        """Return the first detected hand, or None if no hand was found.

        Convenient for single-hand pipelines/models; multi-hand support
        can still access ``.hands`` directly.
        """
        return self.hands[0] if self.hands else None
