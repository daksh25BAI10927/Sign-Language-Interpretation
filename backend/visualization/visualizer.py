"""
Skeleton / landmark visualization module.

Draws the hand skeleton overlay and status/prediction text onto a camera
frame. Contains no model logic and no camera logic — it only knows how
to render ``DetectionResult`` / ``Prediction`` data onto an image.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from backend.hand_detection.landmarks import DetectedHand, DetectionResult
from backend.model.base_model import Prediction

# MediaPipe's canonical 21-point hand connections (pairs of landmark
# indices). Defined locally so this module doesn't need to import
# mediapipe just to draw lines.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index finger
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring finger
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                  # palm base
]

COLOR_SKELETON = (0, 255, 0)
COLOR_JOINT = (0, 200, 255)
COLOR_TEXT = (255, 255, 255)
COLOR_TEXT_BG = (0, 0, 0)
COLOR_WARNING = (0, 165, 255)


class Visualizer:
    """Renders hand skeletons and status/prediction overlays onto frames."""

    def __init__(self, show_confidence: bool = True, show_handedness: bool = True) -> None:
        self._show_confidence = show_confidence
        self._show_handedness = show_handedness

    def draw(
        self,
        frame: np.ndarray,
        detection: Optional[DetectionResult] = None,
        prediction: Optional[Prediction] = None,
        extra_status_lines: Optional[list[str]] = None,
    ) -> np.ndarray:
        """Draw the skeleton overlay and status text onto a copy of the frame.

        Args:
            frame: The BGR camera frame to draw on (not mutated in place;
                a modified copy is returned).
            detection: The hand detection result for this frame, if any.
            prediction: The model's prediction for this frame, if any.
            extra_status_lines: Additional free-form status strings to
                render (e.g. "Camera Running").

        Returns:
            A new frame with overlays drawn.
        """
        output = frame.copy()
        height, width = output.shape[:2]

        if detection is not None:
            for hand in detection.hands:
                self._draw_hand_skeleton(output, hand, width, height)

        status_lines = list(extra_status_lines or [])

        if detection is not None and detection.hand_detected:
            status_lines.append(f"Hand Detected ({detection.hand_count})")
            primary = detection.primary_hand()
            if primary and self._show_handedness:
                label = primary.handedness
                if self._show_confidence:
                    label += f" ({primary.handedness_confidence:.2f})"
                status_lines.append(f"Hand: {label}")
        else:
            status_lines.append("No Hand Detected")

        if prediction is not None:
            line = f"Prediction: {prediction.label}"
            if self._show_confidence:
                line += f" ({prediction.confidence:.2f})"
            status_lines.append(line)

        self._draw_status_text(output, status_lines)
        return output

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _draw_hand_skeleton(
        self, frame: np.ndarray, hand: DetectedHand, width: int, height: int
    ) -> None:
        if not hand.is_valid():
            return

        points = [
            (int(lm.x * width), int(lm.y * height)) for lm in hand.landmarks
        ]

        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(frame, points[start_idx], points[end_idx], COLOR_SKELETON, 2)

        for point in points:
            cv2.circle(frame, point, 4, COLOR_JOINT, thickness=-1)

    def _draw_status_text(self, frame: np.ndarray, lines: list[str]) -> None:
        line_height = 24
        padding = 8
        x, y = 10, 10

        for i, line in enumerate(lines):
            text_y = y + (i + 1) * line_height
            (text_w, text_h), _ = cv2.getTextSize(
                line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                frame,
                (x - padding // 2, text_y - text_h - padding // 2),
                (x + text_w + padding // 2, text_y + padding // 2),
                COLOR_TEXT_BG,
                thickness=-1,
            )
            cv2.putText(
                frame,
                line,
                (x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                COLOR_TEXT,
                2,
                cv2.LINE_AA,
            )
