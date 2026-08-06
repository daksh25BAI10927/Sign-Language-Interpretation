"""
main.py
-------
Real-time sign language interpretation.
Continuously reads webcam frames, detects hand landmarks, predicts the sign
using a trained model, and overlays the prediction at the bottom of the frame.

Requires model.pkl (created by train_model.py) in the same folder.
"""

import cv2
import mediapipe as mp
import pickle
import numpy as np
from collections import deque, Counter

MODEL_PATH = "model.pkl"

# Load trained model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

cap = cv2.VideoCapture(0)

# Smoothing buffer: only update displayed prediction if it's stable over recent frames
prediction_buffer = deque(maxlen=10)
displayed_label = ""

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read from camera.")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]

        # Highlight the hand with landmark skeleton
        mp_drawing.draw_landmarks(
            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
        )

        # Flatten landmarks into feature vector
        landmarks_flat = []
        for lm in hand_landmarks.landmark:
            landmarks_flat.extend([lm.x, lm.y, lm.z])

        features = np.array(landmarks_flat).reshape(1, -1)
        prediction = model.predict(features)[0]
        prediction_buffer.append(prediction)

        # Stabilize: show the most common prediction in the recent buffer
        most_common, count = Counter(prediction_buffer).most_common(1)[0]
        if count >= 5:  # require some consistency before updating display
            displayed_label = most_common
    else:
        prediction_buffer.clear()
        displayed_label = ""

    # Draw a bottom banner with the predicted sign
    h, w, _ = frame.shape
    banner_height = 60
    cv2.rectangle(frame, (0, h - banner_height), (w, h), (0, 0, 0), -1)
    text = displayed_label if displayed_label else "..."
    cv2.putText(
        frame, f"Sign: {text}",
        (10, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )

    cv2.imshow("Sign Language Interpreter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()