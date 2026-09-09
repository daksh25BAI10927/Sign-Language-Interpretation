"""
collect_data.py
----------------
Captures hand landmarks from webcam and saves them to a CSV file,
labeled by the sign you're currently showing.

HOW TO USE:
1. Run the script.
2. Type the label for the sign you're about to show (e.g. "A", "hello") when prompted.
3. Hold the sign in front of the camera and press 'c' repeatedly to capture samples
   (capture ~50-100 samples per sign, moving your hand slightly each time for variety).
4. Press 'n' to move on to a new sign (you'll be asked for a new label).
5. Press 'q' to quit and save everything.
"""

import cv2
import mediapipe as mp
import csv
import os

DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "landmarks.csv")

os.makedirs(DATA_DIR, exist_ok=True)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

# Create CSV with header if it doesn't exist yet
file_exists = os.path.isfile(CSV_PATH)
csv_file = open(CSV_PATH, mode="a", newline="")
csv_writer = csv.writer(csv_file)
if not file_exists:
    header = ["label"] + [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]
    csv_writer.writerow(header)

cap = cv2.VideoCapture(0)

current_label = input("Enter label for the sign you'll show first: ").strip()
sample_count = 0

print("\nControls: 'c' = capture sample | 'n' = new sign label | 'q' = quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read from camera.")
        break

    frame = cv2.flip(frame, 1)  # mirror for natural feel
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    landmarks_flat = None

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        landmarks_flat = []
        for lm in hand_landmarks.landmark:
            landmarks_flat.extend([lm.x, lm.y, lm.z])

    # UI text
    cv2.putText(frame, f"Label: {current_label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Samples: {sample_count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, "c=capture  n=new label  q=quit", (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Data Collection", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        if landmarks_flat is not None:
            csv_writer.writerow([current_label] + landmarks_flat)
            sample_count += 1
            print(f"Captured sample #{sample_count} for '{current_label}'")
        else:
            print("No hand detected — try again.")

    elif key == ord('n'):
        current_label = input("Enter new label: ").strip()
        sample_count = 0

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
csv_file.close()
print(f"\nData saved to {CSV_PATH}")