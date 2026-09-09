"""
test_camera.py
---------------
Bare minimum script to check your camera is working with OpenCV.
Press 'q' to quit.
"""

import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open camera.")
else:
    print("Camera opened successfully. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame.")
        break

    cv2.imshow("Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()