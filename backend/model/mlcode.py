import os
import cv2
import numpy as np
import mediapipe as mp

from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ------------------- CONFIG -------------------

DATA_DIR = "data/"
IMG_SIZE = 224
MODEL_PATH = "modelnet_model.h5"

LABELS = sorted(os.listdir(DATA_DIR))


# MEDIAPIPE HAND DETECTION

mp_hands = mp.solutions.hands

hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


#TRAIN MODEL

def build_and_train_model():

    global LABELS

    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2
    )

    train_gen = datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=32,
        class_mode="categorical",
        subset="training"
    )

    val_gen = datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=32,
        class_mode="categorical",
        subset="validation"
    )

    LABELS = list(train_gen.class_indices.keys())

    print("Detected classes:", LABELS)

    # CNN MODEL

    model = Sequential([
        Conv2D(
            32,
            (3, 3),
            activation="relu",
            input_shape=(IMG_SIZE, IMG_SIZE, 3)
        ),
        MaxPooling2D(2, 2),

        Conv2D(
            64,
            (3, 3),
            activation="relu"
        ),
        MaxPooling2D(2, 2),

        Conv2D(
            128,
            (3, 3),
            activation="relu"
        ),
        MaxPooling2D(2, 2),

        Flatten(),

        Dense(
            128,
            activation="relu"
        ),

        Dropout(0.5),

        Dense(
            len(LABELS),
            activation="softmax"
        )
    ])

    # COMPILE

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # TRAIN

    model.fit(
        train_gen,
        epochs=20,
        validation_data=val_gen
    )

    # SAVE MODEL

    model.save(MODEL_PATH)

    print(f"Model trained and saved to {MODEL_PATH}")

    return model


# LOAD OR TRAIN
if os.path.exists(MODEL_PATH):

    model = load_model(MODEL_PATH)

    print("Loaded pre-trained model")

else:

    print("Model not found, starting training...")

    model = build_and_train_model()


#HAND DETECTION + CROPPING
def detect_and_crop_hand(frame):

    img_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands_detector.process(img_rgb)

    if not results.multi_hand_landmarks:
        return None

    h, w, _ = frame.shape

    for hand_landmarks in results.multi_hand_landmarks:

        x_coords = [
            lm.x * w
            for lm in hand_landmarks.landmark
        ]

        y_coords = [
            lm.y * h
            for lm in hand_landmarks.landmark
        ]

        x_min = int(min(x_coords)) - 20
        x_max = int(max(x_coords)) + 20

        y_min = int(min(y_coords)) - 20
        y_max = int(max(y_coords)) + 20

        x_min = max(0, x_min)
        y_min = max(0, y_min)

        x_max = min(w, x_max)
        y_max = min(h, y_max)

        cropped = frame[
            y_min:y_max,
            x_min:x_max
        ]

        return cropped

    return None


# IMAGE PREPROCESSING
def preprocess_frame(frame):

    img = cv2.resize(
        frame,
        (IMG_SIZE, IMG_SIZE)
    )

    img = img.astype("float32") / 255.0

    img = np.expand_dims(
        img,
        axis=0
    )

    return img


#PREDICTION
def predict_frame(frame):

    cropped = detect_and_crop_hand(frame)

    if cropped is None:
        return "No Hand Detected", 0.0

    processed = preprocess_frame(cropped)

    preds = model.predict(
        processed,
        verbose=0
    )

    class_index = np.argmax(preds)

    confidence = float(
        np.max(preds)
    )

    return LABELS[class_index], confidence