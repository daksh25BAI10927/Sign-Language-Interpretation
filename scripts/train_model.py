"""
train_model.py
--------------
Trains a Multi-Layer Perceptron (MLP) on the extracted hand landmark
features and exports the model to ONNX format.

ONNX is used so the trained model can be loaded in Python 3.14
(the project's runtime) via onnxruntime — no TensorFlow needed at
inference time, avoiding Python version conflicts.

Usage (must be run with Python 3.12 where TensorFlow is installed):
    py -3.12 scripts/train_model.py

Outputs:
    backend/model/sign_model.onnx   — trained model in ONNX format
    backend/model/labels.json       — ordered list of class labels
    data/training_report.png        — accuracy/loss curves + confusion matrix
"""

import json
import os
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import tf2onnx

# ------------------------------------------------------------------ #
#  CONFIG                                                              #
# ------------------------------------------------------------------ #

LANDMARKS_CSV    = Path("data/landmarks_dataset.csv")

# Output paths — written directly to the model files folder used by the backend
_MODEL_FILES     = Path(r"E:\Github\model files")
MODEL_ONNX_PATH  = _MODEL_FILES / "sign_model.onnx"
LABELS_JSON_PATH = _MODEL_FILES / "labels.json"
REPORT_PNG_PATH  = _MODEL_FILES / "training_report.png"
CHECKPOINT_PATH  = Path("data/best_model_checkpoint.keras")

EPOCHS           = 150
BATCH_SIZE       = 128
VALIDATION_SPLIT = 0.15
TEST_SPLIT       = 0.15
RANDOM_SEED      = 42

# ------------------------------------------------------------------ #
#  LOGGING                                                             #
# ------------------------------------------------------------------ #

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  DATA LOADING                                                        #
# ------------------------------------------------------------------ #

def load_data():
    if not LANDMARKS_CSV.exists():
        log.error("Landmark CSV not found: %s", LANDMARKS_CSV)
        log.error("Run scripts/extract_landmarks.py first.")
        sys.exit(1)

    log.info("Loading dataset from %s ...", LANDMARKS_CSV)
    df = pd.read_csv(LANDMARKS_CSV)
    log.info("  Total samples : %d", len(df))

    label_counts = df["label"].value_counts()
    log.info("  Classes       : %d", len(label_counts))
    log.info("  Samples/class : min=%d  max=%d  mean=%.0f",
             label_counts.min(), label_counts.max(), label_counts.mean())

    # Encode labels to integers
    le = LabelEncoder()
    y = le.fit_transform(df["label"].values)
    X = df.drop(columns=["label"]).values.astype(np.float32)

    labels = list(le.classes_)
    log.info("  Label order   : %s", labels)

    return X, y, labels


# ------------------------------------------------------------------ #
#  DATA AUGMENTATION  (ML Fix 3)                                       #
# ------------------------------------------------------------------ #

def _rotate_landmarks(X: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    """Rotate the x,y of each 21-landmark vector by a per-sample angle."""
    out = X.copy()
    for i, angle in enumerate(angles_deg):
        rad = np.deg2rad(angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        coords = out[i].reshape(21, 3)
        x_new = coords[:, 0] * cos_a - coords[:, 1] * sin_a
        y_new = coords[:, 0] * sin_a + coords[:, 1] * cos_a
        coords[:, 0] = x_new
        coords[:, 1] = y_new
        out[i] = coords.flatten()
    return out


def _dropout_landmarks(X: np.ndarray, rng, drop_prob: float = 0.10) -> np.ndarray:
    """Zero-out random landmarks to simulate partial occlusion."""
    out = X.copy().reshape(-1, 21, 3)
    mask = rng.random((out.shape[0], 21)) < drop_prob
    out[mask] = 0.0
    return out.reshape(-1, 63)


def augment_landmarks(X: np.ndarray, y: np.ndarray):
    """Augment landmark vectors → 5x dataset size.

    Passes applied to training data only:
      1. Original
      2. Gaussian noise  (σ=0.01) — sensor jitter / small wiggles
      3. Scale jitter    (±10%)   — different hand sizes / distances
      4. Random rotation (±15°)   — tilted or rotated hands
      5. Landmark dropout (10%)   — simulates partial occlusion

    Args:
        X: Feature matrix (N, 63).
        y: Integer labels (N,).

    Returns:
        (X_aug, y_aug) with 5× the original rows.
    """
    rng = np.random.default_rng(seed=42)

    # Pass 2: Gaussian noise
    noise = rng.normal(0.0, 0.01, X.shape).astype(np.float32)
    X_noisy = X + noise

    # Pass 3: Scale jitter ±10%
    scale = rng.uniform(0.90, 1.10, (X.shape[0], 1)).astype(np.float32)
    X_scaled = X * scale

    # Pass 4: Random rotation ±15°
    angles = rng.uniform(-15.0, 15.0, X.shape[0])
    X_rotated = _rotate_landmarks(X.copy(), angles).astype(np.float32)

    # Pass 5: Landmark dropout
    X_dropped = _dropout_landmarks(X, rng, drop_prob=0.10).astype(np.float32)

    X_aug = np.vstack([X, X_noisy, X_scaled, X_rotated, X_dropped])
    y_aug = np.concatenate([y, y, y, y, y])
    log.info(
        "Augmentation: %d original → %d total (noise + scale + rotation + dropout)",
        len(X), len(X_aug),
    )
    return X_aug, y_aug


# ------------------------------------------------------------------ #
#  MODEL DEFINITION                                                    #
# ------------------------------------------------------------------ #

def build_model(input_dim: int, num_classes: int) -> keras.Model:
    """
    3-layer MLP designed for 63-dimensional hand landmark vectors.
    Architecture matches the backend Preprocessor output shape exactly.

    ML Fix 4: Uses label smoothing (0.1) in the cross-entropy loss to
    prevent overconfident softmax outputs and improve calibration.
    """
    model = keras.Sequential([
        keras.Input(shape=(input_dim,), name="landmarks"),

        layers.Dense(512, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),

        layers.Dense(num_classes, activation="softmax", name="predictions"),
    ], name="sign_language_mlp")

    # ML Fix 4: label_smoothing=0.1 prevents the model from becoming
    # overconfident and makes the confidence bar on the frontend more honest.
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )
    return model


# ------------------------------------------------------------------ #
#  TRAINING                                                            #
# ------------------------------------------------------------------ #

def train(X, y, labels):
    num_classes = len(labels)
    input_dim   = X.shape[1]   # should be 63

    log.info("Input dim: %d  |  Classes: %d", input_dim, num_classes)

    # Split: train / val / test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=TEST_SPLIT,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    val_fraction_of_temp = VALIDATION_SPLIT / (1.0 - TEST_SPLIT)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_fraction_of_temp,
        random_state=RANDOM_SEED,
        stratify=y_temp,
    )

    log.info("Split -> train=%d  val=%d  test=%d", len(X_train), len(X_val), len(X_test))

    # ML Fix 3: Augment only the training set — val/test stay clean for honest eval
    X_train, y_train = augment_landmarks(X_train, y_train)
    log.info("After augmentation: train=%d", len(X_train))

    model = build_model(input_dim, num_classes)
    model.summary()

    cb = [
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=str(CHECKPOINT_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=0,
        ),
    ]

    log.info("Starting training (%d epochs max, batch=%d) ...", EPOCHS, BATCH_SIZE)
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=cb,
        verbose=1,
    )

    return model, history, X_test, y_test


# ------------------------------------------------------------------ #
#  EVALUATION & REPORTING                                              #
# ------------------------------------------------------------------ #

def evaluate_and_plot(model, history, X_test, y_test, labels):
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    log.info("=" * 55)
    log.info("Test accuracy : %.4f  (%.1f%%)", test_acc, test_acc * 100)
    log.info("Test loss     : %.4f", test_loss)
    log.info("=" * 55)

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=labels))

    # --- Plot: accuracy/loss curves + confusion matrix ---
    hist = history.history
    epochs_run = len(hist["accuracy"])

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Sign Language MLP — Training Report", fontsize=15, fontweight="bold")

    # Accuracy
    ax = axes[0]
    ax.plot(hist["accuracy"],     label="Train Acc")
    ax.plot(hist["val_accuracy"], label="Val Acc")
    ax.set_title("Accuracy")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.set_ylim(0, 1)

    # Loss
    ax = axes[1]
    ax.plot(hist["loss"],     label="Train Loss")
    ax.plot(hist["val_loss"], label="Val Loss")
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.legend()

    # Confusion matrix (normalised)
    ax = axes[2]
    cm = confusion_matrix(y_test, y_pred, normalize="true")
    tick_size = max(4, 12 - len(labels) // 5)
    sns.heatmap(
        cm,
        annot=len(labels) <= 20,
        fmt=".0%",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        annot_kws={"size": tick_size},
    )
    ax.set_title("Confusion Matrix (normalised)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.tick_params(axis="both", labelsize=tick_size)

    plt.tight_layout()
    REPORT_PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(REPORT_PNG_PATH, dpi=150, bbox_inches="tight")
    log.info("Training report saved to %s", REPORT_PNG_PATH)
    plt.close()


# ------------------------------------------------------------------ #
#  ONNX EXPORT                                                         #
# ------------------------------------------------------------------ #

def export_onnx(model, input_dim: int):
    """Convert the trained Keras model to ONNX via SavedModel + tf2onnx CLI.

    tf2onnx.convert.from_keras() doesn't support Keras 3.x (shipped with
    TF 2.16+). Workaround: export to TF SavedModel first, then convert
    using the tf2onnx CLI (python -m tf2onnx.convert --saved-model ...).
    """
    import shutil
    import subprocess

    saved_model_dir = Path("data/saved_model_temp")
    MODEL_ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: export Keras model → TF SavedModel
    if saved_model_dir.exists():
        shutil.rmtree(saved_model_dir)
    log.info("Saving as TF SavedModel → %s ...", saved_model_dir)
    model.export(str(saved_model_dir))

    # Step 2: convert SavedModel → ONNX via CLI
    log.info("Converting to ONNX → %s ...", MODEL_ONNX_PATH)
    result = subprocess.run(
        [sys.executable, "-m", "tf2onnx.convert",
         "--saved-model", str(saved_model_dir),
         "--output", str(MODEL_ONNX_PATH),
         "--opset", "13"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        log.error("tf2onnx CLI failed:\n%s", result.stderr)
        raise RuntimeError("ONNX export failed — see error above")

    # Cleanup temp SavedModel
    shutil.rmtree(saved_model_dir, ignore_errors=True)

    log.info("ONNX export complete. Model size: %.2f MB",
             MODEL_ONNX_PATH.stat().st_size / 1e6)


# ------------------------------------------------------------------ #
#  MAIN                                                                #
# ------------------------------------------------------------------ #

def main():
    log.info("TensorFlow version : %s", tf.__version__)
    log.info("Python version     : %s", sys.version)

    # GPU check
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        log.info("GPU detected: %s", gpus)
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    else:
        log.info("No GPU detected — training on CPU")

    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    X, y, labels = load_data()
    model, history, X_test, y_test = train(X, y, labels)
    evaluate_and_plot(model, history, X_test, y_test, labels)

    # Save labels
    LABELS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LABELS_JSON_PATH, "w") as f:
        json.dump(labels, f, indent=2)
    log.info("Labels saved to %s  (%d classes)", LABELS_JSON_PATH, len(labels))

    # Export to ONNX
    export_onnx(model, X.shape[1])

    log.info("")
    log.info("✅  Training complete!")
    log.info("   Model  : %s", MODEL_ONNX_PATH.resolve())
    log.info("   Labels : %s", LABELS_JSON_PATH.resolve())
    log.info("   Report : %s", REPORT_PNG_PATH.resolve())


if __name__ == "__main__":
    main()
