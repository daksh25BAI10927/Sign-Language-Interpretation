"""
extract_landmarks.py
---------------------
Extracts MediaPipe hand landmarks from dataset images and saves them
to a CSV file that the training script can load.

Uses the MediaPipe Tasks API (HandLandmarker) — matching exactly what
the backend pipeline uses — so training and inference are always aligned.

Datasets used:
  - asl_alphabet_train  : 29 classes (A-Z + del/nothing/space), 3000 imgs, 200x200
  - Data (friend's)     : 36 classes (0-9, a-z), 70 imgs, 400x400

Usage:
    py -3.12 scripts/extract_landmarks.py

Output:
    data/landmarks_dataset.csv  — rows: label, x0,y0,z0, ..., x20,y20,z20
"""

import csv
import sys
import logging
import urllib.request
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
from tqdm import tqdm

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

# ------------------------------------------------------------------ #
#  CONFIG                                                             #
# ------------------------------------------------------------------ #

# Label normalisation for ASL train: keep del/nothing/space as-is, uppercase rest
def asl_label(folder_name: str) -> str:
    """Map ASL train folder name to final label."""
    special = {"del": "DEL", "nothing": "NOTHING", "space": "SPACE"}
    return special.get(folder_name.lower(), folder_name.upper())


DATASETS = [
    # (root_folder, label_fn, description)
    (
        r"E:\Github\datasets and friends ml model\asl_alphabet_train\asl_alphabet_train",
        asl_label,
        "ASL Alphabet Train (A-Z + del/nothing/space, 200x200)",
    ),
    (
        r"E:\Github\datasets and friends ml model\Data",
        str.upper,
        "Friend's Data (0-9, A-Z, 400x400)",
    ),
]

OUTPUT_CSV  = Path("data/landmarks_dataset.csv")
IMAGE_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Max images per class from each dataset (None = use all)
# ASL train has 3000/class — cap at 2000 to balance with other classes
MAX_PER_CLASS = 2000

# Hand landmarker model bundle
HAND_LANDMARKER_MODEL = Path("backend/hand_detection/models/hand_landmarker.task")
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

DETECT_CONF = 0.3   # lower = more detections on dataset images

# ------------------------------------------------------------------ #
#  LOGGING                                                            #
# ------------------------------------------------------------------ #

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

WRIST      = 0
MIDDLE_MCP = 9


# ------------------------------------------------------------------ #
#  MEDIAPIPE                                                          #
# ------------------------------------------------------------------ #

def _ensure_model(path: Path) -> str:
    if path.exists():
        return str(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading HandLandmarker model from Google...")
    urllib.request.urlretrieve(_MODEL_URL, str(path))
    log.info("Downloaded to %s", path)
    return str(path)


def build_landmarker(model_path: str) -> HandLandmarker:
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=DETECT_CONF,
        min_tracking_confidence=DETECT_CONF,
    )
    return HandLandmarker.create_from_options(options)


# ------------------------------------------------------------------ #
#  LANDMARK EXTRACTION                                                #
# ------------------------------------------------------------------ #

def extract_landmarks(landmarker: HandLandmarker, image_path: str):
    """
    Run MediaPipe HandLandmarker on a single image.

    Returns:
        list[float] of length 63 (21 x,y,z), wrist-centred and
        scale-normalised — identical to backend Preprocessor output.
        Returns None if no hand detected.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return None

    img_rgb = np.ascontiguousarray(img_bgr[:, :, ::-1])
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

    result = landmarker.detect(mp_image)
    if not result.hand_landmarks:
        return None

    lm_list = result.hand_landmarks[0]
    coords = np.array([[lm.x, lm.y, lm.z] for lm in lm_list], dtype=np.float32)

    # Mirror backend Preprocessor exactly:
    coords -= coords[WRIST]                       # 1. Centre on wrist
    scale = np.linalg.norm(coords[MIDDLE_MCP])
    if scale > 1e-6:
        coords /= scale                           # 2. Scale-normalise

    return coords.flatten().tolist()              # (63,)


# ------------------------------------------------------------------ #
#  DATASET DISCOVERY                                                  #
# ------------------------------------------------------------------ #

def collect_image_paths(root: str, label_fn, desc: str):
    items = []
    root_path = Path(root)
    if not root_path.exists():
        log.warning("Skipping (not found): %s", root)
        return items

    for class_dir in sorted(root_path.iterdir()):
        if not class_dir.is_dir():
            continue
        label = label_fn(class_dir.name)
        images = sorted(
            p for p in class_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTS
        )
        if MAX_PER_CLASS:
            images = images[:MAX_PER_CLASS]
        for img_path in images:
            items.append((label, str(img_path)))

    classes = {lbl for lbl, _ in items}
    log.info("  %-60s → %6d imgs | %2d classes",
             desc, len(items), len(classes))
    return items


# ------------------------------------------------------------------ #
#  MAIN                                                               #
# ------------------------------------------------------------------ #

def main():
    model_path = _ensure_model(HAND_LANDMARKER_MODEL)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    log.info("Scanning datasets...")
    all_items = []
    for root, label_fn, desc in DATASETS:
        items = collect_image_paths(root, label_fn, desc)
        all_items.extend(items)

    if not all_items:
        log.error("No images found — check DATASETS paths.")
        sys.exit(1)

    counts = Counter(lbl for lbl, _ in all_items)
    log.info("─" * 60)
    log.info("Total images  : %d", len(all_items))
    log.info("Total classes : %d", len(counts))
    log.info("Classes       : %s", sorted(counts.keys()))
    log.info("─" * 60)

    header = ["label"] + [f"{ax}{i}" for i in range(21) for ax in ("x", "y", "z")]

    detected  = 0
    skipped   = 0
    per_class = Counter()

    landmarker = build_landmarker(model_path)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for label, img_path in tqdm(all_items, desc="Extracting", unit="img"):
            landmarks = extract_landmarks(landmarker, img_path)
            if landmarks is None:
                skipped += 1
                continue
            writer.writerow([label] + landmarks)
            detected += 1
            per_class[label] += 1

    landmarker.close()

    total = len(all_items)
    log.info("─" * 60)
    log.info("Done.")
    log.info("  Extracted  : %d  (%.1f%%)", detected, 100 * detected / max(total, 1))
    log.info("  Skipped    : %d  (%.1f%%)", skipped, 100 * skipped / max(total, 1))
    log.info("  CSV        : %s", OUTPUT_CSV.resolve())
    log.info("─" * 60)
    log.info("Samples per class:")
    for cls in sorted(per_class):
        log.info("  %-8s : %d", cls, per_class[cls])

    poor = [c for c, n in per_class.items() if n < 30]
    if poor:
        log.warning("Low-sample classes (< 30 samples): %s", poor)


if __name__ == "__main__":
    main()
