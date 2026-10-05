"""
export_onnx.py
--------------
Standalone script to export the trained Keras model checkpoint to ONNX.

Run this after train_model.py if the ONNX export step failed:
    py -3.12 scripts/export_onnx.py

tf2onnx's from_keras() doesn't support Keras 3.x (shipped with TF 2.21+).
The fix: save as TF SavedModel first, then convert with from_saved_model().
"""

import logging
import shutil
from pathlib import Path

import tensorflow as tf
import tf2onnx

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

CHECKPOINT_PATH  = Path("data/best_model_checkpoint.keras")
SAVED_MODEL_DIR  = Path("data/saved_model_temp")
MODEL_ONNX_PATH  = Path("backend/model/sign_model.onnx")


def main():
    if not CHECKPOINT_PATH.exists():
        log.error("Checkpoint not found: %s", CHECKPOINT_PATH)
        log.error("Run scripts/train_model.py first.")
        raise SystemExit(1)

    log.info("Loading checkpoint from %s ...", CHECKPOINT_PATH)
    model = tf.keras.models.load_model(str(CHECKPOINT_PATH))
    model.summary()

    # Step 1: Export to TF SavedModel format (tf2onnx supports this)
    log.info("Saving as TF SavedModel → %s ...", SAVED_MODEL_DIR)
    if SAVED_MODEL_DIR.exists():
        shutil.rmtree(SAVED_MODEL_DIR)
    model.export(str(SAVED_MODEL_DIR))

    # Step 2: Convert SavedModel → ONNX
    MODEL_ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)
    log.info("Converting to ONNX → %s ...", MODEL_ONNX_PATH)
    tf2onnx.convert.from_saved_model(
        str(SAVED_MODEL_DIR),
        opset=13,
        output_path=str(MODEL_ONNX_PATH),
    )

    size_mb = MODEL_ONNX_PATH.stat().st_size / 1e6
    log.info("ONNX export complete! Size: %.2f MB", size_mb)

    # Cleanup temp SavedModel
    shutil.rmtree(SAVED_MODEL_DIR, ignore_errors=True)
    log.info("Cleaned up temp SavedModel.")

    # Quick validation — load ONNX and run a dummy inference
    log.info("Validating ONNX model with dummy input ...")
    import onnxruntime as ort
    import numpy as np
    sess = ort.InferenceSession(str(MODEL_ONNX_PATH), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    dummy = np.zeros((1, 63), dtype=np.float32)
    out = sess.run(None, {input_name: dummy})
    log.info("  Output shape : %s  (num_classes=%d)", out[0].shape, out[0].shape[1])
    log.info("  Input name   : %s", input_name)

    log.info("")
    log.info("Done! Files ready:")
    log.info("  Model  : %s", MODEL_ONNX_PATH.resolve())
    log.info("  Labels : %s", Path('backend/model/labels.json').resolve())
    log.info("")
    log.info("Run 'python main.py' to start the interpreter with the real model.")


if __name__ == "__main__":
    main()
