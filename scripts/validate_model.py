import onnxruntime as ort
import numpy as np
import json
import os
import shutil

MODEL_ONNX   = r"E:\Github\model files\sign_model.onnx"
MODEL_LABELS = r"E:\Github\model files\labels.json"

# Validate ONNX model with Python 3.14 (project runtime)
sess = ort.InferenceSession(MODEL_ONNX, providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0]
out = sess.get_outputs()[0]
print(f"Input  : name='{inp.name}'  shape={inp.shape}")
print(f"Output : name='{out.name}'  shape={out.shape}")

# Run a dummy inference
dummy = np.zeros((1, 63), dtype="float32")
result = sess.run(None, {inp.name: dummy})
print(f"Inference OK — output shape: {result[0].shape}")

# Load and show labels
with open(MODEL_LABELS) as f:
    labels = json.load(f)
print(f"Labels ({len(labels)}): {labels}")

size_kb = os.path.getsize(MODEL_ONNX) / 1024
print(f"Model size: {size_kb:.1f} KB")

print("\nAll good! Run 'python main.py' to start with the real model.")
