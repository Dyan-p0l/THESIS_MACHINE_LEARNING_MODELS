import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession('./models/classification/ann_model.onnx')

# ── Print model info ──────────────────────────────────────────────────────────
print("=== Model Info ===")
print("Input  name :", sess.get_inputs()[0].name)
print("Input  shape:", sess.get_inputs()[0].shape)
print("Output name :", sess.get_outputs()[0].name)
print("All outputs :", [o.name for o in sess.get_outputs()])

# ── Test with a range of capacitance values ───────────────────────────────────
# Replace these with actual values from your dataset
test_values = [4.9955, 7.42175, 2.1385, 1.4895, 7.61675, 5.2965, 4.0555, 3.065]

label_map = {0: 'fresh', 1: 'moderate', 2: 'spoiled'}

print("\n=== Predictions ===")
for val in test_values:
    inp = np.array([[val]], dtype=np.float32)
    result = sess.run(None, {'float_input': inp})
    label_index = result[0][0]
    print(f"  capacitance={val:.1f} pF → raw output={label_index} → {label_map.get(int(label_index), '???')}")

X_train = np.load('./data/preprocessed/X_train.npy')
y_train = np.load('./data/preprocessed/y_train.npy')

print("\n=== Training Data Ranges ===")
for cls, name in label_map.items():
    vals = X_train[y_train == cls]
    if len(vals):
        print(f"  {name:8s} (class {cls}): min={vals.min():.3f}  max={vals.max():.3f}  mean={vals.mean():.3f}  n={len(vals)}")  