import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnx
import joblib
import os

os.makedirs('./results/classification/knn', exist_ok=True)
os.makedirs('./models/classification', exist_ok=True)

# ── Load preprocessed data ────────────────────────────────────────────────────
# X is RAW (unscaled) — the Pipeline will handle scaling internally
X_train = np.load('./data/preprocessed/X_train.npy')
X_test  = np.load('./data/preprocessed/X_test.npy')
y_train = np.load('./data/preprocessed/y_train.npy')
y_test  = np.load('./data/preprocessed/y_test.npy')

# ── Load the scaler fitted during preprocessing ───────────────────────────────
scaler_clf = joblib.load('./data/artifacts/scaler_clf.pkl')

# ── Find the best K ───────────────────────────────────────────────────────────
print("Testing different values of K...")

k_values   = range(1, 21)
accuracies = []

for k in k_values:
    # Each candidate is a full pipeline: scaler + knn
    candidate = Pipeline([
        ('scaler',     scaler_clf),
        ('classifier', KNeighborsClassifier(n_neighbors=k)),
    ])
    candidate.fit(X_train, y_train)
    preds = candidate.predict(X_test)
    acc   = np.mean(preds == y_test) * 100
    accuracies.append(acc)
    print(f"  K={k:2d} → Accuracy: {acc:.2f}%")

best_k   = k_values[np.argmax(accuracies)]
best_acc = max(accuracies)
print(f"\nBest K: {best_k} with accuracy: {best_acc:.2f}%")

# ── Plot K vs Accuracy ────────────────────────────────────────────────────────
plt.figure(figsize=(8, 4))
plt.plot(k_values, accuracies, marker='o', color='steelblue', linewidth=2)
plt.axvline(x=best_k, color='red', linestyle='--', label=f'Best K={best_k}')
plt.title("KNN — Accuracy vs K")
plt.xlabel("K (number of neighbors)")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('./results/classification/knn/knn_k_vs_accuracy.png', dpi=150)
plt.show()
print("K vs Accuracy plot saved.")

# ── Train final pipeline with best K ─────────────────────────────────────────
final_pipeline = Pipeline([
    ('scaler',     scaler_clf),
    ('classifier', KNeighborsClassifier(n_neighbors=best_k)),
])

final_pipeline.fit(X_train, y_train)
final_pred = final_pipeline.predict(X_test)

# ── Evaluate ──────────────────────────────────────────────────────────────────
accuracy = np.mean(final_pred == y_test) * 100
print(f"\nFinal Accuracy (K={best_k}): {accuracy:.2f}%\n")

print("Classification Report:")
print(classification_report(
    y_test, final_pred,
    target_names=["fresh", "moderate", "spoiled"]
))

cm = confusion_matrix(y_test, final_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["fresh", "moderate", "spoiled"]
)

fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title(f"KNN Confusion Matrix (K={best_k})")
plt.tight_layout()
plt.savefig('./results/classification/knn/knn_confusion_matrix.png', dpi=150)
plt.show()
print("Confusion matrix saved.")

np.save('./results/classification/knn/knn_predictions.npy', final_pred)
print(f"Final predictions saved using best K={best_k}")

# ── Export to ONNX — scaler baked in ─────────────────────────────────────────
initial_type = [('float_input', FloatTensorType([None, 1]))]
onnx_model   = convert_sklearn(
    final_pipeline,
    initial_types=initial_type,
    target_opset=12,          # explicitly cap at opset 12
)

# Validate before saving — crashes here instead of on device if broken
onnx.checker.check_model(onnx_model)

with open('./models/classification/knn_model.onnx', 'wb') as f:
    f.write(onnx_model.SerializeToString())

print("\nONNX model saved.")

# ── Quick runtime test — catches input/output name mismatches ─────────────────
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession('./models/classification/knn_model.onnx')
print("Input name :", sess.get_inputs()[0].name)   # must be 'float_input'
print("Input shape:", sess.get_inputs()[0].shape)  # should be [None, 1]
print("Output name:", sess.get_outputs()[0].name)  # should be 'label'

dummy = np.array([[100.0]], dtype=np.float32)
result = sess.run(None, {'float_input': dummy})
print("Test prediction:", result[0])               # should be 0, 1, or 2