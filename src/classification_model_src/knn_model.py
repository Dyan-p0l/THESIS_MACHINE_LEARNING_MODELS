import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, f1_score, balanced_accuracy_score
)
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnx
import onnxruntime as ort
import os

os.makedirs('./results/classification/knn', exist_ok=True)
os.makedirs('./models/classification', exist_ok=True)

# ── Load preprocessed data ────────────────────────────────────────────────────
X_train = np.load('./data/preprocessed/X_train.npy')
X_test  = np.load('./data/preprocessed/X_test.npy')
y_train = np.load('./data/preprocessed/y_train.npy')
y_test  = np.load('./data/preprocessed/y_test.npy')

# ── Hyperparameter search via cross-validation on TRAIN only ──────────────────
# Test set is never touched during tuning
print("Testing different values of K (5-fold CV on train set)...")

k_values  = range(1, 21)
cv_means  = []
cv_stds   = []

for k in k_values:
    candidate = Pipeline([
        ('scaler',     StandardScaler()),
        ('classifier', KNeighborsClassifier(n_neighbors=k)),
    ])
    scores = cross_val_score(
        candidate, X_train, y_train,
        cv=5, scoring='f1_weighted'
    )
    cv_means.append(scores.mean())
    cv_stds.append(scores.std())
    print(f"  K={k:2d} → F1 (weighted): {scores.mean():.4f} ± {scores.std():.4f}")

best_k      = list(k_values)[np.argmax(cv_means)]
best_cv_f1  = max(cv_means)
print(f"\nBest K: {best_k} with CV F1 (weighted): {best_cv_f1:.4f}")

# ── Plot K vs CV F1 ───────────────────────────────────────────────────────────
plt.figure(figsize=(8, 4))
plt.plot(k_values, cv_means, marker='o', color='steelblue', linewidth=2, label='CV F1 mean')
plt.fill_between(
    k_values,
    np.array(cv_means) - np.array(cv_stds),
    np.array(cv_means) + np.array(cv_stds),
    alpha=0.2, color='steelblue', label='±1 std'
)
plt.axvline(x=best_k, color='red', linestyle='--', label=f'Best K={best_k}')
plt.title("KNN — CV F1 (weighted) vs K")
plt.xlabel("K (number of neighbors)")
plt.ylabel("F1 Score (weighted)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('./results/classification/knn/knn_k_vs_f1.png', dpi=150)
plt.show()
print("K vs F1 plot saved.")

# ── Train final pipeline with best K on full train set ────────────────────────
final_pipeline = Pipeline([
    ('scaler',     StandardScaler()),
    ('classifier', KNeighborsClassifier(n_neighbors=best_k)),
])

final_pipeline.fit(X_train, y_train)

# ── Evaluate ONCE on test set ─────────────────────────────────────────────────
final_pred = final_pipeline.predict(X_test)

accuracy      = np.mean(final_pred == y_test) * 100
f1            = f1_score(y_test, final_pred, average='weighted')
bal_acc       = balanced_accuracy_score(y_test, final_pred)

print(f"\n── Test Set Results (K={best_k}) ──")
print(f"  Accuracy          : {accuracy:.2f}%")
print(f"  F1 (weighted)     : {f1:.4f}")
print(f"  Balanced Accuracy : {bal_acc:.4f}")
print(f"  CV F1 mean ± std  : {best_cv_f1:.4f} ± {cv_stds[np.argmax(cv_means)]:.4f}\n")

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

# ── Export to ONNX ────────────────────────────────────────────────────────────
initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
onnx_model   = convert_sklearn(
    final_pipeline,
    initial_types=initial_type,
    target_opset=12,
)

onnx.checker.check_model(onnx_model)

with open('./models/classification/knn_model.onnx', 'wb') as f:
    f.write(onnx_model.SerializeToString())

print("\nONNX model saved.")

# ── Runtime test + inference timing ──────────────────────────────────────────
sess = ort.InferenceSession('./models/classification/knn_model.onnx')
print("Input name :", sess.get_inputs()[0].name)
print("Input shape:", sess.get_inputs()[0].shape)
print("Output name:", sess.get_outputs()[0].name)

dummy = np.array([[100.0]], dtype=np.float32)

# Warm-up
sess.run(None, {'float_input': dummy})

# Timed runs
n_runs = 100
start  = time.perf_counter()
for _ in range(n_runs):
    sess.run(None, {'float_input': dummy})
elapsed_ms = (time.perf_counter() - start) / n_runs * 1000

result = sess.run(None, {'float_input': dummy})
print(f"Test prediction  : {result[0]}")
print(f"Avg inference time: {elapsed_ms:.4f} ms (over {n_runs} runs)")