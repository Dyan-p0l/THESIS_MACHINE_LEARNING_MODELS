import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
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

os.makedirs('./results/classification/svm', exist_ok=True)
os.makedirs('./models/classification', exist_ok=True)

# ── Load preprocessed data ────────────────────────────────────────────────────
X_train = np.load('./data/preprocessed/X_train.npy')
X_test  = np.load('./data/preprocessed/X_test.npy')
y_train = np.load('./data/preprocessed/y_train.npy')
y_test  = np.load('./data/preprocessed/y_test.npy')

# ── Hyperparameter search via cross-validation on TRAIN only ──────────────────
# Test set is never touched during tuning
print("Testing different values of C (5-fold CV on train set)...")

c_values  = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
cv_means  = []
cv_stds   = []

for c in c_values:
    candidate = Pipeline([
        ('scaler',     StandardScaler()),
        ('classifier', SVC(C=c, kernel='rbf', gamma='scale', random_state=42)),
    ])
    scores = cross_val_score(
        candidate, X_train, y_train,
        cv=5, scoring='f1_weighted'
    )
    cv_means.append(scores.mean())
    cv_stds.append(scores.std())
    print(f"  C={c:<8} → F1 (weighted): {scores.mean():.4f} ± {scores.std():.4f}")

best_c      = c_values[np.argmax(cv_means)]
best_cv_f1  = max(cv_means)
print(f"\nBest C: {best_c} with CV F1 (weighted): {best_cv_f1:.4f}")

# ── Plot C vs CV F1 ───────────────────────────────────────────────────────────
plt.figure(figsize=(8, 4))
plt.plot(range(len(c_values)), cv_means, marker='o', color='steelblue', linewidth=2, label='CV F1 mean')
plt.fill_between(
    range(len(c_values)),
    np.array(cv_means) - np.array(cv_stds),
    np.array(cv_means) + np.array(cv_stds),
    alpha=0.2, color='steelblue', label='±1 std'
)
plt.axvline(x=np.argmax(cv_means), color='red', linestyle='--', label=f'Best C={best_c}')
plt.xticks(range(len(c_values)), [str(c) for c in c_values])
plt.title("SVM — CV F1 (weighted) vs C (RBF kernel)")
plt.xlabel("C (regularization parameter)")
plt.ylabel("F1 Score (weighted)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('./results/classification/svm/svm_c_vs_f1.png', dpi=150)
plt.show()
print("C vs F1 plot saved.")

# ── Train final pipeline with best C on full train set ────────────────────────
final_pipeline = Pipeline([
    ('scaler',     StandardScaler()),
    ('classifier', SVC(C=best_c, kernel='rbf', gamma='scale', random_state=42)),
])

final_pipeline.fit(X_train, y_train)

# ── Evaluate ONCE on test set ─────────────────────────────────────────────────
final_pred = final_pipeline.predict(X_test)

accuracy  = np.mean(final_pred == y_test) * 100
f1        = f1_score(y_test, final_pred, average='weighted')
bal_acc   = balanced_accuracy_score(y_test, final_pred)

print(f"\n── Test Set Results (C={best_c}) ──")
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
ax.set_title(f"SVM Confusion Matrix (C={best_c}, RBF kernel)")
plt.tight_layout()
plt.savefig('./results/classification/svm/svm_confusion_matrix.png', dpi=150)
plt.show()
print("Confusion matrix saved.")

np.save('./results/classification/svm/svm_predictions.npy', final_pred)

# ── Export to ONNX ────────────────────────────────────────────────────────────
initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
onnx_model   = convert_sklearn(
    final_pipeline,
    initial_types=initial_type,
    target_opset=12,
)

onnx.checker.check_model(onnx_model)

with open('./models/classification/svm_model.onnx', 'wb') as f:
    f.write(onnx_model.SerializeToString())

print("\nONNX model saved.")

# ── Runtime test + inference timing ──────────────────────────────────────────
sess = ort.InferenceSession('./models/classification/svm_model.onnx')
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
print(f"Test prediction   : {result[0]}")
print(f"Avg inference time: {elapsed_ms:.4f} ms (over {n_runs} runs)")