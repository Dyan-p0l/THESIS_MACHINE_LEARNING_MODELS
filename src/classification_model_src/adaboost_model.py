import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
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

os.makedirs('./results/classification/adaboost', exist_ok=True)
os.makedirs('./models/classification', exist_ok=True)

# ── Load preprocessed data ────────────────────────────────────────────────────
X_train = np.load('./data/preprocessed/X_train.npy')
X_test  = np.load('./data/preprocessed/X_test.npy')
y_train = np.load('./data/preprocessed/y_train.npy')
y_test  = np.load('./data/preprocessed/y_test.npy')

# ── Hyperparameter search via cross-validation on TRAIN only ──────────────────
# Test set is never touched during tuning
# n_estimators and learning_rate are tuned jointly since they interact
print("Testing n_estimators × learning_rate grid (5-fold CV on train set)...")

n_values  = [50, 100, 150]
lr_values = [0.1, 0.5, 1.0]

results   = []   # (n, lr, mean, std)
best_f1   = -1
best_n    = None
best_lr   = None

for n in n_values:
    for lr in lr_values:
        candidate = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=1),
                n_estimators=n,
                learning_rate=lr,
                random_state=42,
            )),
        ])
        scores = cross_val_score(
            candidate, X_train, y_train,
            cv=5, scoring='f1_weighted'
        )
        mean, std = scores.mean(), scores.std()
        results.append((n, lr, mean, std))
        print(f"  n={n:<4} lr={lr:<4} → F1 (weighted): {mean:.4f} ± {std:.4f}")

        if mean > best_f1:
            best_f1, best_n, best_lr = mean, n, lr

print(f"\nBest: n_estimators={best_n}, learning_rate={best_lr} → CV F1: {best_f1:.4f}")

# ── Plot heatmap: n_estimators × learning_rate ────────────────────────────────
grid = np.array([r[2] for r in results]).reshape(len(n_values), len(lr_values))

fig, ax = plt.subplots(figsize=(7, 4))
im = ax.imshow(grid, cmap='Blues', aspect='auto')
ax.set_xticks(range(len(lr_values)))
ax.set_yticks(range(len(n_values)))
ax.set_xticklabels([str(lr) for lr in lr_values])
ax.set_yticklabels([str(n)  for n  in n_values])
ax.set_xlabel("learning_rate")
ax.set_ylabel("n_estimators")
ax.set_title("AdaBoost — CV F1 (weighted) Grid")

for i in range(len(n_values)):
    for j in range(len(lr_values)):
        ax.text(j, i, f"{grid[i, j]:.3f}", ha='center', va='center', fontsize=9)

plt.colorbar(im, ax=ax, label='F1 (weighted)')
plt.tight_layout()
plt.savefig('./results/classification/adaboost/adaboost_grid_f1.png', dpi=150)
plt.show()
print("Grid search heatmap saved.")

# ── Train final pipeline with best params on full train set ───────────────────
final_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=best_n,
        learning_rate=best_lr,
        random_state=42,
    )),
])

final_pipeline.fit(X_train, y_train)

# ── Evaluate ONCE on test set ─────────────────────────────────────────────────
final_pred = final_pipeline.predict(X_test)

accuracy  = np.mean(final_pred == y_test) * 100
f1        = f1_score(y_test, final_pred, average='weighted')
bal_acc   = balanced_accuracy_score(y_test, final_pred)

best_result = next(r for r in results if r[0] == best_n and r[1] == best_lr)

print(f"\n── Test Set Results (n={best_n}, lr={best_lr}) ──")
print(f"  Accuracy          : {accuracy:.2f}%")
print(f"  F1 (weighted)     : {f1:.4f}")
print(f"  Balanced Accuracy : {bal_acc:.4f}")
print(f"  CV F1 mean ± std  : {best_result[2]:.4f} ± {best_result[3]:.4f}\n")

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
ax.set_title(f"AdaBoost Confusion Matrix (n={best_n}, lr={best_lr})")
plt.tight_layout()
plt.savefig('./results/classification/adaboost/adaboost_confusion_matrix.png', dpi=150)
plt.show()
print("Confusion matrix saved.")

np.save('./results/classification/adaboost/adaboost_predictions.npy', final_pred)

# ── Export to ONNX ────────────────────────────────────────────────────────────
initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
onnx_model   = convert_sklearn(
    final_pipeline,
    initial_types=initial_type,
    target_opset=12,
)

onnx.checker.check_model(onnx_model)

with open('./models/classification/adaboost_model.onnx', 'wb') as f:
    f.write(onnx_model.SerializeToString())

print("\nONNX model saved.")

# ── Runtime test + inference timing ──────────────────────────────────────────
sess = ort.InferenceSession('./models/classification/adaboost_model.onnx')
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