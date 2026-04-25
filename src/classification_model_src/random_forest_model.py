import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
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

os.makedirs('./results/classification/random_forest', exist_ok=True)
os.makedirs('./models/classification', exist_ok=True)

# ── Load preprocessed data ────────────────────────────────────────────────────
X_train = np.load('./data/preprocessed/X_train.npy')
X_test  = np.load('./data/preprocessed/X_test.npy')
y_train = np.load('./data/preprocessed/y_train.npy')
y_test  = np.load('./data/preprocessed/y_test.npy')

# ── Hyperparameter search via cross-validation on TRAIN only ──────────────────
# Test set is never touched during tuning
# Tuning: n_estimators × max_depth × min_samples_split
print("Testing n_estimators × max_depth × min_samples_split (5-fold CV on train set)...")

n_values     = [50, 100, 200]
depth_values = [5, 10, None]   # None = fully grown trees
split_values = [2, 5, 10]

results     = []
best_f1     = -1
best_n      = None
best_depth  = None
best_split  = None

for n in n_values:
    for depth in depth_values:
        for split in split_values:
            candidate = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(
                    n_estimators=n,
                    max_depth=depth,
                    min_samples_split=split,
                    random_state=42,
                    n_jobs=-1,
                )),
            ])
            scores = cross_val_score(
                candidate, X_train, y_train,
                cv=5, scoring='f1_weighted'
            )
            mean, std = scores.mean(), scores.std()
            depth_label = str(depth) if depth is not None else 'None'
            results.append((n, depth, split, mean, std))
            print(f"  n={n:<4} depth={depth_label:<5} split={split:<3} → F1 (weighted): {mean:.4f} ± {std:.4f}")

            if mean > best_f1:
                best_f1, best_n, best_depth, best_split = mean, n, depth, split

depth_label = str(best_depth) if best_depth is not None else 'None'
print(f"\nBest: n_estimators={best_n}, max_depth={depth_label}, min_samples_split={best_split} → CV F1: {best_f1:.4f}")

# ── Plot heatmap: max_depth × min_samples_split (at best n_estimators) ────────
plot_depths = [str(d) if d is not None else 'None' for d in depth_values]
grid = np.zeros((len(depth_values), len(split_values)))
for n, depth, split, mean, std in results:
    if n == best_n:
        i = depth_values.index(depth)
        j = split_values.index(split)
        grid[i, j] = mean

fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(grid, cmap='Blues', aspect='auto')
ax.set_xticks(range(len(split_values)))
ax.set_yticks(range(len(depth_values)))
ax.set_xticklabels([str(s) for s in split_values])
ax.set_yticklabels(plot_depths)
ax.set_xlabel("min_samples_split")
ax.set_ylabel("max_depth")
ax.set_title(f"Random Forest — CV F1 (weighted) Grid (n_estimators={best_n})")

for i in range(len(depth_values)):
    for j in range(len(split_values)):
        ax.text(j, i, f"{grid[i, j]:.3f}", ha='center', va='center', fontsize=9)

plt.colorbar(im, ax=ax, label='F1 (weighted)')
plt.tight_layout()
plt.savefig('./results/classification/random_forest/rf_grid_f1.png', dpi=150)
plt.show()
print("Grid search heatmap saved.")

# ── Train final pipeline with best params on full train set ───────────────────
final_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(
        n_estimators=best_n,
        max_depth=best_depth,
        min_samples_split=best_split,
        random_state=42,
        n_jobs=-1,
    )),
])

final_pipeline.fit(X_train, y_train)

# ── Evaluate ONCE on test set ─────────────────────────────────────────────────
final_pred = final_pipeline.predict(X_test)

accuracy  = np.mean(final_pred == y_test) * 100
f1        = f1_score(y_test, final_pred, average='weighted')
bal_acc   = balanced_accuracy_score(y_test, final_pred)

best_result = next(
    r for r in results
    if r[0] == best_n and r[1] == best_depth and r[2] == best_split
)

depth_label = str(best_depth) if best_depth is not None else 'None'
print(f"\n── Test Set Results (n={best_n}, depth={depth_label}, split={best_split}) ──")
print(f"  Accuracy          : {accuracy:.2f}%")
print(f"  F1 (weighted)     : {f1:.4f}")
print(f"  Balanced Accuracy : {bal_acc:.4f}")
print(f"  CV F1 mean ± std  : {best_result[3]:.4f} ± {best_result[4]:.4f}\n")

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
ax.set_title(f"Random Forest Confusion Matrix\n(n={best_n}, depth={depth_label}, split={best_split})")
plt.tight_layout()
plt.savefig('./results/classification/random_forest/rf_confusion_matrix.png', dpi=150)
plt.show()
print("Confusion matrix saved.")

np.save('./results/classification/random_forest/rf_predictions.npy', final_pred)

# ── Export to ONNX via skl2onnx ───────────────────────────────────────────────
# RandomForestClassifier is natively supported by skl2onnx —
# scaler is baked into the pipeline directly, no manual merge needed.
initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
onnx_model   = convert_sklearn(
    final_pipeline,
    initial_types=initial_type,
    target_opset=12,
)

onnx.checker.check_model(onnx_model)

with open('./models/classification/random_forest_model.onnx', 'wb') as f:
    f.write(onnx_model.SerializeToString())

print("\nONNX model saved.")

# ── Runtime test + inference timing ──────────────────────────────────────────
sess       = ort.InferenceSession('./models/classification/random_forest_model.onnx')
input_name = sess.get_inputs()[0].name

print("Input name :", input_name)
print("Input shape:", sess.get_inputs()[0].shape)
print("Output name:", sess.get_outputs()[0].name)

dummy = np.array([[100.0]], dtype=np.float32)

# Warm-up
sess.run(None, {input_name: dummy})

# Timed runs
n_runs = 100
start  = time.perf_counter()
for _ in range(n_runs):
    sess.run(None, {input_name: dummy})
elapsed_ms = (time.perf_counter() - start) / n_runs * 1000

result = sess.run(None, {input_name: dummy})
print(f"Test prediction   : {result[0]}")
print(f"Avg inference time: {elapsed_ms:.4f} ms (over {n_runs} runs)")