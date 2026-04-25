import time
import numpy as np
import matplotlib.pyplot as plt
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

os.makedirs('./results/classification/decision_tree', exist_ok=True)
os.makedirs('./models/classification', exist_ok=True)

# ── Load preprocessed data ────────────────────────────────────────────────────
X_train = np.load('./data/preprocessed/X_train.npy')
X_test  = np.load('./data/preprocessed/X_test.npy')
y_train = np.load('./data/preprocessed/y_train.npy')
y_test  = np.load('./data/preprocessed/y_test.npy')

# ── Hyperparameter search via cross-validation on TRAIN only ──────────────────
# Test set is never touched during tuning
# Tuning: max_depth × min_samples_split × criterion
print("Testing max_depth × min_samples_split × criterion (5-fold CV on train set)...")

depth_values       = [3, 5, 7, 10, None]   # None = fully grown tree
min_split_values   = [2, 5, 10]
criterion_values   = ['gini', 'entropy']

results     = []
best_f1     = -1
best_depth  = None
best_split  = None
best_crit   = None

for depth in depth_values:
    for split in min_split_values:
        for crit in criterion_values:
            candidate = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', DecisionTreeClassifier(
                    max_depth=depth,
                    min_samples_split=split,
                    criterion=crit,
                    random_state=42,
                )),
            ])
            scores = cross_val_score(
                candidate, X_train, y_train,
                cv=5, scoring='f1_weighted'
            )
            mean, std = scores.mean(), scores.std()
            depth_label = str(depth) if depth is not None else 'None'
            results.append((depth, split, crit, mean, std))
            print(f"  depth={depth_label:<5} split={split:<3} crit={crit:<8} → F1 (weighted): {mean:.4f} ± {std:.4f}")

            if mean > best_f1:
                best_f1, best_depth, best_split, best_crit = mean, depth, split, crit

depth_label = str(best_depth) if best_depth is not None else 'None'
print(f"\nBest: max_depth={depth_label}, min_samples_split={best_split}, criterion={best_crit} → CV F1: {best_f1:.4f}")

# ── Plot heatmap: max_depth × min_samples_split (at best criterion) ───────────
plot_depths = [d if d is not None else 'None' for d in depth_values]
grid = np.zeros((len(depth_values), len(min_split_values)))
for depth, split, crit, mean, std in results:
    if crit == best_crit:
        i = depth_values.index(depth)
        j = min_split_values.index(split)
        grid[i, j] = mean

fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(grid, cmap='Blues', aspect='auto')
ax.set_xticks(range(len(min_split_values)))
ax.set_yticks(range(len(depth_values)))
ax.set_xticklabels([str(s) for s in min_split_values])
ax.set_yticklabels([str(d) for d in plot_depths])
ax.set_xlabel("min_samples_split")
ax.set_ylabel("max_depth")
ax.set_title(f"Decision Tree — CV F1 (weighted) Grid (criterion='{best_crit}')")

for i in range(len(depth_values)):
    for j in range(len(min_split_values)):
        ax.text(j, i, f"{grid[i, j]:.3f}", ha='center', va='center', fontsize=9)

plt.colorbar(im, ax=ax, label='F1 (weighted)')
plt.tight_layout()
plt.savefig('./results/classification/decision_tree/dt_grid_f1.png', dpi=150)
plt.show()
print("Grid search heatmap saved.")

# ── Train final pipeline with best params on full train set ───────────────────
final_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', DecisionTreeClassifier(
        max_depth=best_depth,
        min_samples_split=best_split,
        criterion=best_crit,
        random_state=42,
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
    if r[0] == best_depth and r[1] == best_split and r[2] == best_crit
)

depth_label = str(best_depth) if best_depth is not None else 'None'
print(f"\n── Test Set Results (depth={depth_label}, split={best_split}, crit={best_crit}) ──")
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
ax.set_title(f"Decision Tree Confusion Matrix\n(depth={depth_label}, split={best_split}, crit={best_crit})")
plt.tight_layout()
plt.savefig('./results/classification/decision_tree/dt_confusion_matrix.png', dpi=150)
plt.show()
print("Confusion matrix saved.")

np.save('./results/classification/decision_tree/dt_predictions.npy', final_pred)

# ── Export to ONNX via skl2onnx ───────────────────────────────────────────────
# DecisionTreeClassifier is natively supported by skl2onnx —
# no onnxmltools or manual merge needed, scaler baked in directly.
initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
onnx_model   = convert_sklearn(
    final_pipeline,
    initial_types=initial_type,
    target_opset=12,
)

onnx.checker.check_model(onnx_model)

with open('./models/classification/decision_tree_model.onnx', 'wb') as f:
    f.write(onnx_model.SerializeToString())

print("\nONNX model saved.")

# ── Runtime test + inference timing ──────────────────────────────────────────
sess       = ort.InferenceSession('./models/classification/decision_tree_model.onnx')
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