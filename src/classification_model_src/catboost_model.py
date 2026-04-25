import time
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, f1_score, balanced_accuracy_score
)
import onnx
from onnx import compose
import onnxruntime as ort
import os

os.makedirs('./results/classification/catboost', exist_ok=True)
os.makedirs('./models/classification', exist_ok=True)

# ── Load preprocessed data ────────────────────────────────────────────────────
X_train = np.load('./data/preprocessed/X_train.npy')
X_test  = np.load('./data/preprocessed/X_test.npy')
y_train = np.load('./data/preprocessed/y_train.npy')
y_test  = np.load('./data/preprocessed/y_test.npy')

# ── Hyperparameter search via cross-validation on TRAIN only ──────────────────
print("Testing iterations × depth × learning_rate (5-fold CV on train set)...")

iter_values  = [50, 100, 200]
depth_values = [4, 6, 8]
lr_values    = [0.01, 0.1, 0.3]

results    = []
best_f1    = -1
best_iter  = None
best_depth = None
best_lr    = None

for itr in iter_values:
    for depth in depth_values:
        for lr in lr_values:
            candidate = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', CatBoostClassifier(
                    iterations=itr,
                    depth=depth,
                    learning_rate=lr,
                    random_seed=42,
                    verbose=0,
                )),
            ])
            scores = cross_val_score(
                candidate, X_train, y_train,
                cv=5, scoring='f1_weighted'
            )
            mean, std = scores.mean(), scores.std()
            results.append((itr, depth, lr, mean, std))
            print(f"  iter={itr:<4} depth={depth} lr={lr:<5} → F1 (weighted): {mean:.4f} ± {std:.4f}")

            if mean > best_f1:
                best_f1, best_iter, best_depth, best_lr = mean, itr, depth, lr

print(f"\nBest: iterations={best_iter}, depth={best_depth}, learning_rate={best_lr} → CV F1: {best_f1:.4f}")

# ── Plot heatmap ──────────────────────────────────────────────────────────────
grid = np.zeros((len(depth_values), len(lr_values)))
for itr, depth, lr, mean, std in results:
    if itr == best_iter:
        i = depth_values.index(depth)
        j = lr_values.index(lr)
        grid[i, j] = mean

fig, ax = plt.subplots(figsize=(7, 4))
im = ax.imshow(grid, cmap='Blues', aspect='auto')
ax.set_xticks(range(len(lr_values)))
ax.set_yticks(range(len(depth_values)))
ax.set_xticklabels([str(lr)    for lr    in lr_values])
ax.set_yticklabels([str(depth) for depth in depth_values])
ax.set_xlabel("learning_rate")
ax.set_ylabel("depth")
ax.set_title(f"CatBoost — CV F1 (weighted) Grid (iterations={best_iter})")

for i in range(len(depth_values)):
    for j in range(len(lr_values)):
        ax.text(j, i, f"{grid[i, j]:.3f}", ha='center', va='center', fontsize=9)

plt.colorbar(im, ax=ax, label='F1 (weighted)')
plt.tight_layout()
plt.savefig('./results/classification/catboost/catboost_grid_f1.png', dpi=150)
plt.show()
print("Grid search heatmap saved.")

# ── Train final pipeline with best params on full train set ───────────────────
final_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', CatBoostClassifier(
        iterations=best_iter,
        depth=best_depth,
        learning_rate=best_lr,
        random_seed=42,
        verbose=0,
    )),
])

final_pipeline.fit(X_train, y_train)

# ── Evaluate ONCE on test set ─────────────────────────────────────────────────
# Cast to int to avoid CatBoost returning string labels on some versions
final_pred = final_pipeline.predict(X_test).astype(int)

final_pred = final_pred.flatten()

accuracy  = np.mean(final_pred == y_test) * 100
f1        = f1_score(y_test, final_pred, average='weighted')
bal_acc   = balanced_accuracy_score(y_test, final_pred)

best_result = next(r for r in results if r[0] == best_iter and r[1] == best_depth and r[2] == best_lr)

print(f"\n── Test Set Results (iter={best_iter}, depth={best_depth}, lr={best_lr}) ──")
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
ax.set_title(f"CatBoost Confusion Matrix (iter={best_iter}, depth={best_depth}, lr={best_lr})")
plt.tight_layout()
plt.savefig('./results/classification/catboost/catboost_confusion_matrix.png', dpi=150)
plt.show()
print("Confusion matrix saved.")

np.save('./results/classification/catboost/catboost_predictions.npy', final_pred)

# ── Export to ONNX ────────────────────────────────────────────────────────────
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

n_features = X_train.shape[1]
scaler     = final_pipeline.named_steps['scaler']
cat_model  = final_pipeline.named_steps['classifier']

# Step 1: scaler → ONNX via skl2onnx
scaler_onnx = convert_sklearn(
    scaler,
    initial_types=[('float_input', FloatTensorType([None, n_features]))],
    target_opset=12,
)

# Step 2: CatBoost → ONNX via native export
cat_model.save_model(
    './models/classification/catboost_raw.onnx',
    format='onnx',
    export_parameters={'onnx_domain': 'ai.catboost', 'onnx_opset_version': 12},
)
cat_onnx = onnx.load('./models/classification/catboost_raw.onnx')

# Step 3: align IR version
target_ir              = max(scaler_onnx.ir_version, cat_onnx.ir_version)
scaler_onnx.ir_version = target_ir
cat_onnx.ir_version    = target_ir

# Step 4: align ai.onnx.ml opset version
# onnxmltools/CatBoost may produce a different ai.onnx.ml version than skl2onnx
def get_opset_version(model, domain):
    for opset in model.opset_import:
        if opset.domain == domain:
            return opset.version
    return None

def set_opset_version(model, domain, version):
    for opset in model.opset_import:
        if opset.domain == domain:
            opset.version = version
            return
    # domain not present — add it
    new_opset = model.opset_import.add()
    new_opset.domain  = domain
    new_opset.version = version

ml_domain   = 'ai.onnx.ml'
scaler_ml_v = get_opset_version(scaler_onnx, ml_domain)
cat_ml_v    = get_opset_version(cat_onnx,    ml_domain)

if scaler_ml_v is not None and cat_ml_v is not None:
    target_ml = max(scaler_ml_v, cat_ml_v)
    set_opset_version(scaler_onnx, ml_domain, target_ml)
    set_opset_version(cat_onnx,    ml_domain, target_ml)
    print(f"Aligned ai.onnx.ml opset to version {target_ml}")

# Step 5: merge scaler → catboost
scaler_out = scaler_onnx.graph.output[0].name
cat_in     = cat_onnx.graph.input[0].name

merged = compose.merge_models(
    scaler_onnx, cat_onnx,
    io_map=[(scaler_out, cat_in)]
)

onnx.checker.check_model(merged)

with open('./models/classification/catboost_model.onnx', 'wb') as f:
    f.write(merged.SerializeToString())

os.remove('./models/classification/catboost_raw.onnx')

print("\nONNX model saved (scaler + CatBoost merged).")

# ── Runtime test + inference timing ──────────────────────────────────────────
sess       = ort.InferenceSession('./models/classification/catboost_model.onnx')
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

for i in range(10):
    print(y_test[i], final_pred[i])