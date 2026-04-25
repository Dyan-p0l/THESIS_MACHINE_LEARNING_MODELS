import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
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

os.makedirs('./results/classification/ann', exist_ok=True)
os.makedirs('./models/classification', exist_ok=True)

# ── Load preprocessed data ────────────────────────────────────────────────────
X_train = np.load('./data/preprocessed/X_train.npy')
X_test  = np.load('./data/preprocessed/X_test.npy')
y_train = np.load('./data/preprocessed/y_train.npy')
y_test  = np.load('./data/preprocessed/y_test.npy')

# ── Hyperparameter search via cross-validation on TRAIN only ──────────────────
# Test set is never touched during tuning.
# Tuning: hidden_layer_sizes × learning_rate_init × alpha (L2 regularisation)
print("Testing hidden_layer_sizes × learning_rate_init × alpha (5-fold CV on train set)...")

layer_options  = [(64,), (128,), (64, 32), (128, 64), (128, 64, 32)]
lr_options     = [0.001, 0.01]
alpha_options  = [1e-4, 1e-3, 1e-2]

results    = []
best_f1    = -1
best_layers = None
best_lr    = None
best_alpha = None

for layers in layer_options:
    for lr in lr_options:
        for alpha in alpha_options:
            candidate = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', MLPClassifier(
                    hidden_layer_sizes=layers,
                    activation='relu',
                    solver='adam',
                    learning_rate_init=lr,
                    alpha=alpha,
                    max_iter=300,
                    early_stopping=True,   # hold out 10 % of train for val
                    n_iter_no_change=15,
                    random_state=42,
                )),
            ])
            scores = cross_val_score(
                candidate, X_train, y_train,
                cv=5, scoring='f1_weighted'
            )
            mean, std = scores.mean(), scores.std()
            results.append((layers, lr, alpha, mean, std))
            print(f"  layers={str(layers):<16} lr={lr:<6} alpha={alpha:.0e} → F1 (weighted): {mean:.4f} ± {std:.4f}")

            if mean > best_f1:
                best_f1, best_layers, best_lr, best_alpha = mean, layers, lr, alpha

print(f"\nBest: hidden_layer_sizes={best_layers}, lr={best_lr}, alpha={best_alpha:.0e} → CV F1: {best_f1:.4f}")

# ── Plot heatmap: learning_rate_init × alpha (at best hidden_layer_sizes) ─────
grid = np.zeros((len(lr_options), len(alpha_options)))
for layers, lr, alpha, mean, std in results:
    if layers == best_layers:
        i = lr_options.index(lr)
        j = alpha_options.index(alpha)
        grid[i, j] = mean

fig, ax = plt.subplots(figsize=(7, 4))
im = ax.imshow(grid, cmap='Blues', aspect='auto')
ax.set_xticks(range(len(alpha_options)))
ax.set_yticks(range(len(lr_options)))
ax.set_xticklabels([f"{a:.0e}" for a in alpha_options])
ax.set_yticklabels([str(lr) for lr in lr_options])
ax.set_xlabel("alpha (L2 regularisation)")
ax.set_ylabel("learning_rate_init")
ax.set_title(f"ANN — CV F1 (weighted) Grid (layers={best_layers})")

for i in range(len(lr_options)):
    for j in range(len(alpha_options)):
        ax.text(j, i, f"{grid[i, j]:.3f}", ha='center', va='center', fontsize=9)

plt.colorbar(im, ax=ax, label='F1 (weighted)')
plt.tight_layout()
plt.savefig('./results/classification/ann/ann_grid_f1.png', dpi=150)
plt.show()
print("Grid search heatmap saved.")

# ── Train final pipeline with best params on full train set ───────────────────
final_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', MLPClassifier(
        hidden_layer_sizes=best_layers,
        activation='relu',
        solver='adam',
        learning_rate_init=best_lr,
        alpha=best_alpha,
        max_iter=300,
        early_stopping=True,
        n_iter_no_change=15,
        random_state=42,
    )),
])

final_pipeline.fit(X_train, y_train)

# ── Plot training loss curve ──────────────────────────────────────────────────
mlp = final_pipeline.named_steps['classifier']
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(mlp.loss_curve_, label='Training loss')
if hasattr(mlp, 'validation_scores_') and mlp.validation_scores_ is not None:
    ax.plot(mlp.validation_scores_, label='Validation F1 (early stopping)')
    ax.set_ylabel('Loss / Validation F1')
else:
    ax.set_ylabel('Loss')
ax.set_xlabel('Epoch')
ax.set_title(f"ANN Loss Curve (layers={best_layers}, lr={best_lr}, alpha={best_alpha:.0e})")
ax.legend()
plt.tight_layout()
plt.savefig('./results/classification/ann/ann_loss_curve.png', dpi=150)
plt.show()
print("Loss curve saved.")

# ── Evaluate ONCE on test set ─────────────────────────────────────────────────
final_pred = final_pipeline.predict(X_test)

accuracy = np.mean(final_pred == y_test) * 100
f1       = f1_score(y_test, final_pred, average='weighted')
bal_acc  = balanced_accuracy_score(y_test, final_pred)

best_result = next(
    r for r in results
    if r[0] == best_layers and r[1] == best_lr and r[2] == best_alpha
)

print(f"\n── Test Set Results (layers={best_layers}, lr={best_lr}, alpha={best_alpha:.0e}) ──")
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
ax.set_title(f"ANN Confusion Matrix\n(layers={best_layers}, lr={best_lr}, alpha={best_alpha:.0e})")
plt.tight_layout()
plt.savefig('./results/classification/ann/ann_confusion_matrix.png', dpi=150)
plt.show()
print("Confusion matrix saved.")

np.save('./results/classification/ann/ann_predictions.npy', final_pred)

# ── Export to ONNX via skl2onnx ───────────────────────────────────────────────
initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
onnx_model   = convert_sklearn(
    final_pipeline,
    initial_types=initial_type,
    target_opset=12,
)

onnx.checker.check_model(onnx_model)

with open('./models/classification/ann_model.onnx', 'wb') as f:
    f.write(onnx_model.SerializeToString())

print("\nONNX model saved.")

# ── Runtime test + inference timing ──────────────────────────────────────────
sess       = ort.InferenceSession('./models/classification/ann_model.onnx')
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