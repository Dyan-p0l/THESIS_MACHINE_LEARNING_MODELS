import time
import itertools
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, f1_score, balanced_accuracy_score
)
import tf2onnx
import onnx
import onnxruntime as ort
import os

os.makedirs('./results/classification/ann', exist_ok=True)
os.makedirs('./models/classification', exist_ok=True)

# ── Load preprocessed data (RAW, unscaled — same as before) ───────────────────
X_train = np.load('./data/preprocessed/X_train.npy').astype(np.float32)
X_test  = np.load('./data/preprocessed/X_test.npy').astype(np.float32)
y_train = np.load('./data/preprocessed/y_train.npy')
y_test  = np.load('./data/preprocessed/y_test.npy')

NUM_CLASSES  = 3
NUM_FEATURES = X_train.shape[1]

# ── Build Normalization layer (replaces StandardScaler in Pipeline) ────────────
# Adapted on X_train only — equivalent to scaler.fit(X_train)
normalizer = layers.Normalization(axis=-1)
normalizer.adapt(X_train)

# ── Helper: build model with given hyperparameters ────────────────────────────

def build_model(hidden_layers: tuple, lr: float, l2_alpha: float) -> keras.Model:
    model = keras.Sequential()
    model.add(keras.Input(shape=(NUM_FEATURES,)))
    model.add(normalizer)                           # scaling baked into model graph
    for units in hidden_layers:
        model.add(layers.Dense(
            units,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(l2_alpha),
        ))
    model.add(layers.Dense(NUM_CLASSES, activation='softmax'))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',     # accepts integer labels directly
        metrics=['accuracy'],
    )
    return model

# ── Manual k-fold CV (replaces cross_val_score) ───────────────────────────────

def kfold_cv_f1(hidden_layers, lr, l2_alpha, X, y, k=3, epochs=30) -> tuple[float, float]:
    n       = len(X)
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    folds   = np.array_split(indices, k)
    scores  = []

    for i in range(k):
        val_idx   = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])

        X_tr, y_tr   = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx],   y[val_idx]

        # Rebuild normalizer per fold to avoid data leakage
        norm = layers.Normalization(axis=-1)
        norm.adapt(X_tr)

        model = keras.Sequential()
        model.add(keras.Input(shape=(NUM_FEATURES,)))
        model.add(norm)
        for units in hidden_layers:
            model.add(layers.Dense(
                units, activation='relu',
                kernel_regularizer=keras.regularizers.l2(l2_alpha),
            ))
        model.add(layers.Dense(NUM_CLASSES, activation='softmax'))
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss='sparse_categorical_crossentropy',
        )

        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=[early_stop],
            verbose=0,
        )

        preds = np.argmax(model.predict(X_val, verbose=0), axis=1)
        scores.append(f1_score(y_val, preds, average='weighted'))

    return float(np.mean(scores)), float(np.std(scores))

# ── Hyperparameter search ─────────────────────────────────────────────────────
# Trimmed grid — deeper nets and stronger regularisation are overkill for a
# single-feature input. 3×2×2×3 folds = 36 fits vs the original 150.

layer_options = [(64,), (128,), (64, 32)]   # deeper nets overkill for 1 feature
lr_options    = [0.001, 0.01]
alpha_options = [1e-4, 1e-3]               # 1e-2 too aggressive for small models

CV_EPOCHS = 30    # caps search-phase training; final model still trains to 300
CV_FOLDS  = 3

print(f"Testing hidden_layer_sizes x learning_rate_init x alpha ({CV_FOLDS}-fold CV on train set)...")
print(f"  [CV budget] max {CV_EPOCHS} epochs per fold (final model trains up to 300)\n")

cv_results  = []
best_f1     = -1
best_layers = None
best_lr     = None
best_alpha  = None

for layers_cfg, lr, alpha in itertools.product(layer_options, lr_options, alpha_options):
    mean, std = kfold_cv_f1(layers_cfg, lr, alpha, X_train, y_train, k=CV_FOLDS, epochs=CV_EPOCHS)
    cv_results.append((layers_cfg, lr, alpha, mean, std))
    print(f"  layers={str(layers_cfg):<12} lr={lr:<6} alpha={alpha:.0e} -> F1 (weighted): {mean:.4f} +/- {std:.4f}")

    if mean > best_f1:
        best_f1, best_layers, best_lr, best_alpha = mean, layers_cfg, lr, alpha

print(f"\nBest: hidden_layer_sizes={best_layers}, lr={best_lr}, alpha={best_alpha:.0e} -> CV F1: {best_f1:.4f}")

# ── Plot heatmap: lr x alpha at best hidden_layer_sizes ──────────────────────

grid = np.zeros((len(lr_options), len(alpha_options)))
for layers_cfg, lr, alpha, mean, std in cv_results:
    if layers_cfg == best_layers:
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
ax.set_title(f"ANN - CV F1 (weighted) Grid (layers={best_layers})")
for i in range(len(lr_options)):
    for j in range(len(alpha_options)):
        ax.text(j, i, f"{grid[i, j]:.3f}", ha='center', va='center', fontsize=9)
plt.colorbar(im, ax=ax, label='F1 (weighted)')
plt.tight_layout()
plt.savefig('./results/classification/ann/ann_grid_f1.png', dpi=150)
plt.show()
print("Grid search heatmap saved.")

# ── Train final model with best params on full train set ──────────────────────
# Normalizer already adapted on full X_train above
final_model = build_model(best_layers, best_lr, best_alpha)

early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=15, restore_best_weights=True, verbose=1
)

history = final_model.fit(
    X_train, y_train,
    validation_split=0.1,           # holds out 10% of train for early stopping
    epochs=300,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1,
)

# ── Plot training loss curve ──────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(history.history['loss'],     label='Training loss')
ax.plot(history.history['val_loss'], label='Validation loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title(f"ANN Loss Curve (layers={best_layers}, lr={best_lr}, alpha={best_alpha:.0e})")
ax.legend()
plt.tight_layout()
plt.savefig('./results/classification/ann/ann_loss_curve.png', dpi=150)
plt.show()
print("Loss curve saved.")

# ── Evaluate ONCE on test set ─────────────────────────────────────────────────

final_pred_probs = final_model.predict(X_test, verbose=0)
final_pred       = np.argmax(final_pred_probs, axis=1)

accuracy = np.mean(final_pred == y_test) * 100
f1       = f1_score(y_test, final_pred, average='weighted')
bal_acc  = balanced_accuracy_score(y_test, final_pred)

best_result = next(
    r for r in cv_results
    if r[0] == best_layers and r[1] == best_lr and r[2] == best_alpha
)

print(f"\n-- Test Set Results (layers={best_layers}, lr={best_lr}, alpha={best_alpha:.0e}) --")
print(f"  Accuracy          : {accuracy:.2f}%")
print(f"  F1 (weighted)     : {f1:.4f}")
print(f"  Balanced Accuracy : {bal_acc:.4f}")
print(f"  CV F1 mean +/- std: {best_result[3]:.4f} +/- {best_result[4]:.4f}\n")

print("Classification Report:")
print(classification_report(y_test, final_pred, target_names=["fresh", "moderate", "spoiled"]))

cm   = confusion_matrix(y_test, final_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["fresh", "moderate", "spoiled"])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title(f"ANN Confusion Matrix\n(layers={best_layers}, lr={best_lr}, alpha={best_alpha:.0e})")
plt.tight_layout()
plt.savefig('./results/classification/ann/ann_confusion_matrix.png', dpi=150)
plt.show()
print("Confusion matrix saved.")

np.save('./results/classification/ann/ann_predictions.npy', final_pred)

# ── Export to ONNX via tf2onnx ────────────────────────────────────────────────
# Normalization layer is part of the model graph, so scaling is baked in --
# identical behaviour to the original skl2onnx Pipeline export.
#
# tf2onnx requires a SavedModel on disk as an intermediate step; we write it
# to a temp folder, convert, then delete the temp folder so only the final
# .onnx ends up in ./models/classification/.

import subprocess, sys, shutil

_tmp_saved  = './models/classification/_tmp_ann_savedmodel'
onnx_output = './models/classification/ann_model.onnx'

final_model.export(_tmp_saved)

result = subprocess.run(
    [
        sys.executable, '-m', 'tf2onnx.convert',
        '--saved-model', _tmp_saved,
        '--output',      onnx_output,
        '--opset',       '12',
    ],
    capture_output=True, text=True,
)
shutil.rmtree(_tmp_saved)   # remove temp folder immediately after conversion

if result.returncode != 0:
    raise RuntimeError(f"tf2onnx conversion failed:\n{result.stderr}")

onnx_model = onnx.load(onnx_output)

# Rename the input from the auto-generated keras tensor name to 'float_input'
# so it stays consistent with all other models and the benchmarking script.
old_input_name = onnx_model.graph.input[0].name
if old_input_name != 'float_input':
    onnx_model.graph.input[0].name = 'float_input'
    for node in onnx_model.graph.node:
        for i, inp in enumerate(node.input):
            if inp == old_input_name:
                node.input[i] = 'float_input'
    onnx.save(onnx_model, onnx_output)

onnx.checker.check_model(onnx_model)
print(f"  Input renamed: '{old_input_name}' -> 'float_input'")
print("\nONNX model saved.")

# ── Export to TFLite ──────────────────────────────────────────────────────────
converter    = tf.lite.TFLiteConverter.from_keras_model(final_model)
tflite_model = converter.convert()

with open('./models/classification/ann_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("TFLite model saved.")

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