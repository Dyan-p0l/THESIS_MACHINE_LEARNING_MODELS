import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
)
import onnxruntime as ort

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────

MODELS = {
    "KNN":           "./models/classification/knn_model.onnx",
    "SVM":           "./models/classification/svm_model.onnx",
    "XGBoost":       "./models/classification/xgboost_model.onnx",
    "AdaBoost":      "./models/classification/adaboost_model.onnx",
    "CatBoost":      "./models/classification/catboost_model.onnx",
    "Random Forest": "./models/classification/random_forest_model.onnx",
    "Decision Tree": "./models/classification/decision_tree_model.onnx",
    "ANN":           "./models/classification/ann_model.onnx",
}

CLASS_NAMES   = ["fresh", "moderate", "spoiled"]
N_TIMING_RUNS = 200           # runs per model for inference timing
OUT_DIR       = "./results/benchmarking"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────

print("Loading preprocessed data...")
X_test  = np.load('./data/preprocessed/X_test.npy').astype(np.float32)
y_test  = np.load('./data/preprocessed/y_test.npy')
print(f"  X_test : {X_test.shape}  |  y_test : {y_test.shape}\n")

dummy_single = X_test[:1]   # 1-sample array for latency measurement

# ── Helper: load session with fallback input name ─────────────────────────────

def load_session(path: str):
    sess = ort.InferenceSession(path)
    input_name  = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    return sess, input_name, output_name


def batch_predict(sess, input_name, X: np.ndarray) -> np.ndarray:
    result = sess.run(None, {input_name: X})
    out = np.array(result[0])
    if out.ndim == 2:
        # TF/Keras model — output is softmax probabilities (n_samples, n_classes)
        return np.argmax(out, axis=1).astype(int)
    else:
        # sklearn model — output is already class labels (n_samples,)
        return out.flatten().astype(int)


def measure_latency_ms(sess, input_name, sample: np.ndarray,
                        n_runs: int = N_TIMING_RUNS) -> float:
    """Single-sample inference latency (ms), averaged over n_runs."""
    # warm-up
    for _ in range(10):
        sess.run(None, {input_name: sample})
    start = time.perf_counter()
    for _ in range(n_runs):
        sess.run(None, {input_name: sample})
    return (time.perf_counter() - start) / n_runs * 1000


# ── Benchmark loop ────────────────────────────────────────────────────────────

results  = []
all_preds = {}

print(f"{'Model':<16} {'Acc':>7} {'F1-W':>7} {'Bal-Acc':>9} {'Latency(ms)':>12}")
print("─" * 56)

for name, onnx_path in MODELS.items():
    if not os.path.exists(onnx_path):
        print(f"  [SKIP] {name}: file not found → {onnx_path}")
        continue

    try:
        sess, inp, out = load_session(onnx_path)

        # ── Batch prediction on full test set ──────────────────────────────
        t0    = time.perf_counter()
        preds = batch_predict(sess, inp, X_test)
        batch_ms = (time.perf_counter() - t0) * 1000   # total ms for all samples

        # ── Single-sample latency ──────────────────────────────────────────
        lat_ms = measure_latency_ms(sess, inp, dummy_single)

        # ── Metrics ───────────────────────────────────────────────────────
        acc     = np.mean(preds == y_test) * 100
        f1_w    = f1_score(y_test, preds, average='weighted')
        f1_mac  = f1_score(y_test, preds, average='macro')
        f1_per  = f1_score(y_test, preds, average=None, labels=[0, 1, 2])
        bal_acc = balanced_accuracy_score(y_test, preds)

        all_preds[name] = preds

        results.append({
            "Model":              name,
            "Accuracy (%)":       round(acc, 4),
            "F1 Weighted":        round(f1_w, 4),
            "F1 Macro":           round(f1_mac, 4),
            "F1 (fresh)":         round(f1_per[0], 4),
            "F1 (moderate)":      round(f1_per[1], 4),
            "F1 (spoiled)":       round(f1_per[2], 4),
            "Balanced Accuracy":  round(bal_acc, 4),
            "Latency/sample (ms)": round(lat_ms, 4),
            "Total batch (ms)":   round(batch_ms, 2),
        })

        print(f"  {name:<14} {acc:>7.2f}% {f1_w:>7.4f} {bal_acc:>9.4f} {lat_ms:>12.4f}")

    except Exception as e:
        print(f"  [ERROR] {name}: {e}")

print()

# ── DataFrame + CSV ───────────────────────────────────────────────────────────

df = pd.DataFrame(results).sort_values("F1 Weighted", ascending=False).reset_index(drop=True)
df.index += 1   # rank from 1
df.to_csv(f"{OUT_DIR}/benchmark_summary.csv", index_label="Rank")
print("Benchmark summary saved → benchmark_summary.csv")
print(df.to_string())
print()

# ── Colour palette (one per model) ───────────────────────────────────────────

PALETTE = plt.cm.tab10.colors
model_names = df["Model"].tolist()
colors = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(model_names)}

# ═════════════════════════════════════════════════════════════════════════════
# Plot 1 — Grouped bar chart: Accuracy / F1-W / Balanced Accuracy
# ═════════════════════════════════════════════════════════════════════════════

metrics_to_plot = ["Accuracy (%)", "F1 Weighted", "Balanced Accuracy"]
metric_labels   = ["Accuracy (%)", "F1 Weighted", "Bal. Accuracy"]

x     = np.arange(len(model_names))
width = 0.22
fig, ax = plt.subplots(figsize=(14, 6))

for i, (col, label) in enumerate(zip(metrics_to_plot, metric_labels)):
    vals = df[col].values
    # scale Accuracy to 0-1 for comparable y-axis
    if col == "Accuracy (%)":
        vals = vals / 100
    bars = ax.bar(x + i * width, vals, width, label=label,
                  color=[PALETTE[i]] * len(vals), edgecolor='white', linewidth=0.6)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.3f}", ha='center', va='bottom', fontsize=7)

ax.set_xticks(x + width)
ax.set_xticklabels(model_names, rotation=20, ha='right', fontsize=10)
ax.set_ylim(0, 1.12)
ax.set_ylabel("Score (0–1 scale)")
ax.set_title("Model Benchmark — Core Metrics", fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.grid(axis='y', alpha=0.35)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/benchmark_metrics_bar.png", dpi=150)
plt.close()
print("Saved → benchmark_metrics_bar.png")

# ═════════════════════════════════════════════════════════════════════════════
# Plot 2 — Inference latency (log scale horizontal bar)
# ═════════════════════════════════════════════════════════════════════════════

lat_df = df.sort_values("Latency/sample (ms)", ascending=True)
fig, ax = plt.subplots(figsize=(9, 5))

bars = ax.barh(lat_df["Model"], lat_df["Latency/sample (ms)"],
               color=[colors[m] for m in lat_df["Model"]],
               edgecolor='white', linewidth=0.6)
for bar, v in zip(bars, lat_df["Latency/sample (ms)"]):
    ax.text(v * 1.05, bar.get_y() + bar.get_height() / 2,
            f"{v:.4f} ms", va='center', fontsize=9)

ax.set_xscale('log')
ax.set_xlabel("Inference latency per sample (ms, log scale)")
ax.set_title("Inference Speed Comparison", fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.35)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/benchmark_inference_time.png", dpi=150)
plt.close()
print("Saved → benchmark_inference_time.png")

# ═════════════════════════════════════════════════════════════════════════════
# Plot 3 — Radar / Spider chart
# ═════════════════════════════════════════════════════════════════════════════

radar_cols  = ["F1 (fresh)", "F1 (moderate)", "F1 (spoiled)", "F1 Weighted", "Balanced Accuracy"]
radar_label = ["F1\nfresh", "F1\nmoderate", "F1\nspoiled", "F1\nweighted", "Bal.\nAccuracy"]
N           = len(radar_cols)
angles      = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles     += angles[:1]   # close polygon

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

for i, row in df.iterrows():
    values  = [row[c] for c in radar_cols]
    values += values[:1]
    m       = row["Model"]
    ax.plot(angles, values, linewidth=2, label=m, color=colors[m])
    ax.fill(angles, values, alpha=0.07, color=colors[m])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_label, fontsize=10)
ax.set_ylim(0, 1)
ax.set_yticks([0.25, 0.50, 0.75, 1.00])
ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8)
ax.set_title("Model Radar — Per-class & Overall Metrics",
             fontsize=13, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/benchmark_radar.png", dpi=150)
plt.close()
print("Saved → benchmark_radar.png")

# ═════════════════════════════════════════════════════════════════════════════
# Plot 4 — Confusion matrices (grid)
# ═════════════════════════════════════════════════════════════════════════════

n_models = len(all_preds)
ncols    = 4
nrows    = int(np.ceil(n_models / ncols))

fig, axes = plt.subplots(nrows, ncols,
                          figsize=(ncols * 4, nrows * 3.8))
axes = axes.flatten()

for idx, (name, preds) in enumerate(all_preds.items()):
    cm   = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(ax=axes[idx], colorbar=False, cmap="Blues")
    f1w  = df.loc[df["Model"] == name, "F1 Weighted"].values[0]
    axes[idx].set_title(f"{name}\n(F1-W: {f1w:.4f})", fontsize=10, fontweight='bold')
    axes[idx].tick_params(axis='both', labelsize=8)

# hide unused axes
for j in range(idx + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Confusion Matrices — All Models", fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/benchmark_confusion_matrices.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved → benchmark_confusion_matrices.png")

# ═════════════════════════════════════════════════════════════════════════════
# Print full classification reports
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 60)
print("FULL CLASSIFICATION REPORTS")
print("═" * 60)
for name, preds in all_preds.items():
    print(f"\n── {name} ──")
    print(classification_report(y_test, preds, target_names=CLASS_NAMES))

print(f"\nAll benchmark outputs saved to: {OUT_DIR}/")