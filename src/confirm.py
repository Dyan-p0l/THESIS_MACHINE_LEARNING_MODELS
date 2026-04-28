"""
pca_diagnostic.py
─────────────────
Standalone diagnostic script to visualize whether your 3 freshness groups
(Fresh / Moderate / Spoiled) are separable in feature space.

Run this AFTER your preprocessing is done. It only needs:
  - ./data/preprocessed/X_cluster.npy
  - ./data/preprocessed/labels.csv
  - ./data/artifacts/scaler_cluster.pkl

Usage:
    python pca_diagnostic.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

os.makedirs('./results/diagnostics', exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────
print('Loading data...')
X         = np.load('./data/preprocessed/X_cluster.npy')
labels_df = pd.read_csv('./data/preprocessed/labels.csv')
y_true    = labels_df['label'].values
scaler    = joblib.load('./data/artifacts/scaler_cluster.pkl')

# Decode numeric labels to names if needed
# Adjust this mapping to match YOUR label encoding
LABEL_NAMES = {0: 'Fresh', 1: 'Moderate', 2: 'Spoiled'}
COLORS      = {0: '#2ecc71', 1: '#f39c12', 2: '#e74c3c'}  # green / orange / red

unique_labels = np.unique(y_true)
print(f'Unique label values in y_true: {unique_labels}')
print(f'Sample counts per label:')
for lbl in unique_labels:
    name = LABEL_NAMES.get(int(lbl), str(lbl))
    print(f'  {name} ({lbl}): {(y_true == lbl).sum()} samples')

# ── Plot 1: Feature means per class ─────────────────────────────────────────
# This tells you immediately if the 3 groups differ in the raw features.
# If all 3 lines overlap, clustering cannot separate them no matter the algorithm.
print('\nPlotting feature means per class...')
df_X = pd.DataFrame(X, columns=[f'f{i}' for i in range(X.shape[1])])
df_X['label'] = y_true

fig, ax = plt.subplots(figsize=(10, 4))
for lbl in unique_labels:
    name  = LABEL_NAMES.get(int(lbl), str(lbl))
    color = COLORS.get(int(lbl), 'gray')
    means = df_X[df_X['label'] == lbl].drop(columns='label').mean()
    ax.plot(means.values, label=name, color=color, linewidth=2, marker='o', markersize=4)

ax.set(title='Feature means per class  (if lines overlap → groups hard to separate)',
       xlabel='Feature index', ylabel='Scaled value')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('./results/diagnostics/feature_means_per_class.png', dpi=150)
plt.close()
print('Saved → results/diagnostics/feature_means_per_class.png')

# ── Data shape report ───────────────────────────────────────────────────────
print(f'\nX shape: {X.shape}  →  {X.shape[0]} samples, {X.shape[1]} feature(s)')
if X.shape[1] == 1:
    print('  WARNING: Only 1 feature found.')
    print('  PCA/t-SNE need ≥2 features — skipping those plots.')
    print('  Instead, a 1D strip plot will be generated.')

# ── Plot 2: PCA — 2D projection ──────────────────────────────────────────────
# PCA finds the 2 directions of maximum variance.
# If the 3 clusters appear visually distinct here, FCM CAN find them — the
# problem is just the sweep criterion (silhouette prefers c=2).
# If they overlap completely, no clustering algorithm will split them cleanly.
if X.shape[1] >= 2:
    print('Running PCA...')
    pca   = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    var   = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(8, 6))
    for lbl in unique_labels:
        name  = LABEL_NAMES.get(int(lbl), str(lbl))
        color = COLORS.get(int(lbl), 'gray')
        mask  = y_true == lbl
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   label=name, color=color, alpha=0.5, s=15, edgecolors='none')

    ax.set(title=f'PCA projection  (PC1={var[0]*100:.1f}%  PC2={var[1]*100:.1f}% variance explained)',
           xlabel='PC1', ylabel='PC2')
    ax.legend(markerscale=3)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig('./results/diagnostics/pca_projection.png', dpi=150)
    plt.close()
    print('Saved → results/diagnostics/pca_projection.png')
else:
    # ── 1-feature fallback: strip plot + KDE ────────────────────────────────
    # With only 1 feature, the only question is: do the 3 groups occupy
    # different ranges along that single axis?
    print('Single feature detected — generating strip + KDE plot instead of PCA...')
    from scipy.stats import gaussian_kde

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7),
                                    gridspec_kw={'height_ratios': [1, 2]})
    cap_raw = scaler.inverse_transform(X).ravel()

    # Strip plot (jittered)
    rng = np.random.default_rng(42)
    for lbl in unique_labels:
        name  = LABEL_NAMES.get(int(lbl), str(lbl))
        color = COLORS.get(int(lbl), 'gray')
        mask  = y_true == lbl
        jitter = rng.uniform(-0.15, 0.15, mask.sum())
        ax1.scatter(cap_raw[mask], np.full(mask.sum(), int(lbl)) + jitter,
                    color=color, alpha=0.4, s=10, edgecolors='none', label=name)

    ax1.set_yticks(list(unique_labels))
    ax1.set_yticklabels([LABEL_NAMES.get(int(l), str(l)) for l in unique_labels])
    ax1.set(title='Strip plot — does each class occupy a distinct range?',
            xlabel='Capacitance (original scale)')
    ax1.grid(axis='x', alpha=0.3)

    # KDE plot
    x_grid = np.linspace(cap_raw.min(), cap_raw.max(), 500)
    for lbl in unique_labels:
        name  = LABEL_NAMES.get(int(lbl), str(lbl))
        color = COLORS.get(int(lbl), 'gray')
        vals  = cap_raw[y_true == lbl]
        if len(vals) > 1:
            kde = gaussian_kde(vals, bw_method='scott')
            ax2.fill_between(x_grid, kde(x_grid), alpha=0.25, color=color)
            ax2.plot(x_grid, kde(x_grid), color=color, linewidth=2, label=name)

    ax2.set(title='KDE — overlap between classes shows why FCM finds 2 clusters',
            xlabel='Capacitance (original scale)', ylabel='Density')
    ax2.legend()
    ax2.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig('./results/diagnostics/single_feature_distribution.png', dpi=150)
    plt.close()
    print('Saved → results/diagnostics/single_feature_distribution.png')
    print()
    print('  READ THIS PLOT:')
    print('  - If Fresh/Moderate/Spoiled KDE peaks are clearly separated → force c=3')
    print('  - If Fresh and Moderate peaks heavily overlap → those 2 groups look like 1 cluster to FCM')
    print('  - If all 3 overlap into one hump → the sensor alone cannot distinguish freshness levels')

# ── Plot 3: t-SNE — non-linear 2D projection ────────────────────────────────
# t-SNE is better than PCA at revealing cluster structure when clusters are
# non-linearly separable. If clusters appear here but not in PCA, your data
# is non-linearly structured — consider kernel-based features.
# Note: t-SNE is slow on large datasets; reduce perplexity if it hangs.
if X.shape[1] < 2:
    print('Skipping t-SNE — need ≥2 features.')
else:
    print('Running t-SNE (this may take a moment)...')
    perplexity = min(30, len(X) // 4)
    tsne       = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
    X_tsne     = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(8, 6))
    for lbl in unique_labels:
        name  = LABEL_NAMES.get(int(lbl), str(lbl))
        color = COLORS.get(int(lbl), 'gray')
        mask  = y_true == lbl
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                   label=name, color=color, alpha=0.5, s=15, edgecolors='none')

    ax.set(title=f't-SNE projection  (perplexity={perplexity})',
           xlabel='t-SNE 1', ylabel='t-SNE 2')
    ax.legend(markerscale=3)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig('./results/diagnostics/tsne_projection.png', dpi=150)
    plt.close()
    print('Saved → results/diagnostics/tsne_projection.png')

# ── Plot 4: Pairwise feature scatter (if few features) ───────────────────────
# Shows raw feature relationships — useful if X has only 2–5 features.
n_features = X.shape[1]
if 2 <= n_features <= 5:
    print(f'Plotting pairwise scatter ({n_features} features)...')
    fig, axes = plt.subplots(n_features, n_features,
                              figsize=(3 * n_features, 3 * n_features))
    for i in range(n_features):
        for j in range(n_features):
            ax = axes[i, j]
            if i == j:
                for lbl in unique_labels:
                    color = COLORS.get(int(lbl), 'gray')
                    ax.hist(X[y_true == lbl, i], bins=20, alpha=0.5, color=color)
            else:
                for lbl in unique_labels:
                    color = COLORS.get(int(lbl), 'gray')
                    mask  = y_true == lbl
                    ax.scatter(X[mask, j], X[mask, i],
                               color=color, alpha=0.4, s=5, edgecolors='none')
            ax.set_xlabel(f'f{j}', fontsize=8)
            ax.set_ylabel(f'f{i}', fontsize=8)
            ax.tick_params(labelsize=6)

    patches = [mpatches.Patch(color=COLORS[int(l)], label=LABEL_NAMES.get(int(l), str(l)))
               for l in unique_labels]
    fig.legend(handles=patches, loc='upper right', fontsize=10)
    fig.suptitle('Pairwise feature scatter by class', fontsize=13)
    plt.tight_layout()
    plt.savefig('./results/diagnostics/pairwise_scatter.png', dpi=120)
    plt.close()
    print('Saved → results/diagnostics/pairwise_scatter.png')
else:
    print(f'Skipping pairwise scatter (need 2–5 features, got {n_features})')

# ── Summary ──────────────────────────────────────────────────────────────────
print('\n── Interpretation guide ───────────────────────────────────────────')
print('  feature_means_per_class.png')
print('    All 3 lines overlap   → features do NOT distinguish groups; add better features')
print('    Lines separate well   → features are informative; clustering should work')
print()
print('  pca_projection.png')
print('    3 distinct blobs      → FCM CAN find 3 clusters; fix: force c=3 or use FPC sweep')
print('    2 blobs only          → Fresh+Moderate overlap; consider merging or adding features')
print('    1 blob                → No structure; clustering will not work')
print()
print('  tsne_projection.png')
print('    3 blobs (not in PCA)  → Non-linear structure; add polynomial/interaction features')
print('    Still 2 or 1 blob     → Data genuinely has 2 groups; "Moderate" is a gradient, not a cluster')
print()
print('  Next step if you see 3 blobs in PCA/t-SNE:')
print('    → Force c=3 in your FCM script (bypass the sweep)')
print('    → Use FPC instead of silhouette as the sweep criterion')