import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import joblib
import os
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from scipy.cluster.hierarchy import dendrogram, linkage

os.makedirs('./models/clustering', exist_ok=True)
os.makedirs('./results/clustering/hierarchical', exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────
X         = np.load('./data/preprocessed/X_cluster.npy')
labels_df = pd.read_csv('./data/preprocessed/labels.csv')
y_true    = labels_df['label'].values
scaler    = joblib.load('./data/artifacts/scaler_cluster.pkl')
cap_raw   = scaler.inverse_transform(X).ravel()

# ── Step 1: sweep linkage methods to find the best one ──────────────────────
# Hierarchical clustering does NOT determine k automatically — you supply it.
# However, the dendrogram visually suggests where to "cut" to get natural groups.
# We also sweep linkage strategies; each measures inter-cluster distance differently:
#
#   ward     → minimises total within-cluster variance (best for compact, equal clusters)
#   complete → distance between the two FARTHEST points across clusters (avoids chaining)
#   average  → average distance between all cross-cluster point pairs (balanced)
#   single   → distance between the two CLOSEST points (prone to "chaining")

linkage_methods = ['ward', 'complete', 'average', 'single']
n_clusters      = len(np.unique(y_true))    # use ground-truth count as the cut point

print(f'Target n_clusters: {n_clusters}')
print('Sweeping linkage methods ...\n')

best_sil, best_method, best_labels, best_model = -1, None, None, None

for method in linkage_methods:
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=method)
    lbl = agg.fit_predict(X)
    sil = silhouette_score(X, lbl)
    print(f'  linkage={method:<8}  silhouette={sil:.4f}')
    if sil > best_sil:
        best_sil, best_method, best_labels, best_model = sil, method, lbl, agg

print(f'\nBest linkage: {best_method}  (silhouette={best_sil:.4f})')

# ── Step 2: evaluate ─────────────────────────────────────────────────────────
hier_labels = best_labels
sil = silhouette_score(X, hier_labels)
db  = davies_bouldin_score(X, hier_labels)
ari = adjusted_rand_score(y_true, hier_labels)

print(f'\n── Hierarchical Results (linkage={best_method}) ──────────────')
print(f'  Clusters found    : {n_clusters}  (cut at ground-truth count)')
print(f'  Silhouette Score  : {sil:.4f}  (higher → better separated, max 1)')
print(f'  Davies-Bouldin    : {db:.4f}   (lower  → more compact clusters, min 0)')
print(f'  Adj. Rand Index   : {ari:.4f}  (agreement with ground truth, max 1)')

cluster_counts = pd.Series(hier_labels).value_counts().sort_index()
print(f'\n  Samples per cluster:\n{cluster_counts.to_string()}')

# ── Step 3: save artefacts ───────────────────────────────────────────────────
joblib.dump(best_model, './models/clustering/hierarchical_model.pkl')

pd.DataFrame([{
    'algorithm': 'Hierarchical', 'linkage': best_method,
    'n_clusters_found': n_clusters,
    'silhouette': round(sil, 4), 'davies_bouldin': round(db, 4), 'ari': round(ari, 4)
}]).to_csv('./results/clustering/hierarchical/hierarchical_metrics.csv', index=False)

np.save('./data/artifacts/hierarchical_labels.npy', hier_labels)

# ── Step 4: plots ─────────────────────────────────────────────────────────────
# Plot A — Dendrogram
# The dendrogram is the primary tool for visually choosing the number of clusters
# in hierarchical clustering. Each merge is shown as a horizontal bar; the HEIGHT
# of the bar = distance between the two groups being merged.
# → Long vertical gaps before the next merge = natural cluster boundaries.
# → The red dashed line marks the cut for `n_clusters` groups.

MAX_DENDRO = 500
sample_idx = np.random.choice(len(X), size=min(MAX_DENDRO, len(X)), replace=False)
X_sample   = X[sample_idx]

Z = linkage(X_sample, method=best_method)

# Find cut height: the merge that produces n_clusters groups
cut_height = Z[-(n_clusters - 1), 2]

fig, ax = plt.subplots(figsize=(14, 5))
dendrogram(Z, ax=ax, no_labels=True,
           color_threshold=cut_height,
           above_threshold_color='grey',
           truncate_mode='lastp', p=40)
ax.axhline(y=cut_height, color='red', linestyle='--',
           label=f'Cut for {n_clusters} clusters  (height={cut_height:.3f})')
ax.set(title=f'Hierarchical Dendrogram  (linkage={best_method},  sample n={len(X_sample)})',
       xlabel='Samples (collapsed leaves)', ylabel='Merge distance')
ax.legend()
plt.tight_layout()
plt.savefig('./results/clustering/hierarchical/hierarchical_dendrogram.png', dpi=150)
plt.close()
print('Saved → hierarchical_dendrogram.png')

# Plot B — Cluster assignments
sort_idx = np.argsort(cap_raw)
colors   = cm.tab10.colors

fig, ax = plt.subplots(figsize=(14, 4))
for c in range(n_clusters):
    mask = hier_labels[sort_idx] == c
    ax.scatter(np.where(mask)[0], cap_raw[sort_idx][mask],
               s=5, color=colors[c % 10], label=f'Cluster {c}', alpha=0.7)

ax.set(title=f'Hierarchical Cluster Assignments  (linkage={best_method})',
       xlabel='Sample index (sorted by capacitance)', ylabel='Capacitance (pF)')
ax.legend(markerscale=4)
plt.tight_layout()
plt.savefig('./results/clustering/hierarchical/hierarchical_assignments.png', dpi=150)
plt.close()
print('Saved → hierarchical_assignments.png')

print('\nHierarchical clustering done.')