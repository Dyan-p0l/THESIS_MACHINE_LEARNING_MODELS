import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import joblib
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score

os.makedirs('./models/clustering', exist_ok=True)
os.makedirs('./results/clustering/kmeans', exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────
X         = np.load('./data/preprocessed/X_cluster.npy')
labels_df = pd.read_csv('./data/preprocessed/labels.csv')
y_true    = labels_df['label'].values
scaler    = joblib.load('./data/artifacts/scaler_cluster.pkl')
cap_raw   = scaler.inverse_transform(X).ravel()          # back to pF

# ── Step 1: sweep k to find optimal number of clusters ──────────────────────
# K-Means does NOT determine k automatically — you must choose it.
# We sweep k = 2..9 and score each with:
#   • Inertia       → sum of squared distances to cluster center (lower = tighter clusters)
#   • Silhouette    → how well each point fits its own cluster vs. neighbours (higher = better)
# The "elbow" in the inertia curve and the silhouette peak both suggest the best k.

k_range    = range(2, min(10, len(X)))
inertias   = []
sil_scores = []

print('Sweeping k ...')
for k in k_range:
    km  = KMeans(n_clusters=k, n_init=20, random_state=42)
    km.fit(X)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X, km.labels_))
    print(f'  k={k}  inertia={km.inertia_:,.1f}  silhouette={sil_scores[-1]:.4f}')

# Best k by silhouette score
best_k = list(k_range)[int(np.argmax(sil_scores))]
print(f'\nBest k by silhouette: {best_k}')
print(f'(Your dataset has {len(np.unique(y_true))} ground-truth categories)')

# ── Step 2: fit final model ──────────────────────────────────────────────────
kmeans    = KMeans(n_clusters=best_k, n_init=20, random_state=42)
kmeans.fit(X)
km_labels = kmeans.labels_

# ── Step 3: evaluate ─────────────────────────────────────────────────────────
sil = silhouette_score(X, km_labels)
db  = davies_bouldin_score(X, km_labels)
ari = adjusted_rand_score(y_true, km_labels)

print(f'\n── K-Means Results (k={best_k}) ──────────────────')
print(f'  Clusters found    : {best_k}')
print(f'  Silhouette Score  : {sil:.4f}  (higher → better separated, max 1)')
print(f'  Davies-Bouldin    : {db:.4f}   (lower  → more compact clusters, min 0)')
print(f'  Adj. Rand Index   : {ari:.4f}  (agreement with ground truth, max 1)')

cluster_counts = pd.Series(km_labels).value_counts().sort_index()
print(f'\n  Samples per cluster:\n{cluster_counts.to_string()}')

# ── Step 4: save artefacts ───────────────────────────────────────────────────
joblib.dump(kmeans, './models/clustering/kmeans_model.pkl')

pd.DataFrame([{
    'algorithm': 'K-Means', 'n_clusters_found': best_k,
    'silhouette': round(sil, 4), 'davies_bouldin': round(db, 4), 'ari': round(ari, 4)
}]).to_csv('./results/clustering/kmeans/kmeans_metrics.csv', index=False)

np.save('./data/artifacts/kmeans_labels.npy', km_labels)

# ── Step 5: plots ────────────────────────────────────────────────────────────
# Plot A — Elbow + Silhouette sweep
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

axes[0].plot(k_range, inertias, 'bo-', linewidth=2, markersize=7)
axes[0].axvline(best_k, color='red', linestyle='--', label=f'Best k={best_k}')
axes[0].set(title='K-Means — Elbow Curve',
            xlabel='Number of Clusters (k)', ylabel='Inertia (↓ better)')
axes[0].legend()

axes[1].plot(k_range, sil_scores, 'gs-', linewidth=2, markersize=7)
axes[1].axvline(best_k, color='red', linestyle='--', label=f'Best k={best_k}')
axes[1].set(title='K-Means — Silhouette Score vs k',
            xlabel='k', ylabel='Silhouette Score (↑ better)')
axes[1].legend()

plt.suptitle('K-Means: Choosing the Number of Clusters', fontsize=13)
plt.tight_layout()
plt.savefig('./results/clustering/kmeans/kmeans_elbow.png', dpi=150)
plt.close()
print('Saved → kmeans_elbow.png')

# Plot B — Cluster assignments along sorted capacitance axis
sort_idx = np.argsort(cap_raw)
colors   = cm.tab10.colors

fig, ax = plt.subplots(figsize=(14, 4))
for c in range(best_k):
    mask = km_labels[sort_idx] == c
    ax.scatter(np.where(mask)[0], cap_raw[sort_idx][mask],
               s=5, color=colors[c % 10], label=f'Cluster {c}', alpha=0.7)

ax.set(title=f'K-Means Cluster Assignments  (k={best_k})',
       xlabel='Sample index (sorted by capacitance)', ylabel='Capacitance (pF)')
ax.legend(markerscale=4)
plt.tight_layout()
plt.savefig('./results/clustering/kmeans/kmeans_assignments.png', dpi=150)
plt.close()
print('Saved → kmeans_assignments.png')

print('\nK-Means done.')