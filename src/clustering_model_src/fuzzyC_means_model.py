import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import joblib
import os
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
import skfuzzy as fuzz

os.makedirs('./models/clustering', exist_ok=True)
os.makedirs('./results/clustering/fuzzycmeans', exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────
X         = np.load('./data/preprocessed/X_cluster.npy')
labels_df = pd.read_csv('./data/preprocessed/labels.csv')
y_true    = labels_df['label'].values
scaler    = joblib.load('./data/artifacts/scaler_cluster.pkl')
cap_raw   = scaler.inverse_transform(X).ravel()

# skfuzzy expects shape (features × samples)
X_T = X.T

# ── Step 1: sweep c (number of clusters) ─────────────────────────────────────
# FCM does NOT automatically discover c — you choose it, just like K-Means.
# We sweep c = 2..9 and score each with silhouette to find the best value.
# The fuzziness exponent m=2.0 is the standard default for the sweep.

c_range    = range(2, min(10, len(X)))
sil_scores = []

print('Sweeping number of clusters c ...')
for c in c_range:
    centers, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        X_T, c=c, m=2.0, error=1e-6, maxiter=1000, init=None
    )
    labels_c = np.argmax(u, axis=0)
    sil      = silhouette_score(X, labels_c)
    sil_scores.append(sil)
    print(f'  c={c}  silhouette={sil:.4f}')

best_c = list(c_range)[int(np.argmax(sil_scores))]
print(f'\nBest c by silhouette: {best_c}')
print(f'(Your dataset has {len(np.unique(y_true))} ground-truth categories)')

# ── Step 2: sweep fuzziness exponent m at best_c ─────────────────────────────
# m controls how "fuzzy" the boundaries are:
#   m → 1.0  : near-crisp (hard) assignment — each sample belongs to exactly one cluster
#   m = 2.0  : standard default, balanced softness
#   m → ∞    : all memberships collapse to 1/c (uniform, meaningless)
# Typical useful range: 1.5 – 3.0

m_values = [1.5, 2.0, 2.5, 3.0]
best_sil_m, best_m = -1, None
best_labels, best_centers, best_u = None, None, None

print(f'\nSweeping fuzziness exponent m at c={best_c} ...')
for m in m_values:
    centers, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        X_T, c=best_c, m=m, error=1e-6, maxiter=1000, init=None
    )
    labels_m = np.argmax(u, axis=0)
    sil      = silhouette_score(X, labels_m)
    print(f'  m={m}  silhouette={sil:.4f}')
    if sil > best_sil_m:
        best_sil_m = sil
        best_m     = m
        best_labels, best_centers, best_u = labels_m, centers, u

print(f'\nBest m (fuzziness): {best_m}')

# ── Step 3: evaluate ─────────────────────────────────────────────────────────
fcm_labels = best_labels
sil = silhouette_score(X, fcm_labels)
db  = davies_bouldin_score(X, fcm_labels)
ari = adjusted_rand_score(y_true, fcm_labels)

print(f'\n── Fuzzy C-Means Results (c={best_c}, m={best_m}) ───────────')
print(f'  Clusters found    : {best_c}')
print(f'  Fuzziness (m)     : {best_m}')
print(f'  Silhouette Score  : {sil:.4f}  (on hard-assigned labels)')
print(f'  Davies-Bouldin    : {db:.4f}')
print(f'  Adj. Rand Index   : {ari:.4f}  (agreement with ground truth, max 1)')

cluster_counts = pd.Series(fcm_labels).value_counts().sort_index()
print(f'\n  Samples per cluster (hard assignment):\n{cluster_counts.to_string()}')

# Membership entropy: measures how "uncertain" each sample's assignment is
# High entropy → sample sits near a cluster boundary (truly fuzzy)
# Low entropy  → sample clearly belongs to one cluster
entropy = -np.sum(best_u.T * np.log(best_u.T + 1e-10), axis=1)
print(f'\n  Membership entropy (0=certain, higher=fuzzy):')
print(f'    Mean : {entropy.mean():.4f}')
print(f'    Max  : {entropy.max():.4f}')
print(f'    Samples near boundary (entropy > 0.5): {(entropy > 0.5).sum()}')

# ── Step 4: save artefacts ───────────────────────────────────────────────────
# FCM has no sklearn object — save the centers and membership matrix instead.
# To predict new samples: use fuzz.cluster.cmeans_predict(new_X.T, centers, m, ...)
np.save('./models/clustering/fcm_centers.npy', best_centers)
np.save('./models/clustering/fcm_membership.npy', best_u)
joblib.dump({'m': best_m, 'n_clusters': best_c}, './models/clustering/fcm_params.pkl')

pd.DataFrame([{
    'algorithm': 'Fuzzy C-Means', 'n_clusters_found': best_c, 'best_m': best_m,
    'silhouette': round(sil, 4), 'davies_bouldin': round(db, 4), 'ari': round(ari, 4)
}]).to_csv('./results/clustering/fuzzycmeans/fuzzycmeans_metrics.csv', index=False)

np.save('./data/artifacts/fuzzycmeans_labels.npy', fcm_labels)

# ── Step 5: plots ─────────────────────────────────────────────────────────────
sort_idx = np.argsort(cap_raw)
colors   = cm.tab10.colors

# Plot A — Silhouette sweep over c
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(list(c_range), sil_scores, 'bo-', linewidth=2, markersize=7)
ax.axvline(best_c, color='red', linestyle='--', label=f'Best c={best_c}')
ax.set(title='Fuzzy C-Means — Silhouette Score vs Number of Clusters',
       xlabel='Number of Clusters (c)', ylabel='Silhouette Score (↑ better)')
ax.legend()
plt.tight_layout()
plt.savefig('./results/clustering/fuzzycmeans/fuzzycmeans_silhouette_sweep.png', dpi=150)
plt.close()
print('Saved → fuzzycmeans_silhouette_sweep.png')

# Plot B — Hard cluster assignments
fig, ax = plt.subplots(figsize=(14, 4))
for c in range(best_c):
    mask = fcm_labels[sort_idx] == c
    ax.scatter(np.where(mask)[0], cap_raw[sort_idx][mask],
               s=5, color=colors[c % 10], label=f'Cluster {c}', alpha=0.7)

ax.set(title=f'Fuzzy C-Means Assignments  (c={best_c}, m={best_m})  — hard labels via argmax',
       xlabel='Sample index (sorted by capacitance)', ylabel='Capacitance (pF)')
ax.legend(markerscale=4)
plt.tight_layout()
plt.savefig('./results/clustering/fuzzycmeans/fuzzycmeans_assignments.png', dpi=150)
plt.close()
print('Saved → fuzzycmeans_assignments.png')

# Plot C — Soft membership degrees
# This is FCM's unique output: every sample has a degree of membership [0..1] for each cluster.
# All memberships sum to 1 per sample.
# Samples where one cluster dominates (e.g. 0.95) are clearly categorised.
# Samples where values are even (e.g. 0.34 / 0.33 / 0.33) are ambiguous — boundary readings.
fig, ax = plt.subplots(figsize=(14, 4))
for c in range(best_c):
    ax.plot(cap_raw[sort_idx], best_u[c][sort_idx],
            linewidth=1.2, label=f'Cluster {c} membership', alpha=0.8)

ax.set(title=f'Fuzzy C-Means — Soft Membership Degrees  (c={best_c}, m={best_m})',
       xlabel='Capacitance (pF)', ylabel='Membership degree  [0 → 1]')
ax.legend()
plt.tight_layout()
plt.savefig('./results/clustering/fuzzycmeans/fuzzycmeans_membership.png', dpi=150)
plt.close()
print('Saved → fuzzycmeans_membership.png')

print('\nFuzzy C-Means done.')
print('\nTo predict on new data:')
print('  params  = joblib.load("./models/clustering/fcm_params.pkl")')
print('  centers = np.load("./models/clustering/fcm_centers.npy")')
print('  u_new, *_ = fuzz.cluster.cmeans_predict(X_new.T, centers, params["m"], error=1e-6, maxiter=1000)')
print('  labels_new = np.argmax(u_new, axis=0)')