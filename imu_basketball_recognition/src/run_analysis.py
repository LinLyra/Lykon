"""
run_analysis.py — Phase 5-7: Feature selection, separability, cross-subject, classifier.
"""
import sys
sys.path.insert(0, '/Users/owen/Desktop/lykon-motion-lab-v2/imu_basketball_recognition/src')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (confusion_matrix, classification_report,
                             f1_score, accuracy_score)
from scipy.spatial.distance import cdist
from scipy.stats import skew, kurtosis
from config import N_FEATURES_SELECT, CORR_THRESHOLD, RANDOM_STATE, RF_N_ESTIMATORS

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading data...")
feat_df = pd.read_parquet('/Users/owen/Desktop/lykon-motion-lab-v2/imu_basketball_recognition/data/features/features.parquet')
meta = pd.read_parquet('/Users/owen/Desktop/lykon-motion-lab-v2/imu_basketball_recognition/data/windows/meta.parquet')
lib = np.load('/Users/owen/Desktop/lykon-motion-lab-v2/imu_basketball_recognition/data/windows/windows.npz')
y = lib['y']

X = feat_df.values
feature_names = feat_df.columns.tolist()
print(f"Features: {X.shape}, Labels: {len(y)}, Meta: {len(meta)}")

# Ensure alignment
assert len(X) == len(y) == len(meta), "Mismatch!"
meta = meta.reset_index(drop=True)

# ---------------------------------------------------------------------------
# Phase 5: Feature Selection
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("PHASE 5: Feature Selection")
print("="*60)

# Exclude null from feature selection (use only action classes)
action_mask = y != 'null'
X_action = X[action_mask]
y_action = y[action_mask]
meta_action = meta[action_mask].reset_index(drop=True)

# 1. ANOVA F-value
f_scores, f_pvals = f_classif(X_action, y_action)
f_rank = np.argsort(-f_scores)

# 2. Random Forest importance
rf_sel = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
rf_sel.fit(X_action, y_action)
rf_imp = rf_sel.feature_importances_
rf_rank = np.argsort(-rf_imp)

# 3. Combine rankings (average rank)
combined_rank = {}
for i in range(len(feature_names)):
    combined_rank[i] = (np.where(f_rank == i)[0][0] + np.where(rf_rank == i)[0][0]) / 2

sorted_idx = sorted(combined_rank, key=combined_rank.get)

# 4. Sequential selection with redundancy removal
selected = []
selected_names = []
for idx in sorted_idx:
    if len(selected) >= N_FEATURES_SELECT:
        break
    if len(selected) == 0:
        selected.append(idx)
        selected_names.append(feature_names[idx])
        continue
    # Check correlation with already selected
    corr_max = np.max(np.abs(np.corrcoef(X[:, idx], X[:, selected].T)[0, 1:]))
    if corr_max < CORR_THRESHOLD:
        selected.append(idx)
        selected_names.append(feature_names[idx])

print(f"Selected {len(selected)} features (target was {N_FEATURES_SELECT})")
print(f"Top 10: {selected_names[:10]}")

X_sel = X[:, selected]
X_action_sel = X_action[:, selected]

# Save selected feature names
with open('/Users/owen/Desktop/lykon-motion-lab-v2/imu_basketball_recognition/data/features/selected_features.txt', 'w') as f:
    for name in selected_names:
        f.write(name + '\n')

# ---------------------------------------------------------------------------
# Phase 6A: Class Separability Analysis
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("PHASE 6A: Class Separability")
print("="*60)

classes = sorted(np.unique(y_action))
n_classes = len(classes)

# Compute class means and covariances (Ledoit-Wolf shrinkage)
from sklearn.covariance import LedoitWolf

class_means = {}
class_covs = {}
class_counts = {}
for c in classes:
    mask = y_action == c
    class_means[c] = np.mean(X_action_sel[mask], axis=0)
    class_counts[c] = np.sum(mask)
    if np.sum(mask) > 10:
        lw = LedoitWolf()
        lw.fit(X_action_sel[mask])
        class_covs[c] = lw.covariance_
    else:
        class_covs[c] = np.eye(X_action_sel.shape[1]) * 1e-3

# Bhattacharyya distance matrix
bhatt_dist = np.zeros((n_classes, n_classes))
maha_dist = np.zeros((n_classes, n_classes))
for i, ci in enumerate(classes):
    for j, cj in enumerate(classes):
        if i == j:
            continue
        mu_i, mu_j = class_means[ci], class_means[cj]
        cov_i, cov_j = class_covs[ci], class_covs[cj]
        # Symmetrized covariance
        cov_avg = (cov_i + cov_j) / 2
        try:
            inv_avg = np.linalg.inv(cov_avg)
        except:
            inv_avg = np.linalg.pinv(cov_avg)
        diff = mu_i - mu_j
        maha = np.sqrt(diff @ inv_avg @ diff)
        maha_dist[i, j] = maha
        # Bhattacharyya approximation
        d = len(mu_i)
        sign, logdet_avg = np.linalg.slogdet(cov_avg)
        sign_i, logdet_i = np.linalg.slogdet(cov_i)
        sign_j, logdet_j = np.linalg.slogdet(cov_j)
        if sign > 0 and sign_i > 0 and sign_j > 0:
            bhatt = 0.125 * diff @ inv_avg @ diff + 0.5 * (logdet_avg - 0.5*(logdet_i + logdet_j))
        else:
            bhatt = 0.125 * diff @ inv_avg @ diff
        bhatt_dist[i, j] = bhatt

# Plot distance matrices
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, dist, title in [(axes[0], bhatt_dist, 'Bhattacharyya Distance'),
                        (axes[1], maha_dist, 'Mahalanobis Distance')]:
    sns.heatmap(dist, annot=True, fmt='.2f', xticklabels=classes, yticklabels=classes,
                cmap='YlOrRd', ax=ax, square=True)
    ax.set_title(title)
plt.tight_layout()
plt.savefig('/Users/owen/Desktop/lykon-motion-lab-v2/imu_basketball_recognition/reports/separability_distances.png', dpi=150)
print("Saved distance matrices plot")

# Cohen's d per class pair
print("\nTop discriminating features per class pair:")
cohen_report = []
for i, ci in enumerate(classes):
    for j, cj in enumerate(classes):
        if i >= j:
            continue
        Xi = X_action_sel[y_action == ci]
        Xj = X_action_sel[y_action == cj]
        d_vals = []
        for f in range(X_action_sel.shape[1]):
            mi, mj = np.mean(Xi[:, f]), np.mean(Xj[:, f])
            si, sj = np.std(Xi[:, f]), np.std(Xj[:, f])
            pooled = np.sqrt((si**2 + sj**2) / 2 + 1e-12)
            d = abs(mi - mj) / pooled
            d_vals.append((selected_names[f], d))
        d_vals.sort(key=lambda x: -x[1])
        print(f"\n  {ci} vs {cj}: top 5 features")
        for name, d in d_vals[:5]:
            print(f"    {name}: d={d:.2f}")
        cohen_report.append({
            'pair': f"{ci}_vs_{cj}",
            'top_feature': d_vals[0][0],
            'top_d': d_vals[0][1],
        })

# Silhouette coefficient
from sklearn.metrics import silhouette_samples, silhouette_score
sil_score = silhouette_score(X_action_sel, y_action)
sil_per_class = {}
for c in classes:
    mask = y_action == c
    # Use silhouette_samples and average for this class
    sil_samples = silhouette_samples(X_action_sel, y_action)
    sil_per_class[c] = float(np.mean(sil_samples[mask]))

print(f"\nOverall silhouette score: {sil_score:.3f}")
print("Per-class silhouette:")
for c, s in sil_per_class.items():
    print(f"  {c}: {s:.3f}")

# ---------------------------------------------------------------------------
# Phase 6B: Cross-Subject Analysis
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("PHASE 6B: Cross-Subject Analysis")
print("="*60)

# Per-class per-feature Cohen's d between subjects
subjects = ['owen', 'ryan']
cross_sub_report = []

for c in classes:
    mask_c = y_action == c
    meta_c = meta_action[mask_c]
    X_c = X_action_sel[mask_c]

    owen_mask = meta_c['subject'].values == 'owen'
    ryan_mask = meta_c['subject'].values == 'ryan'

    if owen_mask.sum() < 3 or ryan_mask.sum() < 3:
        print(f"  {c}: insufficient samples for cross-subject comparison")
        continue

    Xo, Xr = X_c[owen_mask], X_c[ryan_mask]
    d_vals = []
    for f in range(X_action_sel.shape[1]):
        mo, mr = np.mean(Xo[:, f]), np.mean(Xr[:, f])
        so, sr = np.std(Xo[:, f]), np.std(Xr[:, f])
        pooled = np.sqrt((so**2 + sr**2) / 2 + 1e-12)
        d = abs(mo - mr) / pooled
        d_vals.append((selected_names[f], d))
    d_vals.sort(key=lambda x: x[1])

    print(f"\n  {c}: cross-subject stability")
    print(f"    Most stable (d<0.5): {[n for n,d in d_vals[:3] if d<0.5]}")
    print(f"    Least stable (d>1.0): {[n for n,d in d_vals[-3:] if d>1.0]}")
    cross_sub_report.append({
        'class': c,
        'mean_d': np.mean([d for _, d in d_vals]),
        'median_d': np.median([d for _, d in d_vals]),
    })

# ---------------------------------------------------------------------------
# Phase 7: Classification
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("PHASE 7: Classification")
print("="*60)

# Include null class for full classification
X_full_sel = X[:, selected]
y_full = y
meta_full = meta

# A. Within-subject CV (by rep_id for discrete, by recording for continuous)
def eval_split(X_tr, y_tr, X_te, y_te, name):
    results = {}
    for clf_name, clf in [
        ('RF', RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1)),
        ('SVM', Pipeline([('sc', StandardScaler()), ('svc', SVC(kernel='rbf', class_weight='balanced', random_state=RANDOM_STATE))])),
        ('kNN', Pipeline([('sc', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=5))])),
    ]:
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        acc = accuracy_score(y_te, y_pred)
        f1_macro = f1_score(y_te, y_pred, average='macro', zero_division=0)
        results[clf_name] = {'acc': acc, 'f1_macro': f1_macro, 'pred': y_pred}
        print(f"  {name} / {clf_name}: acc={acc:.3f}, macro_f1={f1_macro:.3f}")
    return results

# 1. Within-subject (owen only)
owen_mask = meta_full['subject'] == 'owen'
Xo, yo, mo = X_full_sel[owen_mask], y_full[owen_mask], meta_full[owen_mask]
# Simple train/test split by rep_id parity for owen
if 'rep_id' in mo.columns and mo['rep_id'].nunique() > 1:
    train_mask_o = mo['rep_id'] % 2 == 0
    test_mask_o = mo['rep_id'] % 2 == 1
else:
    # Fall back: first 70% vs last 30%
    split = int(0.7 * len(Xo))
    train_mask_o = np.zeros(len(Xo), dtype=bool)
    train_mask_o[:split] = True
    test_mask_o = ~train_mask_o

print("\n1. Within-subject (Owen):")
res_owen = eval_split(Xo[train_mask_o], yo[train_mask_o], Xo[test_mask_o], yo[test_mask_o], "owen")

# 2. Within-subject (ryan only)
ryan_mask = meta_full['subject'] == 'ryan'
# Exclude 'random' recording from training for now
ryan_rec_mask = meta_full['recording'] != '26071015随机做'
Xr, yr, mr = X_full_sel[ryan_mask & ryan_rec_mask], y_full[ryan_mask & ryan_rec_mask], meta_full[ryan_mask & ryan_rec_mask]
if 'rep_id' in mr.columns and mr['rep_id'].nunique() > 1:
    train_mask_r = mr['rep_id'] % 2 == 0
    test_mask_r = mr['rep_id'] % 2 == 1
else:
    split = int(0.7 * len(Xr))
    train_mask_r = np.zeros(len(Xr), dtype=bool)
    train_mask_r[:split] = True
    test_mask_r = ~train_mask_r

print("\n2. Within-subject (Ryan):")
res_ryan = eval_split(Xr[train_mask_r], yr[train_mask_r], Xr[test_mask_r], yr[test_mask_r], "ryan")

# 3. Cross-subject (owen train → ryan test)
print("\n3. Cross-subject (train=Owen, test=Ryan):")
res_cross_or = eval_split(Xo, yo, Xr, yr, "owen→ryan")

# 4. Cross-subject (ryan train → owen test)
print("\n4. Cross-subject (train=Ryan, test=Owen):")
res_cross_ro = eval_split(Xr, yr, Xo, yo, "ryan→owen")

# 5. Merged (both subjects, simple 70/30 split by recording)
X_m = X_full_sel[ryan_rec_mask]
y_m = y_full[ryan_rec_mask]
m_m = meta_full[ryan_rec_mask]
# Split by recording
recs = m_m['recording'].unique()
np.random.seed(RANDOM_STATE)
np.random.shuffle(recs)
split_rec = int(0.7 * len(recs))
train_recs = set(recs[:split_rec])
test_recs = set(recs[split_rec:])
train_mask_m = m_m['recording'].isin(train_recs).values
test_mask_m = m_m['recording'].isin(test_recs).values

print("\n5. Merged split (by recording):")
res_merged = eval_split(X_m[train_mask_m], y_m[train_mask_m], X_m[test_mask_m], y_m[test_mask_m], "merged")

# Confusion matrix for best model (RF on merged)
cm = confusion_matrix(y_m[test_mask_m], res_merged['RF']['pred'], labels=sorted(np.unique(y_m)))
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=sorted(np.unique(y_m)),
            yticklabels=sorted(np.unique(y_m)), cmap='Blues', ax=ax)
ax.set_title('Confusion Matrix - RF (Merged Test)')
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
plt.tight_layout()
plt.savefig('/Users/owen/Desktop/lykon-motion-lab-v2/imu_basketball_recognition/reports/confusion_matrix.png', dpi=150)

# Save summary
summary = {
    'n_windows_total': len(y),
    'n_windows_action': len(y_action),
    'n_features_selected': len(selected),
    'silhouette_score': sil_score,
    'classes': classes,
    'owen_rf_acc': res_owen['RF']['acc'],
    'ryan_rf_acc': res_ryan['RF']['acc'],
    'cross_or_rf_acc': res_cross_or['RF']['acc'],
    'cross_ro_rf_acc': res_cross_ro['RF']['acc'],
    'merged_rf_acc': res_merged['RF']['acc'],
    'merged_rf_f1': res_merged['RF']['f1_macro'],
}

import json
with open('/Users/owen/Desktop/lykon-motion-lab-v2/imu_basketball_recognition/reports/summary.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print("\n" + "="*60)
print("Analysis complete. Summary:")
print(f"  Selected features: {len(selected)}")
print(f"  Silhouette score: {sil_score:.3f}")
print(f"  Owen RF acc: {res_owen['RF']['acc']:.3f}")
print(f"  Ryan RF acc: {res_ryan['RF']['acc']:.3f}")
print(f"  Owen→Ryan RF acc: {res_cross_or['RF']['acc']:.3f}")
print(f"  Ryan→Owen RF acc: {res_cross_ro['RF']['acc']:.3f}")
print(f"  Merged RF acc: {res_merged['RF']['acc']:.3f}, macro_f1: {res_merged['RF']['f1_macro']:.3f}")
print("="*60)
