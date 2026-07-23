"""
run_v31_analysis_fix2.py — v3.1 修复二后分析
"""
import sys
sys.path.insert(0, '/Users/owen/Desktop/lykon-motion-lab-v2/imu_basketball_recognition/src')

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
from config import N_FEATURES_SELECT, CORR_THRESHOLD, RANDOM_STATE, RF_N_ESTIMATORS

# Load fix2 data
print("Loading v3.1 fix2 data...")
feat_df = pd.read_parquet('/Users/owen/Desktop/lykon-motion-lab-v2/imu_basketball_recognition/data/features/features_v31_fix2.parquet')
meta = pd.read_parquet('/Users/owen/Desktop/lykon-motion-lab-v2/imu_basketball_recognition/data/windows/meta_v31_fix2.parquet')
lib = np.load('/Users/owen/Desktop/lykon-motion-lab-v2/imu_basketball_recognition/data/windows/windows_v31_fix2.npz')
y = lib['y']
X = feat_df.values
feature_names = feat_df.columns.tolist()
print(f"X={X.shape}, y={len(y)}")

meta = meta.reset_index(drop=True)
action_mask = y != 'null'
X_action = X[action_mask]
y_action = y[action_mask]
meta_action = meta[action_mask].reset_index(drop=True)

# Feature selection
print("Feature selection...")
f_scores, _ = f_classif(X_action, y_action)
rf_sel = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
rf_sel.fit(X_action, y_action)
rf_imp = rf_sel.feature_importances_
f_rank = np.argsort(-f_scores)
rf_rank = np.argsort(-rf_imp)
combined_rank = {i: (np.where(f_rank==i)[0][0] + np.where(rf_rank==i)[0][0])/2 for i in range(len(feature_names))}
sorted_idx = sorted(combined_rank, key=combined_rank.get)

selected = []
for idx in sorted_idx:
    if len(selected) >= N_FEATURES_SELECT:
        break
    if len(selected) == 0:
        selected.append(idx); continue
    corr_max = np.max(np.abs(np.corrcoef(X[:, idx], X[:, selected].T)[0, 1:]))
    if corr_max < CORR_THRESHOLD:
        selected.append(idx)

X_sel = X[:, selected]
X_action_sel = X_action[:, selected]

# Cohen's d
classes = sorted(np.unique(y_action))
for c1 in classes:
    for c2 in classes:
        if c1 >= c2: continue
        X1 = X_action_sel[y_action == c1]
        X2 = X_action_sel[y_action == c2]
        d_vals = [abs(np.mean(X1[:, f]) - np.mean(X2[:, f])) / np.sqrt((np.std(X1[:, f])**2 + np.std(X2[:, f])**2)/2 + 1e-12) for f in range(X_action_sel.shape[1])]
        top_d = max(d_vals)
        print(f"  {c1} vs {c2}: top d = {top_d:.2f} ({feature_names[selected[d_vals.index(top_d)]]})")

# Classification
X_full = X_sel
y_full = y
meta_full = meta

def eval_split(X_tr, y_tr, X_te, y_te, name):
    for clf_name, clf in [
        ('RF', RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1)),
        ('SVM', Pipeline([('sc', StandardScaler()), ('svc', SVC(kernel='rbf', class_weight='balanced', random_state=RANDOM_STATE))])),
        ('kNN', Pipeline([('sc', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=5))])),
    ]:
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        acc = accuracy_score(y_te, y_pred)
        f1_macro = f1_score(y_te, y_pred, average='macro', zero_division=0)
        action_labels = [l for l in np.unique(y_te) if l != 'null']
        f1_action = f1_score(y_te, y_pred, labels=action_labels, average='macro', zero_division=0) if action_labels else 0
        print(f"  {name}/{clf_name}: acc={acc:.3f}, macro_f1={f1_macro:.3f}, action_f1={f1_action:.3f}")

print("\n--- Classification ---")
for split_name, mask_fn in [
    ('Owen', lambda m: m['subject'] == 'owen'),
    ('Ryan', lambda m: (m['subject'] == 'ryan') & (m['recording'] != '26071015随机做')),
]:
    m = meta_full
    sub_mask = mask_fn(m).values
    Xs, ys, ms = X_full[sub_mask], y_full[sub_mask], m[sub_mask]
    if 'rep_id' in ms.columns and ms['rep_id'].nunique() > 1:
        tm = ms['rep_id'] % 2 == 0
    else:
        split = int(0.7 * len(Xs))
        tm = np.zeros(len(Xs), dtype=bool)
        tm[:split] = True
    print(f"\n{split_name}:")
    eval_split(Xs[tm], ys[tm], Xs[~tm], ys[~tm], split_name.lower())

# Cross
m_owen = meta_full['subject'] == 'owen'
m_ryan = (meta_full['subject'] == 'ryan') & (meta_full['recording'] != '26071015随机做')
Xo, yo = X_full[m_owen], y_full[m_owen]
Xr, yr = X_full[m_ryan], y_full[m_ryan]
print("\nOwen->Ryan:")
eval_split(Xo, yo, Xr, yr, "o->r")
print("\nRyan->Owen:")
eval_split(Xr, yr, Xo, yo, "r->o")

# Merged
mask = meta_full['recording'] != '26071015随机做'
Xm, ym, mm = X_full[mask], y_full[mask], meta_full[mask]
recs = mm['recording'].unique()
np.random.seed(RANDOM_STATE)
np.random.shuffle(recs)
split = int(0.7 * len(recs))
train_recs = set(recs[:split])
tm = mm['recording'].isin(train_recs).values
print(f"\nMerged: train={train_recs}, test={set(recs[split:])}")
eval_split(Xm[tm], ym[tm], Xm[~tm], ym[~tm], "merged")
