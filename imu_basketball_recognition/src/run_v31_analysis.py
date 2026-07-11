"""
run_v31_analysis.py — v3.1 修复一后分析
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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.covariance import LedoitWolf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from config import N_FEATURES_SELECT, CORR_THRESHOLD, RANDOM_STATE, RF_N_ESTIMATORS

# Load
print("Loading v3.1 data...")
feat_df = pd.read_parquet('/Users/owen/Desktop/lykon-motion-lab-v2/imu_basketball_recognition/data/features/features_v31.parquet')
meta = pd.read_parquet('/Users/owen/Desktop/lykon-motion-lab-v2/imu_basketball_recognition/data/windows/meta_v31.parquet')
lib = np.load('/Users/owen/Desktop/lykon-motion-lab-v2/imu_basketball_recognition/data/windows/windows_v31.npz')
y = lib['y']

X = feat_df.values
feature_names = feat_df.columns.tolist()
print(f"X={X.shape}, y={len(y)}, meta={len(meta)}")

assert len(X) == len(y) == len(meta)
meta = meta.reset_index(drop=True)

# Action mask
action_mask = y != 'null'
X_action = X[action_mask]
y_action = y[action_mask]
meta_action = meta[action_mask].reset_index(drop=True)

# Feature selection (same as before)
print("\nFeature selection...")
f_scores, f_pvals = f_classif(X_action, y_action)
rf_sel = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
rf_sel.fit(X_action, y_action)
rf_imp = rf_sel.feature_importances_

f_rank = np.argsort(-f_scores)
rf_rank = np.argsort(-rf_imp)
combined_rank = {i: (np.where(f_rank == i)[0][0] + np.where(rf_rank == i)[0][0]) / 2 for i in range(len(feature_names))}
sorted_idx = sorted(combined_rank, key=combined_rank.get)

selected = []
for idx in sorted_idx:
    if len(selected) >= N_FEATURES_SELECT:
        break
    if len(selected) == 0:
        selected.append(idx)
        continue
    corr_max = np.max(np.abs(np.corrcoef(X[:, idx], X[:, selected].T)[0, 1:]))
    if corr_max < CORR_THRESHOLD:
        selected.append(idx)

X_sel = X[:, selected]
X_action_sel = X_action[:, selected]

# Class separability
classes = sorted(np.unique(y_action))
print(f"\nClasses: {classes}")

# Cohen's d: dribble_right_once vs shot
from scipy import stats
for c1 in classes:
    for c2 in classes:
        if c1 >= c2:
            continue
        X1 = X_action_sel[y_action == c1]
        X2 = X_action_sel[y_action == c2]
        d_vals = []
        for f in range(X_action_sel.shape[1]):
            m1, m2 = np.mean(X1[:, f]), np.mean(X2[:, f])
            s1, s2 = np.std(X1[:, f]), np.std(X2[:, f])
            d = abs(m1 - m2) / np.sqrt((s1**2 + s2**2) / 2 + 1e-12)
            d_vals.append(d)
        top_d = max(d_vals)
        top_idx = d_vals.index(top_d)
        print(f"  {c1} vs {c2}: top Cohen's d = {top_d:.2f} ({feature_names[selected[top_idx]]})")

# Classification
print("\n--- Classification ---")
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
        f1_action = f1_score(y_te, y_pred, labels=[l for l in np.unique(y_te) if l != 'null'], average='macro', zero_division=0)
        print(f"  {name}/{clf_name}: acc={acc:.3f}, macro_f1={f1_macro:.3f}, action_f1={f1_action:.3f}")

# Within-subject (Owen)
owen_mask = meta_full['subject'] == 'owen'
Xo, yo, mo = X_full[owen_mask], y_full[owen_mask], meta_full[owen_mask]
if 'rep_id' in mo.columns and mo['rep_id'].nunique() > 1:
    train_mask = mo['rep_id'] % 2 == 0
else:
    split = int(0.7 * len(Xo))
    train_mask = np.zeros(len(Xo), dtype=bool)
    train_mask[:split] = True
print("\n1. Owen:")
eval_split(Xo[train_mask], yo[train_mask], Xo[~train_mask], yo[~train_mask], "owen")

# Within-subject (Ryan, exclude random)
ryan_mask = (meta_full['subject'] == 'ryan') & (meta_full['recording'] != '26071015随机做')
Xr, yr, mr = X_full[ryan_mask], y_full[ryan_mask], meta_full[ryan_mask]
if 'rep_id' in mr.columns and mr['rep_id'].nunique() > 1:
    train_mask = mr['rep_id'] % 2 == 0
else:
    split = int(0.7 * len(Xr))
    train_mask = np.zeros(len(Xr), dtype=bool)
    train_mask[:split] = True
print("\n2. Ryan:")
eval_split(Xr[train_mask], yr[train_mask], Xr[~train_mask], yr[~train_mask], "ryan")

# Cross
print("\n3. Owen→Ryan:")
eval_split(Xo, yo, Xr, yr, "owen->ryan")
print("\n4. Ryan→Owen:")
eval_split(Xr, yr, Xo, yo, "ryan->owen")

# Merged (exclude random)
mask = meta_full['recording'] != '26071015随机做'
Xm, ym, mm = X_full[mask], y_full[mask], meta_full[mask]
recs = mm['recording'].unique()
np.random.seed(RANDOM_STATE)
np.random.shuffle(recs)
split = int(0.7 * len(recs))
train_recs = set(recs[:split])
test_recs = set(recs[split:])
tm = mm['recording'].isin(train_recs).values
print(f"\n5. Merged: train_recs={train_recs}, test_recs={test_recs}")
eval_split(Xm[tm], ym[tm], Xm[~tm], ym[~tm], "merged")
