"""
run_v31_degraded.py — 降级方案: 姿态依赖特征出池 + 替代特征主导
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
feat_df = pd.read_parquet('/Users/owen/Desktop/lykon-motion-lab-v2/imu_basketball_recognition/data/features/features_v31_fix2.parquet')
meta = pd.read_parquet('/Users/owen/Desktop/lykon-motion-lab-v2/imu_basketball_recognition/data/windows/meta_v31_fix2.parquet')
lib = np.load('/Users/owen/Desktop/lykon-motion-lab-v2/imu_basketball_recognition/data/windows/windows_v31_fix2.npz')
y = lib['y']
X = feat_df.values
feature_names = feat_df.columns.tolist()

# C1: 姿态依赖特征出池
attitude_keywords = ['a_vert', 'a_horiz', 'yaw', 'pitch_mean', 'roll_mean', 'pitch_dom', 'roll_dom']
# 保留 pitch_ptp, roll_ptp, pitch_std, roll_std 等
keep_mask = []
for name in feature_names:
    drop = False
    for kw in attitude_keywords:
        if kw in name:
            drop = True
            break
    keep_mask.append(not drop)

keep_idx = [i for i, v in enumerate(keep_mask) if v]
X_filtered = X[:, keep_idx]
filtered_names = [feature_names[i] for i in keep_idx]
print(f"Features after attitude removal: {X_filtered.shape[1]} (from {X.shape[1]})")
print(f"Removed: {X.shape[1] - X_filtered.shape[1]}")

# Feature selection
action_mask = y != 'null'
X_action = X_filtered[action_mask]
y_action = y[action_mask]

f_scores, _ = f_classif(X_action, y_action)
rf_sel = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
rf_sel.fit(X_action, y_action)
rf_imp = rf_sel.feature_importances_

f_rank = np.argsort(-f_scores)
rf_rank = np.argsort(-rf_imp)
combined_rank = {i: (np.where(f_rank==i)[0][0] + np.where(rf_rank==i)[0][0])/2 for i in range(X_filtered.shape[1])}
sorted_idx = sorted(combined_rank, key=combined_rank.get)

selected = []
for idx in sorted_idx:
    if len(selected) >= N_FEATURES_SELECT:
        break
    if len(selected) == 0:
        selected.append(idx); continue
    corr_max = np.max(np.abs(np.corrcoef(X_filtered[:, idx], X_filtered[:, selected].T)[0, 1:]))
    if corr_max < CORR_THRESHOLD:
        selected.append(idx)

X_sel = X_filtered[:, selected]
X_action_sel = X_action[:, selected]

# Cohen's d: dribble_right_once vs shot
classes = sorted(np.unique(y_action))
for c1 in ['dribble_right_once']:
    for c2 in ['shot']:
        X1 = X_action_sel[y_action == c1]
        X2 = X_action_sel[y_action == c2]
        d_vals = [abs(np.mean(X1[:, f]) - np.mean(X2[:, f])) / np.sqrt((np.std(X1[:, f])**2 + np.std(X2[:, f])**2)/2 + 1e-12) for f in range(X_action_sel.shape[1])]
        top_d = max(d_vals)
        print(f"\ndribble_right_once vs shot: top Cohen's d = {top_d:.2f} ({filtered_names[selected[d_vals.index(top_d)]]})")
        # Top 5
        top5 = sorted(enumerate(d_vals), key=lambda x: -x[1])[:5]
        for idx, d in top5:
            print(f"  {filtered_names[selected[idx]]}: d={d:.2f}")

# Classification
X_full = X_sel
y_full = y
meta_full = meta.reset_index(drop=True)

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

print("\n--- Degraded: attitude features removed ---")
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

m_owen = meta_full['subject'] == 'owen'
m_ryan = (meta_full['subject'] == 'ryan') & (meta_full['recording'] != '26071015随机做')
Xo, yo = X_full[m_owen], y_full[m_owen]
Xr, yr = X_full[m_ryan], y_full[m_ryan]
print("\nOwen->Ryan:")
eval_split(Xo, yo, Xr, yr, "o->r")
print("\nRyan->Owen:")
eval_split(Xr, yr, Xo, yo, "r->o")

mask = meta_full['recording'] != '26071015随机做'
Xm, ym, mm = X_full[mask], y_full[mask], meta_full[mask]
recs = mm['recording'].unique()
np.random.seed(RANDOM_STATE)
np.random.shuffle(recs)
split = int(0.7 * len(recs))
tm = mm['recording'].isin(set(recs[:split])).values
print(f"\nMerged:")
eval_split(Xm[tm], ym[tm], Xm[~tm], ym[~tm], "merged")
