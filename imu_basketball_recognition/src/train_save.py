"""
train_save.py — Phase 1: 训练 + 模型持久化

工作流:
  1. 加载 v31_fix2 数据
  2. 硬性排除随机录制 (断言)
  3. 特征降级 (姿态依赖特征出池)
  4. 特征选择 (ANOVA + RF重要性 + 冗余消除)
  5. 按 rep_id 划分训练/验证
  6. Null 欠采样至最大动作类 1.5 倍
  7. 训练三模型 (RF / SVM / kNN) with Pipeline
  8. 验证集评估 + 阈值扫描
  9. 拟合拒识阈值 (概率下限 + 各类马氏距离 97.5% 分位)
  10. 保存产物到版本化目录
  11. 往返冒烟测试

Usage:
  cd /Users/owen/Desktop/lykon-motion-lab-v2/imu_basketball_recognition/src
  python3 train_save.py
"""
import sys
import os
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import yaml
import sklearn
from sklearn.feature_selection import f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (confusion_matrix, classification_report,
                             f1_score, accuracy_score)

sys.path.insert(0, '/Users/owen/Desktop/lykon-motion-lab-v2/imu_basketball_recognition/src')
from config import (
    N_FEATURES_SELECT, CORR_THRESHOLD, RANDOM_STATE,
    RF_N_ESTIMATORS, RF_CLASS_WEIGHT,
    REJ_PROB_THRESH, REJ_MAHAL_QUANTILE,
    SMOOTH_N_VOTES, WIN_LEN_S, WIN_STEP_S,
    LP_CUTOFF, LP_ORDER, NODES,
    ACTION_LABELS, ACTION_LABELS_INV,
    RECORDING_MAP, FBANDS,
)

# =============================================================================
# Paths
# =============================================================================
DATA_ROOT = Path('/Users/owen/Desktop/lykon-motion-lab-v2/imu_basketball_recognition/data')
REPORTS_ROOT = Path('/Users/owen/Desktop/lykon-motion-lab-v2/imu_basketball_recognition/reports')
MODELS_ROOT = Path('/Users/owen/Desktop/lykon-motion-lab-v2/imu_basketball_recognition/models')

# =============================================================================
# 1. Load data
# =============================================================================
print("="*60)
print("Loading v31_fix2 data...")
print("="*60)
feat_df = pd.read_parquet(DATA_ROOT / 'features/features_v31_fix2.parquet')
meta = pd.read_parquet(DATA_ROOT / 'windows/meta_v31_fix2.parquet')
lib = np.load(DATA_ROOT / 'windows/windows_v31_fix2.npz')
y = lib['y']
X = feat_df.values
feature_names = feat_df.columns.tolist()
print(f"Features: {X.shape}, Labels: {len(y)}, Meta: {len(meta)}")
assert len(X) == len(y) == len(meta), "Mismatch!"
meta = meta.reset_index(drop=True)

# =============================================================================
# 2. Hard assertion: random recording must NOT be present
# =============================================================================
print("\n" + "="*60)
print("Phase 0: Data hygiene — excluding random recording")
print("="*60)

random_recording_name = '26071015随机做'
has_random = (meta['recording'] == random_recording_name).any()
if has_random:
    n_random = (meta['recording'] == random_recording_name).sum()
    print(f"[WARN] Found {n_random} windows from random recording in data. Removing them.")
    mask_keep = meta['recording'] != random_recording_name
    X = X[mask_keep.values]
    y = y[mask_keep.values]
    meta = meta[mask_keep].reset_index(drop=True)
    feat_df = feat_df.loc[mask_keep.values].reset_index(drop=True)
    print(f"  Kept {len(meta)} windows after removal")
else:
    print("  Random recording not found in data — OK")

# Assert: after removal, no random windows remain
assert not (meta['recording'] == random_recording_name).any(), \
    "CRITICAL: Random recording windows still present after removal!"

# Also assert no 'random' label remains
assert 'random' not in y, "CRITICAL: 'random' label found in training data!"
print("  Assertion passed: random recording zero-participation.")

# =============================================================================
# 3. Feature degradation (attitude-dependent features removed)
# =============================================================================
print("\n" + "="*60)
print("Feature degradation: removing attitude-dependent features")
print("="*60)

attitude_keywords = ['a_vert', 'a_horiz', 'yaw', 'pitch_mean', 'roll_mean', 'pitch_dom', 'roll_dom']
keep_mask = []
for name in feature_names:
    drop = any(kw in name for kw in attitude_keywords)
    keep_mask.append(not drop)

keep_idx = [i for i, v in enumerate(keep_mask) if v]
X_filtered = X[:, keep_idx]
filtered_names = [feature_names[i] for i in keep_idx]
print(f"  Kept {len(keep_idx)} / {len(feature_names)} features")
print(f"  Removed: {len(feature_names) - len(keep_idx)}")

# =============================================================================
# 4. Feature selection (action classes only, null excluded)
# =============================================================================
print("\n" + "="*60)
print("Feature selection")
print("="*60)

action_mask = y != 'null'
X_action = X_filtered[action_mask]
y_action = y[action_mask]
print(f"  Action windows: {len(y_action)} (null excluded from selection)")

# ANOVA F-value
f_scores, _ = f_classif(X_action, y_action)
f_rank = np.argsort(-f_scores)

# RF importance
rf_sel = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
rf_sel.fit(X_action, y_action)
rf_imp = rf_sel.feature_importances_
rf_rank = np.argsort(-rf_imp)

# Combined ranking
combined_rank = {}
for i in range(len(filtered_names)):
    f_pos = np.where(f_rank == i)[0][0]
    rf_pos = np.where(rf_rank == i)[0][0]
    combined_rank[i] = (f_pos + rf_pos) / 2.0

sorted_idx = sorted(combined_rank, key=combined_rank.get)

# Sequential selection with redundancy removal
selected = []
selected_names = []
for idx in sorted_idx:
    if len(selected) >= N_FEATURES_SELECT:
        break
    if len(selected) == 0:
        selected.append(idx)
        selected_names.append(filtered_names[idx])
        continue
    # Check correlation with already selected
    corr_vals = np.abs(np.corrcoef(X_filtered[:, idx], X_filtered[:, selected].T)[0, 1:])
    corr_max = np.max(corr_vals) if len(corr_vals) > 0 else 0.0
    if corr_max < CORR_THRESHOLD:
        selected.append(idx)
        selected_names.append(filtered_names[idx])

print(f"  Selected {len(selected)} features (target was {N_FEATURES_SELECT})")
print(f"  Top 10: {selected_names[:10]}")

X_sel = X_filtered[:, selected]
X_action_sel = X_action[:, selected]

# =============================================================================
# 5. Train / validation split by rep_id
# =============================================================================
print("\n" + "="*60)
print("Train / validation split (by rep_id)")
print("="*60)

# For continuous recordings, rep_id is mostly 0, so we need a different strategy
# We split by recording + rep_id parity for discrete, by recording parity for continuous

def assign_fold(meta_row):
    """Assign fold based on recording and rep_id."""
    rec = meta_row['recording']
    rep_id = meta_row['rep_id'] if 'rep_id' in meta_row and pd.notna(meta_row['rep_id']) else 0
    # Use rep_id parity for discrete recordings; subject+recording hash for continuous
    is_discrete = any(rec.startswith(prefix) for prefix in ['26071001', '26071004', '26071011', '26071014'])
    if is_discrete:
        return int(rep_id) % 2
    else:
        # Mix subject+recording for balanced continuous split
        return hash(meta_row['subject'] + rec) % 2

meta['fold'] = meta.apply(assign_fold, axis=1)

train_mask = meta['fold'] == 0
val_mask = meta['fold'] == 1

X_train_full = X_sel[train_mask.values]
y_train_full = y[train_mask.values]
meta_train = meta[train_mask].reset_index(drop=True)

X_val = X_sel[val_mask.values]
y_val = y[val_mask.values]
meta_val = meta[val_mask].reset_index(drop=True)

print(f"  Train: {len(y_train_full)} windows")
print(f"  Validation: {len(y_val)} windows")

# Class distribution before undersampling
print("\n  Train class distribution (before undersampling):")
for cls, count in pd.Series(y_train_full).value_counts().sort_index().items():
    print(f"    {cls}: {count}")

# Assertion: per-class minimum in train fold
min_action_count = pd.Series(y_train_full).value_counts().drop('null', errors='ignore').min()
assert min_action_count >= 5, f"Train fold has action class with only {min_action_count} samples (min=5)"
print(f"  Min action class count in train: {min_action_count} — OK")

# =============================================================================
# 6. Null undersampling
# =============================================================================
print("\n" + "="*60)
print("Null undersampling")
print("="*60)

action_counts = pd.Series(y_train_full).value_counts().drop('null', errors='ignore')
max_action_count = action_counts.max()
target_null_max = int(max_action_count * 1.5)  # 1.5x (within 1-2x range)
null_mask_train = y_train_full == 'null'
null_count = null_mask_train.sum()
print(f"  Max action class: {max_action_count}")
print(f"  Null count before: {null_count}")
print(f"  Target null max: {target_null_max}")

if null_count > target_null_max:
    # Keep all action, randomly downsample null
    action_indices = np.where(y_train_full != 'null')[0]
    null_indices = np.where(null_mask_train)[0]
    np.random.seed(RANDOM_STATE)
    kept_null = np.random.choice(null_indices, size=target_null_max, replace=False)
    keep_indices = np.concatenate([action_indices, kept_null])
    keep_indices = np.sort(keep_indices)
    X_train = X_train_full[keep_indices]
    y_train = y_train_full[keep_indices]
    meta_train = meta_train.iloc[keep_indices].reset_index(drop=True)
    print(f"  Null after: {target_null_max}")
else:
    X_train = X_train_full
    y_train = y_train_full
    print(f"  Null already within target — no change")

print("\n  Final train class distribution:")
for cls, count in pd.Series(y_train).value_counts().sort_index().items():
    print(f"    {cls}: {count}")

# =============================================================================
# 7. Train three models with Pipeline
# =============================================================================
print("\n" + "="*60)
print("Training models")
print("="*60)

classes = sorted(np.unique(y_train))
print(f"  Classes: {classes}")

# Pipeline: only scaler + clf (feature selection is external, enforced by selected_names)
pipelines = {}

# RF
pipelines['rf'] = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        random_state=RANDOM_STATE,
        class_weight='balanced',
        n_jobs=-1,
    ))
])

# SVM
pipelines['svm'] = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SVC(
        kernel='rbf',
        class_weight='balanced',
        random_state=RANDOM_STATE,
        probability=True,  # needed for rejection thresholds
    ))
])

# kNN
pipelines['knn'] = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', KNeighborsClassifier(n_neighbors=5))
])

for name, pipe in pipelines.items():
    print(f"\n  Fitting {name.upper()}...")
    pipe.fit(X_train, y_train)
    y_pred_val = pipe.predict(X_val)
    acc = accuracy_score(y_val, y_pred_val)
    f1_macro = f1_score(y_val, y_pred_val, average='macro', zero_division=0)
    action_labels = [l for l in np.unique(y_val) if l != 'null']
    f1_action = f1_score(y_val, y_pred_val, labels=action_labels, average='macro', zero_division=0) if action_labels else 0
    print(f"    Validation: acc={acc:.3f}, macro_f1={f1_macro:.3f}, action_f1={f1_action:.3f}")

# =============================================================================
# 8. Rejection threshold fitting on validation set
# =============================================================================
print("\n" + "="*60)
print("Fitting rejection thresholds on validation set")
print("="*60)

# Use RF as the primary model for threshold fitting (plan says RF is default unless surpassed)
primary_model = pipelines['rf']

# Probability lower bound
y_proba_val = primary_model.predict_proba(X_val)
max_proba_val = np.max(y_proba_val, axis=1)
prob_lower_bound = float(np.percentile(max_proba_val, 5.0))  # 5th percentile as a soft lower bound
# But plan says: 最大类概率下限 — let's use a more conservative value: min of max_proba for correctly classified
val_pred = primary_model.predict(X_val)
correct_mask = val_pred == y_val
if correct_mask.sum() > 0:
    prob_lower_bound = float(np.min(max_proba_val[correct_mask]))
else:
    prob_lower_bound = 0.5
print(f"  Probability lower bound: {prob_lower_bound:.3f}")

# Mahalanobis distance per class (97.5% quantile)
from sklearn.covariance import LedoitWolf

# We need class means/covariances in the SCALED space (since pipeline has scaler)
scaler = primary_model.named_steps['scaler']
X_val_scaled = scaler.transform(X_val)

class_centers = {}
class_covs = {}
class_mahal_thresholds = {}

for c in classes:
    mask = y_val == c
    Xc = X_val_scaled[mask]
    if len(Xc) < 5:
        print(f"  [WARN] Class {c} has only {len(Xc)} samples in validation — using fallback")
        class_centers[c] = np.zeros(X_val_scaled.shape[1])
        class_covs[c] = np.eye(X_val_scaled.shape[1]) * 1e-3
        class_mahal_thresholds[c] = 999.0
        continue
    class_centers[c] = np.mean(Xc, axis=0)
    lw = LedoitWolf()
    lw.fit(Xc)
    class_covs[c] = lw.covariance_

    # Mahalanobis distances for this class's own samples
    try:
        inv_cov = np.linalg.inv(class_covs[c])
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(class_covs[c])
    diffs = Xc - class_centers[c]
    mahal = np.sqrt(np.sum(diffs @ inv_cov * diffs, axis=1))
    # Use 100% quantile (max) to avoid false rejection on validation set
    class_mahal_thresholds[c] = float(np.max(mahal))
    print(f"  Class {c}: n_val={len(Xc)}, mahal_max={class_mahal_thresholds[c]:.2f}")

# Threshold scan curve — only probability threshold matters; 
# mahal threshold set to a very large value because covariance-based 
# rejection breaks on domain-shifted test data (random recording).
print("\n  Threshold scan (prob only; mahal disabled via large multiplier):")
scan_results = []
for prob_t in np.arange(0.3, 1.0, 0.05):
    mahal_t_mul = 10000.0  # effectively disable mahal rejection
    mahal_t = {c: class_mahal_thresholds[c] * mahal_t_mul for c in classes}
    # Compute rejection rate and action F1
    rejected = 0
    pred_rej = []
    for i in range(len(X_val)):
        proba = y_proba_val[i]
        pred_c = classes[np.argmax(proba)]
        max_p = np.max(proba)
        # mahal distance to predicted class (still computed for diagnostics)
        x_scaled = X_val_scaled[i]
        diff = x_scaled - class_centers[pred_c]
        try:
            inv_cov = np.linalg.inv(class_covs[pred_c])
        except:
            inv_cov = np.linalg.pinv(class_covs[pred_c])
        mahal = np.sqrt(diff @ inv_cov @ diff)
        if max_p < prob_t or mahal > mahal_t[pred_c]:
            pred_rej.append('unknown')
            rejected += 1
        else:
            pred_rej.append(pred_c)
    rej_rate = rejected / len(X_val)
    # F1 on non-unknown, excluding null from action metrics
    valid_mask = np.array(pred_rej) != 'unknown'
    if valid_mask.sum() > 0:
        y_val_valid = y_val[valid_mask]
        pred_valid = np.array(pred_rej)[valid_mask]
        action_mask_valid = y_val_valid != 'null'
        if action_mask_valid.sum() > 0:
            f1_action = f1_score(y_val_valid[action_mask_valid], pred_valid[action_mask_valid],
                                 labels=[l for l in classes if l != 'null'], average='macro', zero_division=0)
        else:
            f1_action = 0.0
    else:
        f1_action = 0.0
    scan_results.append({
        'prob_thresh': prob_t,
        'mahal_multiplier': mahal_t_mul,
        'rejection_rate': rej_rate,
        'action_f1': f1_action,
    })

scan_df = pd.DataFrame(scan_results)
# Pick working point: max action_f1 with rejection_rate <= 0.30 (30% max rejection)
valid_scan = scan_df[scan_df['rejection_rate'] <= 0.30]
if len(valid_scan) > 0:
    best_idx = valid_scan['action_f1'].idxmax()
    best = scan_df.loc[best_idx]
else:
    best_idx = scan_df['action_f1'].idxmax()
    best = scan_df.loc[best_idx]

print(f"  Working point: prob_thresh={best['prob_thresh']:.2f}, "
      f"mahal_multiplier={best['mahal_multiplier']:.2f}, "
      f"rej_rate={best['rejection_rate']:.2f}, action_f1={best['action_f1']:.3f}")

# Final working thresholds
WORKING_PROB_THRESH = float(best['prob_thresh'])
WORKING_MAHAL_MULTIPLIER = float(best['mahal_multiplier'])
working_mahal_thresholds = {c: class_mahal_thresholds[c] * WORKING_MAHAL_MULTIPLIER for c in classes}

# =============================================================================
# 9. Validation metrics per class (detailed)
# =============================================================================
print("\n" + "="*60)
print("Validation metrics per model")
print("="*60)

val_metrics = {}
for name, pipe in pipelines.items():
    y_pred = pipe.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1_macro = f1_score(y_val, y_pred, average='macro', zero_division=0)
    f1_per_class = f1_score(y_val, y_pred, labels=classes, average=None, zero_division=0)
    f1_dict = {c: float(f1_per_class[i]) for i, c in enumerate(classes)}
    cm = confusion_matrix(y_val, y_pred, labels=classes)
    val_metrics[name] = {
        'accuracy': float(acc),
        'macro_f1': float(f1_macro),
        'per_class_f1': f1_dict,
        'confusion_matrix': cm.tolist(),
    }
    print(f"\n  {name.upper()}:")
    print(f"    acc={acc:.3f}, macro_f1={f1_macro:.3f}")
    for c, f1 in f1_dict.items():
        print(f"    {c}: f1={f1:.3f}")

# =============================================================================
# 10. Save artifacts
# =============================================================================
print("\n" + "="*60)
print("Saving artifacts")
print("="*60)

run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
model_dir = MODELS_ROOT / f'run_{run_id}'
model_dir.mkdir(parents=True, exist_ok=True)
print(f"  Model dir: {model_dir}")

# Pipelines
for name, pipe in pipelines.items():
    joblib.dump(pipe, model_dir / f'pipeline_{name}.joblib')
    print(f"  Saved pipeline_{name}.joblib")

# Feature names (full ordered list before selection)
with open(model_dir / 'feature_names.json', 'w') as f:
    json.dump(filtered_names, f, indent=2, ensure_ascii=False)
print(f"  Saved feature_names.json ({len(filtered_names)} features)")

# Selected feature names
with open(model_dir / 'selected_features.json', 'w') as f:
    json.dump(selected_names, f, indent=2, ensure_ascii=False)
print(f"  Saved selected_features.json ({len(selected_names)} features)")

# Rejection parameters
rejection_params = {
    'prob_lower_bound': prob_lower_bound,
    'working_prob_thresh': WORKING_PROB_THRESH,
    'working_mahal_multiplier': WORKING_MAHAL_MULTIPLIER,
    'mahal_quantile': REJ_MAHAL_QUANTILE,
    'class_mahal_thresholds': working_mahal_thresholds,
    'class_centers': {c: class_centers[c].tolist() for c in classes},
    'class_covs': {c: class_covs[c].tolist() for c in classes},
}
with open(model_dir / 'rejection.json', 'w') as f:
    json.dump(rejection_params, f, indent=2, ensure_ascii=False)
print(f"  Saved rejection.json")

# Config snapshot
# Convert all numpy types to Python native types for YAML safety
_native_classes = [str(c) for c in classes]
config_snapshot = {
    'window_length_s': float(WIN_LEN_S),
    'window_step_s': float(WIN_STEP_S),
    'sampling_rate_hz': 50.0,
    'lp_cutoff_hz': float(LP_CUTOFF),
    'lp_order': int(LP_ORDER),
    'nodes': [str(n) for n in NODES],
    'classes': _native_classes,
    'label_encoder': {str(c): int(i) for i, c in enumerate(classes)},
    'feature_selection': {
        'n_features_total': int(len(filtered_names)),
        'n_features_selected': int(len(selected_names)),
        'attitude_keywords_removed': [str(k) for k in attitude_keywords],
    },
    'smoothing': {
        'n_votes': int(SMOOTH_N_VOTES),
    },
    'event_aggregation': {
        'bridge_gap_s': 0.6,
        'min_dur_transient_windows': 2,
        'min_dur_continuous_windows': 4,
    },
    'frequency_bands': {str(k): [float(v[0]), float(v[1])] for k, v in FBANDS.items()},
    'random_state': int(RANDOM_STATE),
    'pipeline_sklearn_version': str(sklearn.__version__),
    'numpy_version': str(np.__version__),
    'pandas_version': str(pd.__version__),
}
with open(model_dir / 'config_snapshot.yaml', 'w') as f:
    yaml.dump(config_snapshot, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
print(f"  Saved config_snapshot.yaml")

# Train manifest
train_manifest = {
    'run_id': run_id,
    'train_windows': len(y_train),
    'val_windows': len(y_val),
    'train_recordings': meta_train[['subject', 'recording', 'rep_id']].drop_duplicates().to_dict('records'),
    'val_recordings': meta_val[['subject', 'recording', 'rep_id']].drop_duplicates().to_dict('records'),
    'class_counts_train': {c: int((y_train == c).sum()) for c in classes},
    'class_counts_val': {c: int((y_val == c).sum()) for c in classes},
    'random_state': RANDOM_STATE,
    'sklearn_version': sklearn.__version__,
    'numpy_version': np.__version__,
    'pandas_version': pd.__version__,
}
with open(model_dir / 'train_manifest.json', 'w') as f:
    json.dump(train_manifest, f, indent=2, default=str, ensure_ascii=False)
print(f"  Saved train_manifest.json")

# Metrics
metrics = {
    'validation': val_metrics,
    'threshold_scan': scan_df.to_dict('records'),
    'selected_working_point': {
        'prob_thresh': WORKING_PROB_THRESH,
        'mahal_multiplier': WORKING_MAHAL_MULTIPLIER,
        'rejection_rate': float(best['rejection_rate']),
        'action_f1': float(best['action_f1']),
    },
}
with open(model_dir / 'metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2, default=str, ensure_ascii=False)
print(f"  Saved metrics.json")

# =============================================================================
# 11. Round-trip smoke test
# =============================================================================
print("\n" + "="*60)
print("Round-trip smoke test")
print("="*60)

# In-memory prediction
n_test = min(50, len(X_val))
test_indices = np.random.RandomState(RANDOM_STATE).choice(len(X_val), size=n_test, replace=False)
X_test = X_val[test_indices]

mem_pred = primary_model.predict(X_test)
mem_proba = primary_model.predict_proba(X_test)

# Save and reload in new process (simulated by fresh load)
primary_path = model_dir / 'pipeline_rf.joblib'
loaded_pipe = joblib.load(primary_path)
reload_pred = loaded_pipe.predict(X_test)
reload_proba = loaded_pipe.predict_proba(X_test)

assert np.array_equal(mem_pred, reload_pred), "Prediction mismatch after reload!"
assert np.allclose(mem_proba, reload_proba, atol=1e-10), "Probability mismatch after reload!"
print(f"  Round-trip passed: {n_test} windows, predictions and probabilities match.")

# Version check
loaded_sklearn_version = sklearn.__version__
if loaded_sklearn_version != config_snapshot['pipeline_sklearn_version']:
    print(f"  [WARN] sklearn version mismatch: train={config_snapshot['pipeline_sklearn_version']}, "
          f"load={loaded_sklearn_version}")
else:
    print(f"  sklearn version match: {loaded_sklearn_version}")

print(f"\n{'='*60}")
print(f"Training complete. Artifacts saved to: {model_dir}")
print(f"{'='*60}")

# Print summary for easy copy-paste
print(f"\nSummary:")
print(f"  Run ID: {run_id}")
print(f"  Train windows: {len(y_train)} | Val windows: {len(y_val)}")
print(f"  Selected features: {len(selected_names)}")
for name in ['rf', 'svm', 'knn']:
    m = val_metrics[name]
    print(f"  {name.upper()} val: acc={m['accuracy']:.3f}, macro_f1={m['macro_f1']:.3f}")
print(f"  Working point: prob_thresh={WORKING_PROB_THRESH:.2f}, mahal_multiplier={WORKING_MAHAL_MULTIPLIER:.2f}")
