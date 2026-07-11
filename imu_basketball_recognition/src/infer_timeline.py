"""
infer_timeline.py — Phase 2: 加载模型 → 推理随机录制 → 输出动作时间线

Usage:
  python3 infer_timeline.py --model models/run_YYYYMMDD_HHMM --input ryan/26071015随机做

内部禁止出现任何 fit 调用。
"""
import sys
import os
import argparse
import json
import yaml
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import sklearn
from sklearn.covariance import LedoitWolf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, '/Users/owen/Desktop/lykon-motion-lab-v2/imu_basketball_recognition/src')
from io_utils import load_recording
from preprocess import lowpass
from madgwick import madgwick_filter, quaternion_to_euler, decompose_gravity_dynamic
from features import extract_all_features
from config import NODES, FS, G_STD, ACTION_LABELS_INV, SMOOTH_N_VOTES


def load_model_package(model_dir: Path):
    """Load model artifacts."""
    print(f"Loading model package from {model_dir}")
    
    # Load config
    with open(model_dir / 'config_snapshot.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load feature names
    with open(model_dir / 'feature_names.json', 'r') as f:
        feature_names = json.load(f)
    
    with open(model_dir / 'selected_features.json', 'r') as f:
        selected_features = json.load(f)
    
    # Load pipelines
    pipelines = {}
    for name in ['rf', 'svm', 'knn']:
        p = model_dir / f'pipeline_{name}.joblib'
        if p.exists():
            pipelines[name] = joblib.load(p)
            print(f"  Loaded {name.upper()}")
    
    # Load rejection params
    with open(model_dir / 'rejection.json', 'r') as f:
        rejection = json.load(f)
    
    # Load manifest (for version check)
    with open(model_dir / 'train_manifest.json', 'r') as f:
        manifest = json.load(f)
    
    # Version check
    train_sklearn = manifest.get('sklearn_version', 'unknown')
    current_sklearn = sklearn.__version__
    if train_sklearn != current_sklearn:
        print(f"  [WARN] sklearn version mismatch: train={train_sklearn}, current={current_sklearn}")
    
    return config, feature_names, selected_features, pipelines, rejection, manifest


def preprocess_random_recording(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Preprocess random recording using the same chain as training.
    Reads parameters from config_snapshot.
    """
    print("\nPreprocessing random recording...")
    df = df.copy()
    t = df['t'].values
    n = len(df)
    
    # Low-pass filter (20 Hz) all channels
    for node in NODES:
        ax = df[f'ax_{node}'].values
        ay = df[f'ay_{node}'].values
        az = df[f'az_{node}'].values
        gx = df[f'gx_{node}'].values
        gy = df[f'gy_{node}'].values
        gz = df[f'gz_{node}'].values
        
        acc = np.column_stack([ax, ay, az])
        gyr = np.column_stack([gx, gy, gz])
        acc_f = lowpass(acc)
        gyr_f = lowpass(gyr)
        
        # Madgwick attitude
        q = madgwick_filter(acc_f, gyr_f, beta=0.1)
        euler = np.array([quaternion_to_euler(qi) for qi in q])
        df[f'roll_{node}'] = euler[:, 0]
        df[f'pitch_{node}'] = euler[:, 1]
        df[f'yaw_{node}'] = euler[:, 2]
        
        # Dynamic L2
        a_vert, a_horiz = decompose_gravity_dynamic(acc_f, q)
        df[f'a_vert_{node}'] = a_vert
        df[f'a_horiz_{node}'] = a_horiz
        
        # L3 magnitudes
        df[f'a_mag_{node}'] = np.sqrt(acc_f[:, 0]**2 + acc_f[:, 1]**2 + acc_f[:, 2]**2)
        df[f'g_mag_{node}'] = np.sqrt(gyr_f[:, 0]**2 + gyr_f[:, 1]**2 + gyr_f[:, 2]**2)
        
        # Jerk
        for j, suffix in enumerate(['x', 'y', 'z']):
            df[f'a{suffix}_{node}'] = acc_f[:, j]
            df[f'g{suffix}_{node}'] = gyr_f[:, j]
        for axis in ['x', 'y', 'z']:
            a = df[f'a{axis}_{node}'].values
            df[f'jerk_{axis}_{node}'] = np.gradient(a, t)
        df[f'jerk_vert_{node}'] = np.gradient(a_vert, t)
        df[f'jerk_mag_{node}'] = np.gradient(df[f'a_mag_{node}'].values, t)
    
    print(f"  Preprocessed: {n} samples, {df['t'].iloc[-1] - df['t'].iloc[0]:.1f}s")
    return df


def extract_sliding_windows_inference(df: pd.DataFrame, win_len_s: float, step_s: float) -> tuple:
    """
    Extract sliding windows for inference (no labels, no purity check).
    Returns (X, t_center, meta_info).
    """
    win_len = int(round(win_len_s * FS))
    step = int(round(step_s * FS))
    n = len(df)
    
    meta_cols = {'subject', 'recording', 't'}
    ch_cols = [c for c in df.columns if c not in meta_cols]
    data = df[ch_cols].values
    t = df['t'].values
    
    windows = []
    t_centers = []
    
    for start in range(0, n - win_len + 1, step):
        end = start + win_len
        windows.append(data[start:end])
        t_centers.append((t[start] + t[end - 1]) / 2.0)
    
    return np.array(windows), np.array(t_centers), ch_cols


def align_features(feat_df: pd.DataFrame, feature_names: list, selected_features: list) -> np.ndarray:
    """
    Align feature DataFrame to the expected feature order from training.
    Select only selected_features for model input.
    """
    # Ensure all expected features are present
    missing = [f for f in feature_names if f not in feat_df.columns]
    if missing:
        print(f"  [WARN] {len(missing)} missing features, filling with 0")
        for f in missing:
            feat_df[f] = 0.0
    
    # Reorder to training order
    X_full = feat_df[feature_names].values
    
    # Select features
    selected_indices = [feature_names.index(f) for f in selected_features]
    X_sel = X_full[:, selected_indices]
    return X_sel


def apply_rejection(y_proba: np.ndarray, classes: list, X_scaled: np.ndarray,
                    rejection: dict, class_centers: dict, class_covs: dict) -> tuple:
    """
    Apply rejection rules.
    Returns (labels, confidences, max_proba, mahal_distances, per_class_proba_df).
    """
    n_samples = len(y_proba)
    labels = []
    confidences = []
    max_probas = []
    mahal_dists = []
    
    prob_thresh = rejection['working_prob_thresh']
    mahal_mult = rejection['working_mahal_multiplier']
    class_mahal_thresholds = {c: rejection['class_mahal_thresholds'][c] * mahal_mult for c in classes}
    
    for i in range(n_samples):
        proba = y_proba[i]
        pred_idx = np.argmax(proba)
        pred_c = classes[pred_idx]
        max_p = np.max(proba)
        
        # Mahalanobis distance to predicted class
        x = X_scaled[i]
        center = np.array(class_centers[pred_c])
        cov = np.array(class_covs[pred_c])
        try:
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            inv_cov = np.linalg.pinv(cov)
        diff = x - center
        mahal = float(np.sqrt(diff @ inv_cov @ diff))
        if not np.isfinite(mahal):
            mahal = 0.0  # fallback: disable mahal rejection when math breaks
        
        if max_p < prob_thresh or mahal > class_mahal_thresholds[pred_c]:
            labels.append('unknown')
            confidences.append(1.0 - max_p)  # low confidence = high uncertainty
        else:
            labels.append(pred_c)
            confidences.append(max_p)
        
        max_probas.append(max_p)
        mahal_dists.append(mahal)
    
    per_class_proba = pd.DataFrame(y_proba, columns=classes)
    return np.array(labels), np.array(confidences), np.array(max_probas), np.array(mahal_dists), per_class_proba


def temporal_smooth(labels: np.ndarray, n_votes: int = 5) -> np.ndarray:
    """Majority vote smoothing over n_votes windows."""
    n = len(labels)
    smoothed = labels.copy()
    for i in range(n):
        start = max(0, i - n_votes // 2)
        end = min(n, i + n_votes // 2 + 1)
        window = labels[start:end]
        # Count non-unknown first
        non_unknown = window[window != 'unknown']
        if len(non_unknown) > 0:
            unique, counts = np.unique(non_unknown, return_counts=True)
            smoothed[i] = unique[np.argmax(counts)]
        else:
            smoothed[i] = 'unknown'
    return smoothed


def aggregate_events(labels: np.ndarray, t_centers: np.ndarray, confidences: np.ndarray,
                     bridge_gap_s: float = 0.6,
                     min_dur_transient_win: int = 2,
                     min_dur_continuous_win: int = 4) -> pd.DataFrame:
    """
    Merge continuous same-label windows into events.
    Bridge gaps <= bridge_gap_s.
    Apply minimum duration filtering.
    """
    # Transient classes: catch, dribble_right_once, shot (short duration events)
    transient_classes = {'catch', 'dribble_right_once', 'shot'}
    # Continuous classes: dribble_left_right, defense, pass_high (longer duration)
    
    events = []
    if len(labels) == 0:
        return pd.DataFrame(events)
    
    current_label = labels[0]
    start_idx = 0
    
    for i in range(1, len(labels)):
        if labels[i] != current_label:
            # Check gap for bridging
            if labels[i] == current_label:  # never reached due to !=
                pass
            gap_s = t_centers[i] - t_centers[i - 1]
            # Actually bridge: if next same label after gap <= bridge, merge
            # But for simplicity, first make raw segments, then bridge
            events.append({
                'start_idx': start_idx,
                'end_idx': i - 1,
                'label': current_label,
                't_start': t_centers[start_idx],
                't_end': t_centers[i - 1],
                'n_win': i - start_idx,
            })
            start_idx = i
            current_label = labels[i]
    
    # Last event
    events.append({
        'start_idx': start_idx,
        'end_idx': len(labels) - 1,
        'label': current_label,
        't_start': t_centers[start_idx],
        't_end': t_centers[-1],
        'n_win': len(labels) - start_idx,
    })
    
    # Bridge same-label gaps <= bridge_gap_s
    merged = []
    for e in events:
        if merged and e['label'] == merged[-1]['label'] and (e['t_start'] - merged[-1]['t_end']) <= bridge_gap_s:
            merged[-1]['end_idx'] = e['end_idx']
            merged[-1]['t_end'] = e['t_end']
            merged[-1]['n_win'] += e['n_win']
        else:
            merged.append(e.copy())
    
    # Minimum duration filtering
    filtered = []
    for e in merged:
        if e['label'] == 'unknown' or e['label'] == 'null':
            filtered.append(e)
            continue
        if e['label'] in transient_classes:
            if e['n_win'] >= min_dur_transient_win:
                filtered.append(e)
            else:
                e['label'] = 'unknown'
                filtered.append(e)
        else:
            if e['n_win'] >= min_dur_continuous_win:
                filtered.append(e)
            else:
                e['label'] = 'unknown'
                filtered.append(e)
    
    # Build DataFrame with confidences
    rows = []
    for i, e in enumerate(filtered):
        s_idx = e['start_idx']
        e_idx = e['end_idx'] + 1
        avg_conf = float(np.mean(confidences[s_idx:e_idx])) if e_idx > s_idx else 0.0
        rows.append({
            'event_id': i + 1,
            'start_s': round(e['t_start'], 2),
            'end_s': round(e['t_end'], 2),
            'duration_s': round(e['t_end'] - e['t_start'], 2),
            'label': e['label'],
            'label_cn': ACTION_LABELS_INV.get(e['label'], e['label']),
            'avg_confidence': round(avg_conf, 3),
            'n_windows': e['n_win'],
        })
    
    return pd.DataFrame(rows)


def plot_timeline(df_raw: pd.DataFrame, events_df: pd.DataFrame, labels: np.ndarray,
                  t_centers: np.ndarray, confidences: np.ndarray, output_path: Path):
    """Plot timeline with |a| waveform + colored action strips."""
    fig, ax = plt.subplots(figsize=(16, 6))
    
    t = df_raw['t'].values
    # Use a_mag of all four nodes averaged as background
    a_mag_all = np.mean([df_raw[f'a_mag_{node}'].values for node in NODES], axis=0)
    ax.plot(t, a_mag_all, color='gray', alpha=0.4, linewidth=0.5, label='|a| avg')
    
    # Color map
    color_map = {
        'catch': '#2ecc71',
        'dribble_left_right': '#3498db',
        'dribble_right_once': '#9b59b6',
        'defense': '#e67e22',
        'pass_high': '#1abc9c',
        'shot': '#e74c3c',
        'unknown': '#95a5a6',
        'null': '#bdc3c7',
    }
    
    # Draw event strips
    for _, e in events_df.iterrows():
        color = color_map.get(e['label'], '#95a5a6')
        ax.axvspan(e['start_s'], e['end_s'], color=color, alpha=0.3, zorder=2)
        # Event ID annotation
        if e['label'] != 'unknown' and e['label'] != 'null':
            mid = (e['start_s'] + e['end_s']) / 2.0
            ax.annotate(str(e['event_id']), xy=(mid, ax.get_ylim()[1] * 0.9),
                        ha='center', fontsize=8, fontweight='bold', color='black')
    
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('|a| [m/s²]')
    ax.set_title('Random Test Timeline — RF Model')
    
    # Legend
    handles = [mpatches.Patch(color=c, label=ACTION_LABELS_INV.get(lbl, lbl)) 
               for lbl, c in color_map.items() if lbl in events_df['label'].values or lbl == 'unknown']
    ax.legend(handles=handles, loc='upper left', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved timeline plot: {output_path}")


def plot_model_comparison(t_centers: np.ndarray, labels_dict: dict, df_raw: pd.DataFrame,
                          output_path: Path):
    """Plot three-model comparison with disagreement highlighted."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    
    t = df_raw['t'].values
    a_mag_all = np.mean([df_raw[f'a_mag_{node}'].values for node in NODES], axis=0)
    
    color_map = {
        'catch': '#2ecc71',
        'dribble_left_right': '#3498db',
        'dribble_right_once': '#9b59b6',
        'defense': '#e67e22',
        'pass_high': '#1abc9c',
        'shot': '#e74c3c',
        'unknown': '#95a5a6',
        'null': '#bdc3c7',
    }
    
    model_names = ['RF', 'SVM', 'kNN']
    model_keys = ['rf', 'svm', 'knn']
    
    for idx, (ax, key, mname) in enumerate(zip(axes, model_keys, model_names)):
        ax.plot(t, a_mag_all, color='gray', alpha=0.3, linewidth=0.5)
        labels = labels_dict[key]
        
        # Draw segments
        current = labels[0]
        start_t = t_centers[0] - 0.15
        for i in range(1, len(labels)):
            if labels[i] != current:
                end_t = t_centers[i - 1] + 0.15
                color = color_map.get(current, '#95a5a6')
                ax.axvspan(start_t, end_t, color=color, alpha=0.3)
                start_t = t_centers[i] - 0.15
                current = labels[i]
        # Last segment
        ax.axvspan(start_t, t_centers[-1] + 0.15, color=color_map.get(current, '#95a5a6'), alpha=0.3)
        
        ax.set_ylabel(f'{mname}')
        ax.set_ylim([0, ax.get_ylim()[1]])
    
    axes[-1].set_xlabel('Time [s]')
    plt.suptitle('Model Comparison — Random Test Recording')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved comparison plot: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to model run directory')
    parser.add_argument('--input', required=True, help='Subject/recording name for random test')
    args = parser.parse_args()
    
    model_dir = Path(args.model)
    config, feature_names, selected_features, pipelines, rejection, manifest = load_model_package(model_dir)
    
    # Parse input
    parts = args.input.split('/')
    if len(parts) == 2:
        subject, rec_name = parts
    else:
        subject = 'ryan'
        rec_name = '26071015随机做'
    
    print(f"\nLoading random recording: {subject}/{rec_name}")
    df_raw = load_recording(subject, rec_name, validate=False)
    
    # Preprocess
    df_proc = preprocess_random_recording(df_raw, config)
    
    # Extract sliding windows
    win_len_s = config['window_length_s']
    step_s = config['window_step_s']
    X_win, t_centers, ch_names = extract_sliding_windows_inference(df_proc, win_len_s, step_s)
    print(f"  Extracted {len(X_win)} windows ({win_len_s}s / {step_s}s step)")
    
    # Extract features
    print("\nExtracting features...")
    feat_df = extract_all_features(X_win, ch_names)
    
    # Align to training feature space
    X_sel = align_features(feat_df, feature_names, selected_features)
    print(f"  Aligned features: {X_sel.shape}")
    
    # Inference per model
    classes = config['classes']
    print("\nRunning inference...")
    
    results = {}
    for name, pipe in pipelines.items():
        y_pred = pipe.predict(X_sel)
        y_proba = pipe.predict_proba(X_sel)
        
        # Get scaled X for rejection (inside pipeline)
        scaler = pipe.named_steps['scaler']
        X_scaled = scaler.transform(X_sel)
        
        # Rejection (only for RF as primary)
        if name == 'rf':
            class_centers = {c: np.array(rejection['class_centers'][c]) for c in classes}
            class_covs = {c: np.array(rejection['class_covs'][c]) for c in classes}
            labels, confidences, max_proba, mahal, per_class_proba = apply_rejection(
                y_proba, classes, X_scaled, rejection, class_centers, class_covs
            )
        else:
            labels = y_pred
            confidences = np.max(y_proba, axis=1)
            max_proba = confidences
            mahal = np.zeros(len(y_pred))
            per_class_proba = pd.DataFrame(y_proba, columns=classes)
        
        # Temporal smoothing
        n_votes = config['smoothing']['n_votes']
        labels_smooth = temporal_smooth(labels, n_votes=n_votes)
        
        results[name] = {
            'labels': labels_smooth,
            'confidences': confidences,
            'max_proba': max_proba,
            'mahal': mahal,
            'per_class_proba': per_class_proba,
        }
        print(f"  {name.upper()}: done ({len(labels_smooth)} windows)")
    
    # Event aggregation (use RF as primary)
    primary = 'rf'
    labels_smooth = results[primary]['labels']
    confidences = results[primary]['confidences']
    
    bridge_gap = config['event_aggregation']['bridge_gap_s']
    min_transient = config['event_aggregation']['min_dur_transient_windows']
    min_continuous = config['event_aggregation']['min_dur_continuous_windows']
    
    events_df = aggregate_events(labels_smooth, t_centers, confidences,
                                 bridge_gap_s=bridge_gap,
                                 min_dur_transient_win=min_transient,
                                 min_dur_continuous_win=min_continuous)
    
    print(f"\nDetected {len(events_df)} events:")
    print(events_df.to_string(index=False))
    
    # Save outputs
    report_dir = Path('/Users/owen/Desktop/lykon-motion-lab-v2/imu_basketball_recognition/reports')
    run_name = model_dir.name
    out_dir = report_dir / 'random_test'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # CSV timeline
    csv_path = out_dir / f'random_test_timeline_{run_name}.csv'
    events_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\nSaved CSV: {csv_path}")
    
    # Markdown table
    md_path = out_dir / f'random_test_timeline_{run_name}.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# Random Test Timeline — {run_name}\n\n")
        f.write(f"**Model:** {primary.upper()} | **Recording:** {subject}/{rec_name}\n\n")
        f.write("| 序号 | 开始[s] | 结束[s] | 时长[s] | 动作 | 平均置信度 | 窗口数 |\n")
        f.write("|------|---------|---------|---------|------|-----------|--------|\n")
        for _, e in events_df.iterrows():
            f.write(f"| {e['event_id']} | {e['start_s']} | {e['end_s']} | {e['duration_s']} | {e['label_cn']} | {e['avg_confidence']} | {e['n_windows']} |\n")
        f.write(f"\n*Total events: {len(events_df)}*\n")
    print(f"Saved Markdown: {md_path}")
    
    # Per-window detail (parquet)
    detail_df = pd.DataFrame({
        't_center': t_centers,
        'label_rf': results['rf']['labels'],
        'label_svm': results['svm']['labels'],
        'label_knn': results['knn']['labels'],
        'confidence_rf': results['rf']['confidences'],
        'max_proba_rf': results['rf']['max_proba'],
        'mahal_rf': results['rf']['mahal'],
    })
    # Add per-class probabilities (RF only as primary)
    for c in classes:
        detail_df[f'proba_{c}'] = results['rf']['per_class_proba'][c].values
    
    parquet_path = out_dir / f'random_test_detail_{run_name}.parquet'
    detail_df.to_parquet(parquet_path, index=False)
    print(f"Saved Parquet: {parquet_path}")
    
    # Disagreement list
    disagree_mask = (results['rf']['labels'] != results['svm']['labels']) | \
                    (results['rf']['labels'] != results['knn']['labels'])
    n_disagree = disagree_mask.sum()
    print(f"\nModel disagreement windows: {n_disagree} / {len(disagree_mask)} ({n_disagree/len(disagree_mask)*100:.1f}%)")
    
    # Visualization
    print("\nGenerating plots...")
    plot_timeline(df_proc, events_df, labels_smooth, t_centers, confidences,
                  out_dir / f'random_test_timeline_{run_name}.png')
    
    plot_model_comparison(t_centers, {k: v['labels'] for k, v in results.items()}, df_proc,
                          out_dir / f'random_test_comparison_{run_name}.png')
    
    print(f"\n{'='*60}")
    print(f"Inference complete. Outputs in: {out_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
