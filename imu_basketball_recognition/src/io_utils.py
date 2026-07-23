"""
io_utils.py — Data loading layer for 4-node calibrated IMU recordings.
All outputs carry (subject, recording, rep_id) metadata.
"""
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from config import (
    DATA_ROOT, SUBJECTS, RECORDING_MAP, NODES, NODE_ABBR,
    CALIB_SUFFIX, FS, DT, G_STD
)

# ---------------------------------------------------------------------------
# Filename resolution
# ---------------------------------------------------------------------------

def _find_calibrated_files(subject: str, rec_name: str) -> Dict[str, Path]:
    """
    Find the 4 calibrated node files for a given subject+recording.
    Handles name mismatches (e.g. '跳投' vs '投球').
    """
    calib_dir = Path(DATA_ROOT) / subject / 'calibrated'
    if not calib_dir.exists():
        raise FileNotFoundError(f"Calibrated directory not found: {calib_dir}")

    # Try exact match first
    pattern_base = rec_name.replace('跳投', '*')  # allow 跳投/投球 mismatch
    candidates = []
    for f in calib_dir.iterdir():
        if not f.name.endswith(CALIB_SUFFIX):
            continue
        # Strip suffix and node info to get base name
        base = f.name.replace(CALIB_SUFFIX, '')
        base = re.sub(r'_node\d$', '', base)
        # Check if base matches rec_name (with 跳投/投球 flexibility)
        rec_base = rec_name.replace('跳投', '投球')
        if base == rec_base or base == rec_name:
            candidates.append(f)

    if len(candidates) < 4:
        # Fallback: fuzzy match by prefix (timestamp)
        prefix = rec_name[:8]  # e.g. '26071001'
        candidates = [f for f in calib_dir.iterdir()
                      if f.name.startswith(prefix) and f.name.endswith(CALIB_SUFFIX)]

    if len(candidates) < 4:
        raise FileNotFoundError(
            f"Could not find 4 calibrated files for {subject}/{rec_name} in {calib_dir}"
        )

    # Map to nodes
    result = {}
    for f in candidates:
        m = re.search(r'_node(\d)' + re.escape(CALIB_SUFFIX), f.name)
        if m:
            node_key = f"node{m.group(1)}"
            result[node_key] = f

    if set(result.keys()) != set(NODES):
        missing = set(NODES) - set(result.keys())
        raise FileNotFoundError(
            f"Missing nodes {missing} for {subject}/{rec_name}"
        )
    return result


# ---------------------------------------------------------------------------
# Single-node loader
# ---------------------------------------------------------------------------

def _load_node_csv(path: Path) -> pd.DataFrame:
    """Load a single calibrated node CSV."""
    df = pd.read_csv(path)
    # Expected columns: t_rel_s, acc_x_g, acc_y_g, acc_z_g,
    #                   gyr_x_dps, gyr_y_dps, gyr_z_dps, ...
    required = ['t_rel_s', 'acc_x_g', 'acc_y_g', 'acc_z_g',
                'gyr_x_dps', 'gyr_y_dps', 'gyr_z_dps']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    return df


def _convert_units(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert units:
      acc [g] → [m/s²]
      gyr [°/s] → [rad/s]
    """
    df = df.copy()
    for col in ['acc_x_g', 'acc_y_g', 'acc_z_g']:
        df[col] = df[col] * G_STD
    for col in ['gyr_x_dps', 'gyr_y_dps', 'gyr_z_dps']:
        df[col] = np.deg2rad(df[col])
    # Rename to standard names
    df = df.rename(columns={
        'acc_x_g': 'ax', 'acc_y_g': 'ay', 'acc_z_g': 'az',
        'gyr_x_dps': 'gx', 'gyr_y_dps': 'gy', 'gyr_z_dps': 'gz',
    })
    return df


# ---------------------------------------------------------------------------
# Full recording loader (4 nodes merged)
# ---------------------------------------------------------------------------

def load_recording(subject: str, rec_name: str,
                   validate: bool = True) -> pd.DataFrame:
    """
    Load a full recording: 4 nodes merged into one DataFrame.

    Returns DataFrame with columns:
      t, ax_node1, ay_node1, az_node1, gx_node1, gy_node1, gz_node1,
         ax_node2, ... gz_node4,
         subject, recording

    Units: acc [m/s²], gyr [rad/s], t [s].
    """
    files = _find_calibrated_files(subject, rec_name)

    frames = []
    for node in NODES:
        df = _load_node_csv(files[node])
        df = _convert_units(df)
        # Select only the core columns + time
        cols = ['t_rel_s', 'ax', 'ay', 'az', 'gx', 'gy', 'gz']
        df = df[cols].copy()
        # Prefix columns with node name (except time)
        rename = {c: f"{c}_{node}" for c in cols if c != 't_rel_s'}
        rename['t_rel_s'] = 't'
        df = df.rename(columns=rename)
        frames.append(df)

    # Merge on time using outer join then interpolate
    merged = frames[0]
    for df in frames[1:]:
        merged = pd.merge(merged, df, on='t', how='outer')
    merged = merged.sort_values('t').reset_index(drop=True)

    # Interpolate missing values (linear) then drop any remaining NaN at edges
    numeric_cols = merged.columns.difference(['subject', 'recording'])
    merged[numeric_cols] = merged[numeric_cols].interpolate(method='linear')
    merged = merged.dropna().reset_index(drop=True)

    # Add metadata
    merged['subject'] = subject
    merged['recording'] = rec_name

    # --- Validation ---
    if validate:
        # Check sampling consistency
        dt_actual = np.diff(merged['t'].values)
        dt_median = np.median(dt_actual)
        dt_dev = np.abs(dt_actual - dt_median) / dt_median
        bad_ratio = np.mean(dt_dev > 0.20)
        if bad_ratio > 0.05:
            print(f"[WARN] {subject}/{rec_name}: {bad_ratio*100:.1f}% timestamps deviate >20% from median dt")

        # Check stationary segment gravity magnitude
        # Use first 2s as proxy for stationary (user typically starts still)
        n_static = int(2.0 * FS)
        for node in NODES:
            ax = merged[f'ax_{node}'].iloc[:n_static].values
            ay = merged[f'ay_{node}'].iloc[:n_static].values
            az = merged[f'az_{node}'].iloc[:n_static].values
            mag = np.sqrt(ax**2 + ay**2 + az**2)
            mag_mean = np.mean(mag)
            if not (8.5 < mag_mean < 11.0):
                print(f"[WARN] {subject}/{rec_name} {node}: static |a|={mag_mean:.2f} m/s² (expected ~9.8)")

        # Gyro zero check
        for node in NODES:
            gx = merged[f'gx_{node}'].iloc[:n_static].values
            gy = merged[f'gy_{node}'].iloc[:n_static].values
            gz = merged[f'gz_{node}'].iloc[:n_static].values
            gyr_mag = np.sqrt(gx**2 + gy**2 + gz**2)
            gyr_mean = np.mean(gyr_mag)
            if gyr_mean > 0.5:
                print(f"[WARN] {subject}/{rec_name} {node}: static |g|={gyr_mean:.3f} rad/s (zero bias may be large)")

    return merged


# ---------------------------------------------------------------------------
# Convenience: load all recordings for a subject
# ---------------------------------------------------------------------------

def load_subject(subject: str, validate: bool = True) -> Dict[str, pd.DataFrame]:
    """Load all recordings for one subject. Returns dict {rec_name: df}."""
    if subject not in RECORDING_MAP:
        raise ValueError(f"Unknown subject: {subject}")
    result = {}
    for rec_name in RECORDING_MAP[subject].keys():
        print(f"[LOAD] {subject} / {rec_name} ...")
        result[rec_name] = load_recording(subject, rec_name, validate=validate)
    return result


# ---------------------------------------------------------------------------
# Dropout inspection
# ---------------------------------------------------------------------------

def inspect_dropouts(df: pd.DataFrame, subject: str, rec_name: str, node: str) -> pd.DataFrame:
    """
    Report dropout statistics for a single node within a merged DataFrame.
    Returns a small summary DataFrame.
    """
    t = df['t'].values
    dt = np.diff(t)
    expected_dt = DT
    gaps = dt[dt > expected_dt * 1.5]  # gaps > 1.5× expected
    summary = {
        'subject': subject,
        'recording': rec_name,
        'node': node,
        'total_samples': len(t),
        'duration_s': t[-1] - t[0],
        'expected_samples': int(round((t[-1] - t[0]) / expected_dt)) + 1,
        'dropout_count': len(gaps),
        'dropout_ratio': len(gaps) / len(dt) if len(dt) > 0 else 0,
        'max_gap_s': float(np.max(gaps)) if len(gaps) > 0 else 0.0,
    }
    return pd.DataFrame([summary])


# ---------------------------------------------------------------------------
# Quick sanity test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # Load one recording as smoke test
    df = load_recording('owen', '26071001接球-拍球-跳投', validate=True)
    print(f"Loaded shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Time range: {df['t'].iloc[0]:.2f} - {df['t'].iloc[-1]:.2f} s")
    print(f"dt median: {np.median(np.diff(df['t'].values)):.4f} s")
