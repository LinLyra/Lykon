"""
run_v31_fix2.py — 修复二: Madgwick姿态重建 + L2重建 + 替代特征
基于修复一的锚定窗口数据，替换姿态解算
"""
import sys
sys.path.insert(0, '/Users/owen/Desktop/lykon-motion-lab-v2/imu_basketball_recognition/src')

import numpy as np
import pandas as pd
from io_utils import load_recording
from preprocess import lowpass
from madgwick import madgwick_filter, quaternion_to_euler, decompose_gravity_dynamic
from windows_v31 import extract_anchor_windows, extract_sliding_windows, build_window_library_v31
from features import extract_all_features
from config import NODES, FS, G_STD, RANDOM_STATE, RECORDING_MAP

# =====================================================================
# B1: Madgwick 重算全部录制
# =====================================================================

all_data_v2 = {}
quality_scores = {}

for subject in ['owen', 'ryan']:
    all_data_v2[subject] = {}
    for rec_name in RECORDING_MAP[subject].keys():
        print(f"\n[Madgwick] {subject}/{rec_name}")
        df = load_recording(subject, rec_name, validate=False)
        t = df['t'].values
        n = len(df)

        # Low-pass filter (20 Hz) all channels first
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

            # --- Madgwick attitude ---
            q = madgwick_filter(acc_f, gyr_f, beta=0.1)
            euler = np.array([quaternion_to_euler(qi) for qi in q])
            df[f'roll_{node}'] = euler[:, 0]
            df[f'pitch_{node}'] = euler[:, 1]
            df[f'yaw_{node}'] = euler[:, 2]

            # --- Dynamic L2: gravity-frame decomposition ---
            a_vert, a_horiz = decompose_gravity_dynamic(acc_f, q)
            df[f'a_vert_{node}'] = a_vert
            df[f'a_horiz_{node}'] = a_horiz

            # --- L3 magnitudes (on LP signals) ---
            df[f'a_mag_{node}'] = np.sqrt(acc_f[:, 0]**2 + acc_f[:, 1]**2 + acc_f[:, 2]**2)
            df[f'g_mag_{node}'] = np.sqrt(gyr_f[:, 0]**2 + gyr_f[:, 1]**2 + gyr_f[:, 2]**2)

            # --- Jerk (on L1 filtered acc) ---
            for j, suffix in enumerate(['x', 'y', 'z']):
                df[f'a{suffix}_{node}'] = acc_f[:, j]
                df[f'g{suffix}_{node}'] = gyr_f[:, j]
            for axis in ['x', 'y', 'z']:
                a = df[f'a{axis}_{node}'].values
                df[f'jerk_{axis}_{node}'] = np.gradient(a, t)
            df[f'jerk_vert_{node}'] = np.gradient(a_vert, t)
            df[f'jerk_mag_{node}'] = np.gradient(df[f'a_mag_{node}'].values, t)

        # B2: 准静止一致性评估 (简版: first 2s as proxy)
        angles = {}
        for node in NODES:
            n_static = int(2 * FS)
            ax = df[f'ax_{node}'].values[:n_static]
            ay = df[f'ay_{node}'].values[:n_static]
            az = df[f'az_{node}'].values[:n_static]
            a_mag = np.sqrt(ax**2 + ay**2 + az**2)
            a_norm = np.column_stack([ax, ay, az]) / (a_mag[:, None] + 1e-12)
            # Gravity direction from Madgwick (predicted)
            q = df[[f'roll_{node}', f'pitch_{node}', f'yaw_{node}']].values[:n_static]
            # Simplified: use L2 a_vert to estimate
            a_vert = df[f'a_vert_{node}'].values[:n_static]
            # If a_vert is near 0, gravity is aligned
            angles[node] = {'a_vert_mean': float(np.mean(a_vert)), 'a_vert_std': float(np.std(a_vert))}

        quality_scores[f"{subject}/{rec_name}"] = angles
        print(f"  Quality: {angles}")

        # Label (reuse old labeling, same event segments)
        from labeling import label_recording
        df = label_recording(df, subject, rec_name)
        df.attrs['rec_info'] = RECORDING_MAP[subject][rec_name]
        all_data_v2[subject][rec_name] = df

# =====================================================================
# Rebuild window library with new L2
# =====================================================================
print("\n" + "="*60)
print("Rebuilding window library with Madgwick L2")
print("="*60)
lib = build_window_library_v31(all_data_v2)
print(f"\nFinal library: X={lib['X'].shape}, y={lib['y'].shape}")

from windows import print_class_distribution
print_class_distribution(lib['meta'])

# Save
import os
out_dir = '/Users/owen/Desktop/lykon-motion-lab-v2/imu_basketball_recognition/data/windows'
np.savez_compressed(os.path.join(out_dir, 'windows_v31_fix2.npz'), X=lib['X'], y=lib['y'])
lib['meta'].to_parquet(os.path.join(out_dir, 'meta_v31_fix2.parquet'), index=False)

# =====================================================================
# Feature extraction (same channels as before)
# =====================================================================
ch_names = []
for node in NODES:
    for axis in ['x', 'y', 'z']:
        ch_names.append(f'a{axis}_{node}')
        ch_names.append(f'g{axis}_{node}')
    ch_names.append(f'a_vert_{node}')
    ch_names.append(f'a_horiz_{node}')
    ch_names.append(f'a_mag_{node}')
    ch_names.append(f'g_mag_{node}')
    ch_names.append(f'roll_{node}')
    ch_names.append(f'pitch_{node}')
    ch_names.append(f'yaw_{node}')
    for suffix in ['x', 'y', 'z', 'vert', 'mag']:
        ch_names.append(f'jerk_{suffix}_{node}')

print(f"\nExtracting features from {lib['X'].shape[0]} windows...")
feat_df = extract_all_features(lib['X'], ch_names)
feat_df.to_parquet(os.path.join(out_dir, '../features/features_v31_fix2.parquet'), index=False)
print(f"Features: {feat_df.shape}, saved.")
