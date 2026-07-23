"""
run_v31_fix1.py — 运行修复一(事件锚定取窗),基于旧版预处理数据
"""
import sys
sys.path.insert(0, '/Users/owen/Desktop/lykon-motion-lab-v2/imu_basketball_recognition/src')

import numpy as np
import pandas as pd
from io_utils import load_recording
from preprocess import preprocess_recording
from labeling import label_recording
from windows_v31 import build_window_library_v31
from features import extract_all_features
from config import NODES, RANDOM_STATE

# 加载所有录制并预处理
all_data = {}
for subject in ['owen', 'ryan']:
    all_data[subject] = {}
    from config import RECORDING_MAP
    for rec_name in RECORDING_MAP[subject].keys():
        print(f"[LOAD] {subject} / {rec_name}")
        df = load_recording(subject, rec_name, validate=False)
        df = preprocess_recording(df)
        df = label_recording(df, subject, rec_name)
        # 附加录制信息用于判断离散/连续
        df.attrs['rec_info'] = RECORDING_MAP[subject][rec_name]
        all_data[subject][rec_name] = df

# 构建新版窗口库
print("\n" + "="*60)
print("Building v3.1 window library (Fix 1: anchor windows)")
print("="*60)
lib = build_window_library_v31(all_data)
print(f"\nFinal library: X={lib['X'].shape}, y={lib['y'].shape}")

# 打印分布
from windows import print_class_distribution
print_class_distribution(lib['meta'])

# 保存
import os
out_dir = '/Users/owen/Desktop/lykon-motion-lab-v2/imu_basketball_recognition/data/windows'
os.makedirs(out_dir, exist_ok=True)
np.savez_compressed(os.path.join(out_dir, 'windows_v31.npz'), X=lib['X'], y=lib['y'])
lib['meta'].to_parquet(os.path.join(out_dir, 'meta_v31.parquet'), index=False)
print(f"\nSaved to windows_v31.npz and meta_v31.parquet")

# 特征提取
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
feat_df.to_parquet(os.path.join(out_dir, '../features/features_v31.parquet'), index=False)
print(f"Features: {feat_df.shape}, saved.")
