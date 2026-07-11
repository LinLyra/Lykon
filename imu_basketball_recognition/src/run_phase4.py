import sys
sys.path.insert(0, '/Users/owen/Desktop/lykon-motion-lab-v2/imu_basketball_recognition/src')

import numpy as np
import pandas as pd
from features import extract_all_features

# Load window library
lib = np.load('/Users/owen/Desktop/lykon-motion-lab-v2/imu_basketball_recognition/data/windows/windows.npz')
X = lib['X']
y = lib['y']
meta = pd.read_parquet('/Users/owen/Desktop/lykon-motion-lab-v2/imu_basketball_recognition/data/windows/meta.parquet')

print(f"Input: X={X.shape}, y={y.shape}")
print(f"Channels: {meta.columns if hasattr(meta, 'columns') else 'N/A'}")

# We need channel names. They're not saved in meta. Let's reconstruct from the preprocessing pipeline.
# From preprocess.py, the channels added are:
# L1: ax_node1, ay_node1, az_node1, gx_node1, ... for all nodes
# L2: a_vert_node1, a_horiz_node1, ...
# L3: a_mag_node1, g_mag_node1, ...
# Attitude: roll_node1, pitch_node1, yaw_node1, ...
# Jerk: jerk_x_node1, jerk_y_node1, jerk_z_node1, jerk_vert_node1, jerk_mag_node1, ...

from config import NODES
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

print(f"Expected channels: {len(ch_names)}, X has: {X.shape[2]}")

# Extract features
feat_df = extract_all_features(X, ch_names)
print(f"Features extracted: {feat_df.shape}")

# Save
feat_df.to_parquet('/Users/owen/Desktop/lykon-motion-lab-v2/imu_basketball_recognition/data/features/features.parquet', index=False)
print("Saved features to data/features/features.parquet")
