"""
run_phase3.py — End-to-end: load → preprocess → label → windows
"""
import sys
sys.path.insert(0, '/Users/owen/Desktop/lykon-motion-lab-v2/imu_basketball_recognition/src')

import numpy as np
from io_utils import load_recording
from preprocess import preprocess_recording
from labeling import label_recording
from windows import build_window_library, print_class_distribution
from config import RECORDING_MAP, SUBJECTS

# Process all recordings
all_data = {}
for subject in SUBJECTS:
    all_data[subject] = {}
    for rec_name in RECORDING_MAP[subject].keys():
        print(f"\n{'='*60}")
        print(f"Processing {subject} / {rec_name}")
        print('='*60)

        # Load
        df = load_recording(subject, rec_name, validate=False)
        print(f"  Loaded: {df.shape}")

        # Preprocess
        df = preprocess_recording(df)
        print(f"  Preprocessed: {df.shape}")

        # Label
        df = label_recording(df, subject, rec_name)
        print(f"  Labeled: unique labels = {df['label'].unique()}")

        all_data[subject][rec_name] = df

# Build window library
print("\n" + "="*60)
print("Building window library")
print("="*60)
lib = build_window_library(all_data)
print(f"\nFinal library: X={lib['X'].shape}, y={lib['y'].shape}")
print_class_distribution(lib['meta'])

# Save
import os
out_dir = '/Users/owen/Desktop/lykon-motion-lab-v2/imu_basketball_recognition/data/windows'
os.makedirs(out_dir, exist_ok=True)
np.savez_compressed(
    os.path.join(out_dir, 'windows.npz'),
    X=lib['X'],
    y=lib['y'],
)
lib['meta'].to_parquet(os.path.join(out_dir, 'meta.parquet'), index=False)
print(f"\nSaved to {out_dir}/windows.npz and meta.parquet")
