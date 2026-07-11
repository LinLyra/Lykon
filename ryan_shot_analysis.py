#!/usr/bin/env python3
"""
分析 Ryan 投篮文件 (26071011接球-拍球-投球) 的动作分割情况
"""
import sys
sys.path.insert(0, '/Users/owen/Desktop/lykon-motion-lab-v2/imu_basketball_recognition/src')

import numpy as np
import pandas as pd
from io_utils import load_recording
from labeling import label_recording
from preprocess import lowpass
from config import FS, NODES

# 加载数据
print("=" * 70)
print("Loading Ryan 投篮文件: 26071011接球-拍球-投球")
print("=" * 70)

df = load_recording('ryan', '26071011接球-拍球-投球', validate=False)
print(f"总样本数: {len(df)} | 时间范围: {df['t'].iloc[0]:.2f}s ~ {df['t'].iloc[-1]:.2f}s")

# 预处理 a_mag (labeling 需要)
for node in NODES:
    ax = df[f'ax_{node}'].values
    ay = df[f'ay_{node}'].values
    az = df[f'az_{node}'].values
    acc = np.column_stack([ax, ay, az])
    acc_f = lowpass(acc)
    df[f'a_mag_{node}'] = np.sqrt(acc_f[:, 0]**2 + acc_f[:, 1]**2 + acc_f[:, 2]**2)

# 标注
df = label_recording(df, 'ryan', '26071011接球-拍球-投球')

# 统计各标签
print(f"\n标签分布:")
for lbl, cnt in df['label'].value_counts().sort_index().items():
    print(f"  {lbl}: {cnt} samples ({cnt/len(df)*100:.1f}%)")

# 提取事件段
def extract_segments(df):
    labels = df['label'].values
    t = df['t'].values
    segments = []
    start = 0
    current = labels[0]
    for i in range(1, len(labels)):
        if labels[i] != current:
            segments.append({
                'label': current,
                'start_idx': start,
                'end_idx': i,
                'start_t': t[start],
                'end_t': t[i-1],
                'duration': t[i-1] - t[start],
                'n_samples': i - start,
            })
            start = i
            current = labels[i]
    segments.append({
        'label': current,
        'start_idx': start,
        'end_idx': len(labels),
        'start_t': t[start],
        'end_t': t[-1],
        'duration': t[-1] - t[start],
        'n_samples': len(labels) - start,
    })
    return segments

segments = extract_segments(df)
seg_df = pd.DataFrame(segments)

print(f"\n共提取 {len(segments)} 个连续段:")
print(seg_df[['label', 'start_t', 'end_t', 'duration', 'n_samples']].to_string(index=False))

# 按 cycle 分组统计 (接球-拍球-投球 = 3个动作一周期)
print(f"\n{'=' * 70}")
print("按动作周期统计")
print("=" * 70)

cycles = []
action_segments = [s for s in segments if s['label'] != 'null']
for i, seg in enumerate(action_segments):
    cycle_idx = i // 3
    action_in_cycle = i % 3
    action_names = ['catch', 'dribble_right_once', 'shot']
    expected = action_names[action_in_cycle]
    cycles.append({
        'cycle': cycle_idx + 1,
        'action_idx': action_in_cycle + 1,
        'expected': expected,
        'actual': seg['label'],
        'match': '✓' if seg['label'] == expected else '✗',
        'start_t': seg['start_t'],
        'duration': seg['duration'],
        'n_samples': seg['n_samples'],
    })

cycle_df = pd.DataFrame(cycles)
print(cycle_df.to_string(index=False))

# 汇总每周期
cycle_summary = cycle_df.groupby('cycle').agg(
    total_duration=('duration', 'sum'),
    total_samples=('n_samples', 'sum'),
    actions=('actual', lambda x: ' → '.join(x)),
).reset_index()
print(f"\n周期汇总:")
print(cycle_summary.to_string(index=False))

print(f"\n{'=' * 70}")
print("Summary")
print("=" * 70)
print(f"  总事件段: {len(segments)} (含 null 背景)")
print(f"  动作段: {len(action_segments)} (不含 null)")
print(f"  完整周期: {len(action_segments) // 3} (每周期3个动作)")
print(f"  多余动作: {len(action_segments) % 3}")
print(f"  平均动作时长: {cycle_df['duration'].mean():.2f}s")
print(f"  最短动作: {cycle_df['duration'].min():.2f}s")
print(f"  最长动作: {cycle_df['duration'].max():.2f}s")
