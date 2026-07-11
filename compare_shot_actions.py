#!/usr/bin/env python3
"""
compare_shot_actions.py — Owen vs Ryan 投篮三动作相似度对比

对比动作:
  1. catch (接球)
  2. dribble_right_once (拍球)
  3. shot (投球/跳投)

相似度: Pearson r (纵向时间序列)
"""
import sys
sys.path.insert(0, '/Users/owen/Desktop/lykon-motion-lab-v2/imu_basketball_recognition/src')

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from io_utils import load_recording
from labeling import label_recording
from preprocess import lowpass
from config import FS, NODES

SENSORS = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
ACTIONS = ['catch', 'dribble_right_once', 'shot']

# ---------------------------------------------------------------------------
# 1. 加载两人数据
# ---------------------------------------------------------------------------
print("=" * 70)
print("Loading recordings...")
print("=" * 70)

owen_df = load_recording('owen', '26071001接球-拍球-跳投', validate=False)
ryan_df = load_recording('ryan', '26071011接球-拍球-投球', validate=False)

print(f"Owen:  {len(owen_df)} samples, t=[{owen_df['t'].iloc[0]:.2f}, {owen_df['t'].iloc[-1]:.2f}] s")
print(f"Ryan:  {len(ryan_df)} samples, t=[{ryan_df['t'].iloc[0]:.2f}, {ryan_df['t'].iloc[-1]:.2f}] s")

# 预处理 a_mag
for df in [owen_df, ryan_df]:
    for node in NODES:
        ax = df[f'ax_{node}'].values
        ay = df[f'ay_{node}'].values
        az = df[f'az_{node}'].values
        acc = np.column_stack([ax, ay, az])
        acc_f = lowpass(acc)
        df[f'a_mag_{node}'] = np.sqrt(acc_f[:, 0]**2 + acc_f[:, 1]**2 + acc_f[:, 2]**2)

# 标注
owen_df = label_recording(owen_df, 'owen', '26071001接球-拍球-跳投')
ryan_df = label_recording(ryan_df, 'ryan', '26071011接球-拍球-投球')

# ---------------------------------------------------------------------------
# 2. 提取各动作的事件段
# ---------------------------------------------------------------------------
def extract_action_segments(df, action_label):
    """提取指定动作的所有连续段."""
    labels = df['label'].values
    segments = []
    start = None
    for i, lbl in enumerate(labels):
        if lbl == action_label and (i == 0 or labels[i-1] != action_label):
            start = i
        elif lbl != action_label and start is not None:
            segments.append((start, i))
            start = None
    if start is not None:
        segments.append((start, len(labels)))
    return segments

# ---------------------------------------------------------------------------
# 3. 计算两人同动作的相似度
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Cross-subject similarity per action")
print("=" * 70)

results = {}

for action in ACTIONS:
    owen_segs = extract_action_segments(owen_df, action)
    ryan_segs = extract_action_segments(ryan_df, action)
    n_pairs = min(len(owen_segs), len(ryan_segs))
    
    print(f"\n>>> {action.upper()}: Owen={len(owen_segs)} segments, Ryan={len(ryan_segs)} segments, comparing {n_pairs} pairs")
    
    all_matrices = []
    durations = []
    
    for pi in range(n_pairs):
        os, oe = owen_segs[pi]
        rs, re = ryan_segs[pi]
        
        owen_seg = owen_df.iloc[os:oe].reset_index(drop=True)
        ryan_seg = ryan_df.iloc[rs:re].reset_index(drop=True)
        
        min_len = min(len(owen_seg), len(ryan_seg))
        if min_len < 5:
            continue
        owen_seg = owen_seg.iloc[:min_len]
        ryan_seg = ryan_seg.iloc[:min_len]
        
        sim_matrix = np.zeros((len(NODES), len(SENSORS)))
        for ni, node in enumerate(NODES):
            for si, sensor in enumerate(SENSORS):
                col = f'{sensor}_{node}'
                o_vals = owen_seg[col].values
                r_vals = ryan_seg[col].values
                
                if np.std(o_vals) < 1e-12 or np.std(r_vals) < 1e-12:
                    r_val = 0.0
                else:
                    r_val, _ = pearsonr(o_vals, r_vals)
                    if np.isnan(r_val):
                        r_val = 0.0
                sim_matrix[ni, si] = r_val
        
        all_matrices.append(sim_matrix)
        durations.append({
            'pair': pi + 1,
            'owen_dur': owen_df['t'].iloc[oe] - owen_df['t'].iloc[os],
            'ryan_dur': ryan_df['t'].iloc[re] - ryan_df['t'].iloc[rs],
        })
    
    if len(all_matrices) == 0:
        print(f"  [WARN] No valid pairs for {action}")
        continue
    
    all_matrices = np.array(all_matrices)
    mean_mat = all_matrices.mean(axis=0)
    std_mat = all_matrices.std(axis=0)
    
    mean_df = pd.DataFrame(mean_mat, index=NODES, columns=SENSORS)
    std_df = pd.DataFrame(std_mat, index=NODES, columns=SENSORS)
    
    results[action] = {
        'mean': mean_df,
        'std': std_df,
        'n_pairs': len(all_matrices),
    }
    
    print(f"\n  Mean Pearson r (n={len(all_matrices)} pairs):")
    print(mean_df.round(3).to_string())
    
    print(f"\n  Std Dev:")
    print(std_df.round(3).to_string())
    
    # 按节点和传感器汇总
    print(f"\n  Per-node avg across sensors:")
    for ni, node in enumerate(NODES):
        vals = mean_mat[ni, :]
        print(f"    {node:6s}: avg={vals.mean():+.3f}  (ax={vals[0]:+.3f} ay={vals[1]:+.3f} az={vals[2]:+.3f} gx={vals[3]:+.3f} gy={vals[4]:+.3f} gz={vals[5]:+.3f})")
    
    print(f"\n  Per-sensor avg across nodes:")
    for si, sensor in enumerate(SENSORS):
        vals = mean_mat[:, si]
        print(f"    {sensor:3s}: avg={vals.mean():+.3f}  (n1={vals[0]:+.3f} n2={vals[1]:+.3f} n3={vals[2]:+.3f} n4={vals[3]:+.3f})")

# ---------------------------------------------------------------------------
# 4. 三动作横向对比汇总
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("CROSS-ACTION SUMMARY")
print("=" * 70)

print("\n{:<20s} {:>8s} {:>10s} {:>10s} {:>10s} {:>10s}".format(
    "Action", "N_pairs", "Node1", "Node2", "Node3", "Node4"))
print("-" * 70)
for action in ACTIONS:
    if action not in results:
        continue
    mean_mat = results[action]['mean'].values
    node_avgs = [mean_mat[i, :].mean() for i in range(4)]
    print("{:<20s} {:>8d} {:>+10.3f} {:>+10.3f} {:>+10.3f} {:>+10.3f}".format(
        action, results[action]['n_pairs'], *node_avgs))

print("\n{:<20s} {:>8s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s}".format(
    "Action", "N_pairs", "ax", "ay", "az", "gx", "gy", "gz"))
print("-" * 90)
for action in ACTIONS:
    if action not in results:
        continue
    mean_mat = results[action]['mean'].values
    sensor_avgs = [mean_mat[:, i].mean() for i in range(6)]
    print("{:<20s} {:>8d} {:>+10.3f} {:>+10.3f} {:>+10.3f} {:>+10.3f} {:>+10.3f} {:>+10.3f}".format(
        action, results[action]['n_pairs'], *sensor_avgs))

# ---------------------------------------------------------------------------
# 5. 热力图
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Generating comparison heatmap...")
print("=" * 70)

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(len(ACTIONS), 2, figsize=(12, 14))

for ai, action in enumerate(ACTIONS):
    if action not in results:
        continue
    mean_mat = results[action]['mean'].values
    std_mat = results[action]['std'].values
    n = results[action]['n_pairs']
    
    # Mean
    im1 = axes[ai, 0].imshow(mean_mat, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
    axes[ai, 0].set_xticks(range(len(SENSORS)))
    axes[ai, 0].set_xticklabels(SENSORS)
    axes[ai, 0].set_yticks(range(len(NODES)))
    axes[ai, 0].set_yticklabels(NODES)
    axes[ai, 0].set_title(f'{action} | Mean r (n={n})')
    for i in range(len(NODES)):
        for j in range(len(SENSORS)):
            axes[ai, 0].text(j, i, f'{mean_mat[i,j]:+.2f}', ha='center', va='center', fontsize=8)
    plt.colorbar(im1, ax=axes[ai, 0])
    
    # Std
    im2 = axes[ai, 1].imshow(std_mat, cmap='Reds', vmin=0, vmax=std_mat.max(), aspect='auto')
    axes[ai, 1].set_xticks(range(len(SENSORS)))
    axes[ai, 1].set_xticklabels(SENSORS)
    axes[ai, 1].set_yticks(range(len(NODES)))
    axes[ai, 1].set_yticklabels(NODES)
    axes[ai, 1].set_title(f'{action} | Std r')
    for i in range(len(NODES)):
        for j in range(len(SENSORS)):
            axes[ai, 1].text(j, i, f'{std_mat[i,j]:.2f}', ha='center', va='center', fontsize=8)
    plt.colorbar(im2, ax=axes[ai, 1])

plt.tight_layout()
out_path = '/Users/owen/Desktop/lykon-motion-lab-v2/shot_action_similarity_owen_vs_ryan.png'
plt.savefig(out_path, dpi=150)
print(f"Saved: {out_path}")

# ---------------------------------------------------------------------------
# 6. 保存 CSV
# ---------------------------------------------------------------------------
for action in ACTIONS:
    if action not in results:
        continue
    csv_path = f'/Users/owen/Desktop/lykon-motion-lab-v2/similarity_{action}_mean.csv'
    results[action]['mean'].to_csv(csv_path)
    print(f"Saved: {csv_path}")

print("\n" + "=" * 70)
print("Done!")
print("=" * 70)
