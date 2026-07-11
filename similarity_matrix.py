#!/usr/bin/env python3
"""
similarity_matrix.py — Owen vs Ryan 高位传球纵向相似度矩阵

纵向 = 时间序列维度 (t_rel_s)
相似度 = 皮尔逊相关系数 (Pearson r)

输出:
  1. 按事件段 (pass_high cycle) 的节点-通道相似度矩阵
  2. 汇总均值矩阵
  3. 热力图 PNG + CSV
"""
import sys
sys.path.insert(0, '/Users/owen/Desktop/lykon-motion-lab-v2/imu_basketball_recognition/src')

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from io_utils import load_recording
from labeling import label_recording
from preprocess import lowpass
from config import FS

NODES = ['node1', 'node2', 'node3', 'node4']
SENSORS = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']

# ---------------------------------------------------------------------------
# 1. 加载两人高位传球数据
# ---------------------------------------------------------------------------
print("=" * 70)
print("Loading high-pass recordings...")
print("=" * 70)

owen_df = load_recording('owen', '26071004高位传球', validate=False)
ryan_df = load_recording('ryan', '26071014高位传球', validate=False)

print(f"Owen:  {len(owen_df)} samples, t=[{owen_df['t'].iloc[0]:.2f}, {owen_df['t'].iloc[-1]:.2f}] s")
print(f"Ryan:  {len(ryan_df)} samples, t=[{ryan_df['t'].iloc[0]:.2f}, {ryan_df['t'].iloc[-1]:.2f}] s")

# ---------------------------------------------------------------------------
# 2. 预处理: 计算 a_mag (labeling.py 需要 a_mag_node2 列)
# ---------------------------------------------------------------------------
for df in [owen_df, ryan_df]:
    for node in NODES:
        ax = df[f'ax_{node}'].values
        ay = df[f'ay_{node}'].values
        az = df[f'az_{node}'].values
        acc = np.column_stack([ax, ay, az])
        acc_f = lowpass(acc)
        df[f'a_mag_{node}'] = np.sqrt(acc_f[:, 0]**2 + acc_f[:, 1]**2 + acc_f[:, 2]**2)

# ---------------------------------------------------------------------------
# 3. 标注事件段
# ---------------------------------------------------------------------------
owen_df = label_recording(owen_df, 'owen', '26071004高位传球')
ryan_df = label_recording(ryan_df, 'ryan', '26071014高位传球')

print(f"\nOwen labels: {owen_df['label'].value_counts().to_dict()}")
print(f"Ryan labels: {ryan_df['label'].value_counts().to_dict()}")

# ---------------------------------------------------------------------------
# 4. 提取 pass_high 事件段
# ---------------------------------------------------------------------------
def extract_event_segments(df):
    """从标注后的 DataFrame 中提取连续的 pass_high 事件段."""
    labels = df['label'].values
    segments = []
    start = None
    for i, lbl in enumerate(labels):
        if lbl == 'pass_high' and (i == 0 or labels[i-1] != 'pass_high'):
            start = i
        elif lbl != 'pass_high' and start is not None:
            segments.append((start, i))
            start = None
    if start is not None:
        segments.append((start, len(labels)))
    return segments

owen_segments = extract_event_segments(owen_df)
ryan_segments = extract_event_segments(ryan_df)

print(f"\nOwen pass_high segments: {len(owen_segments)}")
print(f"Ryan pass_high segments: {len(ryan_segments)}")

# ---------------------------------------------------------------------------
# 5. 计算每个事件段的皮尔逊相似度
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Computing Pearson similarity matrices per event segment")
print("=" * 70)

n_segments = min(len(owen_segments), len(ryan_segments))
print(f"Comparing first {n_segments} paired segments")

all_matrices = []

for seg_idx in range(n_segments):
    os, oe = owen_segments[seg_idx]
    rs, re = ryan_segments[seg_idx]

    owen_seg = owen_df.iloc[os:oe].reset_index(drop=True)
    ryan_seg = ryan_df.iloc[rs:re].reset_index(drop=True)

    min_len = min(len(owen_seg), len(ryan_seg))
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
    print(f"\nSegment {seg_idx+1:02d} | Owen t=[{owen_df['t'].iloc[os]:.2f}, {owen_df['t'].iloc[oe]:.2f}] | "
          f"Ryan t=[{ryan_df['t'].iloc[rs]:.2f}, {ryan_df['t'].iloc[re]:.2f}] | len={min_len}")
    mat_df = pd.DataFrame(sim_matrix, index=NODES, columns=SENSORS)
    print(mat_df.round(3).to_string())

# ---------------------------------------------------------------------------
# 6. 汇总：均值 + 标准差矩阵
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SUMMARY: Mean Pearson Correlation across all segments")
print("=" * 70)

all_matrices = np.array(all_matrices)
mean_mat = all_matrices.mean(axis=0)
std_mat = all_matrices.std(axis=0)

mean_df = pd.DataFrame(mean_mat, index=NODES, columns=SENSORS)
std_df = pd.DataFrame(std_mat, index=NODES, columns=SENSORS)

print("\n--- Mean Correlation (r) ---")
print(mean_df.round(3).to_string())

print("\n--- Std Dev of Correlation ---")
print(std_df.round(3).to_string())

# ---------------------------------------------------------------------------
# 7. 按传感器通道的跨节点均值
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Per-sensor mean across nodes")
print("=" * 70)
for si, sensor in enumerate(SENSORS):
    vals = mean_mat[:, si]
    print(f"{sensor:3s}:  node1={vals[0]:+.3f}  node2={vals[1]:+.3f}  node3={vals[2]:+.3f}  node4={vals[3]:+.3f}  | avg={vals.mean():+.3f}")

# ---------------------------------------------------------------------------
# 8. 按节点的跨传感器均值
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Per-node mean across sensors")
print("=" * 70)
for ni, node in enumerate(NODES):
    vals = mean_mat[ni, :]
    print(f"{node:6s}:  ax={vals[0]:+.3f}  ay={vals[1]:+.3f}  az={vals[2]:+.3f}  "
          f"gx={vals[3]:+.3f}  gy={vals[4]:+.3f}  gz={vals[5]:+.3f}  | avg={vals.mean():+.3f}")

# ---------------------------------------------------------------------------
# 9. 热力图可视化
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Generating heatmap...")
print("=" * 70)

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

im1 = axes[0].imshow(mean_mat, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
axes[0].set_xticks(range(len(SENSORS)))
axes[0].set_xticklabels(SENSORS)
axes[0].set_yticks(range(len(NODES)))
axes[0].set_yticklabels(NODES)
axes[0].set_title(f'Mean Pearson r (n={n_segments} segments)')
for i in range(len(NODES)):
    for j in range(len(SENSORS)):
        axes[0].text(j, i, f'{mean_mat[i,j]:+.2f}', ha='center', va='center', fontsize=9)
plt.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(std_mat, cmap='Reds', vmin=0, vmax=std_mat.max(), aspect='auto')
axes[1].set_xticks(range(len(SENSORS)))
axes[1].set_xticklabels(SENSORS)
axes[1].set_yticks(range(len(NODES)))
axes[1].set_yticklabels(NODES)
axes[1].set_title('Std Dev of Pearson r')
for i in range(len(NODES)):
    for j in range(len(SENSORS)):
        axes[1].text(j, i, f'{std_mat[i,j]:.2f}', ha='center', va='center', fontsize=9)
plt.colorbar(im2, ax=axes[1])

plt.tight_layout()
out_path = '/Users/owen/Desktop/lykon-motion-lab-v2/similarity_matrix_owen_vs_ryan.png'
plt.savefig(out_path, dpi=150)
print(f"Saved heatmap: {out_path}")

# ---------------------------------------------------------------------------
# 10. 保存数值 CSV
# ---------------------------------------------------------------------------
csv_path = '/Users/owen/Desktop/lykon-motion-lab-v2/similarity_matrix_mean.csv'
mean_df.to_csv(csv_path)
print(f"Saved CSV: {csv_path}")

print("\n" + "=" * 70)
print("Done!")
print("=" * 70)
