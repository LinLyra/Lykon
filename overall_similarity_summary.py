#!/usr/bin/env python3
"""
汇总 Owen vs Ryan 投篮三动作的总体相似度水平
"""
import sys
sys.path.insert(0, '/Users/owen/Desktop/lykon-motion-lab-v2/imu_basketball_recognition/src')

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from io_utils import load_recording
from labeling import label_recording
from preprocess import lowpass
from config import NODES

SENSORS = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
ACTIONS = ['catch', 'dribble_right_once', 'shot']

# 加载
owen_df = load_recording('owen', '26071001接球-拍球-跳投', validate=False)
ryan_df = load_recording('ryan', '26071011接球-拍球-投球', validate=False)

for df in [owen_df, ryan_df]:
    for node in NODES:
        ax = df[f'ax_{node}'].values
        ay = df[f'ay_{node}'].values
        az = df[f'az_{node}'].values
        acc = np.column_stack([ax, ay, az])
        acc_f = lowpass(acc)
        df[f'a_mag_{node}'] = np.sqrt(acc_f[:, 0]**2 + acc_f[:, 1]**2 + acc_f[:, 2]**2)

owen_df = label_recording(owen_df, 'owen', '26071001接球-拍球-跳投')
ryan_df = label_recording(ryan_df, 'ryan', '26071011接球-拍球-投球')

def extract_segments(df, action):
    labels = df['label'].values
    segs = []
    start = None
    for i, lbl in enumerate(labels):
        if lbl == action and (i == 0 or labels[i-1] != action):
            start = i
        elif lbl != action and start is not None:
            segs.append((start, i))
            start = None
    if start is not None:
        segs.append((start, len(labels)))
    return segs

all_r = []
per_action = {}

for action in ACTIONS:
    owen_segs = extract_segments(owen_df, action)
    ryan_segs = extract_segments(ryan_df, action)
    n_pairs = min(len(owen_segs), len(ryan_segs))
    action_r = []
    for pi in range(n_pairs):
        os, oe = owen_segs[pi]
        rs, re = ryan_segs[pi]
        o_seg = owen_df.iloc[os:oe].reset_index(drop=True)
        r_seg = ryan_df.iloc[rs:re].reset_index(drop=True)
        ml = min(len(o_seg), len(r_seg))
        if ml < 5:
            continue
        o_seg = o_seg.iloc[:ml]
        r_seg = r_seg.iloc[:ml]
        for node in NODES:
            for sensor in SENSORS:
                col = f'{sensor}_{node}'
                ov = o_seg[col].values
                rv = r_seg[col].values
                if np.std(ov) < 1e-12 or np.std(rv) < 1e-12:
                    continue
                r_val, _ = pearsonr(ov, rv)
                if not np.isnan(r_val):
                    action_r.append(r_val)
                    all_r.append(r_val)
    per_action[action] = np.array(action_r)

all_r = np.array(all_r)

print("=" * 60)
print("Owen vs Ryan 投篮三动作 — 总体相似度汇总")
print("=" * 60)

print(f"\n【全局总体】")
print(f"  有效相关系数总数: {len(all_r)}")
print(f"  全局平均 Pearson r: {all_r.mean():+.4f}")
print(f"  中位数: {np.median(all_r):+.4f}")
print(f"  标准差: {all_r.std():.4f}")
print(f"  范围: [{all_r.min():+.4f}, {all_r.max():+.4f}]")
print(f"  |r| > 0.3: {(np.abs(all_r) > 0.3).mean()*100:.1f}%")
print(f"  |r| > 0.5: {(np.abs(all_r) > 0.5).mean()*100:.1f}%")
print(f"  正相关比例: {(all_r > 0).mean()*100:.1f}%")

for action in ACTIONS:
    r = per_action[action]
    print(f"\n【{action}】")
    print(f"  有效 r 数: {len(r)}")
    print(f"  平均: {r.mean():+.4f}")
    print(f"  中位数: {np.median(r):+.4f}")
    print(f"  标准差: {r.std():.4f}")
    print(f"  |r|>0.3: {(np.abs(r) > 0.3).mean()*100:.1f}%")

# 按节点汇总
print(f"\n{'=' * 60}")
print("按节点汇总 (全部动作合并)")
print("=" * 60)
node_r = {n: [] for n in NODES}
for action in ACTIONS:
    owen_segs = extract_segments(owen_df, action)
    ryan_segs = extract_segments(ryan_df, action)
    n_pairs = min(len(owen_segs), len(ryan_segs))
    for pi in range(n_pairs):
        os, oe = owen_segs[pi]
        rs, re = ryan_segs[pi]
        o_seg = owen_df.iloc[os:oe].reset_index(drop=True)
        r_seg = ryan_df.iloc[rs:re].reset_index(drop=True)
        ml = min(len(o_seg), len(r_seg))
        if ml < 5:
            continue
        o_seg = o_seg.iloc[:ml]
        r_seg = r_seg.iloc[:ml]
        for node in NODES:
            vals = []
            for sensor in SENSORS:
                col = f'{sensor}_{node}'
                ov = o_seg[col].values
                rv = r_seg[col].values
                if np.std(ov) < 1e-12 or np.std(rv) < 1e-12:
                    continue
                r_val, _ = pearsonr(ov, rv)
                if not np.isnan(r_val):
                    vals.append(r_val)
            if vals:
                node_r[node].append(np.mean(vals))

for node in NODES:
    r = np.array(node_r[node])
    print(f"  {node}: avg={r.mean():+.4f}, median={np.median(r):+.4f}, std={r.std():.4f}, n={len(r)}")

# 按传感器汇总
print(f"\n{'=' * 60}")
print("按传感器汇总 (全部动作合并)")
print("=" * 60)
sensor_r = {s: [] for s in SENSORS}
for action in ACTIONS:
    owen_segs = extract_segments(owen_df, action)
    ryan_segs = extract_segments(ryan_df, action)
    n_pairs = min(len(owen_segs), len(ryan_segs))
    for pi in range(n_pairs):
        os, oe = owen_segs[pi]
        rs, re = ryan_segs[pi]
        o_seg = owen_df.iloc[os:oe].reset_index(drop=True)
        r_seg = ryan_df.iloc[rs:re].reset_index(drop=True)
        ml = min(len(o_seg), len(r_seg))
        if ml < 5:
            continue
        o_seg = o_seg.iloc[:ml]
        r_seg = r_seg.iloc[:ml]
        for sensor in SENSORS:
            vals = []
            for node in NODES:
                col = f'{sensor}_{node}'
                ov = o_seg[col].values
                rv = r_seg[col].values
                if np.std(ov) < 1e-12 or np.std(rv) < 1e-12:
                    continue
                r_val, _ = pearsonr(ov, rv)
                if not np.isnan(r_val):
                    vals.append(r_val)
            if vals:
                sensor_r[sensor].append(np.mean(vals))

for sensor in SENSORS:
    r = np.array(sensor_r[sensor])
    print(f"  {sensor}: avg={r.mean():+.4f}, median={np.median(r):+.4f}, std={r.std():.4f}, n={len(r)}")

print(f"\n{'=' * 60}")
print("结论")
print("=" * 60)
print(f"  总体水平: r = {all_r.mean():+.4f} (接近 0，几乎不相似)")
print(f"  最相似动作: {max([(a, per_action[a].mean()) for a in ACTIONS], key=lambda x: x[1])[0]}")
print(f"  最相似节点: {max([(n, np.array(node_r[n]).mean()) for n in NODES], key=lambda x: x[1])[0]}")
print(f"  最相似传感器: {max([(s, np.array(sensor_r[s]).mean()) for s in SENSORS], key=lambda x: x[1])[0]}")
