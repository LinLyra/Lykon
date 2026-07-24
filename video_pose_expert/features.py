"""从姿态序列提取动作特征向量。

一个"样本"是一段姿态序列 seq，形状 (T, K, 3)（[x,y,conf]）：
    - 预分割数据集（SpaceJam / 合成）：整段 clip = 一个样本
    - 连续视频：由 windowing 切成滑窗，每个滑窗 = 一个样本

特征分两类，均为尺度/位置无关（配合 geometry.normalize_pose）：
    1. 关节角统计（11 个角 × 多个统计量）
    2. 运动学信号统计（手腕相对高度、髋垂直位移=跳跃、踝速度=冲刺、
       左右对称性、整体运动能量等篮球动作判别信号）

统计量沿用 src/lykon_motion/features/extract.py 的思路
（mean/std/min/max/ptp/dominant_freq/peak_count）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from .config import ANGLES, DEFAULT_FPS
from .geometry import normalize_pose, joint_angles


def _dominant_freq(x: np.ndarray, fs: float) -> float:
    x = np.asarray(x, dtype=float)
    if len(x) < 4:
        return 0.0
    x = x - np.mean(x)
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
    if len(spec) <= 1:
        return 0.0
    idx = int(np.argmax(spec[1:])) + 1
    return float(freqs[idx])


def _peak_count(x: np.ndarray, fs: float) -> int:
    if len(x) < 4:
        return 0
    peaks, _ = find_peaks(np.asarray(x, dtype=float), distance=max(1, int(fs // 6)))
    return int(len(peaks))


def _stats(series: np.ndarray, fs: float, prefix: str, full: bool = True) -> dict:
    s = np.asarray(series, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        s = np.zeros(1)
    out = {
        f"{prefix}_mean": float(np.mean(s)),
        f"{prefix}_std": float(np.std(s)),
        f"{prefix}_max": float(np.max(s)),
    }
    if full:
        out.update({
            f"{prefix}_min": float(np.min(s)),
            f"{prefix}_ptp": float(np.ptp(s)),
            f"{prefix}_dom_freq": _dominant_freq(s, fs),
            f"{prefix}_peak_count": _peak_count(np.abs(s - np.mean(s)), fs),
        })
    return out


def _speed(xy: np.ndarray, fs: float) -> np.ndarray:
    """逐帧位移速度，xy 形状 (T,2)，返回 (T-1,)（单位：躯干长度/秒）。"""
    if xy.shape[0] < 2:
        return np.zeros(1)
    d = np.diff(xy, axis=0)
    return np.linalg.norm(d, axis=1) * fs


def sequence_features(seq: np.ndarray, fps: float = DEFAULT_FPS) -> dict:
    """把一段姿态序列转为特征字典。"""
    xy = normalize_pose(seq)                       # (T, K, 2)，躯干单位、y 向上
    ang = joint_angles(xy, normalized=True)        # (T, A)
    from .geometry import _kp                       # 复用关键点取用

    feats: dict = {}

    # ---- 1. 关节角统计 ----
    for i, name in enumerate(ANGLES):
        feats.update(_stats(ang[:, i], fps, f"ang_{name}", full=True))

    # ---- 2. 运动学信号 ----
    lsho = _kp(xy, "left_shoulder"); rsho = _kp(xy, "right_shoulder")
    lwri = _kp(xy, "left_wrist");    rwri = _kp(xy, "right_wrist")
    lank = _kp(xy, "left_ankle");    rank = _kp(xy, "right_ankle")
    mid_hip = _kp(xy, "mid_hip")

    # 手腕相对肩部的高度（投篮举臂/随球、运球下压的核心信号）
    lw_rel_y = lwri[:, 1] - lsho[:, 1]
    rw_rel_y = rwri[:, 1] - rsho[:, 1]
    feats.update(_stats(lw_rel_y, fps, "lwrist_rely", full=True))
    feats.update(_stats(rw_rel_y, fps, "rwrist_rely", full=True))
    # 双手最高点（投篮/传球出手）
    feats["wrist_rely_max"] = float(np.max([lw_rel_y.max(), rw_rel_y.max()]))

    # 髋垂直位移（跳跃：起跳上冲 + 落地）
    hip_y = mid_hip[:, 1]
    feats.update(_stats(hip_y, fps, "hip_y", full=True))
    hip_v = np.diff(hip_y) * fps if len(hip_y) > 1 else np.zeros(1)
    feats["hip_up_vel_max"] = float(np.max(hip_v)) if hip_v.size else 0.0
    feats["hip_down_vel_max"] = float(-np.min(hip_v)) if hip_v.size else 0.0

    # 髋水平位移速度（冲刺/突破）
    hip_x_speed = np.abs(np.diff(mid_hip[:, 0])) * fps if len(mid_hip) > 1 else np.zeros(1)
    feats.update(_stats(hip_x_speed, fps, "hip_xspeed", full=False))

    # 踝速度 / 步频（跑动、冲刺）
    lank_sp = _speed(lank, fps); rank_sp = _speed(rank, fps)
    feats.update(_stats((lank_sp + rank_sp) / 2.0, fps, "ankle_speed", full=True))

    # 手腕速度与周期（运球高频、传球爆发）
    lwri_sp = _speed(lwri, fps); rwri_sp = _speed(rwri, fps)
    feats.update(_stats(np.maximum(lwri_sp, rwri_sp), fps, "wrist_speed", full=True))

    # 左右对称性（传球=双臂协同；运球/上篮=单侧主导）
    feats["arm_symmetry"] = float(np.mean(np.abs(lw_rel_y - rw_rel_y)))
    feats["wrist_speed_asym"] = float(np.abs(np.mean(lwri_sp) - np.mean(rwri_sp)))

    # 整体运动能量（idle 低、sprint/jump 高）
    all_speed = np.stack([_speed(xy[:, k], fps) for k in range(xy.shape[1])], axis=0)
    feats["body_energy_mean"] = float(np.mean(all_speed))
    feats["body_energy_max"] = float(np.max(all_speed))

    return feats


def features_dataframe(samples: list[dict], fps: float = DEFAULT_FPS) -> pd.DataFrame:
    """把一批样本转成特征 DataFrame。

    每个样本 dict 至少含：
        seq  : (T,K,3) 姿态序列
        label: 动作标签（训练时需要；推理时可缺省为 None）
    其余键（session_id/clip_id/player_id/...）会原样带入 meta 列。
    """
    rows = []
    for s in samples:
        seq = s["seq"]
        row = {k: v for k, v in s.items() if k != "seq"}
        row.update(sequence_features(seq, fps=s.get("fps", fps)))
        rows.append(row)
    return pd.DataFrame(rows)


META_COLS = [
    "label", "session_id", "clip_id", "player_id", "video_id",
    "start_frame", "end_frame", "start_time_us", "end_time_us", "source", "fps",
]


def feature_columns(df: pd.DataFrame) -> list[str]:
    """返回纯特征列（排除 meta 与非数值列）。"""
    cols = [c for c in df.columns if c not in META_COLS]
    num = df[cols].select_dtypes(include=["number"]).columns
    return list(num)
