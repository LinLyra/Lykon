"""姿态几何工具：归一化 + 关节角计算。

姿态序列统一表示为 ndarray，形状 (T, K, 3)，最后一维为 [x, y, conf]，
K = len(KEYPOINTS)，顺序与 config.KEYPOINTS 一致，坐标为图像归一化坐标
(0~1) 或像素坐标均可（会做尺度归一化）。

归一化目的：消除人物体型、在画面中的位置与远近差异，使"专家"模型
可泛化到任意人员。归一化后坐标单位为"躯干长度"，且保留跳跃/运球等
时序动态。
"""
from __future__ import annotations

import numpy as np

from .config import (
    KEYPOINTS, ANGLE_TRIPLES, ANGLE_VERTICAL, ANGLES, MIN_KP_CONFIDENCE,
)

_IDX = {name: i for i, name in enumerate(KEYPOINTS)}


def _kp(seq_xy: np.ndarray, name: str) -> np.ndarray:
    """取某关键点的 (T, 2) 坐标；虚拟点 mid_* 现算。"""
    if name == "mid_hip":
        return 0.5 * (seq_xy[:, _IDX["left_hip"]] + seq_xy[:, _IDX["right_hip"]])
    if name == "mid_shoulder":
        return 0.5 * (seq_xy[:, _IDX["left_shoulder"]] + seq_xy[:, _IDX["right_shoulder"]])
    return seq_xy[:, _IDX[name]]


def clean_confidence(seq: np.ndarray, min_conf: float = MIN_KP_CONFIDENCE) -> np.ndarray:
    """把低置信度关键点坐标置为 NaN，并在时间维线性插值补齐。

    返回形状 (T, K, 2) 的坐标（float），y 轴已翻转为"向上为正"。
    """
    seq = np.asarray(seq, dtype=float)
    T, K, _ = seq.shape
    xy = seq[:, :, :2].copy()
    conf = seq[:, :, 2]
    xy[conf < min_conf] = np.nan
    # y 轴翻转（图像坐标 y 向下）→ 向上为正
    xy[:, :, 1] = -xy[:, :, 1]

    # 时间维插值补 NaN
    t = np.arange(T)
    for k in range(K):
        for c in range(2):
            col = xy[:, k, c]
            good = ~np.isnan(col)
            if good.sum() == 0:
                xy[:, k, c] = 0.0
            elif good.sum() < T:
                xy[:, k, c] = np.interp(t, t[good], col[good])
    return xy


def normalize_pose(seq: np.ndarray) -> np.ndarray:
    """尺度/位置归一化。

    - 原点：首个有效帧的 mid-hip
    - 尺度：整段 clip 的躯干长度（mid_shoulder→mid_hip 距离）中位数
    - 保留时序动态（跳跃的垂直位移、运球的手部周期）

    返回 (T, K, 2)，单位为"躯干长度"。
    """
    xy = clean_confidence(seq)                     # (T, K, 2), y-up
    mid_hip = _kp(xy, "mid_hip")                    # (T, 2)
    mid_sho = _kp(xy, "mid_shoulder")
    torso = np.linalg.norm(mid_sho - mid_hip, axis=1)   # (T,)
    scale = np.median(torso[torso > 1e-6]) if np.any(torso > 1e-6) else 1.0
    scale = float(scale) if scale > 1e-6 else 1.0
    origin = mid_hip[0]                             # (2,)
    return (xy - origin[None, None, :]) / scale


def _angle_at(a: np.ndarray, v: np.ndarray, b: np.ndarray) -> np.ndarray:
    """∠(a - v - b)，返回度数，形状 (T,)。"""
    u = a - v
    w = b - v
    nu = np.linalg.norm(u, axis=1)
    nw = np.linalg.norm(w, axis=1)
    denom = np.clip(nu * nw, 1e-8, None)
    cos = np.clip(np.sum(u * w, axis=1) / denom, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def _angle_vertical(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    """向量 p0→p1 相对竖直向上方向的夹角（度），形状 (T,)。0°=竖直向上。"""
    d = p1 - p0
    # 竖直向上单位向量 (0, 1)（归一化后 y 向上）
    ny = np.linalg.norm(d, axis=1)
    cos = np.clip(d[:, 1] / np.clip(ny, 1e-8, None), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def joint_angles(seq: np.ndarray, normalized: bool = False) -> np.ndarray:
    """从姿态序列计算所有关节角，返回 (T, len(ANGLES))，列顺序同 config.ANGLES。"""
    xy = seq if normalized else normalize_pose(seq)
    T = xy.shape[0]
    out = np.zeros((T, len(ANGLES)), dtype=float)
    col = 0
    for _name, (a, v, b) in ANGLE_TRIPLES.items():
        out[:, col] = _angle_at(_kp(xy, a), _kp(xy, v), _kp(xy, b))
        col += 1
    for _name, (p0, p1) in ANGLE_VERTICAL.items():
        out[:, col] = _angle_vertical(_kp(xy, p0), _kp(xy, p1))
        col += 1
    return out
