"""
preprocess.py — 预处理: 滤波、姿态解算、三层通道构建

三层通道:
  L1 (sensor frame): ax, ay, az, gx, gy, gz 每节点
  L2 (gravity frame): a_vert, a_horiz 每节点
  L3 (magnitude): |a|, |g| 每节点

接口:
  preprocess(nodes_dict) -> dict[node_id, DataFrame]
  check_static_baseline(df_static) -> dict
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from config import (
    FS, DT, NODES, NODE_NAMES,
    ACC_SCALE, GYRO_SCALE_RAD, G_STD,
    COMPLEMENTARY_ALPHA,
    FILTER_LOWPASS_CUTOFF, FILTER_LOWPASS_ORDER,
    FILTER_HIGHPASS_CUTOFF, FILTER_HIGHPASS_ORDER,
)


# ──────────────────────────────
# 滤波器系数缓存
# ──────────────────────────────
def _design_butter(cutoff, order, btype="low"):
    nyq = 0.5 * FS
    normal_cutoff = np.array(cutoff) / nyq
    b, a = butter(order, normal_cutoff, btype=btype, analog=False)
    return b, a


_LP_B, _LP_A = _design_butter(FILTER_LOWPASS_CUTOFF, FILTER_LOWPASS_ORDER, "low")
_HP_B, _HP_A = _design_butter(FILTER_HIGHPASS_CUTOFF, FILTER_HIGHPASS_ORDER, "high")


def _filt(data: np.ndarray, b, a) -> np.ndarray:
    if len(data) < max(len(b), len(a)) * 3:
        return data.copy()
    return filtfilt(b, a, data)


# ──────────────────────────────
# 互补滤波姿态解算 (单节点)
# ──────────────────────────────
def _complementary_filter(
    ax: np.ndarray, ay: np.ndarray, az: np.ndarray,
    gx: np.ndarray, gy: np.ndarray, gz: np.ndarray,
    alpha: float = COMPLEMENTARY_ALPHA,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    简易互补滤波，估计 roll / pitch（rad）。
    返回: (roll, pitch, yaw_placeholder)
    yaw 不可观，置 0。
    """
    n = len(ax)
    roll = np.zeros(n)
    pitch = np.zeros(n)

    for i in range(1, n):
        # 加速度估计 roll/pitch（低频）
        acc_roll = np.arctan2(ay[i], az[i])
        acc_pitch = np.arctan2(-ax[i], np.sqrt(ay[i] ** 2 + az[i] ** 2))

        # 陀螺仪积分（高频）
        roll[i] = alpha * (roll[i - 1] + gx[i] * DT) + (1 - alpha) * acc_roll
        pitch[i] = alpha * (pitch[i - 1] + gy[i] * DT) + (1 - alpha) * acc_pitch

    return roll, pitch, np.zeros(n)


# ──────────────────────────────
# 重力系分解 L2
# ──────────────────────────────
def _gravity_decomposition(
    ax: np.ndarray, ay: np.ndarray, az: np.ndarray,
    roll: np.ndarray, pitch: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    用估计的 roll/pitch 把加速度从 sensor frame 旋转到 world frame。
    World frame: z 轴向上（反重力）。
    去除重力后返回:
      a_vert: 竖直分量（向上为正，静止≈0）
      a_horiz: 水平面内模
    """
    n = len(ax)
    a_vert = np.zeros(n)
    a_horiz = np.zeros(n)

    for i in range(n):
        cr, sr = np.cos(roll[i]), np.sin(roll[i])
        cp, sp = np.cos(pitch[i]), np.sin(pitch[i])

        # 旋转矩阵 R = R_x(roll) * R_y(pitch), 将 sensor->world
        # 先做 pitch 旋转，再做 roll 旋转
        # 实际上我们 want world = R_y(-pitch) @ R_x(-roll) @ sensor
        # 简化：分别投影
        # world_x = ax*cp + az*sp
        # world_y = ay*cr + (az*cp - ax*sp)*sr
        # world_z = -ay*sr + (az*cp - ax*sp)*cr

        world_z = -ay[i] * sr + (az[i] * cp - ax[i] * sp) * cr
        world_x = ax[i] * cp + az[i] * sp
        world_y = ay[i] * cr + (az[i] * cp - ax[i] * sp) * sr

        a_vert[i] = world_z - G_STD   # 去除重力，向上为正
        a_horiz[i] = np.sqrt(world_x ** 2 + world_y ** 2)

    return a_vert, a_horiz


# ──────────────────────────────
# 单节点预处理
# ──────────────────────────────
def _preprocess_node(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # 单位转换
    for col in ["ax", "ay", "az"]:
        if col in out.columns:
            out[col] = out[col].values * ACC_SCALE
    for col in ["gx", "gy", "gz"]:
        if col in out.columns:
            out[col] = out[col].values * GYRO_SCALE_RAD

    # L1: 低通 + 高通（仅加速度）
    for col in ["ax", "ay", "az"]:
        if col in out.columns:
            out[f"{col}_lp"] = _filt(out[col].values, _LP_B, _LP_A)
            out[f"{col}_hp"] = _filt(out[col].values, _HP_B, _HP_A)
    for col in ["gx", "gy", "gz"]:
        if col in out.columns:
            out[f"{col}_lp"] = _filt(out[col].values, _LP_B, _LP_A)
            # 陀螺仪不做高通（避免漂移放大）

    # L3 模值 (原始值)
    out["a_norm"] = np.sqrt(out["ax"] ** 2 + out["ay"] ** 2 + out["az"] ** 2)
    out["g_norm"] = np.sqrt(out["gx"] ** 2 + out["gy"] ** 2 + out["gz"] ** 2)

    # 姿态解算
    roll, pitch, yaw = _complementary_filter(
        out["ax"].values, out["ay"].values, out["az"].values,
        out["gx"].values, out["gy"].values, out["gz"].values,
    )
    out["roll"] = np.degrees(roll)
    out["pitch"] = np.degrees(pitch)
    out["yaw"] = np.degrees(yaw)

    # L2: 重力系分解
    a_vert, a_horiz = _gravity_decomposition(
        out["ax"].values, out["ay"].values, out["az"].values,
        roll, pitch,
    )
    out["a_vert"] = a_vert
    out["a_horiz"] = a_horiz

    # L2 高通（仅竖直分量，捕捉冲击）
    out["a_vert_hp"] = _filt(a_vert, _HP_B, _HP_A)

    # L3 模值 (低通)
    out["a_norm_lp"] = np.sqrt(out["ax_lp"] ** 2 + out["ay_lp"] ** 2 + out["az_lp"] ** 2)

    # jerk (基于 a_vert 和 |a|)
    out["jerk_vert"] = np.gradient(a_vert) * FS
    out["jerk_norm"] = np.gradient(out["a_norm"].values) * FS

    return out


# ──────────────────────────────
# 主入口
# ──────────────────────────────
def preprocess(nodes_dict: dict[int, pd.DataFrame]) -> dict[int, pd.DataFrame]:
    result = {}
    for nid, df in nodes_dict.items():
        print(f"[Preprocess] Node {nid} ({NODE_NAMES.get(nid)})")
        df = _preprocess_node(df)
        print(f"  |a| 范围: {df['a_norm'].min():.2f} ~ {df['a_norm'].max():.2f} m/s²")
        print(f"  a_vert 范围: {df['a_vert'].min():.2f} ~ {df['a_vert'].max():.2f} m/s²")
        print(f"  roll 范围: {df['roll'].min():.1f} ~ {df['roll'].max():.1f} deg")
        print(f"  pitch 范围: {df['pitch'].min():.1f} ~ {df['pitch'].max():.1f} deg")
        result[nid] = df
    return result


# ──────────────────────────────
# 静止段基线检查
# ──────────────────────────────
def check_static_baseline(df_static: pd.DataFrame) -> dict:
    """
    取一段明显静止的数据，检查:
      - |a| ≈ 9.8 m/s²
      - a_vert ≈ 0
      - |g| ≈ 0
    """
    a_mean = df_static["a_norm"].mean()
    a_std = df_static["a_norm"].std()
    vert_mean = df_static["a_vert"].mean()
    vert_std = df_static["a_vert"].std()
    g_mean = df_static["g_norm"].mean()
    g_std = df_static["g_norm"].std()

    report = {
        "a_mean": a_mean, "a_std": a_std,
        "vert_mean": vert_mean, "vert_std": vert_std,
        "g_mean": g_mean, "g_std": g_std,
        "pass": (
            abs(a_mean - G_STD) < 1.0 and a_std < 1.0 and
            abs(vert_mean) < 1.0 and vert_std < 1.0 and
            g_mean < 0.5
        ),
    }
    print(f"[Static check] |a|={a_mean:.2f}±{a_std:.2f}, a_vert={vert_mean:.2f}±{vert_std:.2f}, |g|={g_mean:.2f}±{g_std:.2f}")
    if not report["pass"]:
        print("  ⚠️ 静止段异常，请检查传感器固定或单位配置")
    return report
