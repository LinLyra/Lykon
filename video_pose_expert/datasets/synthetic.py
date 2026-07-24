"""合成姿态数据生成器。

为 config.ACTIONS_V1 的 6 个动作生成物理上合理、类间可分、类内有随机
变化的骨骼序列，用于在任何环境（无 mediapipe / 无网络）下端到端验证
整条管线：提取特征 → 训练识别边界 → Expert 推理。

输出坐标沿用图像坐标习惯（y 向下、范围约 0~1），与 MediaPipe 输出一致，
后续由 geometry.clean_confidence 统一翻转 y。
"""
from __future__ import annotations

import numpy as np

from ..config import KEYPOINTS, ACTIONS_V1, DEFAULT_FPS

# 站立基准骨架（图像坐标，y 向下），顺序同 config.KEYPOINTS
_BASE = np.array([
    [0.50, 0.15],   # nose
    [0.43, 0.30],   # left_shoulder
    [0.57, 0.30],   # right_shoulder
    [0.41, 0.45],   # left_elbow
    [0.59, 0.45],   # right_elbow
    [0.41, 0.60],   # left_wrist
    [0.59, 0.60],   # right_wrist
    [0.46, 0.55],   # left_hip
    [0.54, 0.55],   # right_hip
    [0.46, 0.74],   # left_knee
    [0.54, 0.74],   # right_knee
    [0.46, 0.92],   # left_ankle
    [0.54, 0.92],   # right_ankle
], dtype=float)

_I = {n: i for i, n in enumerate(KEYPOINTS)}


def _sample_action(action: str, T: int, rng: np.random.Generator,
                   noise_scale: float = 1.0) -> np.ndarray:
    """生成单个动作 clip，返回 (T, K, 2) 坐标（y 向下）。"""
    pose = np.repeat(_BASE[None, :, :], T, axis=0).copy()   # (T,K,2)
    t = np.arange(T) / max(T - 1, 1)                        # 0~1
    # 更大的个体差异范围 → 类内方差更真实（困难模式下类间会有重叠）
    amp = rng.uniform(0.70, 1.30)                           # 个体幅度差异
    ph = rng.uniform(0, 2 * np.pi)

    if action == "idle":
        pose += rng.normal(0, 0.004, pose.shape)            # 轻微晃动

    elif action == "dribble":
        # 右手周期性上下（运球），身体略降
        freq = rng.uniform(2.2, 3.2)
        osc = 0.10 * amp * np.sin(2 * np.pi * freq * t + ph)
        pose[:, _I["right_wrist"], 1] += 0.10 + osc
        pose[:, _I["right_elbow"], 1] += 0.05 + 0.5 * osc
        pose[:, [_I["left_hip"], _I["right_hip"]], 1] += 0.02

    elif action == "pass":
        # 双臂短促前伸推出（对称爆发），中段速度峰值
        burst = np.exp(-((t - 0.5) ** 2) / (2 * 0.10 ** 2))
        pose[:, _I["left_wrist"], 0] -= 0.10 * amp * burst
        pose[:, _I["right_wrist"], 0] += 0.10 * amp * burst
        pose[:, [_I["left_wrist"], _I["right_wrist"]], 1] -= 0.05 * burst[:, None]
        pose[:, [_I["left_elbow"], _I["right_elbow"]], 1] -= 0.03 * burst[:, None]

    elif action == "shot":
        # 双手由胸前上举过头 + 出手随球（wrist 升到肩上方，y 变小）
        raise_ = np.clip((t - 0.2) / 0.5, 0, 1) ** 1.5
        pose[:, [_I["left_wrist"], _I["right_wrist"]], 1] -= 0.30 * amp * raise_[:, None]
        pose[:, [_I["left_elbow"], _I["right_elbow"]], 1] -= 0.18 * amp * raise_[:, None]
        pose[:, [_I["left_wrist"], _I["right_wrist"]], 0] += \
            np.array([0.03, -0.03]) * raise_[:, None]

    elif action == "jump":
        # 先屈膝下蹲再全身上冲后落地（全身 y 先增后大幅减再回落）
        crouch = np.exp(-((t - 0.25) ** 2) / (2 * 0.08 ** 2))
        air = np.exp(-((t - 0.6) ** 2) / (2 * 0.12 ** 2))
        dy = 0.05 * crouch - 0.22 * amp * air
        pose[:, :, 1] += dy[:, None]
        pose[:, [_I["left_knee"], _I["right_knee"]], 1] += 0.04 * crouch[:, None]

    elif action == "sprint":
        # 全身水平位移 + 手臂交替摆动 + 大步频
        pose[:, :, 0] += (0.35 * amp * t)[:, None]          # 向右冲
        freq = rng.uniform(2.5, 3.5)
        swing = 0.06 * np.sin(2 * np.pi * freq * t + ph)
        pose[:, _I["left_wrist"], 1] += swing
        pose[:, _I["right_wrist"], 1] -= swing
        pose[:, _I["left_ankle"], 0] += 0.05 * np.sin(2 * np.pi * freq * t)
        pose[:, _I["right_ankle"], 0] -= 0.05 * np.sin(2 * np.pi * freq * t)

    pose += rng.normal(0, 0.006 * noise_scale, pose.shape)  # 观测噪声
    return pose


def _apply_occlusion(seq: np.ndarray, occlusion_prob: float,
                     rng: np.random.Generator) -> np.ndarray:
    """模拟真实视频的关键点遮挡/漏检：随机把部分关键点整段置低置信度。"""
    if occlusion_prob <= 0:
        return seq
    K = seq.shape[1]
    for k in range(K):
        if rng.random() < occlusion_prob:
            seq[:, k, 2] = rng.uniform(0.0, 0.25)            # 低于 MIN_KP_CONFIDENCE
    return seq


def generate_synthetic(n_per_class: int = 60, seconds: float = 1.2,
                       fps: float = DEFAULT_FPS, seed: int = 42,
                       actions: list[str] | None = None,
                       difficulty: str = "easy") -> list[dict]:
    """生成合成数据集 samples。

    difficulty:
        "easy" —— 类间高度可分，仅验证管线正确性（准确率会很高）
        "hard" —— 加大噪声 + 关键点遮挡 + 个体差异，接近真实视频姿态，
                  给出有意义、非平凡的训练效果数字

    Returns: list[dict]，每项 {seq:(T,K,3), label, clip_id, source, fps}
    """
    actions = actions or ACTIONS_V1
    noise_scale, occ = (1.0, 0.0) if difficulty == "easy" else (2.5, 0.15)
    conf_lo = 0.85 if difficulty == "easy" else 0.55
    rng = np.random.default_rng(seed)
    T = int(round(seconds * fps))
    samples: list[dict] = []
    cid = 0
    for action in actions:
        for _ in range(n_per_class):
            xy = _sample_action(action, T, rng, noise_scale=noise_scale)
            conf = rng.uniform(conf_lo, 1.0, (T, len(KEYPOINTS), 1))
            seq = np.concatenate([xy, conf], axis=2)          # (T,K,3)
            seq = _apply_occlusion(seq, occ, rng)
            samples.append({
                "seq": seq, "label": action, "clip_id": f"syn_{cid:05d}",
                "source": "synthetic", "fps": fps,
            })
            cid += 1
    rng.shuffle(samples)
    return samples
