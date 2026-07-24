"""SpaceJam 公开篮球动作数据集加载器（格式自适应）。

SpaceJam（simonefrancia/SpaceJam）：每个 clip 16 帧、聚焦单个球员，提供
OpenPose 提取的关节 .npy（图像平面 x,y）与 JSON 标注。因官方未完整公开
数组细节，这里做自适应：按关节数自动识别 BODY_25 / COCO-18 / COCO-17
布局并映射到本项目 13 点。

用法：
    samples = load_spacejam("data/raw/spacejam")
其中目录下应有关节 .npy（可在子目录）与一个标注 json（clip→类别）。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import KEYPOINTS, MIN_FRAMES

# 各 OpenPose 布局 → 本项目 13 点 的索引映射
_MAPS = {
    25: {  # BODY_25
        "nose": 0, "left_shoulder": 5, "right_shoulder": 2,
        "left_elbow": 6, "right_elbow": 3, "left_wrist": 7, "right_wrist": 4,
        "left_hip": 12, "right_hip": 9, "left_knee": 13, "right_knee": 10,
        "left_ankle": 14, "right_ankle": 11,
    },
    18: {  # COCO-18 (OpenPose)
        "nose": 0, "left_shoulder": 5, "right_shoulder": 2,
        "left_elbow": 6, "right_elbow": 3, "left_wrist": 7, "right_wrist": 4,
        "left_hip": 11, "right_hip": 8, "left_knee": 12, "right_knee": 9,
        "left_ankle": 13, "right_ankle": 10,
    },
    17: {  # COCO-17 (keypoint RCNN 顺序)
        "nose": 0, "left_shoulder": 5, "right_shoulder": 6,
        "left_elbow": 7, "right_elbow": 8, "left_wrist": 9, "right_wrist": 10,
        "left_hip": 11, "right_hip": 12, "left_knee": 13, "right_knee": 14,
        "left_ankle": 15, "right_ankle": 16,
    },
}

# SpaceJam 类别 → LYKON V1 taxonomy（可选）
SPACEJAM_TO_LYKON = {
    "dribble": "dribble", "shoot": "shot", "pass": "pass",
    "run": "sprint", "walk": "idle", "no_action": "idle",
    "ball in hand": "dribble", "ball_in_hand": "dribble",
    "defense": "defense_slide", "block": "jump", "pick": "idle",
}


def _remap_joints(arr: np.ndarray) -> np.ndarray:
    """把 (T, J, C) 关节数组映射为本项目 (T, 13, 3)。C 为 2 或 3。"""
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 2:                       # (J, C) 单帧 → 补时间维
        arr = arr[None, :, :]
    T, J, C = arr.shape
    if J not in _MAPS:
        raise ValueError(f"未知关节数 J={J}，无法映射（支持 25/18/17）")
    idx = _MAPS[J]
    out = np.zeros((T, len(KEYPOINTS), 3), dtype=float)
    for k, name in enumerate(KEYPOINTS):
        j = idx[name]
        out[:, k, :2] = arr[:, j, :2]
        if C >= 3:
            out[:, k, 2] = arr[:, j, 2]
        else:
            # 无置信度：坐标为 (0,0) 视为缺失
            missing = (arr[:, j, 0] == 0) & (arr[:, j, 1] == 0)
            out[:, k, 2] = np.where(missing, 0.0, 1.0)
    return out


def _load_annotations(root: Path) -> dict:
    """加载标注 json，返回 {clip_stem: label(str)}。兼容多种结构。"""
    cand = list(root.rglob("*annotation*.json")) + list(root.rglob("*.json"))
    if not cand:
        return {}
    data = json.loads(cand[0].read_text(encoding="utf-8"))
    classes = None
    if isinstance(data, dict) and "classes" in data and "annotations" in data:
        classes = data["classes"]
        data = data["annotations"]
    out = {}
    for clip, val in data.items():
        stem = Path(str(clip)).stem
        if isinstance(val, dict):
            label = val.get("label") or val.get("class") or val.get("action")
        elif isinstance(val, (int, np.integer)) and classes:
            label = classes[int(val)]
        else:
            label = val
        out[stem] = str(label)
    return out


def load_spacejam(root: str | Path, to_lykon: bool = False,
                  max_per_class: int | None = None) -> list[dict]:
    """加载 SpaceJam 数据集为 samples。

    Args:
        root: 数据集根目录（含关节 .npy 与标注 json）
        to_lykon: 是否把类别映射到 LYKON V1 taxonomy
        max_per_class: 每类最多加载多少（快速试跑用）
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"SpaceJam 目录不存在：{root}")
    ann = _load_annotations(root)
    npys = sorted(root.rglob("*.npy"))
    if not npys:
        raise FileNotFoundError(f"{root} 下未找到关节 .npy 文件")

    samples: list[dict] = []
    per_class: dict[str, int] = {}
    for p in npys:
        stem = p.stem
        label = ann.get(stem)
        if label is None:
            continue
        if to_lykon:
            label = SPACEJAM_TO_LYKON.get(label, label)
        if max_per_class and per_class.get(label, 0) >= max_per_class:
            continue
        try:
            seq = _remap_joints(np.load(p))
        except Exception as e:  # 跳过异常文件
            print(f"[spacejam] 跳过 {p.name}: {e}")
            continue
        if seq.shape[0] < MIN_FRAMES:
            # SpaceJam 每 clip 16 帧，通常满足；不足则重复填充
            reps = int(np.ceil(MIN_FRAMES / seq.shape[0]))
            seq = np.repeat(seq, reps, axis=0)[:MIN_FRAMES]
        samples.append({
            "seq": seq, "label": label, "clip_id": stem,
            "source": "spacejam", "fps": 16.0,
        })
        per_class[label] = per_class.get(label, 0) + 1
    if not samples:
        raise RuntimeError("SpaceJam 加载到 0 个样本：请检查 .npy 与标注是否匹配")
    return samples
