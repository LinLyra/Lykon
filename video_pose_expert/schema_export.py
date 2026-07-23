"""把姿态样本导出为仓库 schema 格式的 CSV，打通已有数据表。

对齐 docs/schemas/：
    pose_keypoints_2d.csv   （宽表：每帧一行，各关键点 x/y/conf）
    joint_angles_2d.csv     （每帧一行，各关节角）
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import KEYPOINTS, ANGLES
from .geometry import joint_angles


def keypoints_to_wide_df(seq: np.ndarray, session_id="s0", video_id="v0",
                         player_id="p0", fps: float = 30.0) -> pd.DataFrame:
    """(T,K,3) → pose_keypoints_2d 宽表。"""
    T = seq.shape[0]
    rows = []
    for f in range(T):
        row = {
            "session_id": session_id, "video_id": video_id, "frame_id": f,
            "timestamp_us": int(f / fps * 1e6), "player_id": player_id, "track_id": 0,
        }
        for k, name in enumerate(KEYPOINTS):
            row[f"{name}_x"] = seq[f, k, 0]
            row[f"{name}_y"] = seq[f, k, 1]
            row[f"{name}_conf"] = seq[f, k, 2]
        row["pose_model"] = "mediapipe_blazepose"
        row["pose_confidence"] = float(np.mean(seq[f, :, 2]))
        rows.append(row)
    return pd.DataFrame(rows)


def angles_to_df(seq: np.ndarray, session_id="s0", video_id="v0",
                 player_id="p0", fps: float = 30.0) -> pd.DataFrame:
    """(T,K,3) → joint_angles_2d 表。"""
    ang = joint_angles(seq)                      # (T, A)
    T = ang.shape[0]
    df = pd.DataFrame(ang, columns=ANGLES)
    df.insert(0, "session_id", session_id)
    df.insert(1, "video_id", video_id)
    df.insert(2, "frame_id", np.arange(T))
    df.insert(3, "timestamp_us", (np.arange(T) / fps * 1e6).astype(int))
    df.insert(4, "player_id", player_id)
    df["source"] = "video_pose_expert"
    return df


def export_samples(samples: list[dict], out_dir: str | Path) -> dict:
    """把一批样本导出为两张 schema CSV（拼接所有 clip）。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kp_parts, ang_parts = [], []
    for s in samples:
        cid = s.get("clip_id", "clip")
        vid = s.get("video_id", cid)
        kp_parts.append(keypoints_to_wide_df(
            s["seq"], session_id=s.get("source", "s0"), video_id=vid, fps=s.get("fps", 30.0)))
        ang_parts.append(angles_to_df(
            s["seq"], session_id=s.get("source", "s0"), video_id=vid, fps=s.get("fps", 30.0)))
    kp_path = out_dir / "pose_keypoints_2d.csv"
    ang_path = out_dir / "joint_angles_2d.csv"
    pd.concat(kp_parts, ignore_index=True).to_csv(kp_path, index=False)
    pd.concat(ang_parts, ignore_index=True).to_csv(ang_path, index=False)
    return {"pose_keypoints_2d": str(kp_path), "joint_angles_2d": str(ang_path)}
