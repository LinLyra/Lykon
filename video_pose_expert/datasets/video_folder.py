"""从用户自有视频提取骨骼（Google MediaPipe Pose / BlazePose）。

用法约定（拖入即用）：
    data/raw/video/<action>/*.mp4     # 按子目录名作为动作标签（训练用）
    data/raw/video/test/*.mp4         # 无标签，供 Expert 识别（推理用）

- 训练：load_labeled_clips(root) → 每个视频作为一个已标注样本
- 推理：sliding_windows_from_video(path) → 连续视频切成滑窗样本

MediaPipe / OpenCV 为惰性导入：未安装时本模块仍可被 import，只有真正
提取时才报错并给出安装提示。提取逻辑与 scripts/29_extract_video_pose.py 一致。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import (
    KEYPOINTS, MEDIAPIPE_INDEX, DEFAULT_FPS, WINDOW_SECONDS, STEP_SECONDS,
    MIN_FRAMES,
)

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def _lazy_imports():
    try:
        import cv2  # noqa
        import mediapipe as mp  # noqa
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "提取视频骨骼需要 mediapipe 与 opencv：\n"
            "    pip install mediapipe opencv-python\n"
            f"（原始错误：{e}）"
        )
    return cv2, mp


def extract_pose_from_video(video_path: str | Path,
                            model_complexity: int = 1,
                            max_frames: int | None = None) -> tuple[np.ndarray, float]:
    """提取单个视频的骨骼序列。

    Returns: (seq, fps)，seq 形状 (T, K, 3) = [x_norm, y_norm, visibility]，
             K/顺序同 config.KEYPOINTS。
    """
    cv2, mp = _lazy_imports()
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    if fps <= 0:
        fps = DEFAULT_FPS

    mp_pose = mp.solutions.pose
    frames: list[np.ndarray] = []
    with mp_pose.Pose(static_image_mode=False, model_complexity=model_complexity,
                      min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        fid = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames and fid >= max_frames:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            kp = np.zeros((len(KEYPOINTS), 3), dtype=float)
            if res.pose_landmarks:
                lms = res.pose_landmarks.landmark
                for j, name in enumerate(KEYPOINTS):
                    lm = lms[MEDIAPIPE_INDEX[name]]
                    kp[j] = [lm.x, lm.y, lm.visibility]
            frames.append(kp)
            fid += 1
    cap.release()
    if not frames:
        raise RuntimeError(f"未从视频解出任何帧：{video_path}")
    return np.stack(frames), float(fps)


def load_labeled_clips(root: str | Path, model_complexity: int = 1) -> list[dict]:
    """按 <root>/<action>/*.mp4 结构加载已标注视频，每个视频 = 一个样本。"""
    root = Path(root)
    samples: list[dict] = []
    cid = 0
    for action_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        label = action_dir.name
        if label == "test":
            continue
        for vid in sorted(action_dir.iterdir()):
            if vid.suffix.lower() not in VIDEO_EXTS:
                continue
            seq, fps = extract_pose_from_video(vid, model_complexity)
            if seq.shape[0] < MIN_FRAMES:
                continue
            samples.append({
                "seq": seq, "label": label, "clip_id": f"vid_{cid:05d}",
                "video_id": vid.stem, "source": "video", "fps": fps,
            })
            cid += 1
    return samples


def sliding_windows(seq: np.ndarray, fps: float,
                    window_seconds: float = WINDOW_SECONDS,
                    step_seconds: float = STEP_SECONDS) -> list[dict]:
    """把连续骨骼序列切成滑窗样本（推理用，无标签）。"""
    win = max(MIN_FRAMES, int(round(window_seconds * fps)))
    step = max(1, int(round(step_seconds * fps)))
    out: list[dict] = []
    T = seq.shape[0]
    for start in range(0, max(1, T - win + 1), step):
        chunk = seq[start:start + win]
        if chunk.shape[0] < MIN_FRAMES:
            continue
        out.append({
            "seq": chunk, "label": None, "fps": fps,
            "start_frame": int(start), "end_frame": int(start + chunk.shape[0]),
            "start_time_us": int(start / fps * 1e6),
            "end_time_us": int((start + chunk.shape[0]) / fps * 1e6),
        })
    return out


def sliding_windows_from_video(video_path: str | Path,
                               window_seconds: float = WINDOW_SECONDS,
                               step_seconds: float = STEP_SECONDS,
                               model_complexity: int = 1) -> list[dict]:
    """对一个连续视频提取骨骼并切滑窗，供 Expert 识别任意人员动作。"""
    seq, fps = extract_pose_from_video(video_path, model_complexity)
    windows = sliding_windows(seq, fps, window_seconds, step_seconds)
    for w in windows:
        w["video_id"] = Path(video_path).stem
        w["source"] = "video"
    return windows
