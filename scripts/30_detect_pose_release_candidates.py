from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


REQUIRED_LANDMARKS = [
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "left_wrist",
]


def robust_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    median = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-9:
        scale = np.nanstd(values)
    if not np.isfinite(scale) or scale < 1e-9:
        scale = 1.0
    return (values - median) / scale


def moving_average(values: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1:
        return values.copy()
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values, kernel, mode="same")


def angle_degrees(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """返回 ∠ABC。"""
    ba = a - b
    bc = c - b
    numerator = np.sum(ba * bc, axis=1)
    denominator = (
        np.linalg.norm(ba, axis=1)
        * np.linalg.norm(bc, axis=1)
        + 1e-9
    )
    cosine = np.clip(numerator / denominator, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def pivot_pose(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "frame_id",
        "timestamp_s",
        "fps",
        "landmark_name",
        "x_norm",
        "y_norm",
        "z_rel",
        "visibility",
    }
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"pose CSV 缺少字段: {missing}")

    available = set(df["landmark_name"].unique())
    missing_landmarks = sorted(set(REQUIRED_LANDMARKS) - available)
    if missing_landmarks:
        raise ValueError(f"缺少关键点: {missing_landmarks}")

    subset = df[df["landmark_name"].isin(REQUIRED_LANDMARKS)].copy()
    wide = subset.pivot_table(
        index=["frame_id", "timestamp_s", "fps"],
        columns="landmark_name",
        values=["x_norm", "y_norm", "z_rel", "visibility"],
        aggfunc="first",
    )
    wide.columns = [f"{value}_{landmark}" for value, landmark in wide.columns]
    return wide.reset_index().sort_values("frame_id").reset_index(drop=True)


def get_xyz(wide: pd.DataFrame, landmark: str) -> np.ndarray:
    return wide[
        [
            f"x_norm_{landmark}",
            f"y_norm_{landmark}",
            f"z_rel_{landmark}",
        ]
    ].to_numpy(dtype=float)


def speed(xyz: np.ndarray, fps: float) -> np.ndarray:
    delta = np.diff(xyz, axis=0, prepend=xyz[[0]])
    return np.linalg.norm(delta, axis=1) * fps


def detect_target_count(
    score: np.ndarray,
    target_events: int,
    min_distance_frames: int,
) -> tuple[np.ndarray, dict]:
    positive = score[np.isfinite(score)]
    if len(positive) == 0:
        raise ValueError("事件分数为空。")

    candidates = []
    q_values = np.linspace(0.95, 0.35, 50)

    for q in q_values:
        prominence = max(
            float(np.quantile(positive, q) - np.quantile(positive, 0.25)),
            1e-6,
        )
        peaks, props = find_peaks(
            score,
            distance=min_distance_frames,
            prominence=prominence,
        )
        candidates.append(
            (
                abs(len(peaks) - target_events),
                len(peaks),
                prominence,
                peaks,
                props,
            )
        )

    candidates.sort(key=lambda item: (item[0], abs(item[1] - target_events)))
    _, count, prominence, peaks, props = candidates[0]

    if len(peaks) > target_events:
        order = np.argsort(props["prominences"])[::-1][:target_events]
        peaks = np.sort(peaks[order])

    return peaks, {
        "candidate_count": count,
        "prominence": prominence,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose-csv", required=True)
    parser.add_argument("--target-events", type=int, default=6)
    parser.add_argument("--min-distance-seconds", type=float, default=1.8)
    parser.add_argument(
        "--output-dir",
        default="data/processed/our_data",
    )
    parser.add_argument(
        "--report-dir",
        default="outputs/reports/our_data",
    )
    args = parser.parse_args()

    pose_path = Path(args.pose_csv)
    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(pose_path)
    wide = pivot_pose(df)

    fps = float(wide["fps"].median())
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("无效 FPS。")

    shoulder = get_xyz(wide, "right_shoulder")
    elbow = get_xyz(wide, "right_elbow")
    wrist = get_xyz(wide, "right_wrist")
    left_wrist = get_xyz(wide, "left_wrist")

    right_wrist_speed = speed(wrist, fps)
    right_elbow_speed = speed(elbow, fps)
    left_wrist_speed = speed(left_wrist, fps)

    elbow_angle = angle_degrees(shoulder, elbow, wrist)
    elbow_extension_velocity = np.gradient(elbow_angle) * fps

    # y 坐标向下增大；手腕向上运动时 -dy/dt 为正。
    wrist_vertical_velocity = -np.gradient(wrist[:, 1]) * fps

    visibility = np.minimum(
        wide["visibility_right_wrist"].to_numpy(dtype=float),
        wide["visibility_right_elbow"].to_numpy(dtype=float),
    )

    score = (
        0.45 * np.maximum(robust_z(right_wrist_speed), 0)
        + 0.20 * np.maximum(robust_z(right_elbow_speed), 0)
        + 0.20 * np.maximum(robust_z(elbow_extension_velocity), 0)
        + 0.10 * np.maximum(robust_z(wrist_vertical_velocity), 0)
        + 0.05 * np.maximum(
            robust_z(np.abs(right_wrist_speed - left_wrist_speed)),
            0,
        )
    )
    score = moving_average(score * np.clip(visibility, 0, 1), window=5)

    min_distance_frames = max(
        1,
        int(round(args.min_distance_seconds * fps)),
    )
    peaks, detection_meta = detect_target_count(
        score=score,
        target_events=args.target_events,
        min_distance_frames=min_distance_frames,
    )

    result_rows = []
    for event_number, peak in enumerate(peaks, start=1):
        local_start = max(0, peak - int(round(0.25 * fps)))
        local_end = min(len(score), peak + int(round(0.25 * fps)) + 1)
        local = score[local_start:local_end]
        baseline = float(np.median(local))
        spread = float(np.std(local) + 1e-9)
        peak_strength = max(0.0, float((score[peak] - baseline) / spread))

        result_rows.append(
            {
                "video_event_id": f"V{event_number:02d}",
                "frame_id": int(wide.loc[peak, "frame_id"]),
                "timestamp_s": float(wide.loc[peak, "timestamp_s"]),
                "pose_release_proxy_score": float(score[peak]),
                "pose_peak_strength": peak_strength,
                "right_elbow_angle_deg": float(elbow_angle[peak]),
                "right_wrist_speed_norm_per_s": float(
                    right_wrist_speed[peak]
                ),
                "right_elbow_extension_velocity_deg_s": float(
                    elbow_extension_velocity[peak]
                ),
                "right_wrist_visibility": float(
                    wide.loc[peak, "visibility_right_wrist"]
                ),
                "label": "shot_pose_anchor_candidate",
                "ground_truth_status": "pending_manual_or_ball_validation",
                "manual_is_correct": "",
                "manual_true_release_frame": "",
                "manual_notes": "",
            }
        )

    result = pd.DataFrame(result_rows)
    output_path = output_dir / "pose_release_candidates.csv"
    result.to_csv(output_path, index=False)

    plt.figure(figsize=(18, 7))
    plt.plot(
        wide["timestamp_s"],
        score,
        linewidth=0.9,
        label="Pose release proxy score",
    )
    for row, peak in zip(result_rows, peaks):
        timestamp = row["timestamp_s"]
        plt.scatter(
            [timestamp],
            [score[peak]],
            s=45,
        )
        plt.text(
            timestamp,
            score[peak],
            row["video_event_id"],
            fontsize=9,
        )

    plt.xlabel("Video time (s)")
    plt.ylabel("Pose release proxy score")
    plt.title(
        f"Video pose event candidates: {len(peaks)} "
        f"| target={args.target_events} "
        f"| prominence={detection_meta['prominence']:.4f}"
    )
    plt.legend()
    plt.tight_layout()
    report_path = report_dir / "pose_release_candidates.png"
    plt.savefig(report_path, dpi=180)
    plt.close()

    print(result.to_string(index=False))
    print("\nSaved:")
    print(output_path)
    print(report_path)
    print(
        "\n注意：这些是姿态锚点候选，不是球离手真值。"
        "请在 manual_is_correct 和 manual_true_release_frame 中人工复核。"
    )


if __name__ == "__main__":
    main()
