from pathlib import Path
import json
import yaml
import numpy as np
import pandas as pd

CONFIG_PATH = Path("configs/event_pipeline.yaml")

def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def normalize_per_channel(X):
    mean = X.mean(axis=2, keepdims=True)
    std = X.std(axis=2, keepdims=True) + 1e-8
    return (X - mean) / std

def main():
    cfg = load_config()
    paths = cfg["paths"]
    out_dir = Path(paths["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(paths["input_windows"]).astype(np.float32)
    index_df = pd.read_csv(paths["input_index"])
    release_df = pd.read_csv(out_dir / "event_release_points.csv")
    phases_df = pd.read_csv(out_dir / "event_phase_segments.csv")
    channels = Path(paths["input_channels"]).read_text(encoding="utf-8").splitlines()

    if bool(cfg["template"]["normalize_per_channel"]):
        X_template = normalize_per_channel(X)
    else:
        X_template = X.copy()

    template = X_template.mean(axis=0)
    np.save(out_dir / "free_throw_event_template.npy", template)

    # 统计阶段时长
    phase_stats = {}
    for phase, g in phases_df.groupby("phase"):
        phase_stats[phase] = {
            "mean_duration_s": float(g["duration_s"].mean()),
            "std_duration_s": float(g["duration_s"].std(ddof=0)),
            "median_duration_s": float(g["duration_s"].median()),
            "n": int(len(g)),
        }

    # 统计 release 位置
    release_summary = {
        "mean_release_index": float(release_df["release_index"].mean()),
        "std_release_index": float(release_df["release_index"].std(ddof=0)),
        "median_release_index": float(release_df["release_index"].median()),
        "mean_release_time_from_window_start_s": float(
            release_df["release_time_from_window_start_s"].mean()
        ),
        "mean_detector_confidence": float(release_df["release_confidence"].mean()),
    }

    event_template = {
        "event_name": "dribble_set_shot_routine",
        "event_family": "shot_attempt_sequence",
        "description": (
            "无球模拟的罚球式复合动作：运球调整、收球、举球、出手、随球、回位。"
        ),
        "product_role": (
            "作为比赛事件语法和 shot-attempt 候选事件的初始模板，"
            "不能直接视为真实比赛中的投篮命中或离手真值。"
        ),
        "sensor_layout": {
            "node_1": "right_upper_arm",
            "node_2": "right_forearm",
            "node_3": "left_forearm",
        },
        "sampling_rate_hz": int(cfg["sampling_rate_hz"]),
        "n_windows": int(X.shape[0]),
        "n_channels": int(X.shape[1]),
        "n_timesteps": int(X.shape[2]),
        "channels": channels,
        "release_summary": release_summary,
        "phase_statistics": phase_stats,
        "required_future_validation": [
            "同步视频中的真实离手帧",
            "有球动作",
            "真实比赛中的防守与身体位移",
            "shot / pass / dribble-only / layup 等负样本",
        ],
    }

    json_path = out_dir / "free_throw_event_template.json"
    json_path.write_text(
        json.dumps(event_template, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(event_template, ensure_ascii=False, indent=2))
    print("\nSaved:")
    print(out_dir / "free_throw_event_template.npy")
    print(json_path)

if __name__ == "__main__":
    main()
