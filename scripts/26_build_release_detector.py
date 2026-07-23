from pathlib import Path
import json
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CONFIG_PATH = Path("configs/event_pipeline.yaml")

def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def robust_z(x):
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    scale = 1.4826 * mad if mad > 1e-9 else np.std(x)
    if scale < 1e-9:
        scale = 1.0
    return (x - med) / scale

def moving_average(x, k):
    if k <= 1:
        return x.copy()
    kernel = np.ones(k, dtype=float) / k
    return np.convolve(x, kernel, mode="same")

def vector_mag(x, cidx, prefix, node):
    return np.sqrt(
        x[cidx[f"{prefix}_x_n{node}"]] ** 2
        + x[cidx[f"{prefix}_y_n{node}"]] ** 2
        + x[cidx[f"{prefix}_z_n{node}"]] ** 2
    )

def local_peak_confidence(score, idx, margin):
    lo = max(0, idx - margin)
    hi = min(len(score), idx + margin + 1)
    local = score[lo:hi]
    if len(local) < 3:
        return 0.0
    peak = score[idx]
    baseline = np.median(local)
    spread = np.std(local) + 1e-9
    return float(max(0.0, (peak - baseline) / spread))

def main():
    cfg = load_config()
    paths = cfg["paths"]
    out_dir = Path(paths["output_dir"])
    report_dir = Path(paths["report_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(paths["input_windows"]).astype(np.float32)
    index_df = pd.read_csv(paths["input_index"])
    channels = Path(paths["input_channels"]).read_text(encoding="utf-8").splitlines()
    cidx = {c: i for i, c in enumerate(channels)}

    det = cfg["release_detector"]
    weights = det["weights"]
    smooth_k = int(det["smooth_window_samples"])
    margin = int(det["confidence_margin_samples"])

    results = []
    score_examples = []

    for i, x in enumerate(X):
        n1_g = robust_z(vector_mag(x, cidx, "gyr", 1))
        n2_g = robust_z(vector_mag(x, cidx, "gyr", 2))
        n3_g = robust_z(vector_mag(x, cidx, "gyr", 3))
        n2_a = robust_z(vector_mag(x, cidx, "acc", 2))

        # 投篮出手通常表现为主导侧前臂角速度峰值；
        # 左右前臂差异有助于避免把双臂同步的传球式动作误当成投篮。
        asym = robust_z(np.abs(n2_g - n3_g))

        score = (
            float(weights["right_forearm_gyro"]) * np.maximum(n2_g, 0)
            + float(weights["right_forearm_acc"]) * np.maximum(n2_a, 0)
            + float(weights["right_upper_arm_gyro"]) * np.maximum(n1_g, 0)
            + float(weights["left_forearm_gyro"]) * np.maximum(n3_g, 0)
            + float(weights["asymmetry"]) * np.maximum(asym, 0)
        )
        score = moving_average(score, smooth_k)

        T = x.shape[1]
        start = max(0, int(round(T * float(det["search_start_ratio"]))))
        end = min(T, int(round(T * float(det["search_end_ratio"]))))
        if end <= start:
            raise ValueError("release 搜索区间无效。")

        local_idx = int(np.argmax(score[start:end]))
        release_idx = start + local_idx
        confidence = local_peak_confidence(score, release_idx, margin)

        row = {
            "window_id": index_df.iloc[i]["window_id"],
            "window_row": i,
            "release_index": release_idx,
            "release_time_from_window_start_s": release_idx / float(cfg["sampling_rate_hz"]),
            "release_score": float(score[release_idx]),
            "release_confidence": confidence,
            "search_start_index": start,
            "search_end_index": end,
            "status": "heuristic_pending_video_validation",
        }
        results.append(row)

        if len(score_examples) < 12:
            score_examples.append((row["window_id"], score, release_idx, start, end))

    release_df = pd.DataFrame(results)
    out_path = out_dir / "event_release_points.csv"
    release_df.to_csv(out_path, index=False)

    # 画若干样本的 release score，人工确认峰值是否稳定
    plt.figure(figsize=(18, 10))
    offset = 0.0
    for window_id, score, release_idx, start, end in score_examples:
        s = robust_z(score) + offset
        plt.plot(s, linewidth=0.9, label=str(window_id))
        plt.scatter([release_idx], [s[release_idx]], s=22)
        plt.axvspan(start, end, alpha=0.04)
        offset += 5.0
    plt.xlabel("Window time index")
    plt.ylabel("Release score (stacked)")
    plt.title("Heuristic release detector overview")
    plt.tight_layout()
    plt.savefig(report_dir / "release_alignment_overview.png", dpi=180)
    plt.close()

    print(release_df[[
        "window_id", "release_index",
        "release_time_from_window_start_s",
        "release_confidence"
    ]].head(15))
    print("\nRelease index summary:")
    print(release_df["release_index"].describe())
    print("Saved:", out_path)

if __name__ == "__main__":
    main()
