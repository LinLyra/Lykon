from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CONFIG_PATH = Path("configs/event_pipeline.yaml")

def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def moving_average(x, k=5):
    kernel = np.ones(k, dtype=float) / k
    return np.convolve(x, kernel, mode="same")

def vector_mag(x, cidx, prefix, node):
    return np.sqrt(
        x[cidx[f"{prefix}_x_n{node}"]] ** 2
        + x[cidx[f"{prefix}_y_n{node}"]] ** 2
        + x[cidx[f"{prefix}_z_n{node}"]] ** 2
    )

def find_dribble_end(x, cidx, gather_start, hz, search_before_s):
    # 在 gather 前的区间里，用右小臂加速度能量寻找最后一个显著运球冲击。
    n2_acc = vector_mag(x, cidx, "acc", 2)
    energy = moving_average(np.abs(np.gradient(n2_acc)), 5)

    search_start = max(0, gather_start - int(round(search_before_s * hz)))
    search_end = max(search_start + 1, gather_start)
    segment = energy[search_start:search_end]

    if len(segment) < 3:
        return search_start

    # 取靠后且能量较高的位置，作为最后一次运球/收球转换点
    threshold = np.quantile(segment, 0.70)
    candidates = np.where(segment >= threshold)[0]
    if len(candidates) == 0:
        return search_start
    return int(search_start + candidates[-1])

def add_phase(rows, window_id, phase, start, end, hz, source):
    if end < start:
        return
    rows.append({
        "window_id": window_id,
        "phase": phase,
        "start_index": int(start),
        "end_index": int(end),
        "start_time_s": float(start / hz),
        "end_time_s": float(end / hz),
        "duration_s": float((end - start + 1) / hz),
        "source": source,
    })

def main():
    cfg = load_config()
    paths = cfg["paths"]
    out_dir = Path(paths["output_dir"])
    report_dir = Path(paths["report_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(paths["input_windows"]).astype(np.float32)
    index_df = pd.read_csv(paths["input_index"])
    release_df = pd.read_csv(out_dir / "event_release_points.csv")
    channels = Path(paths["input_channels"]).read_text(encoding="utf-8").splitlines()
    cidx = {c: i for i, c in enumerate(channels)}

    hz = int(cfg["sampling_rate_hz"])
    seg = cfg["phase_segmentation"]

    gather_before = int(round(float(seg["gather_start_seconds_before_release"]) * hz))
    release_half = int(round(float(seg["release_half_width_seconds"]) * hz))
    follow_after = int(round(float(seg["follow_through_end_seconds_after_release"]) * hz))
    dribble_search = float(seg["dribble_search_seconds_before_gather"])

    phase_rows = []
    example_boundaries = []

    release_map = release_df.set_index("window_id")

    for i, x in enumerate(X):
        window_id = index_df.iloc[i]["window_id"]
        r = release_map.loc[window_id]
        release_idx = int(r["release_index"])
        T = x.shape[1]

        release_start = max(0, release_idx - release_half)
        release_end = min(T - 1, release_idx + release_half)
        gather_start = max(0, release_idx - gather_before)
        follow_end = min(T - 1, release_idx + follow_after)

        dribble_end = find_dribble_end(
            x=x,
            cidx=cidx,
            gather_start=gather_start,
            hz=hz,
            search_before_s=dribble_search,
        )

        # 当前窗口是完整复合动作，因此阶段定义为：
        # preparation/dribble -> gather -> release -> follow_through -> recovery
        add_phase(
            phase_rows, window_id, "preparation_or_dribble",
            0, max(0, dribble_end - 1), hz,
            "heuristic_energy_boundary"
        )
        add_phase(
            phase_rows, window_id, "final_dribble_or_collect",
            dribble_end, max(dribble_end, gather_start - 1), hz,
            "heuristic_last_impact"
        )
        add_phase(
            phase_rows, window_id, "gather_and_raise",
            gather_start, max(gather_start, release_start - 1), hz,
            "relative_to_release"
        )
        add_phase(
            phase_rows, window_id, "release",
            release_start, release_end, hz,
            "release_detector"
        )
        add_phase(
            phase_rows, window_id, "follow_through",
            release_end + 1, follow_end, hz,
            "relative_to_release"
        )
        add_phase(
            phase_rows, window_id, "recovery",
            follow_end + 1, T - 1, hz,
            "window_remainder"
        )

        if len(example_boundaries) < 8:
            example_boundaries.append({
                "window_id": window_id,
                "dribble_end": dribble_end,
                "gather_start": gather_start,
                "release_start": release_start,
                "release_idx": release_idx,
                "release_end": release_end,
                "follow_end": follow_end,
                "x": x,
            })

    phases = pd.DataFrame(phase_rows)
    out_path = out_dir / "event_phase_segments.csv"
    phases.to_csv(out_path, index=False)

    # 画右小臂 gyro magnitude 的阶段切分例子
    fig, axes = plt.subplots(len(example_boundaries), 1, figsize=(17, 3 * len(example_boundaries)))
    if len(example_boundaries) == 1:
        axes = [axes]

    for ax, ex in zip(axes, example_boundaries):
        sig = vector_mag(ex["x"], cidx, "gyr", 2)
        ax.plot(sig, linewidth=0.9)
        ax.axvline(ex["dribble_end"], linestyle="--")
        ax.axvline(ex["gather_start"], linestyle="--")
        ax.axvspan(ex["release_start"], ex["release_end"], alpha=0.18)
        ax.axvline(ex["follow_end"], linestyle="--")
        ax.set_title(str(ex["window_id"]))
        ax.set_ylabel("Right forearm gyro magnitude")
    axes[-1].set_xlabel("Time index")
    plt.tight_layout()
    plt.savefig(report_dir / "phase_segmentation_examples.png", dpi=180)
    plt.close()

    print(phases.head(20))
    print("\nPhase counts:")
    print(phases["phase"].value_counts())
    print("Saved:", out_path)

if __name__ == "__main__":
    main()
