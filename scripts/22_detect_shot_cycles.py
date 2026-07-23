from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

CONFIG_PATH = Path("configs/shot_pipeline.yaml")

def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def detect_near_target(signal, hz, target, min_distance_s, q_start, q_floor):
    signal = np.asarray(signal, dtype=float)
    distance = max(1, int(round(min_distance_s * hz)))
    positive = signal[np.isfinite(signal)]
    quantiles = np.linspace(q_start, q_floor, 30)

    candidates = []
    for q in quantiles:
        prom = max(float(np.quantile(positive, q) - np.quantile(positive, 0.25)), 1e-9)
        peaks, props = find_peaks(signal, distance=distance, prominence=prom)
        candidates.append((abs(len(peaks)-target), len(peaks), prom, peaks, props))
    candidates.sort(key=lambda x: (x[0], abs(x[1]-target)))
    _, n, prom, peaks, props = candidates[0]

    # 多于目标数时，优先保留 prominence 最大的峰，再按时间排序
    if len(peaks) > target:
        order = np.argsort(props["prominences"])[::-1][:target]
        peaks = np.sort(peaks[order])
    return peaks, prom

def main():
    cfg = load_config()
    in_path = Path(cfg["output_dir"]) / "shot_cycles_with_energy.csv"
    out_dir = Path(cfg["output_dir"])
    report_dir = Path(cfg["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    hz = int(cfg["sampling_rate_hz"])
    seg = cfg["segmentation"]
    target = int(cfg["target_cycles"])

    # 当前动作是“运球+投篮”复合循环，融合能量比单个冲击峰更稳
    signal = df["fused_energy_smooth"].to_numpy()
    peaks, prominence = detect_near_target(
        signal=signal,
        hz=hz,
        target=target,
        min_distance_s=float(seg["min_cycle_distance_seconds"]),
        q_start=float(seg["prominence_quantile_start"]),
        q_floor=float(seg["prominence_quantile_floor"]),
    )

    markers = pd.DataFrame({
        "cycle_id": [f"C{i+1:03d}" for i in range(len(peaks))],
        "center_index": peaks,
        "center_time_s": df.loc[peaks, "time_s"].to_numpy(),
        "center_timestamp_us": df.loc[peaks, "master_timestamp_us"].astype(np.int64).to_numpy(),
        "detector": "fused_energy_peak",
        "review_status": "unreviewed",
        "keep": 1,
        "notes": "",
    })
    marker_path = out_dir / "shot_cycle_candidates.csv"
    markers.to_csv(marker_path, index=False)

    plt.figure(figsize=(20,6))
    plt.plot(df["time_s"], signal, linewidth=0.7)
    plt.scatter(df.loc[peaks,"time_s"], signal[peaks], s=18)
    for j, idx in enumerate(peaks):
        if j % 5 == 0:
            plt.text(df.loc[idx,"time_s"], signal[idx], str(j+1), fontsize=7)
    plt.xlabel("Time (s)")
    plt.ylabel("Fused energy")
    plt.title(f"Detected cycle centers: {len(peaks)} | prominence={prominence:.4f}")
    plt.tight_layout()
    plt.savefig(report_dir / "detected_shot_cycles.png", dpi=180)
    plt.close()

    print("Detected cycles:", len(peaks))
    print("Prominence used:", prominence)
    print("Saved:", marker_path)
    print("请先查看 detected_shot_cycles.png；如有错峰，在 CSV 里把 keep 改成 0，或手工调整 center_time_s。")

if __name__ == "__main__":
    main()
