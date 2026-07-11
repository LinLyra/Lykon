from pathlib import Path
import yaml
import numpy as np
import pandas as pd

CONFIG_PATH = Path("configs/shot_pipeline.yaml")
CHANNELS = [
    "acc_x_n1","acc_y_n1","acc_z_n1","gyr_x_n1","gyr_y_n1","gyr_z_n1",
    "acc_x_n2","acc_y_n2","acc_z_n2","gyr_x_n2","gyr_y_n2","gyr_z_n2",
    "acc_x_n3","acc_y_n3","acc_z_n3","gyr_x_n3","gyr_y_n3","gyr_z_n3",
]

def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_config()
    out_dir = Path(cfg["output_dir"])
    df = pd.read_csv(out_dir / "shot_cycles_with_energy.csv")
    markers = pd.read_csv(out_dir / "shot_cycle_candidates.csv")
    markers = markers[markers["keep"] == 1].copy()

    hz = int(cfg["sampling_rate_hz"])
    pre = int(round(float(cfg["segmentation"]["pre_seconds"]) * hz))
    post = int(round(float(cfg["segmentation"]["post_seconds"]) * hz))
    length = pre + post + 1

    windows = []
    index_rows = []
    for _, r in markers.iterrows():
        center = int(r["center_index"])
        start = center - pre
        end = center + post
        if start < 0 or end >= len(df):
            continue
        w = df.iloc[start:end+1]
        x = w[CHANNELS].to_numpy(dtype=np.float32).T  # channels x time
        if x.shape[1] != length:
            continue
        windows.append(x)
        index_rows.append({
            "window_id": r["cycle_id"],
            "label": "dribble_set_shot_cycle",
            "start_index": start,
            "center_index": center,
            "end_index": end,
            "start_time_s": w["time_s"].iloc[0],
            "center_time_s": df["time_s"].iloc[center],
            "end_time_s": w["time_s"].iloc[-1],
            "sampling_rate_hz": hz,
            "n_channels": len(CHANNELS),
            "n_timesteps": length,
            "source": "auto_peak_pending_review",
        })

    X = np.stack(windows, axis=0)
    np.save(out_dir / "shot_cycle_windows.npy", X)
    pd.DataFrame(index_rows).to_csv(out_dir / "shot_cycle_windows_index.csv", index=False)
    (out_dir / "shot_cycle_channels.txt").write_text("\n".join(CHANNELS), encoding="utf-8")

    print("X shape:", X.shape)
    print("Saved windows and index.")
    print("解释: (样本数, 18个通道, 时间点数)")

if __name__ == "__main__":
    main()
