from pathlib import Path
import yaml
import numpy as np
import pandas as pd

CONFIG_PATH = Path("configs/shot_pipeline.yaml")
SIGNAL_COLS = ["acc_x","acc_y","acc_z","gyr_x","gyr_y","gyr_z"]

def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def inspect_node(g, node_name):
    g = g.sort_values("master_timestamp_us")
    t = g["master_timestamp_us"].to_numpy(dtype=np.int64)
    dt = np.diff(t)
    median_dt = np.median(dt)
    hz = 1e6 / median_dt if median_dt > 0 else np.nan
    seq = np.sort(g["seq"].dropna().astype(int).unique())
    expected = seq[-1] - seq[0] + 1 if len(seq) else 0
    loss = 1 - len(seq) / expected if expected else np.nan
    sat = ((g[SIGNAL_COLS] >= 32767) | (g[SIGNAL_COLS] <= -32768)).sum().sum()
    print(f"\nNode {node_name}")
    print(f"  rows={len(g)} duration_s={(t.max()-t.min())/1e6:.3f}")
    print(f"  median_hz={hz:.3f} estimated_packet_loss={loss:.2%}")
    print(f"  saturated_values={sat}")

def resample_one(g, node, node_name, start_us, end_us, dt_us):
    g = g.sort_values("master_timestamp_us").drop_duplicates("master_timestamp_us")
    t_src = g["master_timestamp_us"].to_numpy(dtype=float)
    t_dst = np.arange(start_us, end_us + 1, dt_us, dtype=np.int64)

    out = pd.DataFrame({
        "master_timestamp_us": t_dst,
        "node": node,
        "body_position": node_name,
    })
    for c in SIGNAL_COLS:
        out[c] = np.interp(t_dst, t_src, g[c].to_numpy(dtype=float))

    out["acc_mag"] = np.sqrt(out["acc_x"]**2 + out["acc_y"]**2 + out["acc_z"]**2)
    out["gyro_mag"] = np.sqrt(out["gyr_x"]**2 + out["gyr_y"]**2 + out["gyr_z"]**2)
    out["time_s"] = (out["master_timestamp_us"] - start_us) / 1e6
    return out

def main():
    cfg = load_config()
    raw_path = Path(cfg["raw_path"])
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        raise FileNotFoundError(f"找不到原始文件: {raw_path}")

    df = pd.read_csv(raw_path)
    required = {"group","node","seq","node_timestamp_us","master_timestamp_us",*SIGNAL_COLS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"缺少字段: {missing}")

    node_map = {int(k): v for k, v in cfg["node_map"].items()}
    print("Raw shape:", df.shape)
    print("Columns:", df.columns.tolist())

    for node, g in df.groupby("node"):
        inspect_node(g, node_map.get(int(node), f"node_{node}"))

    starts = [g["master_timestamp_us"].min() for _, g in df.groupby("node")]
    ends = [g["master_timestamp_us"].max() for _, g in df.groupby("node")]
    start_us = int(max(starts))
    end_us = int(min(ends))
    hz = int(cfg["sampling_rate_hz"])
    dt_us = int(round(1e6 / hz))

    parts = []
    for node, g in df.groupby("node"):
        node = int(node)
        if node not in node_map:
            continue
        parts.append(resample_one(g, node, node_map[node], start_us, end_us, dt_us))

    long_df = pd.concat(parts, ignore_index=True)
    long_path = out_dir / "shot_cycles_resampled_long.csv"
    long_df.to_csv(long_path, index=False)

    wide_parts = []
    for node, g in long_df.groupby("node"):
        keep = g[["master_timestamp_us","time_s",*SIGNAL_COLS,"acc_mag","gyro_mag"]].copy()
        rename = {c: f"{c}_n{node}" for c in keep.columns if c not in {"master_timestamp_us","time_s"}}
        wide_parts.append(keep.rename(columns=rename))

    wide = wide_parts[0]
    for p in wide_parts[1:]:
        wide = wide.merge(p, on=["master_timestamp_us","time_s"], how="inner")

    wide_path = out_dir / "shot_cycles_resampled_wide.csv"
    wide.to_csv(wide_path, index=False)

    print("\nSaved:")
    print(long_path)
    print(wide_path)
    print("Wide shape:", wide.shape)
    print("Aligned duration_s:", wide["time_s"].iloc[-1] - wide["time_s"].iloc[0])

if __name__ == "__main__":
    main()
