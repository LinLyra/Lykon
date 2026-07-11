from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CONFIG_PATH = Path("configs/shot_pipeline.yaml")

def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def robust_z(x):
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    scale = 1.4826 * mad if mad > 1e-12 else np.nanstd(x)
    if scale < 1e-12:
        scale = 1.0
    return (x - med) / scale

def main():
    cfg = load_config()
    in_path = Path(cfg["output_dir"]) / "shot_cycles_resampled_wide.csv"
    report_dir = Path(cfg["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(in_path)

    # 三节点稳健标准化后融合，避免原始量程差异支配结果
    acc_energy = np.mean(
        [np.abs(robust_z(df[f"acc_mag_n{i}"])) for i in [1,2,3]], axis=0
    )
    gyro_energy = np.mean(
        [np.abs(robust_z(df[f"gyro_mag_n{i}"])) for i in [1,2,3]], axis=0
    )
    df["fused_energy"] = 0.45 * acc_energy + 0.55 * gyro_energy
    df["fused_energy_smooth"] = (
        pd.Series(df["fused_energy"]).rolling(9, center=True, min_periods=1).mean()
    )

    out_csv = Path(cfg["output_dir"]) / "shot_cycles_with_energy.csv"
    df.to_csv(out_csv, index=False)

    for signal in ["acc_mag", "gyro_mag"]:
        plt.figure(figsize=(18,6))
        for i, name in [(1,"Right upper arm"),(2,"Right forearm"),(3,"Left forearm")]:
            plt.plot(df["time_s"], robust_z(df[f"{signal}_n{i}"]), linewidth=0.7, label=name)
        plt.xlabel("Time (s)")
        plt.ylabel(f"Robust z-score of {signal}")
        plt.title(f"Three-node {signal}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(report_dir / f"{signal}_three_nodes.png", dpi=180)
        plt.close()

    plt.figure(figsize=(18,5))
    plt.plot(df["time_s"], df["fused_energy_smooth"], linewidth=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Fused motion energy")
    plt.title("Three-node fused motion energy")
    plt.tight_layout()
    plt.savefig(report_dir / "fused_motion_energy.png", dpi=180)
    plt.close()

    print("Saved:", out_csv)
    print("Saved plots to:", report_dir)

if __name__ == "__main__":
    main()
