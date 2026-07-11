from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CONFIG_PATH = Path("configs/shot_pipeline.yaml")

def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def zscore_per_channel(X):
    mean = X.mean(axis=2, keepdims=True)
    std = X.std(axis=2, keepdims=True) + 1e-8
    return (X - mean) / std

def main():
    cfg = load_config()
    out_dir = Path(cfg["output_dir"])
    report_dir = Path(cfg["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(out_dir / "shot_cycle_windows.npy").astype(np.float32)
    idx = pd.read_csv(out_dir / "shot_cycle_windows_index.csv")
    Xz = zscore_per_channel(X)
    template = Xz.mean(axis=0)

    # 余弦相似度：每个完整动作窗口 vs 平均模板
    tf = template.reshape(-1)
    sims = []
    for x in Xz:
        xf = x.reshape(-1)
        sim = float(np.dot(xf, tf) / ((np.linalg.norm(xf)*np.linalg.norm(tf)) + 1e-9))
        sims.append(sim)

    result = idx[["window_id"]].copy()
    result["template_similarity"] = sims
    result["consistency_score_100"] = np.clip((result["template_similarity"] + 1) * 50, 0, 100)
    result.to_csv(out_dir / "shot_cycle_consistency.csv", index=False)
    np.save(out_dir / "shot_cycle_template.npy", template)

    # 展示最关键的右小臂 6 通道模板
    channels = (out_dir / "shot_cycle_channels.txt").read_text(encoding="utf-8").splitlines()
    selected = [channels.index(c) for c in [
        "acc_x_n2","acc_y_n2","acc_z_n2","gyr_x_n2","gyr_y_n2","gyr_z_n2"
    ]]
    plt.figure(figsize=(16,7))
    for j in selected:
        plt.plot(template[j], linewidth=1.0, label=channels[j])
    plt.title("Right forearm normalized average action template")
    plt.xlabel("Time index")
    plt.ylabel("Z-score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(report_dir / "right_forearm_action_template.png", dpi=180)
    plt.close()

    print(result.sort_values("template_similarity").head(10))
    print("Mean similarity:", result["template_similarity"].mean())
    print("Saved template and consistency report.")

if __name__ == "__main__":
    main()
