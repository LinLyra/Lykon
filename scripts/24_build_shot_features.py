from pathlib import Path
import yaml
import numpy as np
import pandas as pd
from scipy.signal import welch, find_peaks

CONFIG_PATH = Path("configs/shot_pipeline.yaml")

def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def dominant_frequency(x, hz):
    x = np.asarray(x, dtype=float)
    if len(x) < 4 or np.allclose(x, x[0]):
        return 0.0
    f, p = welch(x - np.mean(x), fs=hz, nperseg=min(len(x), 128))
    if len(p) <= 1:
        return 0.0
    return float(f[1:][np.argmax(p[1:])])

def xcorr_lag(a, b, max_lag):
    a = (a - np.mean(a)) / (np.std(a) + 1e-9)
    b = (b - np.mean(b)) / (np.std(b) + 1e-9)
    corr = np.correlate(a, b, mode="full")
    lags = np.arange(-len(a)+1, len(a))
    mask = np.abs(lags) <= max_lag
    idx = np.argmax(corr[mask])
    return int(lags[mask][idx]), float(corr[mask][idx] / len(a))

def main():
    cfg = load_config()
    out_dir = Path(cfg["output_dir"])
    X = np.load(out_dir / "shot_cycle_windows.npy")
    index_df = pd.read_csv(out_dir / "shot_cycle_windows_index.csv")
    channels = (out_dir / "shot_cycle_channels.txt").read_text(encoding="utf-8").splitlines()
    cidx = {c:i for i,c in enumerate(channels)}
    hz = int(cfg["sampling_rate_hz"])

    rows = []
    for i, x in enumerate(X):
        row = {"window_id": index_df.iloc[i]["window_id"]}
        for node in [1,2,3]:
            ax, ay, az = [x[cidx[f"acc_{axis}_n{node}"]] for axis in ["x","y","z"]]
            gx, gy, gz = [x[cidx[f"gyr_{axis}_n{node}"]] for axis in ["x","y","z"]]
            amag = np.sqrt(ax**2 + ay**2 + az**2)
            gmag = np.sqrt(gx**2 + gy**2 + gz**2)
            for name, sig in [("acc_mag",amag),("gyro_mag",gmag)]:
                peaks, _ = find_peaks(sig, distance=max(1,int(0.12*hz)))
                row[f"n{node}_{name}_mean"] = float(np.mean(sig))
                row[f"n{node}_{name}_std"] = float(np.std(sig))
                row[f"n{node}_{name}_max"] = float(np.max(sig))
                row[f"n{node}_{name}_range"] = float(np.ptp(sig))
                row[f"n{node}_{name}_rms"] = float(np.sqrt(np.mean(sig**2)))
                row[f"n{node}_{name}_peak_count"] = int(len(peaks))
                row[f"n{node}_{name}_dom_freq"] = dominant_frequency(sig, hz)

        g1 = np.sqrt(sum(x[cidx[f"gyr_{a}_n1"]]**2 for a in ["x","y","z"]))
        g2 = np.sqrt(sum(x[cidx[f"gyr_{a}_n2"]]**2 for a in ["x","y","z"]))
        g3 = np.sqrt(sum(x[cidx[f"gyr_{a}_n3"]]**2 for a in ["x","y","z"]))

        lag12, corr12 = xcorr_lag(g1,g2,max_lag=int(0.5*hz))
        lag23, corr23 = xcorr_lag(g2,g3,max_lag=int(0.5*hz))
        row["gyro_lag_n1_to_n2_samples"] = lag12
        row["gyro_corr_n1_n2"] = corr12
        row["gyro_lag_n2_to_n3_samples"] = lag23
        row["gyro_corr_n2_n3"] = corr23
        rows.append(row)

    feat = pd.DataFrame(rows)
    feat.to_csv(out_dir / "shot_cycle_features.csv", index=False)
    print("Saved:", out_dir / "shot_cycle_features.csv")
    print("Feature shape:", feat.shape)

if __name__ == "__main__":
    main()
