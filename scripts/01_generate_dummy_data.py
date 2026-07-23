from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "our_data" / "dummy_imu.csv"
np.random.seed(42)
fs = 50
labels = ["idle", "dribble", "pass", "shot", "jump", "sprint"]
rows = []
seq = 0
for player in ["P01", "P02", "P03"]:
    for side in ["left", "right"]:
        for label in labels:
            for rep in range(20):
                duration = 2.0
                n = int(fs * duration)
                t = np.arange(n) / fs
                ax = np.random.normal(0, 0.3, n)
                ay = np.random.normal(0, 0.3, n)
                az = np.random.normal(9.8, 0.3, n)
                gx = np.random.normal(0, 0.15, n)
                gy = np.random.normal(0, 0.15, n)
                gz = np.random.normal(0, 0.15, n)
                if label == "dribble":
                    az += 4 * np.maximum(0, np.sin(2 * np.pi * 2.5 * t))
                    gy += 1.5 * np.sin(2 * np.pi * 2.5 * t)
                elif label == "pass":
                    center = n // 2
                    ax[center:center+5] += 8
                    gy[center:center+5] += 3
                elif label == "shot":
                    center = n // 2
                    gy += 4 * np.exp(-((np.arange(n)-center)/12)**2)
                    az += 3 * np.exp(-((np.arange(n)-center)/10)**2)
                elif label == "jump":
                    az[n//3:n//3+3] += 10
                    az[2*n//3:2*n//3+4] += 14
                elif label == "sprint":
                    ax += 2.5 * np.sin(2 * np.pi * 1.8 * t)
                    az += 1.5 * np.sin(2 * np.pi * 1.8 * t)
                for i in range(n):
                    rows.append({
                        "session_id": "dummy_session", "player_id": player,
                        "sensor_id": f"{player}_{side}", "side": side, "seq": seq,
                        "timestamp_us": seq * int(1_000_000 / fs),
                        "ax": ax[i], "ay": ay[i], "az": az[i],
                        "gx": gx[i], "gy": gy[i], "gz": gz[i],
                        "mx": 0.0, "my": 0.0, "mz": 0.0,
                        "label": label,
                    })
                    seq += 1
pd.DataFrame(rows).to_csv(OUT, index=False)
print(f"Wrote {OUT}")
