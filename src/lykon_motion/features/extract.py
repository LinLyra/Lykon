from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def _dominant_freq(x: np.ndarray, fs: int) -> float:
    x = x - np.mean(x)
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), d=1 / fs)
    if len(spec) <= 1:
        return 0.0
    idx = np.argmax(spec[1:]) + 1
    return float(freqs[idx])


def extract_window_features(X: np.ndarray, meta: pd.DataFrame, channels: list[str], fs: int = 50) -> pd.DataFrame:
    rows = []
    for i in range(X.shape[0]):
        row = meta.iloc[i].to_dict()
        arr = X[i]
        for ci, ch in enumerate(channels):
            s = arr[ci]
            row[f"{ch}_mean"] = float(np.mean(s))
            row[f"{ch}_std"] = float(np.std(s))
            row[f"{ch}_min"] = float(np.min(s))
            row[f"{ch}_max"] = float(np.max(s))
            row[f"{ch}_ptp"] = float(np.ptp(s))
            row[f"{ch}_dom_freq"] = _dominant_freq(s, fs)
            peaks, _ = find_peaks(np.abs(s), distance=max(1, fs // 8))
            row[f"{ch}_peak_count"] = int(len(peaks))
        # combined norms for useful rule/model features
        ax, ay, az = arr[channels.index("ax")], arr[channels.index("ay")], arr[channels.index("az")]
        gx, gy, gz = arr[channels.index("gx")], arr[channels.index("gy")], arr[channels.index("gz")]
        acc_norm = np.sqrt(ax**2 + ay**2 + az**2)
        gyro_norm = np.sqrt(gx**2 + gy**2 + gz**2)
        row["acc_norm_mean"] = float(np.mean(acc_norm))
        row["acc_norm_std"] = float(np.std(acc_norm))
        row["acc_norm_max"] = float(np.max(acc_norm))
        row["acc_norm_peak_count"] = int(len(find_peaks(acc_norm, distance=max(1, fs // 8))[0]))
        row["acc_norm_dom_freq"] = _dominant_freq(acc_norm, fs)
        row["gyro_norm_mean"] = float(np.mean(gyro_norm))
        row["gyro_norm_std"] = float(np.std(gyro_norm))
        row["gyro_norm_max"] = float(np.max(gyro_norm))
        row["gyro_norm_peak_count"] = int(len(find_peaks(gyro_norm, distance=max(1, fs // 8))[0]))
        row["gyro_norm_dom_freq"] = _dominant_freq(gyro_norm, fs)
        rows.append(row)
    return pd.DataFrame(rows)
