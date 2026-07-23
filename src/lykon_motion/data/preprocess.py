import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


def add_norms(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["acc_norm"] = np.sqrt(out["ax"]**2 + out["ay"]**2 + out["az"]**2)
    out["gyro_norm"] = np.sqrt(out["gx"]**2 + out["gy"]**2 + out["gz"]**2)
    out["mag_norm"] = np.sqrt(out["mx"]**2 + out["my"]**2 + out["mz"]**2)
    return out


def lowpass(series: pd.Series, fs: int = 50, cutoff: float = 8.0, order: int = 3) -> np.ndarray:
    if len(series) < 10:
        return series.to_numpy()
    b, a = butter(order, cutoff / (0.5 * fs), btype="low")
    return filtfilt(b, a, series.to_numpy())


def filter_imu(df: pd.DataFrame, fs: int = 50) -> pd.DataFrame:
    out = df.copy()
    for col in ["ax", "ay", "az", "gx", "gy", "gz"]:
        out[col] = lowpass(out[col], fs=fs)
    return out
