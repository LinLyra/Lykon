from __future__ import annotations
import numpy as np
import pandas as pd


def make_windows(
    df: pd.DataFrame,
    sampling_rate_hz: int = 50,
    window_seconds: float = 2.0,
    step_seconds: float = 0.5,
    channels: list[str] | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Create fixed-length windows from standard LYKON IMU dataframe.

    Returns:
        X: shape (n_windows, n_channels, n_timesteps)
        meta: one row per window with labels and metadata
    """
    if channels is None:
        channels = ["ax", "ay", "az", "gx", "gy", "gz", "mx", "my", "mz"]
    win = int(window_seconds * sampling_rate_hz)
    step = int(step_seconds * sampling_rate_hz)
    xs, metas = [], []

    group_cols = ["session_id", "player_id", "side"]
    for keys, g in df.groupby(group_cols):
        g = g.sort_values("timestamp_us").reset_index(drop=True)
        if len(g) < win:
            continue
        for start in range(0, len(g) - win + 1, step):
            chunk = g.iloc[start:start + win]
            # label by majority vote inside window
            label = chunk["label"].mode().iloc[0]
            xs.append(chunk[channels].to_numpy().T)
            metas.append({
                "session_id": keys[0],
                "player_id": keys[1],
                "side": keys[2],
                "start_timestamp_us": int(chunk["timestamp_us"].iloc[0]),
                "end_timestamp_us": int(chunk["timestamp_us"].iloc[-1]),
                "label": label,
            })
    if not xs:
        return np.empty((0, len(channels), win)), pd.DataFrame(metas)
    return np.stack(xs), pd.DataFrame(metas)
