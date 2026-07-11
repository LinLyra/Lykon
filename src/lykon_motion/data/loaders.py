from pathlib import Path
import pandas as pd
from .schema import REQUIRED_COLUMNS


def load_standard_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path.name}: {missing}")
    df = df.sort_values(["session_id", "player_id", "side", "timestamp_us"]).reset_index(drop=True)
    return df


def load_folder(folder: str | Path) -> pd.DataFrame:
    folder = Path(folder)
    files = sorted(folder.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {folder}")
    return pd.concat([load_standard_csv(f) for f in files], ignore_index=True)
