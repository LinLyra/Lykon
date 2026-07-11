from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from lykon_motion.data.loaders import load_folder
from lykon_motion.data.preprocess import add_norms, filter_imu
from lykon_motion.features.windowing import make_windows
from lykon_motion.features.extract import extract_window_features
from lykon_motion.utils.config import load_config

cfg = load_config()
raw_folder = ROOT / "data" / "raw" / "our_data"
out_csv = ROOT / "data" / "processed" / "features.csv"
channels = cfg["imu_channels"]

df = load_folder(raw_folder)
df = filter_imu(df, fs=cfg["sampling_rate_hz"])
df = add_norms(df)
X, meta = make_windows(df, cfg["sampling_rate_hz"], cfg["window_seconds"], cfg["step_seconds"], channels)
features = extract_window_features(X, meta, channels, fs=cfg["sampling_rate_hz"])
out_csv.parent.mkdir(parents=True, exist_ok=True)
features.to_csv(out_csv, index=False)
print(f"Built {len(features)} windows -> {out_csv}")
