from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = ROOT / "docs" / "schemas"
OUT_DIRS = {
    "sessions.csv": ROOT / "data" / "raw",
    "players.csv": ROOT / "data" / "raw",
    "sensors.csv": ROOT / "data" / "raw",
    "sensor_raw.csv": ROOT / "data" / "raw" / "imu",
    "uwb_raw.csv": ROOT / "data" / "raw" / "uwb",
    "event_markers.csv": ROOT / "data" / "raw" / "markers",
    "device_status.csv": ROOT / "data" / "raw",
    "video_metadata.csv": ROOT / "data" / "raw" / "video",
    "video_frames.csv": ROOT / "data" / "raw" / "video",
    "pose_keypoints_2d.csv": ROOT / "data" / "processed" / "pose",
    "joint_angles_2d.csv": ROOT / "data" / "processed" / "pose",
    "action_labels.csv": ROOT / "data" / "labels",
    "event_labels.csv": ROOT / "data" / "labels",
    "train_windows_index.csv": ROOT / "data" / "processed" / "windows",
    "model_predictions.csv": ROOT / "outputs" / "reports",
}

def main():
    for name, out_dir in OUT_DIRS.items():
        out_dir.mkdir(parents=True, exist_ok=True)
        src = SCHEMA_DIR / name
        dst = out_dir / name
        shutil.copyfile(src, dst)
        print(f"created {dst.relative_to(ROOT)}")

if __name__ == "__main__":
    main()
