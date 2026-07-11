from pathlib import Path
import re
import pandas as pd
import numpy as np

from scipy.signal import find_peaks
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

RAW_DIR = Path("data/raw/hangtime")
OUT_DIR = Path("data/processed/hangtime")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 先跑 locomotion；之后可以改成 basketball / coarse
LABEL_COL = "locomotion"

WINDOW_SIZE = 50   # 约2秒窗口，先这样
STEP_SIZE = 25     # 50% overlap


def parse_filename(path: Path):
    # 例如 a0da_eu.csv / 10f0_na.csv
    m = re.match(r"(.+?)_(eu|na)\.csv$", path.name)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def load_all_csv():
    files = sorted(RAW_DIR.glob("*.csv"))
    all_dfs = []

    for f in files:
        subject, region = parse_filename(f)
        if subject is None:
            continue

        df = pd.read_csv(f)
        df["subject"] = subject
        df["region"] = region
        df["source_file"] = f.name
        all_dfs.append(df)

    merged = pd.concat(all_dfs, ignore_index=True)
    return merged


def add_features_base(df):
    df = df.copy()
    df["acc_mag"] = np.sqrt(df["acc_x"]**2 + df["acc_y"]**2 + df["acc_z"]**2)
    return df


def window_features(w):
    x = w["acc_mag"].values
    peaks, _ = find_peaks(x, distance=3)

    return {
        "acc_x_mean": w["acc_x"].mean(),
        "acc_y_mean": w["acc_y"].mean(),
        "acc_z_mean": w["acc_z"].mean(),
        "acc_x_std": w["acc_x"].std(),
        "acc_y_std": w["acc_y"].std(),
        "acc_z_std": w["acc_z"].std(),
        "acc_x_max": w["acc_x"].max(),
        "acc_y_max": w["acc_y"].max(),
        "acc_z_max": w["acc_z"].max(),
        "acc_x_min": w["acc_x"].min(),
        "acc_y_min": w["acc_y"].min(),
        "acc_z_min": w["acc_z"].min(),
        "acc_mag_mean": w["acc_mag"].mean(),
        "acc_mag_std": w["acc_mag"].std(),
        "acc_mag_max": w["acc_mag"].max(),
        "acc_mag_min": w["acc_mag"].min(),
        "acc_mag_range": w["acc_mag"].max() - w["acc_mag"].min(),
        "peak_count": len(peaks),
    }


def build_windows(df):
    rows = []

    group_cols = ["subject", "region", "source_file"]

    for keys, g in df.groupby(group_cols):
        subject, region, source_file = keys
        g = g.reset_index(drop=True)

        for start in range(0, len(g) - WINDOW_SIZE + 1, STEP_SIZE):
            w = g.iloc[start:start + WINDOW_SIZE]

            label = w[LABEL_COL].mode().iloc[0]
            if label == "not_labeled":
                continue

            feats = window_features(w)
            feats.update({
                "subject": subject,
                "region": region,
                "source_file": source_file,
                "label": label,
                "start_index": start,
                "end_index": start + WINDOW_SIZE,
                "start_timestamp": w["timestamp"].iloc[0],
                "end_timestamp": w["timestamp"].iloc[-1],
            })
            rows.append(feats)

    return pd.DataFrame(rows)


def train_random_forest(windows):
    feature_cols = [
        c for c in windows.columns
        if c not in [
            "subject", "region", "source_file", "label",
            "start_index", "end_index", "start_timestamp", "end_timestamp"
        ]
    ]

    X = windows[feature_cols].fillna(0).values

    le = LabelEncoder()
    y = le.fit_transform(windows["label"].values)

    # 按 subject 分组切分，避免同一个人同时出现在训练和测试
    groups = windows["subject"].values
    splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))

    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    clf.fit(X[train_idx], y[train_idx])
    pred = clf.predict(X[test_idx])

    print("\n=== Classification Report ===")
    print(classification_report(y[test_idx], pred, target_names=le.classes_))

    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y[test_idx], pred))

    importances = pd.DataFrame({
        "feature": feature_cols,
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False)

    importances.to_csv(OUT_DIR / "rf_feature_importance.csv", index=False)
    print("\nTop features:")
    print(importances.head(10))


def main():
    print("Loading CSV files...")
    df = load_all_csv()
    print("Merged shape:", df.shape)
    print("Columns:", df.columns.tolist())

    merged_path = OUT_DIR / "hangtime_merged.csv"
    df.to_csv(merged_path, index=False)
    print("Saved:", merged_path)

    print("\nLabel distribution:")
    print(df[LABEL_COL].value_counts(dropna=False))

    df = add_features_base(df)

    print("\nBuilding windows...")
    windows = build_windows(df)
    print("Windows shape:", windows.shape)
    print(windows["label"].value_counts())

    windows_path = OUT_DIR / f"hangtime_windows_{LABEL_COL}.csv"
    windows.to_csv(windows_path, index=False)
    print("Saved:", windows_path)

    if len(windows) == 0:
        raise ValueError("没有可训练窗口。你可能选了一个几乎全是 not_labeled 的 LABEL_COL。")

    train_random_forest(windows)


if __name__ == "__main__":
    main()