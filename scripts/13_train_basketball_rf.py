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

LABEL_COL = "basketball"
WINDOW_SIZE = 50
STEP_SIZE = 25

def parse_filename(path):
    m = re.match(r"(.+?)_(eu|na)\.csv$", path.name)
    if not m:
        return None, None
    return m.group(1), m.group(2)

def load_all_csv():
    rows = []
    for f in sorted(RAW_DIR.glob("*.csv")):
        subject, region = parse_filename(f)
        if subject is None:
            continue
        df = pd.read_csv(f)
        df["subject"] = subject
        df["region"] = region
        df["source_file"] = f.name
        rows.append(df)
    return pd.concat(rows, ignore_index=True)

def add_base(df):
    df = df.copy()
    df["acc_mag"] = np.sqrt(df.acc_x**2 + df.acc_y**2 + df.acc_z**2)
    return df

def window_features(w):
    x = w["acc_mag"].values
    peaks, _ = find_peaks(x, distance=3)
    return {
        "acc_x_mean": w.acc_x.mean(),
        "acc_y_mean": w.acc_y.mean(),
        "acc_z_mean": w.acc_z.mean(),
        "acc_x_std": w.acc_x.std(),
        "acc_y_std": w.acc_y.std(),
        "acc_z_std": w.acc_z.std(),
        "acc_mag_mean": w.acc_mag.mean(),
        "acc_mag_std": w.acc_mag.std(),
        "acc_mag_max": w.acc_mag.max(),
        "acc_mag_min": w.acc_mag.min(),
        "acc_mag_range": w.acc_mag.max() - w.acc_mag.min(),
        "peak_count": len(peaks),
    }

def build_windows(df):
    rows = []
    for keys, g in df.groupby(["subject", "region", "source_file"]):
        subject, region, source_file = keys
        g = g.reset_index(drop=True)

        for start in range(0, len(g) - WINDOW_SIZE + 1, STEP_SIZE):
            w = g.iloc[start:start+WINDOW_SIZE]
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
            })
            rows.append(feats)
    return pd.DataFrame(rows)

def main():
    df = load_all_csv()
    print("Raw:", df.shape)
    print(df[LABEL_COL].value_counts(dropna=False))

    df = add_base(df)
    windows = build_windows(df)

    out_path = OUT_DIR / "hangtime_windows_basketball.csv"
    windows.to_csv(out_path, index=False)

    print("Windows:", windows.shape)
    print(windows.label.value_counts())

    if len(windows) == 0:
        print("没有 basketball 可训练标签，说明该列大多是 not_labeled。")
        return

    feature_cols = [c for c in windows.columns if c not in [
        "subject", "region", "source_file", "label", "start_index", "end_index"
    ]]

    X = windows[feature_cols].fillna(0).values
    le = LabelEncoder()
    y = le.fit_transform(windows.label.values)
    groups = windows.subject.values

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

    print("\n=== Basketball Classification Report ===")
    print(classification_report(y[test_idx], pred, target_names=le.classes_))

    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y[test_idx], pred))

    fi = pd.DataFrame({
        "feature": feature_cols,
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False)

    fi.to_csv(OUT_DIR / "rf_feature_importance_basketball.csv", index=False)
    print(fi.head(10))

if __name__ == "__main__":
    main()