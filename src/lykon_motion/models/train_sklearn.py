from __future__ import annotations
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def train_random_forest(features_csv: str | Path, out_dir: str | Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(features_csv)
    y = df["label"]
    drop_cols = ["session_id", "player_id", "side", "start_timestamp_us", "end_timestamp_us", "label"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    X = X.select_dtypes(include=["number"]).fillna(0)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    stratify = y_enc if len(set(y_enc)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.25, random_state=42, stratify=stratify
    )
    clf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    report = classification_report(y_test, pred, target_names=le.classes_, zero_division=0)
    cm = confusion_matrix(y_test, pred)

    joblib.dump({"model": clf, "label_encoder": le, "features": list(X.columns)}, out_dir / "rf_model.joblib")
    (out_dir / "rf_report.txt").write_text(report, encoding="utf-8")
    pd.DataFrame(cm, index=le.classes_, columns=le.classes_).to_csv(out_dir / "rf_confusion_matrix.csv")
    return {"report": report, "model_path": str(out_dir / "rf_model.joblib")}
