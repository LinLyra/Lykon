from pathlib import Path
import yaml
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, welch

IN_PATH = Path("data/processed/hangtime/hangtime_windows_basketball.csv")
OUT_PATH = Path("data/processed/hangtime/rule_predictions_basketball_v2.csv")
RULE_PATH = Path("configs/rules_v2.yaml")

def load_rules():
    with open(RULE_PATH, "r") as f:
        return yaml.safe_load(f)

def safe_get(row, col, default=np.nan):
    return row[col] if col in row.index else default

def rule_idle(row, th):
    return (
        row["acc_mag_std"] <= th["acc_std_max"]
        and row["acc_mag_range"] <= th["acc_range_max"]
    )

def rule_dribble(row, th):
    return (
        row["acc_mag_std"] >= th["acc_std_min"]
        and row["peak_count"] >= th["peak_count_min_short"]
        and row["peak_count"] <= th["peak_count_max_short"]
    )

def rule_shot(row, th):
    return (
        row["acc_mag_max"] >= th["acc_max_min"]
        and row["acc_mag_range"] >= th["acc_range_min"]
    )

def rule_pass(row, th):
    return (
        row["acc_mag_max"] >= th["acc_max_min"]
        and row["peak_count"] >= th["peak_count_min"]
        and row["peak_count"] <= th["peak_count_max"]
    )

def rule_jump(row, th):
    return (
        row["acc_mag_max"] >= th["acc_max_min"]
        and row["acc_mag_range"] >= th["acc_range_min"]
    )

def rule_sprint(row, th):
    uwb_vel = safe_get(row, "uwb_vel_xy", np.nan)
    imu_ok = row["acc_mag_std"] >= th["acc_std_min"]
    if not np.isnan(uwb_vel):
        return imu_ok and uwb_vel >= th["uwb_vel_min"]
    return imu_ok

def rule_collision(row, th):
    return (
        row["acc_mag_max"] >= th["acc_max_min"]
        and row["peak_count"] <= th["peak_count_max"]
    )

RULE_FUNCS = {
    "idle": rule_idle,
    "dribble": rule_dribble,
    "shot": rule_shot,
    "pass": rule_pass,
    "jump": rule_jump,
    "sprint": rule_sprint,
    "collision": rule_collision,
}

def predict_rule(row, rules):
    ths = rules["thresholds"]
    priority = rules["priority"]

    scores = {}
    for label, fn in RULE_FUNCS.items():
        if label in ths:
            scores[label] = int(fn(row, ths[label]))

    for label in priority:
        if scores.get(label, 0) == 1:
            return pd.Series({
                "rule_pred": label,
                "rule_confidence": 1.0,
                **{f"rule_{k}": v for k, v in scores.items()}
            })

    return pd.Series({
        "rule_pred": "unknown",
        "rule_confidence": 0.0,
        **{f"rule_{k}": v for k, v in scores.items()}
    })

def main():
    rules = load_rules()
    df = pd.read_csv(IN_PATH)

    preds = df.apply(lambda r: predict_rule(r, rules), axis=1)
    out = pd.concat([df, preds], axis=1)
    out.to_csv(OUT_PATH, index=False)

    print("Saved:", OUT_PATH)
    print("\nRule prediction distribution:")
    print(out["rule_pred"].value_counts())

    if "label" in out.columns:
        print("\nCrosstab label vs rule_pred:")
        print(pd.crosstab(out["label"], out["rule_pred"], normalize="index").round(2))

if __name__ == "__main__":
    main()