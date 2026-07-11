from pathlib import Path
import yaml
import pandas as pd
import numpy as np

IN_PATH = Path("data/processed/hangtime/hangtime_windows_locomotion.csv")
OUT_PATH = Path("data/processed/hangtime/rule_predictions.csv")
RULE_PATH = Path("configs/rules.yaml")

def load_rules():
    with open(RULE_PATH, "r") as f:
        return yaml.safe_load(f)["rules"]

def score_idle(row, r):
    return float(
        row["acc_mag_std"] < r["acc_mag_std_max"] and
        row["acc_mag_range"] < r["acc_mag_range_max"]
    )

def score_jump(row, r):
    return float(
        row["acc_mag_max"] > r["acc_mag_max_min"] and
        row["acc_mag_range"] > r["acc_mag_range_min"] and
        row["peak_count"] >= r["peak_count_min"]
    )

def score_dribble(row, r):
    # 现在没有 dominant_freq，先用 peak_count + std 代替
    return float(
        row["peak_count"] >= r["peak_count_min"] and
        row["acc_mag_std"] > r["acc_mag_std_min"]
    )

def score_sprint(row, r):
    return float(
        row["acc_mag_std"] > r["acc_mag_std_min"] and
        row["peak_count"] >= r["peak_count_min"]
    )

def score_collision(row, r):
    return float(row["acc_mag_max"] > r["acc_mag_max_min"])

def score_pass(row, r):
    return float(
        row["acc_mag_max"] > r["acc_mag_max_min"] and
        r["peak_count_min"] <= row["peak_count"] <= r["peak_count_max"]
    )

def score_shot(row, r):
    return float(
        row["acc_mag_max"] > r["acc_mag_max_min"] and
        row["acc_mag_range"] > r["acc_mag_range_min"] and
        row["peak_count"] >= r["peak_count_min"]
    )

SCORERS = {
    "idle": score_idle,
    "jump": score_jump,
    "dribble": score_dribble,
    "sprint": score_sprint,
    "collision": score_collision,
    "pass": score_pass,
    "shot": score_shot,
}

def predict_row(row, rules):
    scores = {}
    for label, fn in SCORERS.items():
        if label in rules:
            scores[label] = fn(row, rules[label])

    best_label = max(scores, key=scores.get)
    best_score = scores[best_label]

    if best_score == 0:
        best_label = "unknown"

    return pd.Series({
        "rule_pred": best_label,
        "rule_confidence": best_score,
        **{f"rule_{k}": v for k, v in scores.items()}
    })

def main():
    rules = load_rules()
    df = pd.read_csv(IN_PATH)

    preds = df.apply(lambda row: predict_row(row, rules), axis=1)
    out = pd.concat([df, preds], axis=1)
    out.to_csv(OUT_PATH, index=False)

    print("Saved:", OUT_PATH)
    print(out["rule_pred"].value_counts())

if __name__ == "__main__":
    main()