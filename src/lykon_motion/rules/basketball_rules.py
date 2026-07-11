from __future__ import annotations
import pandas as pd


def predict_rule_row(row: pd.Series) -> tuple[str, float]:
    """Very rough heuristic rules. Thresholds must be calibrated on real LYKON data."""
    acc_max = row.get("acc_norm_max", 0)
    acc_freq = row.get("acc_norm_dom_freq", 0)
    acc_peaks = row.get("acc_norm_peak_count", 0)
    gyro_max = row.get("gyro_norm_max", 0)
    gyro_freq = row.get("gyro_norm_dom_freq", 0)

    # collision/jump-like impact first
    if acc_max > 25 and gyro_max < 8:
        return "jump", 0.65
    # dribble has repeated rhythmic acceleration peaks
    if 1.5 <= acc_freq <= 4.5 and acc_peaks >= 3:
        return "dribble", 0.70
    # shot has larger arm rotation and longer movement
    if gyro_max > 5.0 and acc_max > 12:
        return "shot", 0.60
    # pass is shorter/sharper, often gyro peak without periodicity
    if gyro_max > 3.0 and acc_peaks <= 2:
        return "pass", 0.55
    # sprint has repeated acceleration pattern but less hand rhythm than dribble
    if acc_freq > 0.8 and acc_max > 13:
        return "sprint", 0.50
    return "idle", 0.50


def apply_rules(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    preds = out.apply(predict_rule_row, axis=1)
    out["rule_pred"] = [p[0] for p in preds]
    out["rule_confidence"] = [p[1] for p in preds]
    return out
