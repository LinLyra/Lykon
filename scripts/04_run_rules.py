from pathlib import Path
import sys
import pandas as pd
from sklearn.metrics import classification_report
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from lykon_motion.rules.basketball_rules import apply_rules

features_csv = ROOT / "data" / "processed" / "features.csv"
out_csv = ROOT / "outputs" / "reports" / "rule_predictions.csv"
df = pd.read_csv(features_csv)
out = apply_rules(df)
out_csv.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(out_csv, index=False)
print(classification_report(out["label"], out["rule_pred"], zero_division=0))
print("Saved:", out_csv)
