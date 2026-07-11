from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from lykon_motion.models.train_sklearn import train_random_forest

result = train_random_forest(ROOT / "data" / "processed" / "features.csv", ROOT / "outputs" / "reports")
print(result["report"])
print("Saved:", result["model_path"])
