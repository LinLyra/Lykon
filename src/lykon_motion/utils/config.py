from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[3]


def load_config(path: str | Path = ROOT / "configs" / "pipeline.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
