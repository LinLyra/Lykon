from pathlib import Path
import json
import pandas as pd

WINDOWS_PATH = Path("data/processed/hangtime/hangtime_windows_locomotion.csv")
META_PATH = Path("data/raw/hangtime/meta.txt")
OUT_PATH = Path("data/processed/hangtime/hangtime_windows_locomotion_with_meta.csv")

def load_meta():
    # 如果 meta.txt 是标准 json，这里可以直接读
    text = META_PATH.read_text()
    meta = json.loads(text)

    rows = []
    for region, subjects in meta.items():
        for subject, info in subjects.items():
            row = {"region": region, "subject": subject}
            row.update(info)
            rows.append(row)

    return pd.DataFrame(rows)

def main():
    windows = pd.read_csv(WINDOWS_PATH)
    meta_df = load_meta()

    merged = windows.merge(meta_df, on=["subject", "region"], how="left")
    merged.to_csv(OUT_PATH, index=False)

    print("windows:", windows.shape)
    print("meta:", meta_df.shape)
    print("merged:", merged.shape)
    print(merged[["subject", "region", "height", "weight", "skill", "gender"]].head())

if __name__ == "__main__":
    main()