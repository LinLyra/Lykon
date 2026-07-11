from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


OUT_DIR = Path("data/processed/our_data")
REPORT_DIR = Path("outputs/reports/our_data")
RELEASE_PATH = OUT_DIR / "event_release_points.csv"
WINDOW_INDEX_PATH = OUT_DIR / "shot_cycle_windows_index.csv"


def robust_z(values: pd.Series) -> pd.Series:
    values = values.astype(float)
    median = values.median()
    mad = (values - median).abs().median()
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-9:
        scale = values.std(ddof=0)
    if not np.isfinite(scale) or scale < 1e-9:
        scale = 1.0
    return (values - median) / scale


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not RELEASE_PATH.exists():
        raise FileNotFoundError(RELEASE_PATH)

    release = pd.read_csv(RELEASE_PATH)

    # 兼容旧字段名。原值不是概率，而是峰值显著度。
    if "release_confidence" in release.columns:
        release = release.rename(
            columns={"release_confidence": "release_peak_strength"}
        )

    required = {
        "window_id",
        "release_index",
        "release_time_from_window_start_s",
        "release_peak_strength",
    }
    missing = sorted(required - set(release.columns))
    if missing:
        raise ValueError(f"release 文件缺少字段: {missing}")

    review = release.copy()
    review["release_index_robust_z"] = robust_z(
        review["release_index"]
    )
    review["peak_strength_robust_z"] = robust_z(
        review["release_peak_strength"]
    )

    median_release = float(review["release_index"].median())
    review["release_index_deviation_frames"] = (
        review["release_index"] - median_release
    ).abs()

    # 当前没有真实标签，所以这里只能做预测稳定性质检。
    review["timing_outlier"] = (
        review["release_index_robust_z"].abs() > 3.0
    )
    review["weak_peak"] = (
        review["release_peak_strength"] < 0.50
    )
    review["very_weak_peak"] = (
        review["release_peak_strength"] < 0.25
    )

    review["qa_label"] = "stable_candidate"
    review.loc[
        review["weak_peak"],
        "qa_label",
    ] = "low_strength_review"
    review.loc[
        review["timing_outlier"],
        "qa_label",
    ] = "timing_outlier_review"
    review.loc[
        review["timing_outlier"] & review["weak_peak"],
        "qa_label",
    ] = "timing_and_strength_review"

    review["predicted_correct"] = ""
    review["manual_correct_release_index"] = ""
    review["manual_error_frames"] = ""
    review["manual_notes"] = ""

    # 自动建议优先复核名单
    review["review_priority"] = 3
    review.loc[review["weak_peak"], "review_priority"] = 2
    review.loc[review["very_weak_peak"], "review_priority"] = 1
    review.loc[review["timing_outlier"], "review_priority"] = 1

    review = review.sort_values(
        ["review_priority", "release_peak_strength"],
        ascending=[True, True],
    ).reset_index(drop=True)

    output_path = OUT_DIR / "imu_release_prediction_review.csv"
    review.to_csv(output_path, index=False)

    # 可视化每一个 IMU 预测点
    plot_df = release.copy().reset_index(drop=True)
    plot_df["qa_label"] = review.set_index("window_id").loc[
        plot_df["window_id"], "qa_label"
    ].to_numpy()

    x = np.arange(len(plot_df))
    plt.figure(figsize=(20, 8))
    plt.scatter(
        x,
        plot_df["release_index"],
        s=50,
    )
    plt.axhline(
        plot_df["release_index"].median(),
        linestyle="--",
        linewidth=1.0,
        label="Median release index",
    )

    for i, row in plot_df.iterrows():
        plt.text(
            i,
            row["release_index"] + 0.8,
            str(row["window_id"]),
            rotation=90,
            fontsize=7,
            ha="center",
        )

    plt.xlabel("Prediction order")
    plt.ylabel("Predicted release index")
    plt.title(
        "All IMU release predictions — per-window review"
    )
    plt.legend()
    plt.tight_layout()
    report_path = REPORT_DIR / "imu_release_review.png"
    plt.savefig(report_path, dpi=180)
    plt.close()

    print("\nQA label counts:")
    print(review["qa_label"].value_counts())
    print("\nHighest-priority review items:")
    print(
        review[
            [
                "window_id",
                "release_index",
                "release_peak_strength",
                "qa_label",
                "review_priority",
            ]
        ].head(15).to_string(index=False)
    )
    print("\nSaved:")
    print(output_path)
    print(report_path)
    print(
        "\n当前 predicted_correct 为空，因为没有真实离手标签。"
        "人工核对后填写 1/0，才可计算真正 precision/recall/error。"
    )


if __name__ == "__main__":
    main()
