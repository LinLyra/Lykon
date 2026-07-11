"""
run_pipeline.py — 执行 Phase 0~6 流水线 (v2)
"""
import sys
import pandas as pd
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))

from config import DATA_RAW, REPORTS, DATA_SEGMENTS, DATA_FEATURES, FS
from io_utils import load_raw, resample_to_grid, merge_nodes, report_sync_quality
from preprocess import preprocess, check_static_baseline
from segment import (
    segment_activity, segment_by_peaks, auto_label, validate_segments,
    fix_length, plot_segmentation, save_label_table, save_segments_npz,
)
from features import extract_features, select_features
from classify import train_and_evaluate

print("=" * 60)
print("Phase 1: Load & Sync")
print("=" * 60)
nodes = load_raw(DATA_RAW)

# 同步质量图
report_sync_quality(nodes, REPORTS / "sync_check.png")

# 重采样到统一网格
nodes_resampled = resample_to_grid(nodes)

# 合并为宽表
df_merged = merge_nodes(nodes_resampled)

print("\n" + "=" * 60)
print("Phase 2: Preprocess")
print("=" * 60)
nodes_preprocessed = preprocess(nodes_resampled)
# 重新合并预处理后的数据
df_merged = merge_nodes(nodes_preprocessed)

# 静态基线检查（取前 1 秒）
static_df = df_merged[df_merged["t"] <= 1.0].copy()
if not static_df.empty:
    a_norms, g_norms, vert_vals = [], [], []
    for prefix in ["LF", "RF", "RU"]:
        for lst, suffix in [(a_norms, "_a_norm"), (g_norms, "_g_norm"), (vert_vals, "_a_vert")]:
            col = f"{prefix}{suffix}"
            if col in static_df.columns:
                lst.append(static_df[col])
    if a_norms and g_norms:
        static_combined = pd.DataFrame({
            "a_norm": pd.concat(a_norms),
            "g_norm": pd.concat(g_norms)
        })
        if vert_vals:
            static_combined["a_vert"] = pd.concat(vert_vals)
        check_static_baseline(static_combined)
    else:
        print("  [Static check] 缺少 a_norm/g_norm 列，跳过基线检查")

print("\n" + "=" * 60)
print("Phase 3: Segment & Label")
print("=" * 60)
# v2: 触发信号 prominence 已换算为物理单位 (m/s²)
segments = segment_by_peaks(df_merged, use_node=2, distance_s=1.0, prominence=5.0)

# 自动交替标注
segments = auto_label(segments, start_label="dribble")

# 校验物理规则 (v2 增强版)
segments = validate_segments(segments, df_merged)

# 定长截取
segments = fix_length(segments, df_merged)

# 保存标签表供人工修改
save_label_table(segments, REPORTS / "segment_labels.csv")

# 绘制分页校验图
plot_segmentation(df_merged, segments, REPORTS / "segmentation", max_duration=30.0)

# 保存 NPZ
save_segments_npz(segments, df_merged, DATA_SEGMENTS / "segments.npz")

print("\n" + "=" * 60)
print("Phase 4: Feature Extraction")
print("=" * 60)
features_df = extract_features(DATA_SEGMENTS / "segments.npz")
features_df.to_parquet(DATA_FEATURES / "features.parquet")
print(f"[Features] Saved -> {DATA_FEATURES / 'features.parquet'}")

print("\n" + "=" * 60)
print("Phase 5: Feature Selection")
print("=" * 60)
selected_cols = select_features(features_df, n_top=40)

print("\n" + "=" * 60)
print("Phase 6: Classification")
print("=" * 60)
summary = train_and_evaluate(features_df, selected_cols, REPORTS)

print("\n" + "=" * 60)
print("Phase 0-6 完成")
print("=" * 60)
print(f"报告目录: {REPORTS}")
print(f"数据目录: {DATA_SEGMENTS}")
print(f"特征目录: {DATA_FEATURES}")

if summary.get("pass"):
    print("\n✅ 验收通过: LOOCV 准确率 ≥ 95%")
else:
    print(f"\n⚠️ 未达标: 最佳准确率 = {summary.get('best_accuracy', 0):.3f}")
    print("建议: 检查标签准确性 / 调整特征 / 增加数据")
