"""video_pose_expert 端到端基准脚本。

跑两档合成数据，给出诚实的训练效果对比：
    easy —— 类间高度可分，验证管线正确性
    hard —— 加噪声 + 关键点遮挡 + 大个体差异，接近真实视频姿态

并演示 Expert 对独立"测试人员"的识别与专家匹配度。
产物写入 outputs/video_pose_expert/。

用法：
    python scripts/40_video_pose_expert_demo.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from video_pose_expert.datasets.synthetic import generate_synthetic
from video_pose_expert.features import features_dataframe
from video_pose_expert.expert import Expert, train_expert


def run_one(difficulty: str, out_dir: Path) -> dict:
    train_samples = generate_synthetic(n_per_class=80, seed=42, difficulty=difficulty)
    test_samples = generate_synthetic(n_per_class=20, seed=777, difficulty=difficulty)

    feat_df = features_dataframe(train_samples)
    metrics = train_expert(feat_df, out_dir / difficulty)

    expert = Expert.load(out_dir / difficulty / "expert.joblib")
    pred = expert.predict_samples(test_samples)
    pred["true_label"] = [s["label"] for s in test_samples]
    acc = float((pred["predicted_action"] == pred["true_label"]).mean())
    pred.to_csv(out_dir / difficulty / "holdout_predictions.csv", index=False)
    summary = expert.summarize(pred)

    return {
        "difficulty": difficulty,
        "n_train": len(train_samples),
        "n_features": metrics["n_features"],
        "internal_test_accuracy": round(metrics["test_accuracy"], 3),
        "macro_f1": round(metrics["macro_f1"], 3),
        "holdout_person_accuracy": round(acc, 3),
        "mean_expert_match": round(summary["mean_expert_match"], 1),
        "within_boundary_rate": round(summary["within_boundary_rate"], 3),
        "report_text": metrics["report_text"],
    }


def main():
    out_dir = ROOT / "outputs" / "video_pose_expert"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for diff in ["easy", "hard"]:
        print(f"\n{'='*60}\n难度: {diff}\n{'='*60}")
        r = run_one(diff, out_dir)
        results[diff] = {k: v for k, v in r.items() if k != "report_text"}
        print(f"内部测试准确率 : {r['internal_test_accuracy']}")
        print(f"Macro-F1       : {r['macro_f1']}")
        print(f"独立人员准确率 : {r['holdout_person_accuracy']}")
        print(f"平均专家匹配度 : {r['mean_expert_match']}/100")
        print(f"边界内比例     : {r['within_boundary_rate']}")
        print("\n分类报告(内部测试集):")
        print(r["report_text"])

    (out_dir / "benchmark.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n汇总已写入: {out_dir / 'benchmark.json'}")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
