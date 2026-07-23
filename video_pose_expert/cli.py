"""命令行入口：训练 Expert / 用 Expert 识别 / 端到端 demo。

    # 端到端 demo（合成数据，无需 mediapipe/网络，保证可跑通）
    python -m video_pose_expert.cli demo

    # 训练（数据源可选 synthetic / spacejam / video）
    python -m video_pose_expert.cli train --source spacejam --data data/raw/spacejam --out outputs/expert_spacejam
    python -m video_pose_expert.cli train --source video --data data/raw/video --out outputs/expert_video

    # 用训练好的 Expert 识别"任意人员"的测试数据
    python -m video_pose_expert.cli predict --expert outputs/expert/expert.joblib --video path/to/test.mp4
    python -m video_pose_expert.cli predict --expert outputs/expert/expert.joblib --test-dir data/raw/video/test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .features import features_dataframe
from .expert import Expert, train_expert
from .datasets.synthetic import generate_synthetic


def _load_samples(source: str, data: str | None, max_per_class: int | None):
    if source == "synthetic":
        return generate_synthetic(n_per_class=max_per_class or 60)
    if source == "spacejam":
        from .datasets.spacejam import load_spacejam
        return load_spacejam(data, to_lykon=True, max_per_class=max_per_class)
    if source == "video":
        from .datasets.video_folder import load_labeled_clips
        return load_labeled_clips(data)
    raise ValueError(f"未知 source: {source}")


def cmd_train(args):
    samples = _load_samples(args.source, args.data, args.max_per_class)
    print(f"[train] 载入 {len(samples)} 个样本（source={args.source}）")
    feat_df = features_dataframe(samples)
    metrics = train_expert(feat_df, args.out)
    print("\n===== 训练结果 =====")
    print(f"样本数    : {metrics['n_samples']}")
    print(f"特征维度  : {metrics['n_features']}")
    print(f"动作类别  : {metrics['classes']}")
    print(f"测试准确率: {metrics['test_accuracy']:.3f}")
    print(f"Macro-F1  : {metrics['macro_f1']:.3f}")
    print(f"识别边界  : {json.dumps(metrics['boundaries'], ensure_ascii=False)}")
    print("\n" + metrics["report_text"])
    print(f"Expert 已保存: {metrics['expert_path']}")
    (Path(args.out) / "metrics.json").write_text(
        json.dumps({k: v for k, v in metrics.items() if k != "report_text"},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def cmd_predict(args):
    expert = Expert.load(args.expert)
    if args.video:
        from .datasets.video_folder import sliding_windows_from_video
        samples = sliding_windows_from_video(args.video)
        pred = expert.predict_samples(samples)
    elif args.test_dir:
        samples = _collect_test(args.test_dir)
        pred = expert.predict_samples(samples)
    else:
        raise SystemExit("需指定 --video 或 --test-dir")

    out_dir = Path(args.out or "outputs/expert/predictions")
    out_dir.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out_dir / "predictions.csv", index=False)
    summary = expert.summarize(pred)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("===== Expert 识别结果 =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n逐样本明细: {out_dir / 'predictions.csv'}")
    return summary


def _collect_test(test_dir: str):
    """从测试目录收集样本：优先视频；若为 .npy 骨骼也支持。"""
    from .datasets.video_folder import extract_pose_from_video, sliding_windows, VIDEO_EXTS
    test_dir = Path(test_dir)
    samples = []
    for p in sorted(test_dir.iterdir()):
        if p.suffix.lower() in VIDEO_EXTS:
            seq, fps = extract_pose_from_video(p)
            for w in sliding_windows(seq, fps):
                w["video_id"] = p.stem
                samples.append(w)
    return samples


def cmd_demo(args):
    """端到端演示：合成数据训练 Expert，再对独立的"测试人员"识别。"""
    out = Path(args.out or "outputs/video_pose_expert")
    print("[demo] 1) 生成合成训练数据（6 个篮球动作）...")
    train_samples = generate_synthetic(n_per_class=60, seed=42)
    feat_df = features_dataframe(train_samples)
    print(f"        训练样本 {len(train_samples)}，特征维度 {feat_df.shape[1]}")

    print("[demo] 2) 训练 Expert（多类识别器 + 各动作识别边界）...")
    metrics = train_expert(feat_df, out)
    print(f"        测试准确率 {metrics['test_accuracy']:.3f} | Macro-F1 {metrics['macro_f1']:.3f}")

    print("[demo] 3) 模拟'任意人员'的测试数据（不同随机种子）并识别...")
    test_samples = generate_synthetic(n_per_class=15, seed=777)
    expert = Expert.load(out / "expert.joblib")
    pred = expert.predict_samples(test_samples)
    # 附带真实标签以核对（真实场景无标签）
    pred["true_label"] = [s["label"] for s in test_samples]
    acc = float((pred["predicted_action"] == pred["true_label"]).mean())
    pred.to_csv(out / "demo_predictions.csv", index=False)
    summary = expert.summarize(pred)
    summary["holdout_person_accuracy"] = acc
    (out / "demo_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n===== DEMO 结果 =====")
    print(f"测试集准确率(训练内划分): {metrics['test_accuracy']:.3f}")
    print(f"独立测试人员识别准确率  : {acc:.3f}")
    print(f"平均专家匹配度          : {summary['mean_expert_match']:.1f}/100")
    print(f"落在识别边界内比例      : {summary['within_boundary_rate']:.3f}")
    print(f"动作分布                : {json.dumps(summary['action_distribution'], ensure_ascii=False)}")
    print(f"\n产物目录: {out}")
    return summary


def build_parser():
    p = argparse.ArgumentParser(prog="video_pose_expert", description="LYKON 视频姿态 Expert")
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train", help="训练 Expert")
    pt.add_argument("--source", choices=["synthetic", "spacejam", "video"], default="synthetic")
    pt.add_argument("--data", default=None, help="数据集根目录（spacejam/video 必填）")
    pt.add_argument("--out", default="outputs/video_pose_expert")
    pt.add_argument("--max-per-class", type=int, default=None)
    pt.set_defaults(func=cmd_train)

    pp = sub.add_parser("predict", help="用 Expert 识别任意人员测试数据")
    pp.add_argument("--expert", required=True, help="expert.joblib 路径")
    pp.add_argument("--video", default=None)
    pp.add_argument("--test-dir", default=None)
    pp.add_argument("--out", default=None)
    pp.set_defaults(func=cmd_predict)

    pd_ = sub.add_parser("demo", help="端到端合成数据演示")
    pd_.add_argument("--out", default="outputs/video_pose_expert")
    pd_.set_defaults(func=cmd_demo)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    main()
