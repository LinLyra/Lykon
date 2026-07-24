# 视频数据放置约定（video_pose_expert）

把你的视频拖到这里，即可用 Google MediaPipe 骨骼提取 + Expert 训练/识别。

## 训练用（按动作分子目录，子目录名即标签）

```
data/raw/video/
├── idle/        *.mp4
├── dribble/     *.mp4
├── pass/        *.mp4
├── shot/        *.mp4
├── jump/        *.mp4
├── sprint/      *.mp4
└── test/        *.mp4   ← 无标签，供 Expert 识别（任意人员）
```

- 每个子目录放该动作的视频片段（建议每段聚焦单个球员、1–3 秒）。
- 支持格式：`.mp4 .mov .avi .mkv .m4v`。
- 动作标签体系见 `docs/action_taxonomy_lykon.md`（V1：idle/dribble/pass/shot/jump/sprint）。

## 一键训练（用你的视频）

```bash
pip install mediapipe opencv-python        # 首次需要
python -m video_pose_expert.cli train --source video --data data/raw/video --out outputs/expert_video
```

## 用训练好的 Expert 识别任意人员

```bash
# 单个视频
python -m video_pose_expert.cli predict --expert outputs/expert_video/expert.joblib --video path/to/test.mp4
# 整个 test 目录
python -m video_pose_expert.cli predict --expert outputs/expert_video/expert.joblib --test-dir data/raw/video/test
```

输出：逐窗口的 `predicted_action / class_proba / expert_match_score / within_boundary` 与人员级汇总。

> 说明：视频骨骼仅用于**训练阶段**构建真值（Teacher/Expert），与 README 的
> `docs/video_pose_training_plan.md` 一致；产品阶段最终以 IMU + UWB 为输入。
> 大数据量训练建议先按动作准备均衡样本，效果不足时继续补充视频再训练。
