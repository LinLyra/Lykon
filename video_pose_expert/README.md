# video_pose_expert — 视频姿态 Expert（动作识别边界）

用视频骨骼（**Google MediaPipe Pose / BlazePose**）作为训练阶段的
**Ground-Truth Teacher / Expert**，完成：

```
视频 → 骨骼关键点 → 关节角/动作特征 → 各动作识别边界 → Expert 推理
                                                        ↓
                              对"任意人员"的测试数据识别动作 + 给出专家匹配度
```

与 `docs/video_pose_training_plan.md`、`docs/action_taxonomy_lykon.md`、
`docs/schemas/{pose_keypoints_2d,joint_angles_2d}.csv` 完全对齐。视频姿态仅用于
训练阶段构建真值；产品阶段最终以 IMU + UWB 为输入。

## 管线组成

| 模块 | 作用 |
|---|---|
| `datasets/synthetic.py` | 合成篮球动作骨骼（easy/hard），无需 mediapipe/网络即可端到端验证 |
| `datasets/video_folder.py` | 用户视频 → MediaPipe 骨骼（33→13 点映射，复用 `scripts/29`） |
| `datasets/spacejam.py` | SpaceJam 公开数据集加载器（自适应 BODY_25 / COCO-18 / COCO-17） |
| `geometry.py` | 姿态归一化（消除体型/位置差异）+ 关节角计算 |
| `features.py` | 关节角 + 运动学信号（手腕相对高度、髋垂直位移、踝速度、左右对称…）统计特征 |
| `boundaries.py` | **各动作识别边界**：PCA + Mahalanobis 距离 + 卡方阈值（识别边界椭球） |
| `expert.py` | 多类识别器(RandomForest) + 边界 → 训练/推理 Expert |
| `schema_export.py` | 骨骼导出为仓库 schema 的 `pose_keypoints_2d.csv` / `joint_angles_2d.csv` |
| `cli.py` | `train` / `predict` / `demo` 命令行 |

## 快速开始

```bash
# 1) 端到端演示（合成数据，零依赖，验证管线正确性）
python -m video_pose_expert.cli demo

# 2) easy/hard 基准（诚实训练效果）
python scripts/40_video_pose_expert_demo.py

# 3) 用你的视频训练（需 pip install mediapipe opencv-python）
python -m video_pose_expert.cli train --source video --data data/raw/video --out outputs/expert_video

# 4) 用 SpaceJam 公开数据训练
python -m video_pose_expert.cli train --source spacejam --data data/raw/spacejam --out outputs/expert_spacejam

# 5) Expert 识别任意人员测试数据
python -m video_pose_expert.cli predict --expert outputs/expert_video/expert.joblib --test-dir data/raw/video/test
```

## 合成基准结果（参考）

| 难度 | 内部测试准确率 | 独立人员准确率 | 专家匹配度 | 边界内比例 |
|---|---|---|---|---|
| easy | 1.00 | 1.00 | 80.5/100 | 0.99 |
| hard（加噪声+关键点遮挡+个体差异） | 0.97 | 0.93 | 76.1/100 | 0.97 |

> easy 仅证明管线正确；hard 接近真实视频姿态噪声，给出有意义的判别效果。
> 真实 NBA / SpaceJam / 自有视频数据接入后即可复现同样的训练→边界→识别流程。

## "识别边界"是什么

对每个动作，在标准化特征空间里拟合一个分布并给出 Mahalanobis 距离阈值
（卡方分位数）。于是每个动作有一个"识别边界椭球"：

- 测试样本落在椭球内 ⇒ 属于该动作的专家模式（`within_boundary=True`）
- 距离越小 ⇒ 与专家动作越接近 ⇒ `expert_match_score` 越高（0~100）

Expert 同时输出多类分类结果与到**每个动作**的匹配度分布，可用于
"这个人的投篮/运球有多像专家"这类评估。

## 数据接口

见 `data/raw/video/README.md`：按 `data/raw/video/<动作>/*.mp4` 拖入即用；
`test/` 目录放无标签视频供识别。效果不足时继续补充视频再训练。
