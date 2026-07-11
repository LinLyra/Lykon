# LYKON Motion Lab v2

篮球智能护臂 / IMU + UWB + Video Pose 训练数据仓库。

这个仓库的目标不是做一个普通 demo，而是搭建未来产品最核心的“动作识别与篮球数字孪生数据底座”。

## 你现在要先跑通什么？

```bash
conda create -n lykon python=3.10 -y
conda activate lykon
pip install -r requirements.txt

python scripts/00_check_env.py
python scripts/05_create_empty_tables.py
python scripts/01_generate_dummy_data.py
python scripts/02_build_features.py
python scripts/03_train_baseline.py
python scripts/04_run_rules.py
```

## v2 新增了什么？

相比 v1，v2 新增了完整的训练真值数据系统：

- video_metadata.csv
- video_frames.csv
- pose_keypoints_2d.csv
- joint_angles_2d.csv
- event_labels.csv
- train_windows_index.csv
- model_predictions.csv

也就是说，仓库现在支持：

```text
训练阶段：IMU + UWB + Video + Pose + Manual Label
产品阶段：IMU + UWB → 模型预测动作/事件 → 回放
```

## 文档入口

- `docs/data_schema_v2.md`：完整数据表设计
- `docs/hardware_engineer_checklist.md`：给硬件工程师的清单
- `docs/video_pose_training_plan.md`：视频骨骼作为 Teacher 的训练方案
- `docs/schemas/`：所有 CSV 表头模板
- `docs/action_taxonomy.md`：动作标签体系
- `docs/algorithm_plan.md`：规则、XGBoost、tsai 路线

## 数据目录

```text
data/
  raw/
    imu/              # sensor_raw.csv
    uwb/              # uwb_raw.csv
    video/            # video_metadata.csv, video_frames.csv
    markers/          # event_markers.csv
  processed/
    pose/             # pose_keypoints_2d.csv, joint_angles_2d.csv
    windows/          # train_windows_index.csv
    features/         # features.csv
  labels/             # action_labels.csv, event_labels.csv
```

## 第一批动作建议

先采 6 类：

```text
idle
walk_run
sprint_drive
jump
dribble
shot
```

第二批再加：

```text
pass
layup
catch
rebound
defense_slide
collision
```

## 核心原则

1. raw 表只存原始数据，不混 label。
2. label 单独用 action_labels.csv / event_labels.csv 记录。
3. 所有数据必须有 session_id。
4. 所有硬件数据必须有 timestamp_us / t_node_us。
5. 视频骨骼是训练老师，不是产品必需输入。
