# LYKON Motion Lab v2 — Multimodal Data Schema

目标：同时支持硬件真实产品数据（IMU/UWB）和训练真值数据（Video/Pose/Manual Labels）。

核心原则：

1. raw 表只放原始数据，不混 label。
2. label 用 start_time_us / end_time_us 独立记录。
3. 所有表必须有 session_id；涉及个人必须有 player_id。
4. 训练阶段可以使用 video/pose 做 Ground Truth；产品阶段默认只依赖 IMU + UWB。
5. 时间同步优先级：t_node_us > synced_timestamp_us > host_received_us > video_timestamp_us。

## 表清单

### 硬件侧必须交付

- sessions.csv
- players.csv
- sensors.csv
- sensor_raw.csv
- uwb_raw.csv
- event_markers.csv
- device_status.csv

### 视频/标注/训练侧

- video_metadata.csv
- video_frames.csv
- pose_keypoints_2d.csv
- joint_angles_2d.csv
- action_labels.csv
- event_labels.csv
- train_windows_index.csv
- model_predictions.csv

## 数据流

sensor_raw.csv + uwb_raw.csv + video_metadata.csv
→ 时间同步
→ pose_keypoints_2d.csv / joint_angles_2d.csv
→ action_labels.csv / event_labels.csv
→ train_windows_index.csv
→ features.csv / windows.npy
→ rules / RandomForest / XGBoost / tsai
→ model_predictions.csv
