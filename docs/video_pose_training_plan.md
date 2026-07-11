# Video/Pose 训练真值方案

视频不是未来产品的必需输入，而是训练阶段的 Ground Truth Teacher。

## 训练阶段

Video + Pose + Manual Label + IMU + UWB
→ 生成高质量 action_labels / event_labels
→ 训练 IMU/UWB student model

## 产品阶段

IMU + UWB
→ student model
→ action/event prediction
→ 2D/NBA2K-style replay

## 推荐 Pose 输出

优先用 MediaPipe Pose 或 YOLO Pose。第一版只需要 2D keypoints：

- nose
- left/right shoulder
- left/right elbow
- left/right wrist
- left/right hip
- left/right knee
- left/right ankle

## Pose 能校正什么？

- 投篮开始/出手/随球动作
- 肩肘腕角度
- 上篮左右手
- 左脚/右脚起跳（视频可做，IMU双护臂难做）
- 防守姿态
- 跳跃落地

## 对齐要求

video_metadata.csv 记录 video_start_timestamp_us。
video_frames.csv 记录每一帧 frame_id 和 timestamp_us。
pose_keypoints_2d.csv 使用同一个 timestamp_us 与 IMU/UWB 对齐。
