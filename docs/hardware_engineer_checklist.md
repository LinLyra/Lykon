# 给硬件工程师的可执行清单

## 设备配置

- 8 个 IMU = 4 个人同时训练
- 每人左右手各 1 个护臂
- 每个护臂 1 个 IMU/BNO085
- 如果有 UWB，每个玩家至少 1 个 UWB tag；如果左右手都有 UWB，需要记录 mount_position

## 必须采集字段

### IMU

- session_id
- player_id
- sensor_id
- side: left/right
- seq
- t_node_us
- host_received_us 或 t_head_recv_us
- ax, ay, az
- gx, gy, gz
- mx, my, mz

### BNO085 强烈建议输出

- qw, qx, qy, qz
- roll, pitch, yaw
- linear_acc_x, linear_acc_y, linear_acc_z
- gravity_x, gravity_y, gravity_z
- calibration_status

### UWB

- session_id
- player_id
- uwb_tag_id
- seq
- uwb_timestamp_us
- uwb_x, uwb_y, uwb_z
- uwb_quality
- uwb_anchor_count

## 采样要求

- IMU：100Hz 优先，最低 50Hz
- UWB：20Hz 优先，最低 10Hz
- 左右手时间误差：尽量 <20ms
- 所有设备必须统一 session_id
- raw 表不要写 label
- 必须保留设备本地时间 t_node_us，不要只保留蓝牙接收时间

## 第一批采集动作

先做 6 类：

- idle
- walk_run
- sprint_drive
- jump
- dribble
- shot

如果时间够，再加：

- pass
- layup

## 第一批数据量

- 最小：4人 × 6动作 × 每人每动作50次 ≈ 1200段
- 更稳：4人 × 8动作 × 每人每动作100次 ≈ 3200段

## 标注方式

至少一种：

1. 手机网页按钮打点 event_marker
2. 视频 + 口令
3. 护臂双击打点
4. 视频骨骼自动候选 + 人工确认
