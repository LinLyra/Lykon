# LYKON 数据表设计

## 1. sessions

一场训练、球局或采集实验。

| 字段 | 含义 |
|---|---|
| session_id | 唯一场次 ID |
| court_id | 球场 ID，可为空 |
| start_time | 开始时间 |
| end_time | 结束时间 |
| mode | training / pickup_game / lab_collection |
| sampling_rate_hz | IMU采样率 |
| notes | 备注 |

## 2. sensor_raw

护臂原始数据。产品最核心资产。

| 字段 | 含义 |
|---|---|
| session_id | 场次 ID |
| player_id | 球员 ID |
| sensor_id | 设备 ID |
| side | left / right |
| seq | 设备序号 |
| timestamp_us | 统一时间戳 |
| ax, ay, az | 加速度 |
| gx, gy, gz | 角速度 |
| mx, my, mz | 磁场 |
| qw, qx, qy, qz | 四元数，建议 BNO085 输出 |
| roll, pitch, yaw | 姿态角，建议输出 |
| uwb_x, uwb_y, uwb_z | UWB位置 |
| uwb_quality | 定位质量 |
| label | 训练数据阶段的动作标签 |

## 3. action_labels

人工标注、视频骨骼标注、模型预测动作都放这里。

| 字段 | 含义 |
|---|---|
| session_id | 场次 ID |
| player_id | 球员 ID |
| start_timestamp_us | 动作开始 |
| end_timestamp_us | 动作结束 |
| label | 动作标签 |
| confidence | 置信度 |
| source | manual / rule / xgboost / tsai / video_pose |

## 4. game_events

由动作推断出的篮球事件。

| 字段 | 含义 |
|---|---|
| session_id | 场次 ID |
| event_id | 事件 ID |
| start_timestamp_us | 开始 |
| end_timestamp_us | 结束 |
| event_type | pass / shot / rebound / drive / steal 等 |
| actor_player_id | 主体球员 |
| target_player_id | 传球目标等 |
| confidence | 置信度 |
| source | rule_engine / model_engine / manual |

## 5. replay_timeline

给 NBA2K 风格回放引擎用的时间线。

| 字段 | 含义 |
|---|---|
| session_id | 场次 ID |
| t | 秒级时间 |
| player_id | 球员 |
| x, y | 球场坐标 |
| animation_state | run / jump / shot / pass / idle |
| ball_owner | 当前推断持球人 |
| caption | AI解说文本 |
