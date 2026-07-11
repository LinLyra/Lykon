# LYKON Basketball Action Taxonomy

## V1 必做标签

| label | 中文 | 说明 | 主要信号 |
|---|---|---|---|
| idle | 静止/等待 | 无明显动作 | 低 acc_std，低 UWB velocity |
| dribble | 运球 | 单臂周期性触球 | peak_count，dominant frequency |
| pass | 传球 | 短促爆发，双臂协同 | acc peak，arm symmetry |
| shot | 投篮 | 举臂、出手、随球 | acc range，gyro peak，follow-through |
| jump | 跳跃 | 起跳/落地冲击 | acc max，microgravity |
| sprint | 冲刺/突破 | 高速位移 | UWB velocity，arm swing |

## V2 扩展标签

layup, catch, rebound, defense_slide, collision, stop, crossover

## V3 细分标签

set_shot, jump_shot, pull_up, catch_and_shoot,
chest_pass, bounce_pass, overhead_pass,
low_dribble, high_dribble, behind_back, between_legs,
spin_move, left_hand_layup, right_hand_layup,
block, steal, box_out

## 训练原则

1. V1 先做稳定，不追求 V3。
2. V2/V3 通过事件序列识别，不只看单个窗口。
3. 视频姿态和球检测只作为训练 Teacher，不作为产品强依赖。
4. 产品阶段默认输入 IMU + UWB。