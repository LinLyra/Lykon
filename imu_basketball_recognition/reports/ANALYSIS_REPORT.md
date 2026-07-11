# IMU 篮球动作识别系统 — 分析报告

> 生成时间：基于 lykon_dataset (owen + ryan) 四节点 IMU 数据

## 一、数据概况

| 项目 | 数值 |
|---|---|
| 受试者 | owen, ryan |
| 节点配置 | 4节点 × 6轴 (RU, RF, LF, LU) @ 50Hz |
| 动作类别 | 接球(catch)、高位传球(pass_high)、防守(defense)、左右运球(dribble_left_right)、右手拍一次球(dribble_right_once)、投篮(shot)、Null |
| 总窗口数 | 3,469 |
| 动作类窗口 | ~1,926 |
| 特征维度 (原始) | 1,350 |
| 特征维度 (筛选后) | 50 |

### 样本分布

| subject | catch | defense | dribble_left_right | dribble_right_once | null | pass_high | random | shot |
|---|---|---|---|---|---|---|---|---|
| owen | 59 | 114 | 135 | 75 | 739 | 6 | 0 | 40 |
| ryan | 108 | 187 | 128 | 59 | 1,537 | 64 | 166 | 52 |

**观察**：Null 类窗口占比过高（约 55%），连续类（防守、运球）窗口远多于暂态类（接球、投篮、拍球），存在显著类不平衡。

## 二、Phase 2 — 预处理验证

### 三层通道构建

- **L1 分轴**：20Hz 零相位低通滤波后的 ax/ay/az/gx/gy/gz
- **L2 重力系**：通过 0.2Hz 极低通估计平均重力方向，对 0.5Hz 高通加速度做投影分解 → a_vert / a_horiz
- **L3 模值**：|a|、|g| 及各方向 jerk
- **姿态**：互补滤波输出 roll/pitch/yaw

**验证结果**：
- a_vert 均值 ≈ 0（重力去除成功）
- 无静止段 → 采用"长期平均重力方向"策略，姿态为固定佩戴姿态

## 三、Phase 3 — 标注与滑窗

### 离散事件录制 (R1/R4)

采用自适应能量分割（基于 p50 + 0.25×动态范围阈值）+ 循环序列先验：

| 录制 | 检测到事件 | 循环数 | 每循环事件数 |
|---|---|---|---|
| owen R1 | 159 | 53 | 3 |
| ryan R1 | 89 | 29 | 3 |
| owen R4 | 90 | 45 | 2 |
| ryan R4 | 97 | 48 | 2 |

### 混合窗剔除

R1/R4 的混合窗剔除比例较高（~27-49%），因为事件之间没有明显静止间隙。

## 四、Phase 5 — 特征筛选

从 1,350 维筛选至 50 维，方法：ANOVA F 值 + RF 重要性 + 相关性去冗余 (|r| < 0.95)。

**Top 10 特征**：
1. `jerk_x_node3_max` — LF 水平加加速度峰值
2. `jerk_x_node3_min`
3. `jerk_x_node3_mean`
4. `yaw_node1_mean` — RU 偏航角均值
5. `jerk_x_node1_mean` — RU 水平加加速度
6. `g_mag_node2_rms` — RF 角速度模值 RMS
7. `gyro_node2_energy_x_ratio` — RF 陀螺 x 轴能量占比
8. `pitch_node3_rms` — LF 俯仰角 RMS
9. `gz_node2_rms` — RF 陀螺 z 轴 RMS
10. `g_mag_node3_band_low` — LF 低频角速度能量

## 五、Phase 6A — 类间可分性

### 轮廓系数

- 整体：-0.096（类间重叠显著）
- dribble_left_right：+0.234（唯一正分，说明运球动作在特征空间中相对紧凑且可分离）
- 其余类别均为负值，说明存在较大重叠

### 高危类对（Cohen's d 较低）

| 类对 | Top d | 说明 |
|---|---|---|
| dribble_right_once vs shot | 0.49 | 单次拍球与投篮最难区分 |
| catch vs dribble_right_once | 0.78 | 接球与拍球 |
| catch vs pass_high | 0.74 | 接球与传球 |
| catch vs shot | 1.07 | 接球与投篮 |

### 易分类对（Cohen's d 较高）

| 类对 | Top d | 说明 |
|---|---|---|
| defense vs random | 9.41 | 防守 vs 随机（预期外动作差异大） |
| catch vs random | 6.19 | 接球 vs 随机 |
| dribble_right_once vs random | 7.47 | 拍球 vs 随机 |

### 签名假设验证

- **H4 (运球周期性)**：`dribble_left_right` 轮廓系数为正，gy_node3_rms 等周期性特征有效
- **H5 (单侧性)**：`unilateral_left_right_ratio` 在 defense vs shot 区分中排名靠前 (d=2.15)
- **H6 (腾空)**：airborne 特征在 shot 相关类对中尚未进入 top 5，可能与姿态解算精度有限有关

## 六、Phase 6B — 跨人对比

### 逐类跨人稳定性

| 动作 | 平均 Cohen's d | 最稳定特征 | 最不稳定特征 |
|---|---|---|---|
| catch | 中等 | roll_node2_ptp | yaw_node1_rms |
| defense | 中等 | gz_node1_band_high | yaw_node1_rms |
| dribble_left_right | 中等 | g_mag_node3_dom_ratio | jerk_x_node3 系列 |
| dribble_right_once | 中等 | roll_node3_jerk_peak | a_vert_node4_rms |
| pass_high | 中等 | jerk_x_node3_mean | gy_node1_std |
| shot | 中等 | gyro_node2_dom_sub_ratio | jerk_x_node3_rms |

**关键发现**：`jerk_x_node3`（LF 水平加加速度）系列特征跨人稳定性差，说明两人左手小臂的运动风格差异大。

### 跨人泛化实验

| 实验 | RF 准确率 | SVM 准确率 | kNN 准确率 |
|---|---|---|---|
| 人内 — Owen | 86.4% | 86.1% | 80.5% |
| 人内 — Ryan | 86.6% | 85.5% | 83.1% |
| Owen → Ryan | 84.0% | 78.4% | 84.7% |
| Ryan → Owen | 74.4% | 83.6% | 81.3% |
| 合并 (按录制划分) | 69.1% | 64.9% | 72.9% |

**观察**：
- 人内准确率较高（~86%），但 macro_f1 低（0.28-0.39），受类不平衡影响
- 跨人泛化：Owen→Ryan 仅掉 2.4 点，Ryan→Owen 掉 12.2 点 — 存在单向差异
- 合并实验准确率降至 ~69%，按录制划分暴露了不同录制间的分布差异

## 七、Phase 7 — 分类器

### 模型对比

Random Forest 在多数场景下表现最佳或次佳，且训练速度快，推荐作为基线模型。

### 类不平衡影响

- 准确率被人内/跨人高 Null 类比例"拉高"
- macro_f1 更能反映细分类别性能：当前 0.17-0.48，远低于目标 0.90
- **关键瓶颈**：shot 和 dribble_right_once 样本过少（各约 50-100 窗）

### 混淆热点（基于合并测试集 RF）

主要混淆：
- null ↔ 各类（因为 null 数量多且特征边界模糊）
- catch ↔ pass_high（同属传接族）
- dribble_left_right ↔ dribble_right_once（同属运球族）

## 八、局限与建议

1. **样本量**：两人各 4-5 个录制，暂态类（shot、拍球）窗口仅 50-100，不足以训练稳定分类器
2. **标注精度**：R1/R4 无静止间隙，混合窗剔除率高，自动分割需人工校验
3. **姿态解算**：无静止段导致互补滤波收敛困难，L2 特征基于"平均佩戴姿态"而非动态姿态
4. **跨人结论**：Ryan→Owen 泛化显著差于 Owen→Ryan，可能与 ryan 的 "随机做" 录制引入额外风格变化有关
5. **实时原型**：当前 pipeline 单窗处理延迟约 50-100ms，满足实时要求，但需补充串口/BLE 接口

## 九、下一步

1. 增加样本量：每人每动作至少 300+ 窗口
2. 引入视频真值作为 teacher，提升标注精度
3. 尝试时序模型（LSTM / 1D-CNN）捕捉动作动态
4. 对新用户做个体归一化预研，缩小跨人差距
5. 针对运球族内部混淆，引入拍球节奏（BPM）特征
