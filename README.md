# LYKON Motion Lab v2 — 篮球 IMU 动作识别数据底座

篮球智能护臂 / IMU 多节点动作识别与数字孪生数据仓库。支持多受试者 IMU 数据采集、标定、特征提取与跨人动作识别模型训练。

---

## 一、数据集概况

当前采集了 **3 名受试者** 的篮球动作数据，每人佩戴 4 节点 IMU（右臂上/下，左臂上/下）：

| 受试者 | 录制文件 | 动作 |
|--------|----------|------|
| **owen** | 4 个 | 接球-拍球-跳投、左右运球、防守、高位传球 |
| **ryan** | 5 个 | 接球-拍球-投球、左右运球、防守、高位传球、随机做（开集测试） |
| **white** | 4 个 | 接球-拍球-投球、左右运球、防守、高位传球 |

每个录制文件包含 4 节点（node1~node4）的原始 IMU 数据：
- `acc_x/y/z` — 加速度（LSB）
- `gyr_x/y/z` — 陀螺仪（LSB）
- `master_timestamp_us` — 统一时间基准

---

## 二、预处理流程

### 2.1 Node 2 坐标轴修正（已完成）

发现 node 2 的传感器 x/y 轴安装方向与其他节点不一致，已统一交换所有数据中的 x/y 轴：
- `acc_x ↔ acc_y`
- `gyr_x ↔ gyr_y`

原始数据已备份至 `backup/` 目录。

### 2.2 50Hz 标定与重采样（已完成）

对 3 人共 13 个录制文件执行：
1. **ADC → 物理单位**：加速度 ÷ 16384 → g，陀螺仪 ÷ 131 → °/s
2. **时间标定**：以各节点首条 `master_timestamp_us` 为 t=0
3. **线性插值重采样**：统一至 50Hz（Δt = 0.02s）
4. **输出**：单节点文件（`calibrated/*_nodeN_calibrated_50hz.csv`）

### 2.3 Madgwick 姿态重建 + L2 分解（已完成）

基于 `run_v31_fix2.py`：
- 对 4 节点分别做 **20Hz 低通滤波**
- 运行 **Madgwick AHRS** 求解姿态四元数 → 欧拉角（roll/pitch/yaw）
- **动态重力分解**：将加速度分解为垂直重力分量（a_vert）和水平运动分量（a_horiz）
- 计算 **jerk**（加速度变化率）和 **模长**（a_mag / g_mag）

---

## 三、训练流程

### 3.1 窗口提取（v3.1 锚定策略）

| 录制类型 | 策略 | 窗口数 |
|----------|------|--------|
| 离散事件型（接球-拍球-投球、高位传球） | **锚定窗口**：以能量峰值为中心，前后各取 0.5s/0.7s | 按事件数生成 |
| 连续动作型（左右运球、防守） | **滑动窗口**：1.2s 窗长，0.3s 步长，75% 重叠 | 按时间滑动 |

共生成 **4,303 个窗口**，去除 ryan 的 `random` 录制后剩余 **3,855** 个有效窗口用于训练。

### 3.2 特征提取

从每个窗口提取 **1,350 维** 特征：
- 时域统计：mean / std / rms / max / min / range / skew / kurtosis
- 频域：spectral entropy / band energy（低/中/高 三频段）
- 零点/峰值计数
- 共 24 通道 × 多特征 = 1,350 维

经 **ANOVA F-value + RF 重要性 + 冗余消除** 筛选后保留 **50 维** 核心特征。

### 3.3 模型训练

训练 3 个分类器（Pipeline：StandardScaler + Classifier）：

| 模型 | 配置 |
|------|------|
| **RF** | 300 棵树，class_weight='balanced' |
| **SVM** | RBF 核，class_weight='balanced'，概率输出 |
| **kNN** | k=5 |

**Fold 划分**：按 `rep_id` 奇偶划分（离散录制），连续录制按 subject+recording hash 分配。

**Null 欠采样**：将 null 类下采样至最大动作类的 1.5 倍，缓解类别不平衡。

### 3.4 开集拒识（Open-set Rejection）

拟合两类拒识阈值：
- **概率下限**：基于验证集正确分类样本的最小最大概率
- **马氏距离**：各类在缩放空间的 97.5% 分位距离（当前因数值稳定性问题已禁用）

---

## 四、当前训练结果（三人统一数据集）

### 验证集性能

| 模型 | Accuracy | Macro F1 | Action F1 |
|------|----------|----------|-----------|
| **RF** | 0.333 | 0.289 | 0.258 |
| **SVM** | 0.345 | 0.306 | 0.278 |
| **KNN** | 0.315 | 0.281 | 0.246 |

### 各类别 F1（RF）

| 类别 | F1 | 说明 |
|------|-----|------|
| catch | 0.293 | 接球识别一般 |
| defense | 0.107 | ⚠️ 防守跨人差异大，识别困难 |
| dribble_left_right | **0.000** | ⚠️ 验证集无样本（fold 划分导致） |
| dribble_right_once | 0.266 | 单次拍球尚可 |
| null | 0.737 | 静止段识别良好 |
| pass_high | 0.369 | 高位传球一般 |
| shot | 0.254 | 投篮识别较弱 |

### 与历史对比（owen + ryan 两人）

| 模型 | 两人 Acc | 两人 Macro F1 | **三人 Acc** | **三人 Macro F1** | 变化 |
|------|----------|---------------|--------------|-------------------|------|
| RF | 0.481 | 0.442 | **0.333** | **0.289** | ↓ -30% |
| SVM | 0.409 | 0.405 | **0.345** | **0.306** | ↓ -24% |
| KNN | 0.499 | 0.450 | **0.315** | **0.281** | ↓ -37% |

> **核心发现**：加入 white 后跨人泛化性能显著下降，这是 IMU 动作识别的经典挑战 — 传感器个体差异、佩戴位置差异、动作风格差异导致域迁移（domain shift）。

---

## 五、已知问题与改进方向

### 5.1 当前问题

1. **`dribble_left_right` 验证集缺失**：连续录制 fold 划分策略导致该类别在验证集上为 0 样本
2. **防守类识别极差**：三人防守动作风格差异大，特征区分度不足
3. **跨人域迁移**：不同受试者的传感器基线、动作幅度、节奏差异显著

### 5.2 建议改进

| 优先级 | 改进项 | 方法 |
|--------|--------|------|
| 🔴 高 | **修复 fold 划分** | 对连续录制改用受试者级留一法或随机分层划分 |
| 🔴 高 | **受试者标准化** | 在特征层做 z-score 标准化，消除个体差异 |
| 🟡 中 | **Domain Adaptation** | CORAL / DANN 减少跨人分布差异 |
| 🟡 中 | **数据增广** | 时间拉伸、幅值抖动、节点 dropout |
| 🟢 低 | **增加 white 采集量** | 当前 white 每动作仅 1 次录制，样本偏少 |

---

## 六、项目目录结构

```
lykon-motion-lab-v2/
├── lykon_dataset/
│   ├── owen/                    # 受试者 owen 原始数据 + calibrated/
│   ├── ryan/                    # 受试者 ryan 原始数据 + calibrated/
│   └── white/                   # 受试者 white 原始数据 + calibrated/
│       ├── *.csv                # 原始录制文件
│       └── calibrated/
│           └── *_nodeN_calibrated_50hz.csv   # 单节点 50Hz 标定文件
│
├── backup/                      # 预处理前的原始数据备份
│
├── imu_basketball_recognition/
│   ├── src/
│   │   ├── config.py            # 全局配置（采样率、窗口参数、特征数等）
│   │   ├── io_utils.py          # 数据加载层（4 节点合并）
│   │   ├── preprocess.py        # 滤波、单位转换
│   │   ├── madgwick.py          # Madgwick AHRS 姿态解算
│   │   ├── labeling.py          # 事件分割与标签分配
│   │   ├── windows_v31.py       # 窗口提取（锚定 + 滑动）
│   │   ├── features.py          # 特征提取（1350 维）
│   │   ├── run_v31_fix2.py      # 端到端：Madgwick → 窗口 → 特征
│   │   ├── train_save.py        # 训练 + 模型保存 + 拒识阈值
│   │   └── infer_timeline.py    # 推理时间线生成
│   ├── data/
│   │   ├── windows/             # 窗口数据（.npz + .parquet）
│   │   └── features/            # 特征矩阵（.parquet）
│   ├── models/
│   │   └── run_YYYYMMDD_HHMMSS/ # 版本化模型目录
│   │       ├── pipeline_rf.joblib
│   │       ├── pipeline_svm.joblib
│   │       ├── pipeline_knn.joblib
│   │       ├── metrics.json     # 验证指标
│   │       ├── train_manifest.json  # 训练/验证划分记录
│   │       ├── selected_features.json
│   │       └── config_snapshot.yaml
│   └── reports/                 # 混淆矩阵、分析报告
│
├── scripts/
│   ├── calibrate_50hz.py        # 50Hz 标定脚本（owen/ryan）
│   ├── calibrate_white_50hz.py # white 50Hz 标定脚本
│   ├── align_white_calibration.py   # white 数据格式对齐
│   └── run_v31_fix2.py          # 训练数据生成（Madgwick + 窗口）
│
├── data/                        # 数据表模板
├── docs/                        # 文档与 Schema 设计
├── outputs/                     # 早期输出文件
└── requirements.txt
```

---

## 七、快速运行

### 环境准备

```bash
conda create -n lykon python=3.10 -y
conda activate lykon
pip install -r requirements.txt
```

### 重新生成训练数据（含 Madgwick 姿态重建）

```bash
cd imu_basketball_recognition/src
python3 run_v31_fix2.py
```

### 重新训练模型

```bash
cd imu_basketball_recognition/src
python3 train_save.py
```

模型自动保存到 `models/run_YYYYMMDD_HHMMSS/` 目录。

---

## 八、核心原则

1. **raw 表只存原始数据，不混 label**
2. **label 单独用 `action_labels.csv` / `event_labels.csv` 记录**
3. **所有数据必须有 `session_id` 和统一时间戳**
4. **视频骨骼是训练老师，不是产品必需输入**
5. **跨人泛化是最大挑战，需要 domain adaptation 或标准化**

---

> **最新训练运行 ID**: `20260711_212943`（三人统一数据集）
> **模型目录**: `imu_basketball_recognition/models/run_20260711_212943/`
