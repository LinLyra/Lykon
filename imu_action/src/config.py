"""
config.py — 全局参数集中管理

================================================================================
信号三层表示约定（全项目统一）
================================================================================
每个节点维护三层通道，层次与用途严格绑定：

| 层 | 通道 | 用途 |
|---|---|---|
| L1 分轴 (sensor frame) | ax, ay, az, gx, gy, gz (6×3节点=18ch) | 特征提取主力；陀螺分轴区分"绕哪个轴转" |
| L2 重力系分解 | a_vert (竖直, 向上为正), a_horiz (水平模) 每节点2ch | 方向性特征，佩戴朝向不变；拍球竖直向下 vs 投篮竖直向上 |
| L3 模值 | |a|, |g| 每节点各1ch | 仅用于：分割触发、总能量(RMS/SMA)、跨节点能量比 |

硬性规则：
1. 加速度与角速度永远是两族独立通道，禁止合成任何跨物理量标量。
2. L3 模值不进入形状类特征（峰形、频谱形状用 L1/L2）。
3. 所有阈值旁注明物理含义和调整方向。
================================================================================
"""
import numpy as np
from pathlib import Path

# ========== 路径配置 ==========
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_SEGMENTS = ROOT / "data" / "segments"
DATA_FEATURES = ROOT / "data" / "features"
REPORTS = ROOT / "reports"

# ========== 采样与硬件 ==========
FS = 50                      # Hz, 采样率
DT = 1.0 / FS                # 0.02 s
NODES = [1, 2, 3]            # LF, RF, RU（左小臂、右小臂、右大臂）
NODE_NAMES = {1: "LF", 2: "RF", 3: "RU"}

# ADC 分辨率: 16-bit signed → 32768 counts 满量程
ADC_BITS = 16
ADC_MAX = 2 ** (ADC_BITS - 1)  # 32768

# 加速度量程 ±4g → m/s²
ACC_RANGE_G = 4.0
ACC_SCALE = ACC_RANGE_G * 9.80665 / ADC_MAX   # ≈ 0.001197 m/s²/count
ACC_UNIT = "m/s2"

# 陀螺仪量程 ±512°/s → rad/s
GYRO_RANGE_DPS = 512.0
GYRO_SCALE_DEG = GYRO_RANGE_DPS / ADC_MAX      # ≈ 0.015625 °/s/count
GYRO_SCALE_RAD = np.deg2rad(GYRO_RANGE_DPS) / ADC_MAX  # ≈ 0.0002727 rad/s/count
GYRO_UNIT = "rad/s"

G_STD = 9.80665              # m/s², 标准重力加速度

# ========== 姿态解算参数 ==========
# 互补滤波系数 α: 0.98 偏向陀螺(高频), 0.02 偏向加速度(低频)
# 对应时间常数 τ ≈ DT * α/(1-α) ≈ 1.0 s
COMPLEMENTARY_ALPHA = 0.98

# ========== 滤波参数 ==========
# 低通：用于运动学特征，保留主要动作波形
FILTER_LOWPASS_CUTOFF = 20   # Hz
FILTER_LOWPASS_ORDER = 4
# 高通：用于捕捉拍球冲击成分(仅对加速度分轴与竖直分量做)
FILTER_HIGHPASS_CUTOFF = 8   # Hz
FILTER_HIGHPASS_ORDER = 4

# ========== 分割参数 ==========
SEGMENT_WINDOW_S = 0.5       # 滑窗长度（秒）
SEGMENT_STEP_S = 0.1         # 滑窗步长（秒）
# 双阈值迟滞（基于静止段 RMS 的倍数）
# 调整方向: 检出段数过少则降低倍数, 过多则提高倍数
SEGMENT_RMS_MULTIPLIER_HIGH = 5.0   # 进入活动段阈值
SEGMENT_RMS_MULTIPLIER_LOW = 2.0    # 退出活动段阈值
SEGMENT_MIN_GAP_S = 0.3      # 合并相邻活动段的最大间隔
SEGMENT_MIN_DURATION_S = 0.3 # 丢弃短于该时长的碎片
SEGMENT_FIXED_LENGTH_S = 2.0 # 定长片段时长（秒）
SEGMENT_PRE_PEAK_S = 0.8     # 能量峰前截取时长
SEGMENT_POST_PEAK_S = 1.2    # 能量峰后截取时长

# 拍球/投篮 物理校验阈值
DRIBBLE_IMPACT_WIDTH_MAX_S = 0.1       # 冲击峰宽度上限
DRIBBLE_IMPACT_PROMINENCE_FACTOR = 3.0 # 峰 prominence 相对于段内均值
SHOT_PITCH_MIN_DEG = 30.0              # 投篮时大臂俯仰角变化量下限

# ========== 特征参数 ==========
FFT_N_PER_SEG = 256          # FFT 窗长（点）
FFT_BANDS = [(0, 3), (3, 8), (8, 25)]  # Hz 频段划分
FEATURE_FIND_PEAKS_DISTANCE = FS // 8  # 峰检测最小间距（约 6 点 ≈ 120ms）
FEATURE_LIST = [
    "mean", "std", "rms", "max", "min", "ptp",
    "zero_crossing_rate", "n_peaks", "skewness", "kurtosis", "sma", "jerk_max"
]

# ========== 分类器参数 ==========
RANDOM_STATE = 42
RF_N_ESTIMATORS = 300
SVM_C_GRID = [0.1, 1.0, 10.0, 100.0]
SVM_GAMMA_GRID = ["scale", "auto", 0.001, 0.01, 0.1]
KNN_K = 5

# ========== 验收指标 ==========
TARGET_LOOCV_ACC = 0.95
TARGET_ROBUST_ACC = 0.90
TARGET_ONLINE_ACC = 0.95
TARGET_LATENCY_MS = 100
