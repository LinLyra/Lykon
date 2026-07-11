"""
LyKon Motion Lab v2 - Owen 数据集预处理脚本
=============================================
流程：加载 → ADC转物理单位 → 多节点时间对齐 → 重采样 → 异常值处理 → 标准化 → 滑动窗口切片
"""
import pandas as pd
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
import os
import json
from pathlib import Path

# ============ 配置 ============
DATA_DIR = Path("/Users/owen/Desktop/lykon-motion-lab-v2/lykon_dataset/owen")
OUTPUT_DIR = Path("/Users/owen/Desktop/lykon-motion-lab-v2/outputs/preprocessed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 传感器转换参数（基于 int16 满量程 ±32768 推测）
# 加速度计 ±2g  -> 16384 LSB/g
# 陀螺仪  ±250°/s -> 131 LSB/(°/s)
ACC_SCALE = 16384.0   # LSB/g
GYR_SCALE = 131.0     # LSB/(°/s)

# 重采样目标频率 (Hz)
TARGET_FS = 200.0

# 滑动窗口参数
WINDOW_SIZE = 128     # 样本数 (128/200Hz = 0.64s)
STRIDE = 64           # 步长 (50% 重叠)

# 动作标签映射
LABEL_MAP = {
    "26071001接球-拍球-跳投.csv": 0,
    "26071002左右运球.csv": 1,
    "26071003防守.csv": 2,
    "26071004高位传球.csv": 3,
}
LABEL_NAMES = {
    0: "catch_dribble_jump_shot",
    1: "left_right_dribble",
    2: "defense",
    3: "high_pass",
}

NODES = [1, 2, 3, 4]
SENSOR_COLS = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']


def load_file(filepath: Path, label: int) -> pd.DataFrame:
    """加载单个CSV文件，添加标签列"""
    df = pd.read_csv(filepath)
    df['label'] = label
    df['label_name'] = LABEL_NAMES[label]
    return df


def adc_to_physical(df: pd.DataFrame) -> pd.DataFrame:
    """将原始ADC值转换为物理单位"""
    df = df.copy()
    df['acc_x_g'] = df['acc_x'] / ACC_SCALE
    df['acc_y_g'] = df['acc_y'] / ACC_SCALE
    df['acc_z_g'] = df['acc_z'] / ACC_SCALE
    df['gyr_x_dps'] = df['gyr_x'] / GYR_SCALE
    df['gyr_y_dps'] = df['gyr_y'] / GYR_SCALE
    df['gyr_z_dps'] = df['gyr_z'] / GYR_SCALE
    return df


def align_nodes_to_wide(df: pd.DataFrame, target_fs: float = TARGET_FS) -> pd.DataFrame:
    """
    将4个节点的数据按master_timestamp_us对齐，合并为宽格式。
    对每个节点分别插值到统一时间网格上。
    
    返回: 宽格式DataFrame，列名为 nodeN_acc_x_g, nodeN_gyr_x_dps 等
    """
    df = df.sort_values('master_timestamp_us').copy()
    
    # 建立相对时间 (从0开始，单位为秒)
    t_min = df['master_timestamp_us'].min()
    df['time_s'] = (df['master_timestamp_us'] - t_min) / 1_000_000.0
    
    # 确定统一的时间网格
    t_max = df['time_s'].max()
    # 目标采样点数
    n_samples = int(np.floor(t_max * target_fs)) + 1
    t_grid = np.linspace(0, (n_samples - 1) / target_fs, n_samples)
    
    # 物理量列
    phys_cols = ['acc_x_g', 'acc_y_g', 'acc_z_g', 'gyr_x_dps', 'gyr_y_dps', 'gyr_z_dps']
    
    wide_data = {'time_s': t_grid}
    
    for node in NODES:
        node_df = df[df['node'] == node].copy()
        if len(node_df) == 0:
            # 如果该节点没有数据，用NaN填充
            for col in phys_cols:
                wide_data[f'node{node}_{col}'] = np.full(len(t_grid), np.nan)
            continue
        
        # 确保时间单调递增（去重）
        node_df = node_df.drop_duplicates(subset=['time_s'])
        node_df = node_df.sort_values('time_s')
        
        t_node = node_df['time_s'].values
        
        for col in phys_cols:
            y = node_df[col].values
            # 线性插值（边界外使用最近值填充）
            if len(t_node) >= 2:
                f_interp = interp1d(
                    t_node, y, kind='linear',
                    bounds_error=False, fill_value=(y[0], y[-1])
                )
                wide_data[f'node{node}_{col}'] = f_interp(t_grid)
            else:
                wide_data[f'node{node}_{col}'] = np.full(len(t_grid), y[0] if len(y) > 0 else np.nan)
    
    wide_df = pd.DataFrame(wide_data)
    return wide_df


def detect_outliers_iqr(df: pd.DataFrame, cols: list, k: float = 3.0) -> pd.DataFrame:
    """使用IQR方法检测并裁剪异常值"""
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - k * IQR
        upper = Q3 + k * IQR
        # 用上下界裁剪，而不是删除
        df[col] = df[col].clip(lower=lower, upper=upper)
    return df


def apply_lowpass_filter(df: pd.DataFrame, cols: list, fs: float, cutoff: float = 20.0) -> pd.DataFrame:
    """应用低通滤波器去除高频噪声"""
    df = df.copy()
    # Butterworth 低通滤波器
    nyq = fs / 2.0
    normal_cutoff = cutoff / nyq
    # 4阶滤波器
    b, a = signal.butter(4, normal_cutoff, btype='low', analog=False)
    
    for col in cols:
        if col not in df.columns:
            continue
        # 处理NaN: 先用线性插值填充，滤波后再恢复NaN位置
        valid_mask = df[col].notna()
        if valid_mask.sum() < 10:
            continue
        y = df[col].interpolate(method='linear', limit_direction='both').values
        # 零相位滤波（避免相位偏移）
        y_filt = signal.filtfilt(b, a, y)
        df[col] = y_filt
    return df


def zscore_normalize(df: pd.DataFrame, cols: list) -> tuple[pd.DataFrame, dict]:
    """Z-score标准化，返回标准化后的数据和统计参数"""
    df = df.copy()
    stats = {}
    for col in cols:
        if col not in df.columns:
            continue
        mean = df[col].mean()
        std = df[col].std()
        if std > 1e-9:
            df[col] = (df[col] - mean) / std
        else:
            df[col] = df[col] - mean
        stats[col] = {'mean': float(mean), 'std': float(std)}
    return df, stats


def create_sliding_windows(df: pd.DataFrame, window_size: int, stride: int, label: int, label_name: str) -> tuple:
    """
    创建滑动窗口样本。
    返回: (windows, labels, label_names, time_starts)
    """
    feature_cols = [c for c in df.columns if c != 'time_s']
    data = df[feature_cols].values
    times = df['time_s'].values
    
    windows = []
    time_starts = []
    n = len(data)
    
    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        window = data[start:end]
        # 丢弃包含过多NaN的窗口
        if np.isnan(window).sum() / window.size > 0.1:
            continue
        windows.append(window)
        time_starts.append(times[start])
    
    if len(windows) == 0:
        return None, None, None, None
    
    windows = np.array(windows)
    labels = np.full(len(windows), label, dtype=np.int32)
    label_names_arr = np.full(len(windows), label_name, dtype=object)
    time_starts = np.array(time_starts)
    
    return windows, labels, label_names_arr, time_starts


def preprocess_all():
    """主预处理流程"""
    all_windows = []
    all_labels = []
    all_label_names = []
    all_time_starts = []
    all_session_ids = []
    all_stats_per_file = {}
    
    files = sorted(DATA_DIR.glob("*.csv"))
    print(f"找到 {len(files)} 个数据文件")
    print("=" * 60)
    
    for i, filepath in enumerate(files):
        filename = filepath.name
        label = LABEL_MAP[filename]
        label_name = LABEL_NAMES[label]
        session_id = f"session_{i:02d}"
        
        print(f"\n📂 处理: {filename}")
        print(f"   标签: {label} ({label_name})")
        
        # 1. 加载
        df = load_file(filepath, label)
        print(f"   原始行数: {len(df)}")
        
        # 2. ADC 转物理单位
        df = adc_to_physical(df)
        print(f"   传感器单位转换完成 (acc→g, gyr→°/s)")
        
        # 3. 多节点对齐 → 宽格式
        wide_df = align_nodes_to_wide(df, target_fs=TARGET_FS)
        print(f"   对齐后样本数: {len(wide_df)} (目标采样率: {TARGET_FS}Hz)")
        
        # 4. 异常值处理 (IQR裁剪)
        feature_cols = [c for c in wide_df.columns if c != 'time_s']
        wide_df = detect_outliers_iqr(wide_df, feature_cols, k=3.0)
        print(f"   IQR异常值裁剪完成")
        
        # 5. 低通滤波 (去除高频噪声)
        wide_df = apply_lowpass_filter(wide_df, feature_cols, fs=TARGET_FS, cutoff=20.0)
        print(f"   低通滤波完成 (截止频率: 20Hz)")
        
        # 6. Z-score 标准化 (按文件单独标准化)
        wide_df_norm, stats = zscore_normalize(wide_df, feature_cols)
        all_stats_per_file[filename] = stats
        print(f"   Z-score标准化完成")
        
        # 保存中间结果
        wide_df_norm.to_csv(OUTPUT_DIR / f"{session_id}_{label_name}_wide.csv", index=False)
        
        # 7. 滑动窗口切片
        windows, labels_arr, label_names_arr, time_starts = create_sliding_windows(
            wide_df_norm, WINDOW_SIZE, STRIDE, label, label_name
        )
        
        if windows is not None:
            all_windows.append(windows)
            all_labels.append(labels_arr)
            all_label_names.append(label_names_arr)
            all_time_starts.append(time_starts)
            all_session_ids.extend([session_id] * len(windows))
            print(f"   生成窗口数: {len(windows)} (窗口大小: {WINDOW_SIZE}, 步长: {STRIDE})")
        else:
            print(f"   ⚠️ 未生成有效窗口")
    
    print("\n" + "=" * 60)
    print("合并所有数据...")
    
    if len(all_windows) == 0:
        print("没有生成任何窗口！")
        return
    
    X = np.concatenate(all_windows, axis=0)
    y = np.concatenate(all_labels, axis=0)
    y_names = np.concatenate(all_label_names, axis=0)
    t_starts = np.concatenate(all_time_starts, axis=0)
    sessions = np.array(all_session_ids)
    
    print(f"总窗口数: {len(X)}")
    print(f"窗口形状: {X.shape} (样本数, 时间步, 特征数)")
    print(f"特征数: {X.shape[2]} (4节点 × 6轴)")
    print(f"各类别分布:")
    for label_id, name in LABEL_NAMES.items():
        count = (y == label_id).sum()
        print(f"   [{label_id}] {name}: {count}")
    
    # 保存为 NumPy 格式
    np.savez(
        OUTPUT_DIR / "owen_preprocessed.npz",
        X=X,
        y=y,
        y_names=y_names,
        time_starts=t_starts,
        sessions=sessions,
        feature_names=[c for c in wide_df_norm.columns if c != 'time_s'],
        target_fs=TARGET_FS,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
    )
    
    # 保存标准化参数
    with open(OUTPUT_DIR / "normalization_stats.json", "w", encoding="utf-8") as f:
        json.dump(all_stats_per_file, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 预处理完成！")
    print(f"   输出目录: {OUTPUT_DIR}")
    print(f"   数据文件: owen_preprocessed.npz")
    print(f"   统计文件: normalization_stats.json")
    
    return X, y, y_names


if __name__ == "__main__":
    preprocess_all()
