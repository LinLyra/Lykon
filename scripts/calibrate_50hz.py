"""
LyKon Motion Lab v2 - Owen 数据集标定与 50Hz 重采样
=====================================================
标定规则：
1. 每个节点第一次出现的 seq 视为同一个时间原点 (t=0)
2. 以 master_timestamp_us 为统一时间基准
3. 各节点按相对时间独立插值，统一重采样到 50Hz
4. 输出宽格式：4 节点 × 6 轴 = 24 个特征列
"""
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from pathlib import Path

# ============ 配置 ============
DATA_DIR = Path("/Users/owen/Desktop/lykon-motion-lab-v2/lykon_dataset/owen")
OUTPUT_DIR = Path("/Users/owen/Desktop/lykon-motion-lab-v2/outputs/calibrated_50hz")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 传感器转换参数
ACC_SCALE = 16384.0   # LSB/g
GYR_SCALE = 131.0     # LSB/(°/s)

# 目标采样率
TARGET_FS = 50.0      # Hz
TS_INTERVAL_S = 1.0 / TARGET_FS  # 0.02 s

NODES = [1, 2, 3, 4]
RAW_SENSOR_COLS = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']
PHYS_COLS = ['acc_x_g', 'acc_y_g', 'acc_z_g', 'gyr_x_dps', 'gyr_y_dps', 'gyr_z_dps']


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


def calibrate_and_resample_node(node_df: pd.DataFrame) -> pd.DataFrame:
    """
    对单个节点的数据进行标定和 50Hz 重采样。
    标定：以该节点第一条记录的 master_timestamp_us 为 t=0。
    """
    node_df = node_df.sort_values('master_timestamp_us').copy()
    
    # 标定：第一条记录的 master_timestamp_us 为 t=0
    t0 = node_df['master_timestamp_us'].iloc[0]
    node_df['t_rel_s'] = (node_df['master_timestamp_us'] - t0) / 1_000_000.0
    
    # 去重（确保时间单调）
    node_df = node_df.drop_duplicates(subset=['t_rel_s'])
    
    t_raw = node_df['t_rel_s'].values
    t_max = t_raw[-1]
    
    # 构建 50Hz 时间网格
    n_samples = int(np.floor(t_max / TS_INTERVAL_S)) + 1
    t_grid = np.arange(n_samples) * TS_INTERVAL_S
    
    # 对每个物理量进行线性插值
    resampled = {'t_rel_s': t_grid}
    for col in PHYS_COLS:
        y_raw = node_df[col].values
        if len(t_raw) >= 2:
            f = interp1d(t_raw, y_raw, kind='linear',
                         bounds_error=False, fill_value=(y_raw[0], y_raw[-1]))
            resampled[col] = f(t_grid)
        else:
            resampled[col] = np.full(len(t_grid), y_raw[0] if len(y_raw) > 0 else np.nan)
    
    return pd.DataFrame(resampled), t_max


def merge_nodes_to_wide(node_dfs: dict) -> pd.DataFrame:
    """
    将4个节点的重采样数据合并为宽格式。
    取所有节点的共同时间范围 [0, min_duration]。
    """
    # 找出共同的时间长度
    min_n = min(len(df) for df in node_dfs.values())
    
    wide_data = {}
    
    for node in NODES:
        df = node_dfs[node].iloc[:min_n].copy()
        if node == 1:
            wide_data['t_rel_s'] = df['t_rel_s'].values
        for col in PHYS_COLS:
            wide_data[f'node{node}_{col}'] = df[col].values
    
    return pd.DataFrame(wide_data)


def process_file(filepath: Path) -> pd.DataFrame:
    """处理单个文件：标定 + 50Hz 重采样 + 宽格式合并"""
    filename = filepath.name
    print(f"\n📂 处理文件: {filename}")
    
    # 1. 加载
    df = pd.read_csv(filepath)
    print(f"   原始总行数: {len(df)}")
    
    # 2. ADC → 物理单位
    df = adc_to_physical(df)
    
    # 3. 按节点分离、标定、重采样
    node_resampled = {}
    node_durations = {}
    
    for node in NODES:
        node_df = df[df['node'] == node].copy()
        if len(node_df) == 0:
            print(f"   ⚠️ Node {node} 无数据")
            continue
        
        first_seq = int(node_df['seq'].iloc[0])
        first_master_ts = int(node_df['master_timestamp_us'].iloc[0])
        
        resampled_df, t_max = calibrate_and_resample_node(node_df)
        node_resampled[node] = resampled_df
        node_durations[node] = t_max
        
        print(f"   Node {node}: 原始 {len(node_df)} 条 → 50Hz 重采样 {len(resampled_df)} 条"
              f" | 首 seq={first_seq} | 时长={t_max:.3f}s")
    
    # 4. 合并为宽格式（取共同时间范围）
    wide_df = merge_nodes_to_wide(node_resampled)
    print(f"   宽格式合并完成: {len(wide_df)} 行 × {len(wide_df.columns)} 列")
    print(f"   各节点原始时长: " + ", ".join([f"N{k}={v:.3f}s" for k, v in node_durations.items()]))
    print(f"   共同时间范围: 0 ~ {wide_df['t_rel_s'].iloc[-1]:.3f}s")
    
    return wide_df


def main():
    files = sorted(DATA_DIR.glob("*.csv"))
    print(f"找到 {len(files)} 个数据文件")
    print("=" * 70)
    
    for filepath in files:
        wide_df = process_file(filepath)
        
        # 保存结果
        out_name = filepath.stem + "_calibrated_50hz.csv"
        out_path = OUTPUT_DIR / out_name
        wide_df.to_csv(out_path, index=False)
        print(f"   ✅ 已保存: {out_path}")
    
    print("\n" + "=" * 70)
    print("🎉 所有文件标定与 50Hz 重采样完成！")
    print(f"   输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
