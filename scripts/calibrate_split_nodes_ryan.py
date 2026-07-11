"""
LyKon Motion Lab v2 - Owen 数据集标定：按节点拆分输出
=========================================================
标定规则：
1. 每个节点第一次出现的 seq 的 master_timestamp_us 视为该节点的时间原点 (t=0)
2. 以 master_timestamp_us 为统一时间基准，按相对时间插值到 50Hz
3. 每个节点输出为独立的 CSV 文件
4. 输出到原目录下的 calibrated/ 子文件夹
"""
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from pathlib import Path

# ============ 配置 ============
DATA_DIR = Path("/Users/owen/Desktop/lykon-motion-lab-v2/lykon_dataset/ryan")
OUTPUT_DIR = DATA_DIR / "calibrated"
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


def calibrate_and_resample_node(node_df: pd.DataFrame, node_id: int) -> pd.DataFrame:
    """
    对单个节点的数据进行标定和 50Hz 重采样。
    标定：以该节点第一条记录的 master_timestamp_us 为 t=0。
    返回：标定后的 DataFrame (t_rel_s, 6个物理量)
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
    
    resampled_df = pd.DataFrame(resampled)
    
    # 添加节点标识
    resampled_df['node'] = node_id
    
    # 添加原始标定信息
    resampled_df['calib_first_seq'] = int(node_df['seq'].iloc[0])
    resampled_df['calib_first_master_ts_us'] = int(t0)
    resampled_df['calib_first_node_ts_us'] = int(node_df['node_timestamp_us'].iloc[0])
    
    return resampled_df


def process_file(filepath: Path):
    """处理单个原始文件，输出4个节点文件到原目录"""
    filename = filepath.stem  # 不含扩展名，如 "26071001接球-拍球-跳投"
    print(f"\n📂 处理文件: {filepath.name}")
    
    # 1. 加载
    df = pd.read_csv(filepath)
    print(f"   原始总行数: {len(df)}")
    
    # 2. ADC → 物理单位
    df = adc_to_physical(df)
    
    # 3. 按节点分离、标定、重采样，每个节点独立输出
    for node in NODES:
        node_df = df[df['node'] == node].copy()
        if len(node_df) == 0:
            print(f"   ⚠️ Node {node} 无数据，跳过")
            continue
        
        # 标定 + 50Hz 重采样
        calibrated_df = calibrate_and_resample_node(node_df, node)
        
        # 输出文件名
        out_name = f"{filename}_node{node}_calibrated_50hz.csv"
        out_path = OUTPUT_DIR / out_name
        
        # 保存
        calibrated_df.to_csv(out_path, index=False)
        
        first_seq = int(node_df['seq'].iloc[0])
        print(f"   ✅ Node {node}: 原始 {len(node_df)} 条 → 50Hz 重采样 {len(calibrated_df)} 条"
              f" | 首 seq={first_seq} | 已保存: {out_name}")


def main():
    files = sorted(DATA_DIR.glob("*.csv"))
    print(f"找到 {len(files)} 个原始数据文件")
    print(f"标定输出目录: {OUTPUT_DIR}")
    print("=" * 70)
    
    for filepath in files:
        process_file(filepath)
    
    print("\n" + "=" * 70)
    print("🎉 所有文件标定完成！")
    print(f"   输出目录: {OUTPUT_DIR}")
    print(f"   每个原始文件 → 4 个节点标定文件")


if __name__ == "__main__":
    main()
