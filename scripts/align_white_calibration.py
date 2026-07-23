"""
White 数据校准文件对齐 owen/ryan 格式
- 从 split 单节点数据做标定 + 50Hz 重采样
- 输出到 calibrated/ 目录，单节点文件格式
"""
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from pathlib import Path
import shutil

BASE_DIR = Path("/Users/owen/Desktop/lykon-motion-lab-v2/lykon_dataset/white")
SPLIT_DIR = BASE_DIR / "split"
CALIB_DIR = BASE_DIR / "calibrated"
CALIB_DIR.mkdir(exist_ok=True)

ACC_SCALE = 16384.0
GYR_SCALE = 131.0
TARGET_FS = 50.0
TS_INTERVAL_S = 1.0 / TARGET_FS
PHYS_COLS = ['acc_x_g', 'acc_y_g', 'acc_z_g', 'gyr_x_dps', 'gyr_y_dps', 'gyr_z_dps']

def adc_to_physical(df):
    df = df.copy()
    df['acc_x_g'] = df['acc_x'] / ACC_SCALE
    df['acc_y_g'] = df['acc_y'] / ACC_SCALE
    df['acc_z_g'] = df['acc_z'] / ACC_SCALE
    df['gyr_x_dps'] = df['gyr_x'] / GYR_SCALE
    df['gyr_y_dps'] = df['gyr_y'] / GYR_SCALE
    df['gyr_z_dps'] = df['gyr_z'] / GYR_SCALE
    return df

def calibrate_and_resample(node_df):
    node_df = node_df.sort_values('master_timestamp_us').copy()
    t0 = int(node_df['master_timestamp_us'].iloc[0])
    first_seq = int(node_df['seq'].iloc[0])
    node_df['t_rel_s'] = (node_df['master_timestamp_us'] - t0) / 1_000_000.0
    node_df = node_df.drop_duplicates(subset=['t_rel_s'])

    t_raw = node_df['t_rel_s'].values
    t_max = t_raw[-1]

    n_samples = int(np.floor(t_max / TS_INTERVAL_S)) + 1
    t_grid = np.arange(n_samples) * TS_INTERVAL_S

    resampled = {'t_rel_s': t_grid}
    for col in PHYS_COLS:
        y_raw = node_df[col].values
        if len(t_raw) >= 2:
            f = interp1d(t_raw, y_raw, kind='linear',
                         bounds_error=False, fill_value=(y_raw[0], y_raw[-1]))
            resampled[col] = f(t_grid)
        else:
            resampled[col] = np.full(len(t_grid), y_raw[0] if len(y_raw) > 0 else np.nan)

    df = pd.DataFrame(resampled)
    df['node'] = node_df['node'].iloc[0]
    df['calib_first_seq'] = first_seq
    df['calib_first_master_ts_us'] = t0
    return df

def main():
    split_files = sorted(SPLIT_DIR.glob("*_node*.csv"))
    print(f"找到 {len(split_files)} 个 split 文件")
    print("=" * 70)

    for f in split_files:
        df = pd.read_csv(f)
        df = adc_to_physical(df)
        calib_df = calibrate_and_resample(df)

        out_name = f.name.replace('.csv', '_calibrated_50hz.csv')
        out_path = CALIB_DIR / out_name
        calib_df.to_csv(out_path, index=False)

        print(f"   ✅ {out_name} | {len(calib_df)} 行 × {len(calib_df.columns)} 列")

    print("=" * 70)
    print(f"🎉 全部完成！输出: {CALIB_DIR}")
    
    # 清理多余目录
    old_dir = BASE_DIR / "calibrated_50hz"
    if old_dir.exists():
        shutil.rmtree(old_dir)
        print(f"   🗑 已清理: {old_dir}")
    if SPLIT_DIR.exists():
        shutil.rmtree(SPLIT_DIR)
        print(f"   🗑 已清理: {SPLIT_DIR}")
    backup_dir = BASE_DIR / "backup"
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
        print(f"   🗑 已清理: {backup_dir}")

if __name__ == "__main__":
    main()
