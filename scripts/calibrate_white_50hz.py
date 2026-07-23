"""
LyKon Motion Lab v2 - White 数据集 50Hz 标定与重采样
复用 scripts/calibrate_50hz.py 逻辑
"""
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from pathlib import Path
import sys

# 配置
DATA_DIR = Path("/Users/owen/Desktop/lykon-motion-lab-v2/lykon_dataset/white")
OUTPUT_DIR = DATA_DIR / "calibrated_50hz"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ACC_SCALE = 16384.0
GYR_SCALE = 131.0
TARGET_FS = 50.0
TS_INTERVAL_S = 1.0 / TARGET_FS

NODES = [1, 2, 3, 4]
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


def calibrate_and_resample_node(node_df):
    node_df = node_df.sort_values('master_timestamp_us').copy()
    t0 = node_df['master_timestamp_us'].iloc[0]
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

    return pd.DataFrame(resampled), t_max


def merge_nodes_to_wide(node_dfs):
    min_n = min(len(df) for df in node_dfs.values())
    wide_data = {}
    for node in NODES:
        df = node_dfs[node].iloc[:min_n].copy()
        if node == 1:
            wide_data['t_rel_s'] = df['t_rel_s'].values
        for col in PHYS_COLS:
            wide_data[f'node{node}_{col}'] = df[col].values
    return pd.DataFrame(wide_data)


def main():
    files = sorted([f for f in DATA_DIR.glob("*.csv") if f.parent.name == 'white'])
    print(f"找到 {len(files)} 个数据文件")
    print("=" * 70)

    for filepath in files:
        filename = filepath.name
        print(f"\n📂 处理文件: {filename}")

        df = pd.read_csv(filepath)
        print(f"   原始总行数: {len(df)}")

        df = adc_to_physical(df)

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

        wide_df = merge_nodes_to_wide(node_resampled)
        print(f"   宽格式合并完成: {len(wide_df)} 行 × {len(wide_df.columns)} 列")
        print(f"   各节点原始时长: " + ", ".join([f"N{k}={v:.3f}s" for k, v in node_durations.items()]))
        print(f"   共同时间范围: 0 ~ {wide_df['t_rel_s'].iloc[-1]:.3f}s")

        out_name = filepath.stem + "_calibrated_50hz.csv"
        out_path = OUTPUT_DIR / out_name
        wide_df.to_csv(out_path, index=False)
        print(f"   ✅ 已保存: {out_path}")

    print("\n" + "=" * 70)
    print("🎉 所有文件标定与 50Hz 重采样完成！")
    print(f"   输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
