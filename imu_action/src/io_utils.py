"""
io_utils.py — 数据加载与节点同步

接口:
  load_raw(data_dir) -> dict[node_id, DataFrame]
  resample_to_grid(dict, fs=50) -> dict[node_id, DataFrame]
  report_sync_quality(dict, out_path)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from config import FS, DT, NODES, NODE_NAMES, REPORTS


# 列名映射: 原始 -> 标准
_RAW_COL_MAP = {
    "acc_x": "ax",
    "acc_y": "ay",
    "acc_z": "az",
    "gyr_x": "gx",
    "gyr_y": "gy",
    "gyr_z": "gz",
    "time_from_seq_s": "t",
}


# ──────────────────────────────
# 加载单节点 (保持原始 counts)
# ──────────────────────────────
def _load_node(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.rename(columns=_RAW_COL_MAP)
    cols = ["group", "node", "seq", "t", "ax", "ay", "az", "gx", "gy", "gz"]
    available = [c for c in cols if c in df.columns]
    df = df[available].copy()
    # 单位转换由 preprocess.py 统一处理，此处保持原始 counts
    df = df.sort_values("t").reset_index(drop=True)
    return df


# ──────────────────────────────
# 检查丢包与插值
# ──────────────────────────────
def _check_gaps(df: pd.DataFrame, fs: int = FS) -> pd.DataFrame:
    dt = 1.0 / fs
    tol = dt * 0.2
    gaps = df["t"].diff().dropna()
    bad = gaps[(gaps < dt - tol) | (gaps > dt + tol)]
    if not bad.empty:
        n_bad = len(bad)
        print(f"  [Node {df['node'].iloc[0]}] 发现 {n_bad} 个异常间隔 (丢包/抖动)")
        t_start = df["t"].iloc[0]
        t_end = df["t"].iloc[-1]
        n_expected = int(round((t_end - t_start) / dt)) + 1
        t_grid = np.linspace(t_start, t_end, n_expected)
        df_interp = pd.DataFrame({"t": t_grid})
        for col in ["ax", "ay", "az", "gx", "gy", "gz"]:
            if col in df.columns:
                df_interp[col] = np.interp(t_grid, df["t"], df[col])
        df_interp["node"] = df["node"].iloc[0]
        df_interp["group"] = df["group"].iloc[0]
        return df_interp
    return df


# ──────────────────────────────
# 加载全部节点
# ──────────────────────────────
def load_raw(data_dir: Path | str, pattern: str = "imu_data_*_node*_seq_aligned.csv") -> dict[int, pd.DataFrame]:
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"未找到匹配文件: {data_dir / pattern}")
    result = {}
    for f in files:
        node_id = int(f.stem.split("_node")[-1].split("_")[0])
        print(f"Loading {f.name} -> Node {node_id} ({NODE_NAMES.get(node_id, '?')})")
        df = _load_node(f)
        df = _check_gaps(df)
        result[node_id] = df
        print(f"  点数: {len(df)}, t 范围: {df['t'].min():.3f} ~ {df['t'].max():.3f} s")
    return result


# ──────────────────────────────
# 重采样到统一 50Hz 网格
# ──────────────────────────────
def resample_to_grid(nodes_dict: dict[int, pd.DataFrame], fs: int = FS) -> dict[int, pd.DataFrame]:
    t_min = max(df["t"].min() for df in nodes_dict.values())
    t_max = min(df["t"].max() for df in nodes_dict.values())
    n = int(np.floor((t_max - t_min) * fs)) + 1
    t_grid = np.round(np.linspace(t_min, t_min + (n - 1) / fs, n), decimals=6)
    print(f"\n公共时间轴: {t_min:.3f} ~ {t_max:.3f} s, 共 {n} 点 ({n / fs:.1f} s)")

    resampled = {}
    for nid, df in nodes_dict.items():
        cols = [c for c in ["ax", "ay", "az", "gx", "gy", "gz"] if c in df.columns]
        interp = {col: np.interp(t_grid, df["t"], df[col]) for col in cols}
        df_new = pd.DataFrame({"t": t_grid, **interp})
        df_new["node"] = nid
        df_new["group"] = df["group"].iloc[0] if "group" in df.columns else 1
        resampled[nid] = df_new
    return resampled


# ──────────────────────────────
# 同步质量图
# ──────────────────────────────
def report_sync_quality(nodes_dict: dict[int, pd.DataFrame], out_path: Path | str | None = None):
    if out_path is None:
        out_path = REPORTS / "sync_check.png"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 4))
    for nid, df in nodes_dict.items():
        a = np.sqrt(df["ax"] ** 2 + df["ay"] ** 2 + df["az"] ** 2)
        mask = df["t"] <= 30.0
        ax.plot(df.loc[mask, "t"], a.loc[mask], label=NODE_NAMES.get(nid, f"N{nid}"), alpha=0.7)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("|a| (raw counts)")
    ax.set_title("Sync quality — |a| overlay (first 30 s)")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[Sync report] saved -> {out_path}")


# ──────────────────────────────
# 合并为单 DataFrame（多节点宽表）
# ──────────────────────────────
def merge_nodes(nodes_dict: dict[int, pd.DataFrame]) -> pd.DataFrame:
    merged = None
    for nid, df in nodes_dict.items():
        prefix = NODE_NAMES.get(nid, f"N{nid}")
        id_cols = {"t", "node", "group"}
        data_cols = [c for c in df.columns if c not in id_cols]
        cols_map = {c: f"{prefix}_{c}" for c in data_cols}
        sub = df[["t"] + data_cols].rename(columns=cols_map)
        if merged is None:
            merged = sub
        else:
            merged = merged.merge(sub, on="t", how="inner")
    merged = merged.sort_values("t").reset_index(drop=True)
    print(f"[Merge] 宽表 shape: {merged.shape}, t 范围: {merged['t'].min():.3f} ~ {merged['t'].max():.3f} s")
    return merged
