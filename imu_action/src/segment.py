"""
segment.py — 动作分割与半自动标注

接口:
  segment_activity(df_merged, use_node=2) -> list[dict]
  auto_label(segments, start_label='dribble') -> list[dict]
  plot_segmentation(df_merged, segments, out_path)
  save_segments(segments, out_path)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks

from config import (
    FS, DT,
    SEGMENT_WINDOW_S, SEGMENT_STEP_S,
    SEGMENT_RMS_MULTIPLIER_HIGH, SEGMENT_RMS_MULTIPLIER_LOW,
    SEGMENT_MIN_GAP_S, SEGMENT_MIN_DURATION_S,
    SEGMENT_FIXED_LENGTH_S, SEGMENT_PRE_PEAK_S, SEGMENT_POST_PEAK_S,
    DRIBBLE_IMPACT_WIDTH_MAX_S, DRIBBLE_IMPACT_PROMINENCE_FACTOR,
    SHOT_PITCH_MIN_DEG,
    REPORTS, DATA_SEGMENTS,
)


def _rolling_rms(signal: np.ndarray, win: int, step: int) -> tuple[np.ndarray, np.ndarray]:
    """滑窗 RMS，返回 (中心索引, rms值)"""
    n = len(signal)
    centers = []
    rms_vals = []
    for i in range(0, n - win + 1, step):
        centers.append(i + win // 2)
        rms_vals.append(np.sqrt(np.mean(signal[i:i + win] ** 2)))
    return np.array(centers), np.array(rms_vals)


# ──────────────────────────────
# 活动段检测 (双阈值迟滞)
# ──────────────────────────────
def segment_activity(
    df_merged: pd.DataFrame,
    use_node: int = 2,
    rms_win_s: float = SEGMENT_WINDOW_S,
    rms_step_s: float = SEGMENT_STEP_S,
    multiplier_high: float = SEGMENT_RMS_MULTIPLIER_HIGH,
    multiplier_low: float = SEGMENT_RMS_MULTIPLIER_LOW,
    min_gap_s: float = SEGMENT_MIN_GAP_S,
    min_duration_s: float = SEGMENT_MIN_DURATION_S,
) -> list[dict]:
    """
    在指定节点 (默认 RF=2) 的 a_norm_hp 上滑窗检测活动段。
    返回 list of dict: {start, end, start_idx, end_idx, duration, energy_peak, ...}
    """
    prefix = {1: "LF", 2: "RF", 3: "RU"}.get(use_node, "N")
    col = f"{prefix}_a_norm_hp"
    if col not in df_merged.columns:
        col = f"{prefix}_a_norm"
    signal = df_merged[col].values
    t = df_merged["t"].values
    n = len(signal)

    win = int(round(rms_win_s * FS))
    step = int(round(rms_step_s * FS))
    centers, rms_vals = _rolling_rms(signal, win, step)

    # 估计静止段 RMS (取整体 RMS 的最低 10% 分位作为静止基线)
    static_rms = np.percentile(rms_vals, 10)
    th_high = static_rms * multiplier_high
    th_low = static_rms * multiplier_low
    print(f"[Segment] 静止 RMS ≈ {static_rms:.2f}, high={th_high:.2f}, low={th_low:.2f}")

    # 双阈值迟滞
    in_segment = False
    segments = []
    cur_start = None
    for i, r in enumerate(rms_vals):
        if not in_segment and r > th_high:
            in_segment = True
            cur_start = i
        elif in_segment and r < th_low:
            in_segment = False
            segments.append((cur_start, i))
            cur_start = None
    if in_segment and cur_start is not None:
        segments.append((cur_start, len(rms_vals) - 1))

    # 映射回原始索引
    raw_segments = []
    for s, e in segments:
        idx_s = max(0, centers[s] - win // 2)
        idx_e = min(n - 1, centers[e] + win // 2)
        raw_segments.append((idx_s, idx_e))

    # 合并近邻
    merged = []
    min_gap_samples = int(min_gap_s * FS)
    for idx_s, idx_e in raw_segments:
        if not merged:
            merged.append([idx_s, idx_e])
        else:
            if idx_s - merged[-1][1] <= min_gap_samples:
                merged[-1][1] = idx_e
            else:
                merged.append([idx_s, idx_e])

    # 过滤短片段
    min_dur_samples = int(min_duration_s * FS)
    results = []
    for idx_s, idx_e in merged:
        if idx_e - idx_s < min_dur_samples:
            continue
        seg_signal = signal[idx_s:idx_e]
        peak_rel = np.argmax(seg_signal)
        peak_idx = idx_s + peak_rel
        results.append({
            "start_idx": int(idx_s),
            "end_idx": int(idx_e),
            "start": float(t[idx_s]),
            "end": float(t[idx_e]),
            "duration": float(t[idx_e] - t[idx_s]),
            "energy_peak_idx": int(peak_idx),
            "energy_peak_t": float(t[peak_idx]),
            "rms_max": float(np.max(seg_signal)),
        })

    print(f"[Segment] 检出 {len(results)} 个活动段")
    return results


# ──────────────────────────────
# 峰值检测分割 (fallback / 针对连续数据)
# ──────────────────────────────
def segment_by_peaks(
    df_merged: pd.DataFrame,
    use_node: int = 2,
    distance_s: float = 1.0,
    prominence: float = 4000.0,
    min_duration_s: float = SEGMENT_MIN_DURATION_S,
) -> list[dict]:
    """
    对连续活跃数据使用 find_peaks 检测活动中心，然后向两侧扩展成活动段。
    默认在 RF_a_norm (raw) 上运行，可检出 dribble/shot 的独立峰值。
    """
    prefix = {1: "LF", 2: "RF", 3: "RU"}.get(use_node, "N")
    col = f"{prefix}_a_norm"
    if col not in df_merged.columns:
        raise ValueError(f"Column {col} not found in merged dataframe")
    signal = df_merged[col].values
    t = df_merged["t"].values
    n = len(signal)

    peaks, props = find_peaks(
        signal,
        distance=int(round(distance_s * FS)),
        prominence=prominence,
    )
    print(f"[Segment by peaks] {len(peaks)} peaks detected (dist={distance_s}s, prom={prominence})")

    # 扩展: 以峰之间局部最小值作为边界
    results = []
    for i, p in enumerate(peaks):
        # 左边界
        if i == 0:
            left = max(0, p - int(1.0 * FS))
        else:
            prev = peaks[i - 1]
            left = prev + int(np.argmin(signal[prev:p]))
        # 右边界
        if i == len(peaks) - 1:
            right = min(n - 1, p + int(1.0 * FS))
        else:
            nxt = peaks[i + 1]
            right = p + int(np.argmin(signal[p:nxt]))
        # 确保最小长度
        min_half = int(round(min_duration_s * FS / 2))
        if p - left < min_half:
            left = max(0, p - min_half)
        if right - p < min_half:
            right = min(n - 1, p + min_half)
        results.append({
            "start_idx": int(left),
            "end_idx": int(right),
            "start": float(t[left]),
            "end": float(t[right]),
            "duration": float(t[right] - t[left]),
            "energy_peak_idx": int(p),
            "energy_peak_t": float(t[p]),
            "rms_max": float(signal[p]),
        })

    print(f"[Segment by peaks] 生成 {len(results)} 个活动段")
    return results


# ──────────────────────────────
# 序列先验自动标注
# ──────────────────────────────
def auto_label(segments: list[dict], start_label: str = "dribble") -> list[dict]:
    """
    按出现顺序交替标注: dribble, shot, dribble, shot...
    """
    labels = ["dribble", "shot"]
    for i, seg in enumerate(segments):
        seg["label"] = labels[(i + (0 if start_label == "dribble" else 1)) % 2]
        seg["cycle"] = i // 2 + 1
    return segments


# ──────────────────────────────
# 物理规则校验与可疑段标记
# ──────────────────────────────
def validate_segments(
    segments: list[dict],
    df_merged: pd.DataFrame,
) -> list[dict]:
    """
    对每段做物理校验:
      - dribble: 高通通道上应有窄冲击峰
      - shot: RU 俯仰角变化应 > 阈值
    标记 suspicious=True 的段。
    """
    prefix = {1: "LF", 2: "RF", 3: "RU"}
    for seg in segments:
        seg["suspicious"] = False
        seg["reason"] = ""

        idx_s, idx_e = seg["start_idx"], seg["end_idx"]
        label = seg["label"]

        if label == "dribble":
            # v2: RF 高通竖直分量冲击峰检查 (L2)
            rf_vert_hp = df_merged[f"{prefix[2]}_a_vert_hp"].values[idx_s:idx_e]
            rf_horiz = df_merged[f"{prefix[2]}_a_horiz"].values[idx_s:idx_e]
            if len(rf_vert_hp) == 0:
                seg["suspicious"] = True
                seg["reason"] = "empty"
                continue
            mean_hp = np.mean(np.abs(rf_vert_hp))
            prominence = max(mean_hp * DRIBBLE_IMPACT_PROMINENCE_FACTOR, 1e-6)
            peaks, _ = find_peaks(np.abs(rf_vert_hp), prominence=prominence, distance=max(1, int(FS * 0.05)))
            valid = False
            vert_dominant = False
            for p in peaks:
                left = p
                while left > 0 and np.abs(rf_vert_hp[left]) > prominence:
                    left -= 1
                right = p
                while right < len(rf_vert_hp) - 1 and np.abs(rf_vert_hp[right]) > prominence:
                    right += 1
                width = (right - left) / FS
                if width < DRIBBLE_IMPACT_WIDTH_MAX_S:
                    valid = True
                if np.abs(rf_vert_hp[p]) > rf_horiz[p]:
                    vert_dominant = True
            if not valid:
                seg["suspicious"] = True
                seg["reason"] = "no_narrow_impact"
            elif not vert_dominant:
                seg["suspicious"] = True
                seg["reason"] = "not_vert_dominant"
            seg["dribble_peak_count"] = int(len(peaks))

        elif label == "shot":
            # v2: RU 俯仰角 + RF a_vert 正向主峰
            ru_pitch = df_merged[f"{prefix[3]}_pitch"].values[idx_s:idx_e]
            rf_vert = df_merged[f"{prefix[2]}_a_vert"].values[idx_s:idx_e]
            if len(ru_pitch) == 0:
                seg["suspicious"] = True
                seg["reason"] = "empty"
                continue
            pitch_range = float(np.max(ru_pitch) - np.min(ru_pitch))
            seg["pitch_range"] = pitch_range
            vert_pos_peak = float(np.max(rf_vert)) if len(rf_vert) > 0 else 0.0
            seg["vert_pos_peak"] = vert_pos_peak
            if pitch_range < 0.01:
                seg["reason"] = "pitch_unavailable"
            elif pitch_range < SHOT_PITCH_MIN_DEG:
                seg["suspicious"] = True
                seg["reason"] = f"pitch_range={pitch_range:.1f}<{SHOT_PITCH_MIN_DEG}"
            elif vert_pos_peak < 2.0:
                seg["suspicious"] = True
                seg["reason"] = f"vert_peak={vert_pos_peak:.1f}<2.0"

    n_susp = sum(1 for s in segments if s["suspicious"])
    print(f"[Validate] 可疑段: {n_susp} / {len(segments)}")
    return segments


# ──────────────────────────────
# 定长截取
# ──────────────────────────────
def fix_length(segments: list[dict], df_merged: pd.DataFrame) -> list[dict]:
    """
    以段内能量峰为中心截取固定长度窗口。
    """
    t = df_merged["t"].values
    n = len(t)
    fixed_len = int(round(SEGMENT_FIXED_LENGTH_S * FS))
    pre = int(round(SEGMENT_PRE_PEAK_S * FS))
    post = int(round(SEGMENT_POST_PEAK_S * FS))

    for seg in segments:
        peak = seg["energy_peak_idx"]
        s = max(0, peak - pre)
        e = min(n, peak + post)
        # 若长度不足，调整到 fixed_len
        if e - s < fixed_len:
            if s == 0:
                e = min(n, fixed_len)
            elif e == n:
                s = max(0, n - fixed_len)
            else:
                # 以 peak 为中心尽量扩展
                s = max(0, peak - fixed_len // 2)
                e = min(n, s + fixed_len)
                s = max(0, e - fixed_len)
        seg["fixed_start_idx"] = int(s)
        seg["fixed_end_idx"] = int(e)
        seg["fixed_start"] = float(t[s])
        seg["fixed_end"] = float(t[e])
        seg["fixed_duration"] = float(t[e] - t[s])
    return segments


# ──────────────────────────────
# 可视化校验图
# ──────────────────────────────
def plot_segmentation(
    df_merged: pd.DataFrame,
    segments: list[dict],
    out_path: Path | str | None = None,
    max_duration: float = 60.0,
):
    """
    绘制 RF |a| 曲线 + 彩色分割区间。
    分页输出，每页 max_duration 秒。
    """
    if out_path is None:
        out_path = REPORTS / "segmentation"
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    t = df_merged["t"].values
    signal = df_merged["RF_a_norm"].values
    t_max = t[-1]
    n_pages = int(np.ceil(t_max / max_duration))

    color_map = {"dribble": "#3498db", "shot": "#e67e22", "unknown": "#95a5a6"}

    for page in range(n_pages):
        t0 = page * max_duration
        t1 = min((page + 1) * max_duration, t_max)
        mask = (t >= t0) & (t <= t1)

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(t[mask], signal[mask], color="black", lw=0.8, alpha=0.6, label="|a|")

        for seg in segments:
            if seg["end"] < t0 or seg["start"] > t1:
                continue
            c = "#e74c3c" if seg.get("suspicious") else color_map.get(seg.get("label", "unknown"), "gray")
            alpha = 0.25 if seg.get("suspicious") else 0.15
            ax.axvspan(seg["start"], seg["end"], color=c, alpha=alpha)
            mid = (seg["start"] + seg["end"]) / 2
            if t0 <= mid <= t1:
                label_text = f"{seg['label']}"
                if seg.get("suspicious"):
                    label_text += "\n⚠️"
                ax.text(mid, np.max(signal[mask]) * 0.9, label_text,
                       ha="center", va="top", fontsize=8, color="white" if seg.get("suspicious") else c,
                       bbox=dict(boxstyle="round,pad=0.2", facecolor=c, alpha=0.8, edgecolor="none"))

        ax.set_xlim(t0, t1)
        ax.set_xlabel("t (s)")
        ax.set_ylabel("RF |a| (m/s²)")
        ax.set_title(f"Segmentation verification — Page {page + 1}/{n_pages} ({t0:.1f}–{t1:.1f} s)")
        ax.grid(True, ls="--", alpha=0.3)
        fig.tight_layout()
        fname = out_path / f"segmentation_page_{page + 1:02d}.png"
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"[Segment plot] {fname}")


# ──────────────────────────────
# 保存段标签表 (CSV，可供人工修改)
# ──────────────────────────────
def save_label_table(segments: list[dict], out_path: Path | str | None = None):
    if out_path is None:
        out_path = REPORTS / "segment_labels.csv"
    out_path = Path(out_path)
    rows = []
    for seg in segments:
        rows.append({
            "idx": seg.get("idx", segments.index(seg)),
            "cycle": seg.get("cycle", ""),
            "label": seg.get("label", ""),
            "suspicious": seg.get("suspicious", False),
            "reason": seg.get("reason", ""),
            "start": seg["start"],
            "end": seg["end"],
            "duration": seg["duration"],
            "energy_peak_t": seg["energy_peak_t"],
            "dribble_peak_count": seg.get("dribble_peak_count", ""),
            "pitch_range": seg.get("pitch_range", ""),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"[Label table] {out_path}")
    return df


# ──────────────────────────────
# 提取片段数据并保存 NPZ
# ──────────────────────────────
def save_segments_npz(
    segments: list[dict],
    df_merged: pd.DataFrame,
    out_path: Path | str | None = None,
):
    """
    保存定长片段到 npz: X[N, T, C], y[N], meta
    """
    if out_path is None:
        out_path = DATA_SEGMENTS / "segments.npz"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    feature_cols = [c for c in df_merged.columns if c != "t"]
    n_seg = len(segments)
    fixed_len = int(round(SEGMENT_FIXED_LENGTH_S * FS))

    X = np.zeros((n_seg, fixed_len, len(feature_cols)), dtype=np.float32)
    y = []
    meta = []

    for i, seg in enumerate(segments):
        s = seg["fixed_start_idx"]
        e = seg["fixed_end_idx"]
        # 截取
        chunk = df_merged[feature_cols].values[s:e]
        if chunk.shape[0] < fixed_len:
            # 零填充到定长
            pad = np.zeros((fixed_len - chunk.shape[0], len(feature_cols)), dtype=np.float32)
            chunk = np.vstack([chunk, pad])
        elif chunk.shape[0] > fixed_len:
            chunk = chunk[:fixed_len]
        X[i] = chunk.astype(np.float32)
        y.append(seg.get("label", "unknown"))
        meta.append({
            "start_t": seg["start"],
            "end_t": seg["end"],
            "peak_t": seg["energy_peak_t"],
            "label": seg.get("label", "unknown"),
            "suspicious": seg.get("suspicious", False),
            "cycle": seg.get("cycle", -1),
        })

    y = np.array(y)
    np.savez(out_path, X=X, y=y, meta=meta, feature_cols=feature_cols)
    print(f"[Segments NPZ] shape={X.shape}, labels={dict(zip(*np.unique(y, return_counts=True)))} -> {out_path}")
    return X, y, meta
