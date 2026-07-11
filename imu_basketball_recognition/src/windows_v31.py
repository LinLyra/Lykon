"""
windows_v31.py — v3.1 修复: 事件锚定取窗 + 抖动增强
"""
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from config import FS, WIN_LEN, WIN_LEN_S, ANCHOR_PRE_S, ANCHOR_POST_S, ANCHOR_JITTER_SHIFTS, RANDOM_STATE

np.random.seed(RANDOM_STATE)


def _find_anchor_in_segment(a_mag_segment: np.ndarray, t_segment: np.ndarray) -> float:
    """
    在事件段内找锚点: |a| 的局部最大峰; 若无显著峰则取能量质心。
    返回锚点在 t_segment 中的时刻 [s]。
    """
    # 局部找峰 (prominence 取段内 std 的 1.5 倍)
    prom = 1.5 * np.std(a_mag_segment)
    peaks, props = find_peaks(a_mag_segment, prominence=prom, distance=int(0.1 * FS))
    if len(peaks) > 0:
        # 取最大峰
        idx = peaks[np.argmax(a_mag_segment[peaks])]
        return float(t_segment[idx])
    else:
        # 能量质心
        energy = a_mag_segment ** 2
        if np.sum(energy) < 1e-12:
            return float(t_segment[len(t_segment) // 2])
        centroid_idx = int(np.sum(np.arange(len(t_segment)) * energy) / np.sum(energy))
        centroid_idx = np.clip(centroid_idx, 0, len(t_segment) - 1)
        return float(t_segment[centroid_idx])


def extract_anchor_windows(df: pd.DataFrame, subject: str, rec_name: str) -> dict:
    """
    对离散事件录制(R1/R4)提取锚定窗。
    返回 dict: {X, y, meta, channel_names}
    """
    labels = df['label'].values
    t = df['t'].values
    n = len(df)

    # 找事件段边界
    event_segments = []
    current_label = labels[0]
    start = 0
    for i in range(1, n):
        if labels[i] != current_label and current_label != 'null':
            event_segments.append((start, i, current_label))
            start = i
            current_label = labels[i]
        elif labels[i] == current_label:
            continue
        else:
            # entering null, close current event
            if current_label != 'null':
                event_segments.append((start, i, current_label))
            start = i
            current_label = labels[i]
    if current_label != 'null':
        event_segments.append((start, n, current_label))

    # 去重: 只保留标签不是 null 且不是重复（与上一段相同）的段
    clean_segments = []
    for s, e, lbl in event_segments:
        if lbl != 'null' and (len(clean_segments) == 0 or clean_segments[-1][2] != lbl):
            clean_segments.append((s, e, lbl))

    # 重新合并连续同标签的段（应对分割错误）
    merged_segments = []
    for s, e, lbl in clean_segments:
        if len(merged_segments) > 0 and merged_segments[-1][2] == lbl and s - merged_segments[-1][1] < int(0.3 * FS):
            merged_segments[-1] = (merged_segments[-1][0], e, lbl)
        else:
            merged_segments.append((s, e, lbl))

    # 记录已覆盖的时间区间
    anchor_covered = []
    windows = []
    ys = []
    metas = []

    # 通道列
    meta_cols = {'subject', 'recording', 'rec_id', 'rep_id', 'cycle_idx',
                 'event_idx', 'label', 't'}
    ch_cols = [c for c in df.columns if c not in meta_cols]
    data = df[ch_cols].values

    # 记录 rep_id (cycle_idx)
    rep_ids = df['cycle_idx'].values if 'cycle_idx' in df.columns else np.zeros(n, dtype=int)

    for seg_idx, (s, e, lbl) in enumerate(merged_segments):
        a_mag_seg = df['a_mag_node2'].iloc[s:e].values
        t_seg = df['t'].iloc[s:e].values
        anchor_t = _find_anchor_in_segment(a_mag_seg, t_seg)
        rep_id = int(rep_ids[s]) if s < len(rep_ids) else -1

        # 抖动偏移: 0, +0.1, -0.2 (在3个预设值中随机组合，但稳定化)
        jitters = [0.0, 0.1, 0.2]
        # 为确定性，用固定偏移组合
        jitter_offsets = [0.0, 0.1, -0.2]

        for aug_idx, offset in enumerate(jitter_offsets):
            shifted_t = anchor_t + offset
            t_start = shifted_t - ANCHOR_PRE_S
            t_end = shifted_t + ANCHOR_POST_S
            # 在df中找最接近的索引
            start_idx = np.argmin(np.abs(t - t_start))
            end_idx = np.argmin(np.abs(t - t_end))
            if end_idx - start_idx >= 30:  # 至少 30 点 (0.6s) 才保留
                # 截窗
                win = data[start_idx:end_idx]
                # 零填充到 WIN_LEN
                if win.shape[0] < WIN_LEN:
                    pad = np.zeros((WIN_LEN - win.shape[0], win.shape[1]))
                    win = np.vstack([win, pad])
                elif win.shape[0] > WIN_LEN:
                    win = win[:WIN_LEN]

                windows.append(win)
                ys.append(lbl)
                metas.append({
                    'subject': subject,
                    'recording': rec_name,
                    'rep_id': rep_id,
                    'window_id': len(windows) - 1,
                    't_start': t_start,
                    't_end': t_end,
                    'label': lbl,
                    'parent_event_id': seg_idx,
                    'is_augmented': offset != 0.0,
                    'anchor_t': anchor_t,
                    'purity': 1.0,
                })
                anchor_covered.append((t_start - 0.3, t_end + 0.3))  # 0.3s 隔离带

    if len(windows) == 0:
        return None

    return {
        'X': np.array(windows),
        'y': np.array(ys),
        'meta': pd.DataFrame(metas),
        'channel_names': ch_cols,
        'anchor_covered': anchor_covered,
    }


def extract_sliding_windows(df: pd.DataFrame, exclude_regions: list = None,
                            max_windows: int = None) -> dict:
    """
    连续类录制的滑窗提取，可排除已覆盖区域。
    """
    labels = df['label'].values
    t = df['t'].values
    n = len(df)

    meta_cols = {'subject', 'recording', 'rec_id', 'rep_id', 'cycle_idx',
                 'event_idx', 'label', 't'}
    ch_cols = [c for c in df.columns if c not in meta_cols]
    data = df[ch_cols].values

    rep_ids = df['rep_id'].values if 'rep_id' in df.columns else np.zeros(n, dtype=int)

    win_step = int(0.3 * FS)  # 固定步长

    windows = []
    ys = []
    metas = []

    for start in range(0, n - WIN_LEN + 1, win_step):
        end = start + WIN_LEN
        t_start = t[start]
        t_end = t[end - 1]

        # 排除与 anchor 覆盖区域重叠
        if exclude_regions is not None:
            overlap = False
            for rs, re in exclude_regions:
                if not (t_end < rs or t_start > re):
                    overlap = True
                    break
            if overlap:
                continue

        seg_labels = labels[start:end]
        unique, counts = np.unique(seg_labels, return_counts=True)
        win_label = unique[counts.argmax()]

        windows.append(data[start:end])
        ys.append(win_label)
        metas.append({
            'subject': df['subject'].iloc[0],
            'recording': df['recording'].iloc[0],
            'rep_id': rep_ids[start],
            'window_id': len(windows) - 1,
            't_start': t_start,
            't_end': t_end,
            'label': win_label,
            'parent_event_id': -1,
            'is_augmented': False,
            'anchor_t': np.nan,
            'purity': counts.max() / len(seg_labels),
        })

    if len(windows) == 0:
        return None

    result = {
        'X': np.array(windows),
        'y': np.array(ys),
        'meta': pd.DataFrame(metas),
        'channel_names': ch_cols,
    }
    return result


def build_window_library_v31(all_data: dict) -> dict:
    """
    构建 v3.1 统一窗口库:
    - R1/R4: 事件锚定 + 抖动增强
    - R2/R3: 滑窗网格
    - Null: 滑窗，但排除锚定窗覆盖区域
    """
    all_X = []
    all_y = []
    all_meta = []

    for subject, rec_dict in all_data.items():
        for rec_name, df in rec_dict.items():
            info = df.attrs.get('rec_info') if hasattr(df, 'attrs') else None
            if info is None:
                # 从 recording name 推断
                from config import RECORDING_MAP
                info = RECORDING_MAP[subject][rec_name]
            rec_id, action_type, event_names, events_per_cycle = info

            print(f"[v31] {subject}/{rec_name} ({action_type}) ...")

            if action_type == 'discrete':
                # 锚定取窗
                out = extract_anchor_windows(df, subject, rec_name)
                if out is not None:
                    all_X.append(out['X'])
                    all_y.append(out['y'])
                    all_meta.append(out['meta'])
                    print(f"  Anchor windows: {len(out['y'])}")
                else:
                    print(f"  No anchor windows!")
                continue
            else:
                # 连续类: 滑窗
                out = extract_sliding_windows(df, exclude_regions=None)
                if out is not None:
                    all_X.append(out['X'])
                    all_y.append(out['y'])
                    all_meta.append(out['meta'])
                    print(f"  Sliding windows: {len(out['y'])}")
                else:
                    print(f"  No sliding windows!")

    if len(all_X) == 0:
        return None

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    meta = pd.concat(all_meta, ignore_index=True)
    meta['window_id'] = np.arange(len(meta))

    return {
        'X': X,
        'y': y,
        'meta': meta,
        'channel_names': all_X[0].shape[2] if len(all_X) > 0 else None,
    }
