"""
windows.py — Sliding window segmentation and sample library construction.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from config import WIN_LEN, WIN_STEP, LABEL_PURITY, FS


def extract_windows(df: pd.DataFrame, win_len: int = WIN_LEN,
                    step: int = WIN_STEP, label_purity: float = LABEL_PURITY) -> dict:
    """
    Extract sliding windows from a labeled DataFrame.

    Parameters
    ----------
    df : DataFrame with columns including all sensor channels + 'label', 'subject',
         'recording', 'rep_id', 't'

    Returns
    -------
    dict with keys:
        X : (N_windows, win_len, n_channels) array
        y : (N_windows,) label array
        meta : DataFrame with (subject, recording, rep_id, window_id, t_start, t_end, label)
        removed_ratio : fraction of windows removed due to mixed labels
    """
    labels = df['label'].values
    t = df['t'].values

    # Determine channel columns (exclude metadata)
    meta_cols = {'subject', 'recording', 'rec_id', 'rep_id', 'cycle_idx',
                 'event_idx', 'label', 't'}
    ch_cols = [c for c in df.columns if c not in meta_cols]
    data = df[ch_cols].values  # (T, C)

    n = len(df)
    windows = []
    ys = []
    metas = []
    removed = 0
    total = 0

    for start in range(0, n - win_len + 1, step):
        end = start + win_len
        total += 1
        seg_labels = labels[start:end]
        unique, counts = np.unique(seg_labels, return_counts=True)
        purity = counts.max() / len(seg_labels)

        if purity >= label_purity:
            win_label = unique[counts.argmax()]
            windows.append(data[start:end])
            ys.append(win_label)
            metas.append({
                'subject': df['subject'].iloc[0],
                'recording': df['recording'].iloc[0],
                'rec_id': df['rec_id'].iloc[0],
                'rep_id': df['rep_id'].iloc[start] if 'rep_id' in df.columns else 0,
                'window_id': len(windows) - 1,
                't_start': t[start],
                't_end': t[end - 1],
                'label': win_label,
                'purity': purity,
            })
        else:
            removed += 1

    if len(windows) == 0:
        return None

    result = {
        'X': np.array(windows),
        'y': np.array(ys),
        'meta': pd.DataFrame(metas),
        'removed_ratio': removed / total if total > 0 else 0,
        'channel_names': ch_cols,
    }
    return result


def build_window_library(subject_data_dict: dict) -> dict:
    """
    Build unified window library from all preprocessed & labeled recordings.

    Parameters
    ----------
    subject_data_dict : {subject: {rec_name: labeled_df}}

    Returns
    -------
    dict with keys X, y, meta, channel_names
    """
    all_X = []
    all_y = []
    all_meta = []

    for subject, rec_dict in subject_data_dict.items():
        for rec_name, df in rec_dict.items():
            print(f"[WINDOW] {subject}/{rec_name} ...")
            out = extract_windows(df)
            if out is None:
                print(f"  No windows extracted!")
                continue
            all_X.append(out['X'])
            all_y.append(out['y'])
            all_meta.append(out['meta'])
            print(f"  Windows: {len(out['y'])}, removed: {out['removed_ratio']*100:.1f}%")

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


def print_class_distribution(meta: pd.DataFrame):
    """Print sample count per class × subject."""
    print("\n=== Window Class Distribution ===")
    dist = meta.groupby(['subject', 'label']).size().unstack(fill_value=0)
    print(dist.to_string())
    print(f"\nTotal windows: {len(meta)}")
    print(f"Classes: {meta['label'].nunique()}")
