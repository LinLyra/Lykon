"""
labeling.py — Per-sample label assignment for all recordings.
"""
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from config import (
    FS, SEG_RMS_WIN_S, SEG_RMS_STEP_S,
    RECORDING_MAP, ACTION_LABELS
)


def _energy_envelope(a_mag: np.ndarray, win_s: float, step_s: float) -> tuple:
    """Compute RMS energy envelope with sliding window."""
    win_n = int(round(win_s * FS))
    step_n = int(round(step_s * FS))
    n = len(a_mag)
    centers = []
    rms = []
    for i in range(0, n - win_n + 1, step_n):
        centers.append(i + win_n // 2)
        rms.append(np.sqrt(np.mean(a_mag[i:i+win_n]**2)))
    return np.array(centers), np.array(rms)


def _detect_events_adaptive(a_mag: np.ndarray,
                            min_dur_s: float = 0.2,
                            merge_gap_s: float = 0.3) -> list:
    """
    Detect event segments using adaptive thresholding on |a| peaks.
    Designed for recordings WITHOUT clear stationary periods.
    """
    # Use short window for time resolution
    centers, env = _energy_envelope(a_mag, win_s=0.2, step_s=0.05)
    t_env = centers / FS

    # Adaptive threshold: detect regions where energy is significantly above local baseline
    # Use percentile-based threshold on the envelope
    p10 = np.percentile(env, 10)
    p50 = np.percentile(env, 50)
    p90 = np.percentile(env, 90)

    # Threshold = p50 + 0.2 * (p90 - p10)  — captures moderate-to-high activity
    thresh = p50 + 0.25 * (p90 - p10)

    # Also require a minimum absolute threshold (can't be too low)
    thresh = max(thresh, p10 * 1.5)

    # Binary activity mask
    active = env > thresh

    # Convert to segments
    min_step = int(round(min_dur_s / 0.05))  # in envelope steps
    merge_step = int(round(merge_gap_s / 0.05))

    segments = []
    in_seg = False
    seg_start = 0
    for i in range(len(active)):
        if active[i] and not in_seg:
            in_seg = True
            seg_start = i
        elif not active[i] and in_seg:
            in_seg = False
            if i - seg_start >= min_step:
                segments.append((seg_start, i))
    if in_seg and len(active) - seg_start >= min_step:
        segments.append((seg_start, len(active)))

    # Merge close segments
    if len(segments) < 2:
        pass
    else:
        merged = [segments[0]]
        for s, e in segments[1:]:
            if s - merged[-1][1] < merge_step:
                merged[-1] = (merged[-1][0], e)
            else:
                merged.append((s, e))
        segments = merged

    # Map envelope indices back to sample indices
    sample_segments = []
    for s, e in segments:
        samp_start = max(0, centers[s] - int(round(0.2 * FS / 2)))
        samp_end = min(len(a_mag), centers[min(e, len(centers)-1)] + int(round(0.2 * FS / 2)))
        if samp_end - samp_start >= int(round(min_dur_s * FS)):
            sample_segments.append((samp_start, samp_end))

    return sample_segments


def label_continuous_recording(df: pd.DataFrame, action_label: str) -> pd.DataFrame:
    """
    Label a continuous recording (R2, R3, R5).
    Whole segment gets action_label, with energy dips marked Null.
    """
    df = df.copy()
    n = len(df)
    labels = np.full(n, action_label, dtype=object)

    # Detect very low energy regions
    a_mag = df['a_mag_node2'].values
    centers, env = _energy_envelope(a_mag, win_s=0.5, step_s=0.1)
    rest_level = np.percentile(env, 10)
    pause_thresh = rest_level * 1.5

    env_full = np.zeros(n)
    win_n = int(round(0.5 * FS))
    for i, c in enumerate(centers):
        start = max(0, c - win_n // 2)
        end = min(n, c + win_n // 2)
        env_full[start:end] = np.maximum(env_full[start:end], env[i])

    pause_mask = env_full < pause_thresh
    labels[pause_mask] = 'null'

    # Trim first and last 1s
    trim = int(FS)
    labels[:trim] = 'null'
    labels[-trim:] = 'null'

    df['label'] = labels
    df['event_idx'] = -1
    df['cycle_idx'] = -1
    return df


def label_discrete_recording(df: pd.DataFrame, event_names: list,
                             events_per_cycle: int,
                             subject: str, rec_name: str) -> pd.DataFrame:
    """
    Label a discrete event sequence recording (R1, R4).
    Uses adaptive energy segmentation + cyclic prior.
    """
    df = df.copy()
    n = len(df)
    labels = np.full(n, 'null', dtype=object)

    # Primary detection channel: RF (node2) |a|
    a_mag = df['a_mag_node2'].values

    events = _detect_events_adaptive(a_mag, min_dur_s=0.2, merge_gap_s=0.3)

    # Apply cyclic prior
    n_events = len(events)
    if n_events == 0:
        print(f"[WARN] {subject}/{rec_name}: no events detected, all null")
        df['label'] = labels
        df['event_idx'] = -1
        df['cycle_idx'] = -1
        df['rep_id'] = -1
        return df

    # Try to align events into cycles
    n_expected_cycles = n_events // events_per_cycle
    remainder = n_events % events_per_cycle

    if remainder != 0:
        # If close to a multiple, trim excess; otherwise warn
        if n_expected_cycles > 0:
            print(f"[WARN] {subject}/{rec_name}: {n_events} events, not divisible by {events_per_cycle}"
                  f" (trimming last {remainder})")
            events = events[:n_expected_cycles * events_per_cycle]
        else:
            # Very few events — try to use all, assign labels cyclically
            print(f"[WARN] {subject}/{rec_name}: only {n_events} events, less than {events_per_cycle} per cycle")
            n_expected_cycles = 1
    else:
        print(f"[LABEL] {subject}/{rec_name}: {n_events} events → {n_expected_cycles} cycles")

    # Assign labels
    for cycle_idx in range(n_expected_cycles):
        for evt_idx_in_cycle in range(events_per_cycle):
            global_evt_idx = cycle_idx * events_per_cycle + evt_idx_in_cycle
            if global_evt_idx >= len(events):
                break
            s, e = events[global_evt_idx]
            lbl = event_names[evt_idx_in_cycle % len(event_names)]
            labels[s:e] = lbl

    df['label'] = labels
    df['event_idx'] = -1
    for ei, (s, e) in enumerate(events[:n_expected_cycles * events_per_cycle]):
        df.loc[s:e, 'event_idx'] = ei

    df['cycle_idx'] = -1
    for ci in range(n_expected_cycles):
        for ei in range(events_per_cycle):
            gidx = ci * events_per_cycle + ei
            if gidx >= len(events):
                break
            s, e = events[gidx]
            df.loc[s:e, 'cycle_idx'] = ci

    df['rep_id'] = df['cycle_idx']
    return df


def label_recording(df: pd.DataFrame, subject: str, rec_name: str) -> pd.DataFrame:
    """Route to appropriate labeling strategy."""
    info = RECORDING_MAP[subject][rec_name]
    rec_id, action_type, event_names, events_per_cycle = info

    if action_type == 'continuous':
        action_label = event_names[0]
        df = label_continuous_recording(df, action_label)
        df['rep_id'] = 0
    else:
        df = label_discrete_recording(df, event_names, events_per_cycle, subject, rec_name)

    df['subject'] = subject
    df['recording'] = rec_name
    df['rec_id'] = rec_id
    return df
