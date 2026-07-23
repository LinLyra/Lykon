"""
features.py — Window-level feature extraction.
Base features + signature-driven incremental features.
"""
import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.signal import correlate, find_peaks
from scipy.stats import skew, kurtosis
from config import FS, FBANDS, NODES, NODE_ABBR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dominant_freq(x: np.ndarray) -> tuple:
    """Dominant frequency [Hz] and its normalized power."""
    yf = np.abs(rfft(x))
    freqs = rfftfreq(len(x), 1.0 / FS)
    # Exclude DC
    idx = np.argmax(yf[1:]) + 1
    total_power = np.sum(yf[1:]**2)
    dom_power = yf[idx]**2
    return freqs[idx], dom_power / (total_power + 1e-12)


def _spectral_centroid(x: np.ndarray) -> float:
    yf = np.abs(rfft(x))
    freqs = rfftfreq(len(x), 1.0 / FS)
    yf = yf[1:]
    freqs = freqs[1:]
    return np.sum(freqs * yf) / (np.sum(yf) + 1e-12)


def _spectral_entropy(x: np.ndarray) -> float:
    yf = np.abs(rfft(x))
    yf = yf[1:]
    ps = yf**2
    ps = ps / (np.sum(ps) + 1e-12)
    return -np.sum(ps * np.log2(ps + 1e-12))


def _band_energy_ratio(x: np.ndarray, flo: float, fhi: float) -> float:
    yf = np.abs(rfft(x))
    freqs = rfftfreq(len(x), 1.0 / FS)
    mask = (freqs >= flo) & (freqs < fhi)
    total = np.sum(yf[1:]**2)
    band = np.sum(yf[mask]**2)
    return band / (total + 1e-12)


def _zcr(x: np.ndarray) -> int:
    return int(np.sum((x[:-1] * x[1:]) < 0))


def _n_peaks(x: np.ndarray, prominence=None) -> int:
    if prominence is None:
        prominence = 0.5 * np.std(x)
    peaks, _ = find_peaks(x, prominence=prominence)
    return len(peaks)


def _jerk_peak(x: np.ndarray, dt: float = 1.0/FS) -> float:
    jerk = np.gradient(x, dt)
    return float(np.max(np.abs(jerk)))


# ---------------------------------------------------------------------------
# Base features per channel
# ---------------------------------------------------------------------------

def extract_base_features(win: np.ndarray, ch_name: str) -> dict:
    """
    Extract base features from a single window channel.
    win: (win_len,) array
    """
    feats = {}
    prefix = ch_name

    # Time domain
    feats[f'{prefix}_mean'] = np.mean(win)
    feats[f'{prefix}_std'] = np.std(win)
    feats[f'{prefix}_rms'] = np.sqrt(np.mean(win**2))
    feats[f'{prefix}_max'] = np.max(win)
    feats[f'{prefix}_min'] = np.min(win)
    feats[f'{prefix}_ptp'] = np.ptp(win)
    feats[f'{prefix}_zcr'] = _zcr(win)
    feats[f'{prefix}_n_peaks'] = _n_peaks(win)
    feats[f'{prefix}_skew'] = skew(win)
    feats[f'{prefix}_kurt'] = kurtosis(win)
    feats[f'{prefix}_jerk_peak'] = _jerk_peak(win)

    # Frequency domain
    dom_f, dom_ratio = _dominant_freq(win)
    feats[f'{prefix}_dom_freq'] = dom_f
    feats[f'{prefix}_dom_ratio'] = dom_ratio
    feats[f'{prefix}_spec_centroid'] = _spectral_centroid(win)
    feats[f'{prefix}_spec_entropy'] = _spectral_entropy(win)
    for bname, (flo, fhi) in FBANDS.items():
        feats[f'{prefix}_band_{bname}'] = _band_energy_ratio(win, flo, fhi)

    return feats


# ---------------------------------------------------------------------------
# Gyro-specific features
# ---------------------------------------------------------------------------

def extract_gyro_features(win_dict: dict, node: str) -> dict:
    """
    win_dict: {'gx': array, 'gy': array, 'gz': array}
    """
    feats = {}
    prefix = f'gyro_{node}'
    e_x = np.mean(win_dict['gx']**2)
    e_y = np.mean(win_dict['gy']**2)
    e_z = np.mean(win_dict['gz']**2)
    e_total = e_x + e_y + e_z + 1e-12
    feats[f'{prefix}_energy_x_ratio'] = e_x / e_total
    feats[f'{prefix}_energy_y_ratio'] = e_y / e_total
    feats[f'{prefix}_energy_z_ratio'] = e_z / e_total
    # Dominant axis
    energies = [e_x, e_y, e_z]
    dom_axis = int(np.argmax(energies))
    feats[f'{prefix}_dom_axis'] = dom_axis
    sorted_e = sorted(energies, reverse=True)
    feats[f'{prefix}_dom_sub_ratio'] = sorted_e[0] / (sorted_e[1] + 1e-12)
    return feats


# ---------------------------------------------------------------------------
# Signature incremental features
# ---------------------------------------------------------------------------

def extract_signature_features(X_win: np.ndarray, ch_map: dict) -> dict:
    """
    Extract hypothesis-driven features from one window.
    X_win: (win_len, n_channels) — raw window matrix
    ch_map: dict mapping channel names to column indices in X_win
    """
    feats = {}
    wlen = X_win.shape[0]
    t = np.arange(wlen) / FS

    # --- H1/H2: Bilateral synchronicity ---
    # LF-RF energy envelope cross-correlation
    for pair, name in [(('node2', 'node3'), 'RF_LF')]:
        a_mag1 = np.abs(X_win[:, ch_map[f'a_mag_{pair[0]}']])
        a_mag2 = np.abs(X_win[:, ch_map[f'a_mag_{pair[1]}']])
        # Simple envelope: moving average
        env1 = np.convolve(a_mag1, np.ones(5)/5, mode='same')
        env2 = np.convolve(a_mag2, np.ones(5)/5, mode='same')
        cc = correlate(env1 - env1.mean(), env2 - env2.mean(), mode='full')
        lags = np.arange(-len(env1)+1, len(env1))
        peak_idx = np.argmax(cc)
        feats[f'sync_{name}_ccmax'] = cc[peak_idx] / (np.std(env1)*np.std(env2)*(len(env1)-1) + 1e-12)
        feats[f'sync_{name}_lag'] = lags[peak_idx] / FS

    # Four-node energy peak time spread
    peak_times = []
    for node in NODES:
        a_mag = X_win[:, ch_map[f'a_mag_{node}']]
        pks, _ = find_peaks(a_mag, prominence=0.5*np.std(a_mag))
        if len(pks) > 0:
            peak_times.append(pks[0] / FS)
    if len(peak_times) >= 2:
        feats['sync_peak_spread'] = np.max(peak_times) - np.min(peak_times)
    else:
        feats['sync_peak_spread'] = 0.0

    # --- H1/H2: Push directionality ---
    for node in ['node2', 'node3']:  # RF, LF
        a_horiz = X_win[:, ch_map[f'a_horiz_{node}']]
        a_vert = X_win[:, ch_map[f'a_vert_{node}']]
        feats[f'push_{node}_horiz_peak'] = np.max(a_horiz)
        feats[f'push_{node}_vert_horiz_ratio'] = np.mean(a_vert**2) / (np.mean(a_horiz**2) + 1e-12)
        feats[f'push_{node}_vert_sign'] = np.sign(np.mean(a_vert))

    # --- H4: Periodicity (dribble) ---
    # LF/RF energy envelope cross-correlation lag stability
    for pair, name in [(('node2', 'node3'), 'RF_LF')]:
        a_mag1 = np.abs(X_win[:, ch_map[f'a_mag_{pair[0]}']])
        a_mag2 = np.abs(X_win[:, ch_map[f'a_mag_{pair[1]}']])
        env1 = np.convolve(a_mag1, np.ones(5)/5, mode='same')
        # Autocorrelation of envelope
        ac = correlate(env1 - env1.mean(), env1 - env1.mean(), mode='full')
        ac = ac[len(ac)//2:]
        if len(ac) > 5:
            # Find first peak after lag 0
            pks, _ = find_peaks(ac[3:], distance=3)
            if len(pks) > 0:
                feats[f'period_{name}_ac_peak'] = ac[3:][pks[0]] / (ac[0] + 1e-12)
                feats[f'period_{name}_ac_lag'] = (3 + pks[0]) / FS
            else:
                feats[f'period_{name}_ac_peak'] = 0.0
                feats[f'period_{name}_ac_lag'] = 0.0
        else:
            feats[f'period_{name}_ac_peak'] = 0.0
            feats[f'period_{name}_ac_lag'] = 0.0
        # Count impact peaks
        feats[f'period_{name}_n_impacts'] = _n_peaks(a_mag1, prominence=0.3*np.std(a_mag1))

    # --- H5: Unilaterality (right-hand dribble) ---
    e_left = np.mean(X_win[:, ch_map['a_mag_node3']]**2) + np.mean(X_win[:, ch_map['a_mag_node4']]**2)
    e_right = np.mean(X_win[:, ch_map['a_mag_node2']]**2) + np.mean(X_win[:, ch_map['a_mag_node1']]**2)
    feats['unilateral_left_right_ratio'] = e_left / (e_right + 1e-12)
    for node in ['node3', 'node4']:
        feats[f'unilateral_{node}_rms'] = np.sqrt(np.mean(X_win[:, ch_map[f'a_mag_{node}']]**2))

    # --- H6: Airborne (shot) ---
    for node in NODES:
        a_mag = X_win[:, ch_map[f'a_mag_{node}']]
        low_mask = a_mag < 2.0  # m/s² threshold
        # Find longest consecutive run
        if np.any(low_mask):
            runs = []
            cur = 0
            for v in low_mask:
                if v:
                    cur += 1
                else:
                    if cur > 0:
                        runs.append(cur)
                        cur = 0
            if cur > 0:
                runs.append(cur)
            feats[f'airborne_{node}_max_dur'] = (max(runs) if runs else 0) / FS
        else:
            feats[f'airborne_{node}_max_dur'] = 0.0
    # All four nodes simultaneously low
    all_low = np.all([X_win[:, ch_map[f'a_mag_{n}']] < 2.0 for n in NODES], axis=0)
    if np.any(all_low):
        runs = []
        cur = 0
        for v in all_low:
            if v:
                cur += 1
            else:
                if cur > 0:
                    runs.append(cur)
                    cur = 0
        if cur > 0:
            runs.append(cur)
        feats['airborne_all_max_dur'] = (max(runs) if runs else 0) / FS
    else:
        feats['airborne_all_max_dur'] = 0.0

    # --- H2/H3/H6: Posture features ---
    for node in ['node1', 'node4']:  # RU, LU
        pitch = X_win[:, ch_map[f'pitch_{node}']]
        feats[f'posture_{node}_pitch_mean'] = np.mean(pitch)
        feats[f'posture_{node}_pitch_max'] = np.max(pitch)
        feats[f'posture_{node}_pitch_range'] = np.ptp(pitch)

    # --- Cross-node correlations ---
    for pair in [('node2', 'node1'), ('node3', 'node4')]:
        for axis in ['x', 'y', 'z']:
            c1 = X_win[:, ch_map[f'a{axis}_{pair[0]}']]
            c2 = X_win[:, ch_map[f'a{axis}_{pair[1]}']]
            feats[f'corr_{pair[0]}_{pair[1]}_a{axis}'] = np.corrcoef(c1, c2)[0, 1]

    for pair in [('node2', 'node3'), ('node1', 'node4')]:
        e1 = np.mean(X_win[:, ch_map[f'a_mag_{pair[0]}']]**2)
        e2 = np.mean(X_win[:, ch_map[f'a_mag_{pair[1]}']]**2)
        feats[f'energy_ratio_{pair[0]}_{pair[1]}'] = e1 / (e2 + 1e-12)

    return feats


# ---------------------------------------------------------------------------
# Main feature extraction
# ---------------------------------------------------------------------------

def extract_all_features(X: np.ndarray, channel_names: list) -> pd.DataFrame:
    """
    Extract all features from window array.

    Parameters
    ----------
    X : (N, win_len, n_channels)
    channel_names : list of channel name strings

    Returns
    -------
    feat_df : DataFrame (N, n_features)
    """
    ch_map = {name: i for i, name in enumerate(channel_names)}
    feat_rows = []

    # Determine which channels are gyro (for gyro-specific features)
    gyro_nodes = {}
    for node in NODES:
        axes = {}
        for axis in ['x', 'y', 'z']:
            key = f'g{axis}_{node}'
            if key in ch_map:
                axes[axis] = ch_map[key]
        if len(axes) == 3:
            gyro_nodes[node] = axes

    for i in range(X.shape[0]):
        if i % 500 == 0:
            print(f"  Feature extraction: {i}/{X.shape[0]}")
        win = X[i]  # (win_len, n_channels)
        feats = {}

        # Base features for all channels
        for j, ch_name in enumerate(channel_names):
            feats.update(extract_base_features(win[:, j], ch_name))

        # Gyro-specific features
        for node, axes in gyro_nodes.items():
            win_dict = {f'g{axis}': win[:, idx] for axis, idx in axes.items()}
            feats.update(extract_gyro_features(win_dict, node))

        # Signature features
        feats.update(extract_signature_features(win, ch_map))

        feat_rows.append(feats)

    feat_df = pd.DataFrame(feat_rows)
    # Drop NaN/Inf
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan)
    feat_df = feat_df.fillna(0)
    return feat_df
