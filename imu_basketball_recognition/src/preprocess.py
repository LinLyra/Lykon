"""
preprocess.py — Filtering, attitude estimation, and L2/L3 channel construction.
"""
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from config import FS, LP_CUTOFF, LP_ORDER, HP_CUTOFF, HP_ORDER, NODES, G_STD

# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

def _design_butter(cutoff, order, btype='low'):
    nyq = 0.5 * FS
    normal_cutoff = np.atleast_1d(cutoff) / nyq
    b, a = butter(order, normal_cutoff, btype=btype)
    return b, a


_LP_B, _LP_A = _design_butter(LP_CUTOFF, LP_ORDER, 'low')
_HP_B, _HP_A = _design_butter(HP_CUTOFF, HP_ORDER, 'high')


def lowpass(sig: np.ndarray) -> np.ndarray:
    """Zero-phase low-pass filter (20 Hz)."""
    return filtfilt(_LP_B, _LP_A, sig, axis=0)


def highpass(sig: np.ndarray) -> np.ndarray:
    """Zero-phase high-pass filter (0.5 Hz) — for gravity removal on L1 if needed."""
    return filtfilt(_HP_B, _HP_A, sig, axis=0)


def bandpass(sig: np.ndarray, low: float, high: float, order: int = 4) -> np.ndarray:
    """Band-pass filter for energy-band isolation."""
    nyq = 0.5 * FS
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, sig, axis=0)


# ---------------------------------------------------------------------------
# Quaternion helpers (for complementary filter)
# ---------------------------------------------------------------------------

def quaternion_multiply(q1, q2):
    """Hamilton product q1 ⊗ q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def quaternion_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quaternion_rotate_vector(q, v):
    """Rotate vector v by quaternion q."""
    qv = np.array([0, v[0], v[1], v[2]])
    qr = quaternion_multiply(quaternion_multiply(q, qv), quaternion_conjugate(q))
    return qr[1:]


def quaternion_to_euler(q):
    """
    Convert quaternion [w,x,y,z] to roll, pitch, yaw [rad].
    Sequence: ZYX (yaw-pitch-roll).
    """
    w, x, y, z = q
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


# ---------------------------------------------------------------------------
# Complementary filter
# ---------------------------------------------------------------------------

def complementary_filter(acc: np.ndarray, gyr: np.ndarray,
                         alpha: float = 0.98) -> np.ndarray:
    """
    Estimate orientation quaternion from accelerometer + gyroscope.
    """
    N = acc.shape[0]
    q = np.zeros((N, 4))
    q[0] = np.array([1.0, 0.0, 0.0, 0.0])
    dt = 1.0 / FS
    for i in range(1, N):
        w = gyr[i]
        w_norm = np.linalg.norm(w)
        if w_norm > 1e-6:
            half_angle = 0.5 * w_norm * dt
            s = np.sin(half_angle) / w_norm
            q_delta = np.array([
                np.cos(half_angle),
                w[0] * s, w[1] * s, w[2] * s
            ])
            q_pred = quaternion_multiply(q[i-1], q_delta)
        else:
            q_pred = q[i-1].copy()
        q_pred = q_pred / (np.linalg.norm(q_pred) + 1e-12)

        a = acc[i]
        a_norm = np.linalg.norm(a)
        if a_norm > 1e-6 and a_norm < 2.5 * G_STD:
            a_unit = a / a_norm
            v = np.array([0.0, 0.0, -1.0])
            v_pred = quaternion_rotate_vector(quaternion_conjugate(q_pred), v)
            error = np.cross(a_unit, v_pred)
            kp = 2.0
            w_corrected = w + kp * error
            wc_norm = np.linalg.norm(w_corrected)
            if wc_norm > 1e-6:
                half_angle = 0.5 * wc_norm * dt
                s = np.sin(half_angle) / wc_norm
                q_delta = np.array([
                    np.cos(half_angle),
                    w_corrected[0] * s, w_corrected[1] * s, w_corrected[2] * s
                ])
                q[i] = quaternion_multiply(q[i-1], q_delta)
            else:
                q[i] = q[i-1].copy()
        else:
            q[i] = q_pred.copy()
        q[i] = q[i] / (np.linalg.norm(q[i]) + 1e-12)
    return q


# ---------------------------------------------------------------------------
# Gravity-frame decomposition (L2)
# ---------------------------------------------------------------------------

def estimate_gravity_direction(acc: np.ndarray, method: str = 'mean') -> np.ndarray:
    """
    Estimate gravity direction from accelerometer data.
    For recordings without stationary periods, use long-term mean
    (motion acceleration averages to zero over time).
    """
    if method == 'mean':
        g_dir = np.mean(acc, axis=0)
    else:
        g_dir = np.median(acc, axis=0)
    g_norm = np.linalg.norm(g_dir)
    if g_norm < 1e-6:
        return np.array([0.0, 0.0, 1.0])
    return g_dir / g_norm


def build_gravity_rotation(g_dir: np.ndarray) -> np.ndarray:
    """
    Build rotation matrix R: sensor frame → gravity-aligned frame.
    In gravity frame: z-axis points UP (opposite to gravity).
    """
    z_new = -g_dir
    if np.abs(z_new[2]) < 0.9:
        tmp = np.array([0.0, 0.0, 1.0])
    else:
        tmp = np.array([1.0, 0.0, 0.0])
    x_new = np.cross(tmp, z_new)
    x_new = x_new / (np.linalg.norm(x_new) + 1e-12)
    y_new = np.cross(z_new, x_new)
    R = np.vstack([x_new, y_new, z_new])
    return R


def decompose_gravity_frame(acc: np.ndarray, g_dir: np.ndarray) -> tuple:
    """
    Rotate acceleration into gravity-aligned frame.

    Parameters
    ----------
    acc   : (N, 3) — sensor-frame acceleration [m/s²]
    g_dir : (3,)   — gravity direction in sensor frame (unit vector)

    Returns
    -------
    a_vert  : (N,) — vertical component [m/s²] (upward positive, gravity removed)
    a_horiz : (N,) — horizontal magnitude [m/s²]
    """
    R = build_gravity_rotation(g_dir)
    a_grav = (R @ acc.T).T  # (N,3)
    a_vert = a_grav[:, 2]   # z points up
    a_horiz = np.sqrt(a_grav[:, 0]**2 + a_grav[:, 1]**2)
    return a_vert, a_horiz


# ---------------------------------------------------------------------------
# Main preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess_recording(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing for one merged recording DataFrame.

    Steps:
      1. Low-pass filter all IMU channels (20 Hz motion band)
      2. High-pass filter acceleration (0.5 Hz) → gravity-removed motion acc
      3. Estimate average gravity direction from very-low-pass acceleration
      4. L2: project HP acceleration onto gravity-aligned frame
      5. Complementary filter for dynamic attitude (Euler angles)
      6. L3: magnitudes |a|, |g| (on LP signals, for energy)
      7. Jerk per node

    Returns expanded DataFrame with new columns.
    """
    df = df.copy()
    t = df['t'].values
    nyq = 0.5 * FS

    # Pre-design 0.2 Hz LP filter for gravity direction estimation
    b_grav, a_grav = butter(4, 0.2 / nyq, btype='low')

    for node in NODES:
        ax = df[f'ax_{node}'].values
        ay = df[f'ay_{node}'].values
        az = df[f'az_{node}'].values
        gx = df[f'gx_{node}'].values
        gy = df[f'gy_{node}'].values
        gz = df[f'gz_{node}'].values

        acc = np.column_stack([ax, ay, az])
        gyr = np.column_stack([gx, gy, gz])

        # --- L1: 20 Hz low-pass (motion band) ---
        acc_lp = lowpass(acc)
        gyr_lp = lowpass(gyr)
        for j, suffix in enumerate(['x', 'y', 'z']):
            df[f'a{suffix}_{node}'] = acc_lp[:, j]
            df[f'g{suffix}_{node}'] = gyr_lp[:, j]

        # --- L1-HP: 0.5 Hz high-pass (gravity-removed motion acceleration) ---
        acc_hp = highpass(acc)

        # --- Attitude: complementary filter on LP signals ---
        q = complementary_filter(acc_lp, gyr_lp)
        euler = np.array([quaternion_to_euler(qi) for qi in q])
        df[f'roll_{node}'] = euler[:, 0]
        df[f'pitch_{node}'] = euler[:, 1]
        df[f'yaw_{node}'] = euler[:, 2]

        # --- L2: gravity-frame decomposition ---
        # Estimate average gravity direction from very-low-pass acceleration
        acc_vlp = filtfilt(b_grav, a_grav, acc, axis=0)
        g_dir = estimate_gravity_direction(acc_vlp, method='median')
        # Project HIGH-PASS (gravity-removed) acceleration into gravity frame
        a_vert, a_horiz = decompose_gravity_frame(acc_hp, g_dir)
        df[f'a_vert_{node}'] = a_vert
        df[f'a_horiz_{node}'] = a_horiz

        # --- L3: magnitudes (on LP signals for energy features) ---
        df[f'a_mag_{node}'] = np.sqrt(acc_lp[:, 0]**2 + acc_lp[:, 1]**2 + acc_lp[:, 2]**2)
        df[f'g_mag_{node}'] = np.sqrt(gyr_lp[:, 0]**2 + gyr_lp[:, 1]**2 + gyr_lp[:, 2]**2)

        # --- Jerk ---
        for axis in ['x', 'y', 'z']:
            a = df[f'a{axis}_{node}'].values
            df[f'jerk_{axis}_{node}'] = np.gradient(a, t)
        df[f'jerk_vert_{node}'] = np.gradient(a_vert, t)
        df[f'jerk_mag_{node}'] = np.gradient(df[f'a_mag_{node}'].values, t)

    return df
    """
    Full preprocessing for one merged recording DataFrame.

    Steps:
      1. Low-pass filter all IMU channels
      2. Complementary filter → quaternion + euler angles per node
      3. L2 decomposition (fixed gravity direction from long-term mean)
      4. L3 magnitude: |a|, |g| per node
      5. Jerk per node

    Returns expanded DataFrame with new columns.
    """
    df = df.copy()
    t = df['t'].values

    for node in NODES:
        ax = df[f'ax_{node}'].values
        ay = df[f'ay_{node}'].values
        az = df[f'az_{node}'].values
        gx = df[f'gx_{node}'].values
        gy = df[f'gy_{node}'].values
        gz = df[f'gz_{node}'].values

        acc = np.column_stack([ax, ay, az])
        gyr = np.column_stack([gx, gy, gz])
        acc_f = lowpass(acc)
        gyr_f = lowpass(gyr)

        # --- L1 filtered ---
        for j, suffix in enumerate(['x', 'y', 'z']):
            df[f'a{suffix}_{node}'] = acc_f[:, j]
            df[f'g{suffix}_{node}'] = gyr_f[:, j]

        # --- Attitude (complementary filter) ---
        q = complementary_filter(acc_f, gyr_f)
        euler = np.array([quaternion_to_euler(qi) for qi in q])
        df[f'roll_{node}'] = euler[:, 0]
        df[f'pitch_{node}'] = euler[:, 1]
        df[f'yaw_{node}'] = euler[:, 2]

        # --- L2: gravity-frame (fixed direction from long-term mean) ---
        g_dir = estimate_gravity_direction(acc_f, method='mean')
        a_vert, a_horiz = decompose_gravity_frame(acc_f, g_dir)
        df[f'a_vert_{node}'] = a_vert
        df[f'a_horiz_{node}'] = a_horiz

        # --- L3: magnitudes ---
        df[f'a_mag_{node}'] = np.sqrt(acc_f[:, 0]**2 + acc_f[:, 1]**2 + acc_f[:, 2]**2)
        df[f'g_mag_{node}'] = np.sqrt(gyr_f[:, 0]**2 + gyr_f[:, 1]**2 + gyr_f[:, 2]**2)

        # --- Jerk ---
        for axis in ['x', 'y', 'z']:
            a = df[f'a{axis}_{node}'].values
            df[f'jerk_{axis}_{node}'] = np.gradient(a, t)
        df[f'jerk_vert_{node}'] = np.gradient(a_vert, t)
        df[f'jerk_mag_{node}'] = np.gradient(df[f'a_mag_{node}'].values, t)

    return df


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_static_gravity(df: pd.DataFrame, node: str, t_start: float = 0.0,
                            t_dur: float = 2.0) -> dict:
    """Check that static segment has a_vert ≈ 0."""
    mask = (df['t'] >= t_start) & (df['t'] < t_start + t_dur)
    a_vert = df.loc[mask, f'a_vert_{node}'].values
    return {
        'node': node,
        'a_vert_mean': float(np.mean(a_vert)),
        'a_vert_std': float(np.std(a_vert)),
        'check': 'PASS' if np.abs(np.mean(a_vert)) < 1.0 else 'FAIL',
    }
