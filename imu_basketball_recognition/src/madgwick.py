"""
madgwick.py — Madgwick AHRS 姿态解算实现 (self-contained, numpy-based)
"""
import numpy as np
from config import FS, G_STD

DT = 1.0 / FS

def madgwick_update(acc, gyr, q, beta=0.1):
    """
    Single-step Madgwick update.
    acc, gyr: (3,) arrays [m/s², rad/s]
    q: (4,) current quaternion [w, x, y, z]
    beta: filter gain (0.05 = slow/smooth, 0.2 = fast/noisy)
    """
    a = acc / (np.linalg.norm(acc) + 1e-12)
    w = gyr

    # Quaternion rate from gyroscope
    q_dot = 0.5 * np.array([
        -q[1]*w[0] - q[2]*w[1] - q[3]*w[2],
         q[0]*w[0] + q[2]*w[2] - q[3]*w[1],
         q[0]*w[1] - q[1]*w[2] + q[3]*w[0],
         q[0]*w[2] + q[1]*w[1] - q[2]*w[0]
    ])

    # Gradient descent step (accelerometer correction)
    f = np.array([
        2*(q[1]*q[3] - q[0]*q[2]) - a[0],
        2*(q[0]*q[1] + q[2]*q[3]) - a[1],
        2*(0.5 - q[1]**2 - q[2]**2) - a[2]
    ])

    J = np.array([
        [-2*q[2],  2*q[3], -2*q[0], 2*q[1]],
        [ 2*q[1],  2*q[0],  2*q[3], 2*q[2]],
        [ 0,       -4*q[1], -4*q[2], 0]
    ])

    step = J.T @ f
    step = step / (np.linalg.norm(step) + 1e-12)

    q_dot -= beta * step
    q_new = q + q_dot * DT
    q_new = q_new / (np.linalg.norm(q_new) + 1e-12)
    return q_new


def madgwick_filter(acc_arr, gyr_arr, beta=0.1):
    """
    Run Madgwick on entire recording.
    acc_arr: (N, 3) [m/s²]
    gyr_arr: (N, 3) [rad/s]
    Returns: (N, 4) quaternions [w, x, y, z]
    """
    N = len(acc_arr)
    q = np.zeros((N, 4))
    q[0] = [1.0, 0.0, 0.0, 0.0]

    for i in range(1, N):
        q[i] = madgwick_update(acc_arr[i], gyr_arr[i], q[i-1], beta)

    return q


def quaternion_rotate_vector(q, v):
    """Rotate vector v by quaternion q."""
    w, x, y, z = q
    qv = np.array([0, v[0], v[1], v[2]])
    qr = np.array([
        w*qv[0] - x*qv[1] - y*qv[2] - z*qv[3],
        w*qv[1] + x*qv[0] + y*qv[3] - z*qv[2],
        w*qv[2] - x*qv[3] + y*qv[0] + z*qv[1],
        w*qv[3] + x*qv[2] - y*qv[1] + z*qv[0]
    ])
    # qr = q ⊗ qv
    # then q* ⊗ qr
    qc = np.array([w, -x, -y, -z])
    result = np.array([
        qc[0]*qr[0] - qc[1]*qr[1] - qc[2]*qr[2] - qc[3]*qr[3],
        qc[0]*qr[1] + qc[1]*qr[0] + qc[2]*qr[3] - qc[3]*qr[2],
        qc[0]*qr[2] - qc[1]*qr[3] + qc[2]*qr[0] + qc[3]*qr[1],
        qc[0]*qr[3] + qc[1]*qr[2] - qc[2]*qr[1] + qc[3]*qr[0]
    ])
    return result[1:]


def quaternion_to_euler(q):
    """Convert quaternion to roll, pitch, yaw [rad]."""
    w, x, y, z = q
    sinr = 2 * (w*x + y*z)
    cosr = 1 - 2*(x*x + y*y)
    roll = np.arctan2(sinr, cosr)
    sinp = 2 * (w*y - z*x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)
    siny = 2 * (w*z + x*y)
    cosy = 1 - 2*(y*y + z*z)
    yaw = np.arctan2(siny, cosy)
    return roll, pitch, yaw


def decompose_gravity_dynamic(acc_arr, q_arr):
    """
    Dynamic gravity-frame decomposition using per-sample quaternion.
    Returns: a_vert, a_horiz
    """
    N = len(acc_arr)
    a_vert = np.zeros(N)
    a_horiz = np.zeros(N)
    for i in range(N):
        a_world = quaternion_rotate_vector(q_arr[i], acc_arr[i])
        a_vert[i] = a_world[2] + G_STD
        a_horiz[i] = np.sqrt(a_world[0]**2 + a_world[1]**2)
    return a_vert, a_horiz
