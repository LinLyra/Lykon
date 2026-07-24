"""统一配置：关键点、动作标签、关节角定义、窗口与训练参数。

这些常量与仓库已有 schema 对齐：
    docs/schemas/pose_keypoints_2d.csv
    docs/schemas/joint_angles_2d.csv
    docs/action_taxonomy_lykon.md
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# 1. 关键点（与 pose_keypoints_2d.csv 一致，2D + 置信度）
#    共 13 个：nose + 左右(肩/肘/腕/髋/膝/踝)
# ---------------------------------------------------------------------------
KEYPOINTS: list[str] = [
    "nose",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

# MediaPipe BlazePose (33 点) → 本项目 13 点 的索引映射
# 参考 scripts/29_extract_video_pose.py 的 LANDMARK_NAMES 顺序
MEDIAPIPE_INDEX: dict[str, int] = {
    "nose": 0,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
}

# ---------------------------------------------------------------------------
# 2. 动作标签（docs/action_taxonomy_lykon.md 的 V1 必做标签）
# ---------------------------------------------------------------------------
ACTIONS_V1: list[str] = ["idle", "dribble", "pass", "shot", "jump", "sprint"]

# ---------------------------------------------------------------------------
# 3. 关节角定义（与 joint_angles_2d.csv 一致）
#    每个角由三个关键点构成 (a, vertex, b)，角度为 ∠(a-vertex-b)。
#    trunk / arm_raise 为"相对竖直方向"的朝向角，单独计算。
# ---------------------------------------------------------------------------
ANGLE_TRIPLES: dict[str, tuple[str, str, str]] = {
    "left_elbow_angle":    ("left_shoulder", "left_elbow", "left_wrist"),
    "right_elbow_angle":   ("right_shoulder", "right_elbow", "right_wrist"),
    "left_shoulder_angle": ("left_elbow", "left_shoulder", "left_hip"),
    "right_shoulder_angle":("right_elbow", "right_shoulder", "right_hip"),
    "left_hip_angle":      ("left_shoulder", "left_hip", "left_knee"),
    "right_hip_angle":     ("right_shoulder", "right_hip", "right_knee"),
    "left_knee_angle":     ("left_hip", "left_knee", "left_ankle"),
    "right_knee_angle":    ("right_hip", "right_knee", "right_ankle"),
}
# 相对竖直方向的朝向角（0°=竖直向上）
ANGLE_VERTICAL: dict[str, tuple[str, str]] = {
    # 躯干：mid_hip → mid_shoulder，越大越前倾
    "trunk_angle":            ("mid_hip", "mid_shoulder"),
    "left_arm_raise_angle":   ("left_shoulder", "left_wrist"),
    "right_arm_raise_angle":  ("right_shoulder", "right_wrist"),
}
ANGLES: list[str] = list(ANGLE_TRIPLES.keys()) + list(ANGLE_VERTICAL.keys())

# ---------------------------------------------------------------------------
# 4. 采样与窗口参数
# ---------------------------------------------------------------------------
DEFAULT_FPS: float = 30.0          # 视频姿态默认帧率
WINDOW_SECONDS: float = 1.0        # 连续视频滑窗长度
STEP_SECONDS: float = 0.5          # 滑窗步长
MIN_FRAMES: int = 8                # 一个样本/窗口的最少帧数

# ---------------------------------------------------------------------------
# 5. 训练 / 边界参数
# ---------------------------------------------------------------------------
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.25
# 各动作识别边界：Mahalanobis 距离阈值取卡方分位数（自由度=PCA维度）
BOUNDARY_CHI2_Q: float = 0.975
BOUNDARY_PCA_DIM: int = 8          # 每个动作边界建模前降到的维度上限

# 低置信度关键点阈值（visibility / confidence 低于此视为缺失）
MIN_KP_CONFIDENCE: float = 0.3
