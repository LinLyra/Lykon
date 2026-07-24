"""LYKON Video-Pose Expert.

用视频骨骼（Google MediaPipe Pose / BlazePose）作为训练阶段的
Ground-Truth Teacher（Expert），完成：

    视频 → 骨骼关键点 → 关节角/动作特征 → 各动作识别边界 → Expert 推理

Expert 训练好后，可对"任意人员"的测试数据（视频或已提取的骨骼）
进行动作识别，并给出与专家动作模式的匹配度（是否落在该动作的识别边界内）。

设计与 README `docs/video_pose_training_plan.md` 一致：视频姿态仅用于
训练阶段构建真值，产品阶段最终以 IMU + UWB 为输入。
"""

from .config import ACTIONS_V1, KEYPOINTS, ANGLES

__all__ = ["ACTIONS_V1", "KEYPOINTS", "ANGLES"]

__version__ = "0.1.0"
