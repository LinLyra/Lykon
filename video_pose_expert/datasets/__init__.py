"""数据集加载器：统一产出 samples（list[dict]，每项含 seq/label/...）。

- synthetic     : 合成姿态，保证任何环境可端到端跑通（无需 mediapipe/网络）
- video_folder  : 用户自有视频目录，用 MediaPipe 提取（需 mediapipe+opencv）
- spacejam      : SpaceJam 公开篮球动作数据集（预提取 2D 关节 + 标签）
"""
from .synthetic import generate_synthetic

__all__ = ["generate_synthetic"]
