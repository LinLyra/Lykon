# v3.1 修复基线快照
## 记录时间
修复前
## 核心指标

| 指标 | 数值 |
|---|---|
| pass_high 窗数 | owen 6, ryan 64 |
| shot 窗数 | owen 40, ryan 52 |
| catch 窗数 | owen 59, ryan 108 |
| dribble_right_once 窗数 | owen 75, ryan 59 |
| 混合窗剔除率 (R1/R4) | owen R1 42.7%, owen R4 49.2%, ryan R1 27.9%, ryan R4 24.0% |
| 排除 Null 的 macro F1 (merged) | 0.208 |
| 整体 silhouette | -0.096 |
| dribble_right_once vs shot top Cohen's d | 0.49 |

## 问题诊断
1. 暂态类（pass_high 6窗, shot 40-52窗）样本严重损耗 → 修复一
2. L2 基于固定平均姿态，精度不足 → 修复二
3. macro F1 仅 0.21，远低于目标 0.90 → 综合修复
