# IMU 篮球动作识别系统

基于四节点（双大臂 + 双小臂）6轴 IMU 的篮球动作识别 pipeline，支持跨人泛化分析。

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行完整 pipeline（Phase 1-7）
python src/run_phase3.py   # 加载 → 预处理 → 标注 → 滑窗
python src/run_phase4.py   # 特征提取
python src/run_analysis.py # 筛选 + 可分性 + 跨人 + 分类器
```

## 项目结构

```
imu_basketball_recognition/
├── data/
│   ├── windows/          # 滑窗样本库 (windows.npz, meta.parquet)
│   └── features/         # 特征矩阵 (features.parquet, selected_features.txt)
├── src/
│   ├── config.py         # 全局参数
│   ├── io_utils.py       # 四节点数据加载
│   ├── preprocess.py     # 滤波 + 姿态解算 + L2/L3 通道
│   ├── labeling.py       # 连续/离散标注策略
│   ├── windows.py        # 滑窗切分
│   ├── features.py       # 窗级特征提取 (1350维)
│   ├── run_phase3.py     # 端到端 Phase 1-3
│   ├── run_phase4.py     # 特征提取
│   └── run_analysis.py   # Phase 5-7 综合分析
├── reports/
│   ├── ANALYSIS_REPORT.md    # 完整分析报告
│   ├── separability_distances.png
│   └── confusion_matrix.png
└── requirements.txt
```

## 数据约定

### 四节点映射

| 节点 | 部位 | 缩写 |
|---|---|---|
| node1 | 右大臂 | RU |
| node2 | 右小臂 | RF |
| node3 | 左小臂 | LF |
| node4 | 左大臂 | LU |

### 三层通道

- **L1 分轴**：ax, ay, az, gx, gy, gz（24通道）— 特征主力
- **L2 重力系**：a_vert, a_horiz（8通道）— 方向性特征
- **L3 模值**：|a|, |g|, jerk（8通道）— 能量/冲击特征

### 动作类别

接球、高位传球、防守、左右运球、右手拍一次球、投篮、Null

## 核心结果

| 指标 | 数值 |
|---|---|
| 人内准确率 (RF) | ~86% |
| 跨人 Owen→Ryan | 84.0% |
| 跨人 Ryan→Owen | 74.4% |
| 轮廓系数 (整体) | -0.096 |
| 筛选后特征 | 50维 |

## 已知局限

1. 两人样本偏少，暂态类（shot、拍球）窗口仅 50-100
2. 录制中无静止段，姿态解算基于平均佩戴姿态
3. R1/R4 边界混合窗剔除率高（~27-49%）
4. 类不平衡严重，macro_f1 远低于目标 0.90

## License

内部研究用
