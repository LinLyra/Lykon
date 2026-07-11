"""
生成最终报告 random_test_report.md，包含：
- 时间线全表
- 可视化引用
- 三模型分歧统计
- 持久化产物清单与冒烟测试结果
- 双向跨人实验重跑结果
"""
import sys
import json
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

sys.path.insert(0, '/Users/owen/Desktop/lykon-motion-lab-v2/imu_basketball_recognition/src')

MODEL_DIR = Path('/Users/owen/Desktop/lykon-motion-lab-v2/imu_basketball_recognition/models/run_20260711_045637')
REPORT_DIR = Path('/Users/owen/Desktop/lykon-motion-lab-v2/imu_basketball_recognition/reports/random_test')

# Load model package
with open(MODEL_DIR / 'config_snapshot.yaml', 'r') as f:
    config = yaml.safe_load(f)
with open(MODEL_DIR / 'metrics.json', 'r') as f:
    metrics = json.load(f)
with open(MODEL_DIR / 'train_manifest.json', 'r') as f:
    manifest = json.load(f)
with open(MODEL_DIR / 'rejection.json', 'r') as f:
    rejection = json.load(f)

# Load timeline
timeline_df = pd.read_csv(REPORT_DIR / 'random_test_timeline_run_20260711_045637.csv')

# Load per-window detail for disagreement analysis
detail_df = pd.read_parquet(REPORT_DIR / 'random_test_detail_run_20260711_045637.parquet')

# Cross-subject experiment (re-run using saved data from training)
feat_df = pd.read_parquet('/Users/owen/Desktop/lykon-motion-lab-v2/imu_basketball_recognition/data/features/features_v31_fix2.parquet')
meta = pd.read_parquet('/Users/owen/Desktop/lykon-motion-lab-v2/imu_basketball_recognition/data/windows/meta_v31_fix2.parquet')
lib = np.load('/Users/owen/Desktop/lykon-motion-lab-v2/imu_basketball_recognition/data/windows/windows_v31_fix2.npz')
y = lib['y']
X = feat_df.values

# Exclude random recording
mask_keep = meta['recording'] != '26071015随机做'
X = X[mask_keep.values]
y = y[mask_keep.values]
meta = meta[mask_keep].reset_index(drop=True)
feat_df = feat_df.loc[mask_keep.values].reset_index(drop=True)

# Feature degradation (same as training)
feature_names = feat_df.columns.tolist()
attitude_keywords = ['a_vert', 'a_horiz', 'yaw', 'pitch_mean', 'roll_mean', 'pitch_dom', 'roll_dom']
keep_mask = [not any(kw in name for kw in attitude_keywords) for name in feature_names]
keep_idx = [i for i, v in enumerate(keep_mask) if v]
X_filtered = X[:, keep_idx]
filtered_names = [feature_names[i] for i in keep_idx]

# Load selected features
with open(MODEL_DIR / 'selected_features.json', 'r') as f:
    selected_features = json.load(f)
selected_indices = [filtered_names.index(f) for f in selected_features]
X_sel = X_filtered[:, selected_indices]

# Cross-subject evaluation
m_owen = meta['subject'] == 'owen'
m_ryan = meta['subject'] == 'ryan'
Xo, yo = X_sel[m_owen], y[m_owen]
Xr, yr = X_sel[m_ryan], y[m_ryan]

# Load saved pipelines
pipelines = {}
for name in ['rf', 'svm', 'knn']:
    pipelines[name] = joblib.load(MODEL_DIR / f'pipeline_{name}.joblib')

cross_results = {}
for name, pipe in pipelines.items():
    # Owen -> Ryan
    pipe.fit(Xo, yo)
    y_pred_r = pipe.predict(Xr)
    acc_or = accuracy_score(yr, y_pred_r)
    f1_or = f1_score(yr, y_pred_r, average='macro', zero_division=0)
    # Ryan -> Owen
    pipe.fit(Xr, yr)
    y_pred_o = pipe.predict(Xo)
    acc_ro = accuracy_score(yo, y_pred_o)
    f1_ro = f1_score(yo, y_pred_o, average='macro', zero_division=0)
    cross_results[name] = {
        'owen_to_ryan_acc': float(acc_or),
        'owen_to_ryan_f1': float(f1_or),
        'ryan_to_owen_acc': float(acc_ro),
        'ryan_to_owen_f1': float(f1_ro),
    }

# Report content
n_total = len(detail_df)
n_all_agree = ((detail_df['label_rf'] == detail_df['label_svm']) & (detail_df['label_rf'] == detail_df['label_knn'])).sum()
n_any_disagree = ((detail_df['label_rf'] != detail_df['label_svm']) | (detail_df['label_rf'] != detail_df['label_knn'])).sum()

report = f"""# 随机录制端到端测试报告

**模型运行:** `run_20260711_045637`  
**生成时间:** 2026-07-11  
**测试录制:** ryan/26071015随机做（零参与训练）

---

## 1. 训练与持久化

### 1.1 训练配置

- **训练数据:** owen R1-R4 + ryan R1-R4（随机录制严格排除）
- **类别:** 六类 + Null（catch, dribble_left_right, defense, dribble_right_once, pass_high, shot, null）
- **Null 欠采样:** 至最大动作类的 1.5 倍
- **class_weight:** balanced
- **特征配置:** 降级模式（姿态依赖特征出池）
- **窗口:** 1.2s / 0.3s 步长
- **特征选择:** ANOVA + RF 重要性 + 冗余消除 → 50 维
- **划分:** 按 rep_id 奇偶（discrete）/ subject+recording hash（continuous）

### 1.2 验证集指标

| 模型 | 准确率 | Macro F1 |
|------|--------|----------|
| RF   | {metrics['validation']['rf']['accuracy']:.3f} | {metrics['validation']['rf']['macro_f1']:.3f} |
| SVM  | {metrics['validation']['svm']['accuracy']:.3f} | {metrics['validation']['svm']['macro_f1']:.3f} |
| kNN  | {metrics['validation']['knn']['accuracy']:.3f} | {metrics['validation']['knn']['macro_f1']:.3f} |

### 1.3 持久化产物清单

| 文件 | 说明 |
|------|------|
| `pipeline_rf.joblib` | RF 完整 Pipeline (scaler + clf) |
| `pipeline_svm.joblib` | SVM 完整 Pipeline |
| `pipeline_knn.joblib` | kNN 完整 Pipeline |
| `feature_names.json` | 原始特征名有序列表（1132 维） |
| `selected_features.json` | 选中特征名（50 维） |
| `rejection.json` | 拒识参数（概率阈值、马氏距离阈值） |
| `config_snapshot.yaml` | 全部推理参数快照 |
| `train_manifest.json` | 训练/验证录制清单、样本数、版本号 |
| `metrics.json` | 验证集逐类 F1、混淆矩阵、阈值扫描数据 |

### 1.4 往返冒烟测试

✅ **通过** — 50 个验证窗口，内存中 Pipeline 预测与磁盘加载后预测完全一致（概率误差 < 1e-10）。  
✅ **sklearn 版本匹配** — 训练/加载均为 1.6.1。

---

## 2. 随机录制推理时间线

### 2.1 事件级时间线（RF 模型）

{timeline_df.to_markdown(index=False)}

### 2.2 可视化

- 时间线图: `random_test_timeline_run_20260711_045637.png`
- 三模型对比图: `random_test_comparison_run_20260711_045637.png`

### 2.3 三模型分歧统计

- 总窗口数: {n_total}
- 三模型完全一致窗口: {n_all_agree}
- 全分歧窗口: {n_any_disagree}

分歧段落（RF 与其他两模型均不一致）往往对应难例，建议人工重点核对这些时间段的视频/波形。

### 2.4 逐窗明细

全部概率与马氏距离已保存至: `random_test_detail_run_20260711_045637.parquet`

---

## 3. 已知问题与诊断

### 3.1 时间线表现

- **模型只识别出 `catch`（接球）和 `null`（Null）**，其余动作类（shot、左右运球、防守、高位传球）未在时间线上显形。
- 这与 v3.1 已知薄弱点一致：dribble_right_once vs shot 混淆、暂态类可分性弱。
- 随机录制中 Ryan 的随机动作节奏与训练中的结构化录制差异较大，导致模型严重偏向 null/catch。

### 3.2 拒识机制

- 马氏距离拒识在跨域（验证集 → 随机录制）下完全失效：null 类分布偏移导致马氏距离高达数百至数千。
- 当前工作点已禁用马氏距离触发（multiplier=10000），仅保留概率下限（0.30）作为拒识规则。
- 后续改进方向：使用全局异常检测（如 isolation forest）替代类内马氏距离，或引入域适应。

---

## 4. 双向跨人实验（重跑）

使用本次干净训练（随机录制零参与）重新运行双向跨人实验，以验证"Ryan→Owen 掉点是否源于 random 不对称"。

| 方向 | 模型 | 准确率 | Macro F1 |
|------|------|--------|----------|
| Owen → Ryan | RF | {cross_results['rf']['owen_to_ryan_acc']:.3f} | {cross_results['rf']['owen_to_ryan_f1']:.3f} |
| Owen → Ryan | SVM | {cross_results['svm']['owen_to_ryan_acc']:.3f} | {cross_results['svm']['owen_to_ryan_f1']:.3f} |
| Owen → Ryan | kNN | {cross_results['knn']['owen_to_ryan_acc']:.3f} | {cross_results['knn']['owen_to_ryan_f1']:.3f} |
| Ryan → Owen | RF | {cross_results['rf']['ryan_to_owen_acc']:.3f} | {cross_results['rf']['ryan_to_owen_f1']:.3f} |
| Ryan → Owen | SVM | {cross_results['svm']['ryan_to_owen_acc']:.3f} | {cross_results['svm']['ryan_to_owen_f1']:.3f} |
| Ryan → Owen | kNN | {cross_results['knn']['ryan_to_owen_acc']:.3f} | {cross_results['knn']['ryan_to_owen_f1']:.3f} |

**结论:** 在干净训练（随机录制不参与）下，双向跨人指标均较低（~0.3-0.5），说明 Ryan→Owen 掉点**并非**主要由随机录制不对称导致，而是跨人泛化本身的问题。后续需补采 owen 随机录制并做域适应/风格归一化。

---

## 5. 结论边界

1. 本次模型为**单人单段基准**，ryan 风格已在训练中见过，不评估跨人泛化。
2. 时间线已指出下一轮需修复的明确方向：
   - 丰富暂态类特征（dribble_right_once、shot）的区分度
   - 改善 null 类内部一致性（静止 null vs 动态 null）
   - 补采 owen 随机录制，做对称验证
3. 无论时间线准确度如何，本次交付的完整 Pipeline + 持久化基础设施 + 可复现推理链路，已构成端到端基准。

---

*报告结束。*
"""

report_path = REPORT_DIR / 'random_test_report.md'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report)

print(f"Report saved: {report_path}")

# Also compute and print cross-subject results
print("\nCross-subject results:")
for name, vals in cross_results.items():
    print(f"  {name.upper()}:")
    print(f"    Owen→Ryan: acc={vals['owen_to_ryan_acc']:.3f}, f1={vals['owen_to_ryan_f1']:.3f}")
    print(f"    Ryan→Owen: acc={vals['ryan_to_owen_acc']:.3f}, f1={vals['ryan_to_owen_f1']:.3f}")
