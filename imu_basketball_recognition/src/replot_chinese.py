"""
Quick re-plot with Chinese font support.
"""
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, '/Users/owen/Desktop/lykon-motion-lab-v2/imu_basketball_recognition/src')
from config import NODES, ACTION_LABELS_INV

# Set Chinese fonts
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'PingFang HK', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

REPORT_DIR = '/Users/owen/Desktop/lykon-motion-lab-v2/imu_basketball_recognition/reports/random_test'
RUN_NAME = 'run_20260711_045637'

# Load data
events_df = pd.read_csv(f'{REPORT_DIR}/random_test_timeline_{RUN_NAME}.csv')
detail_df = pd.read_parquet(f'{REPORT_DIR}/random_test_detail_{RUN_NAME}.parquet')

# We need raw |a| data for background — load from parquet if we saved it, otherwise skip
# For simplicity, re-plot without |a| background, just event strips

# Timeline plot (simplified)
fig, ax = plt.subplots(figsize=(16, 6))

color_map = {
    'catch': '#2ecc71',
    'dribble_left_right': '#3498db',
    'dribble_right_once': '#9b59b6',
    'defense': '#e67e22',
    'pass_high': '#1abc9c',
    'shot': '#e74c3c',
    'unknown': '#95a5a6',
    'null': '#bdc3c7',
}

# Draw event strips
for _, e in events_df.iterrows():
    color = color_map.get(e['label'], '#95a5a6')
    ax.axvspan(e['start_s'], e['end_s'], color=color, alpha=0.3, zorder=2)
    if e['label'] != 'unknown' and e['label'] != 'null':
        mid = (e['start_s'] + e['end_s']) / 2.0
        ax.annotate(str(e['event_id']), xy=(mid, 0.9), ha='center', fontsize=8,
                    fontweight='bold', color='black', transform=ax.get_xaxis_transform())

ax.set_xlabel('时间 [s]')
ax.set_ylabel('事件存在')
ax.set_title(f'随机录制时间线 — {RUN_NAME}')
ax.set_ylim(0, 1)
ax.set_yticks([])

handles = [mpatches.Patch(color=c, label=ACTION_LABELS_INV.get(lbl, lbl))
           for lbl, c in color_map.items() if lbl in events_df['label'].values or lbl == 'unknown']
ax.legend(handles=handles, loc='upper left', fontsize=8)

plt.tight_layout()
plt.savefig(f'{REPORT_DIR}/random_test_timeline_{RUN_NAME}.png', dpi=150)
plt.close()
print(f'Re-plotted timeline: {REPORT_DIR}/random_test_timeline_{RUN_NAME}.png')

# Model comparison plot
fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

t_centers = detail_df['t_center'].values
model_names = ['RF', 'SVM', 'kNN']
model_keys = ['label_rf', 'label_svm', 'label_knn']

for idx, (ax, key, mname) in enumerate(zip(axes, model_keys, model_names)):
    labels = detail_df[key].values
    current = labels[0]
    start_t = t_centers[0] - 0.15
    for i in range(1, len(labels)):
        if labels[i] != current:
            end_t = t_centers[i - 1] + 0.15
            color = color_map.get(current, '#95a5a6')
            ax.axvspan(start_t, end_t, color=color, alpha=0.3)
            start_t = t_centers[i] - 0.15
            current = labels[i]
    ax.axvspan(start_t, t_centers[-1] + 0.15, color=color_map.get(current, '#95a5a6'), alpha=0.3)
    ax.set_ylabel(mname)
    ax.set_ylim(0, 1)
    ax.set_yticks([])

axes[-1].set_xlabel('时间 [s]')
plt.suptitle('三模型对比 — 随机录制')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f'{REPORT_DIR}/random_test_comparison_{RUN_NAME}.png', dpi=150)
plt.close()
print(f'Re-plotted comparison: {REPORT_DIR}/random_test_comparison_{RUN_NAME}.png')
