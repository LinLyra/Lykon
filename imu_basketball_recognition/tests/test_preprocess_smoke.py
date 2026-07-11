import sys
sys.path.insert(0, '/Users/owen/Desktop/lykon-motion-lab-v2/imu_basketball_recognition/src')
from io_utils import load_recording
from preprocess import preprocess_recording
import numpy as np

print("Loading recording...")
df = load_recording('owen', '26071001接球-拍球-跳投', validate=False)
print(f"Before preprocess: shape={df.shape}, cols={len(df.columns)}")

print("Preprocessing...")
df_p = preprocess_recording(df)
print(f"After preprocess: shape={df_p.shape}, cols={len(df_p.columns)}")

# Check a_vert is centered near 0 (gravity removed)
for node in ['node1', 'node2', 'node3', 'node4']:
    av = df_p[f'a_vert_{node}'].values
    ah = df_p[f'a_horiz_{node}'].values
    print(f"  {node}: a_vert mean={av.mean():+.4f} std={av.std():.2f} | "
          f"a_horiz mean={ah.mean():.2f} std={ah.std():.2f}")

# Check pitch of RU during a shot segment (R1, around 30-50s)
mask = (df_p['t'] > 30) & (df_p['t'] < 50)
pitch_range = df_p.loc[mask, 'pitch_node1'].agg(['min', 'max'])
print(f"\nRU pitch (30-50s): {pitch_range['min']:.2f} to {pitch_range['max']:.2f} rad")
