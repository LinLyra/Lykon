"""
IMU Basketball Action Recognition — Central Configuration
All parameters in one place. No magic numbers elsewhere.
"""
import numpy as np

# =============================================================================
# Hardware & Sampling
# =============================================================================
FS = 50.0                       # Sampling frequency [Hz]
DT = 1.0 / FS                   # Sampling interval [s]
NODES = ['node1', 'node2', 'node3', 'node4']
NODE_ABBR = {                   # Anatomical mapping
    'node1': 'RU',             # Right Upper arm
    'node2': 'RF',             # Right Forearm
    'node3': 'LF',             # Left Forearm
    'node4': 'LU',             # Left Upper arm
}

# Physical constants
G_STD = 9.80665                 # Standard gravity [m/s^2]

# =============================================================================
# Data Paths
# =============================================================================
DATA_ROOT = '/Users/owen/Desktop/lykon-motion-lab-v2/lykon_dataset'
SUBJECTS = ['owen', 'ryan']

# Recording name → (rec_id, action_type, events_per_cycle)
# action_type: 'discrete' = event sequence (R1, R4), 'continuous' = whole-segment (R2, R3)
RECORDING_MAP = {
    'owen': {
        '26071001接球-拍球-跳投': ('R1', 'discrete', ['catch', 'dribble_right_once', 'shot'], 3),
        '26071002左右运球':        ('R2', 'continuous', ['dribble_left_right'], None),
        '26071003防守':            ('R3', 'continuous', ['defense'], None),
        '26071004高位传球':        ('R4', 'discrete', ['catch', 'pass_high'], 2),
    },
    'ryan': {
        '26071011接球-拍球-投球': ('R1', 'discrete', ['catch', 'dribble_right_once', 'shot'], 3),
        '26071012左右运球':        ('R2', 'continuous', ['dribble_left_right'], None),
        '26071013防守':            ('R3', 'continuous', ['defense'], None),
        '26071014高位传球':        ('R4', 'discrete', ['catch', 'pass_high'], 2),
        '26071015随机做':          ('R5', 'continuous', ['random'], None),  # extra / open-set test
    },
}

# Calibrated file suffix pattern
CALIB_SUFFIX = '_calibrated_50hz.csv'

# =============================================================================
# Filtering
# =============================================================================
# Low-pass for general smoothing (anti-aliasing / motion band)
LP_CUTOFF = 20.0                # [Hz] — covers basketball motion band
LP_ORDER = 4

# High-pass for acceleration (gravity removal on L1 if needed)
HP_CUTOFF = 0.5                 # [Hz] — keep above gravity drift
HP_ORDER = 4

# Band for motion energy (used in event segmentation)
ENERGY_BAND = (0.5, 20.0)       # [Hz]

# =============================================================================
# Sliding Window
# =============================================================================
WIN_LEN_S = 1.2                 # Window length [s]
WIN_STEP_S = 0.3                # Step size [s]  (75% overlap default)
WIN_LEN = int(round(WIN_LEN_S * FS))   # samples
WIN_STEP = int(round(WIN_STEP_S * FS)) # samples

# Window length candidates for hyper-parameter scan (Phase 7)
WIN_LEN_CANDIDATES = [0.8, 1.2, 1.6, 2.0]

# v3.1 Fix: Event-anchor window parameters
ANCHOR_PRE_S = 0.5              # Time before anchor point [s]
ANCHOR_POST_S = 0.7             # Time after anchor point [s]
ANCHOR_JITTER_SHIFTS = [0.0, 0.1, 0.2]  # Jitter offsets for augmentation [s]
ANCHOR_JITTER_SIGNS = [-1, 1]   # Random sign for jitter

# Label purity threshold for a window to receive a class label
# v3.1: DISABLED for anchor windows (always 1.0 for anchor-based)
LABEL_PURITY = 0.70             # ≥70% samples must share same label
# (Only used for continuous sliding windows)

# =============================================================================
# Event Segmentation (Phase 3b — labeling only, never used at inference)
# =============================================================================
SEG_RMS_WIN_S = 0.5             # RMS sliding window for energy [s]
SEG_RMS_STEP_S = 0.1            # RMS step [s]
SEG_THRESH_HI_MUL = 5.0         # High threshold = resting RMS × this
SEG_THRESH_LO_MUL = 2.0         # Low threshold = resting RMS × this
SEG_MERGE_GAP_S = 0.3           # Merge events closer than this [s]
SEG_MIN_DUR_S = 0.3             # Discard events shorter than this [s]

# =============================================================================
# Feature Extraction
# =============================================================================
# Frequency bands for spectral features [Hz]
FBANDS = {
    'low':    (0.0, 3.0),
    'mid':    (3.0, 8.0),
    'high':   (8.0, 25.0),
}

# Number of features to retain after selection
N_FEATURES_SELECT = 50
CORR_THRESHOLD = 0.95           # Remove redundant features with |corr| > this

# =============================================================================
# Classification
# =============================================================================
RANDOM_STATE = 42
RF_N_ESTIMATORS = 300
RF_CLASS_WEIGHT = 'balanced'

# Open-set rejection thresholds
REJ_PROB_THRESH = 0.50          # Max class probability < this → reject
REJ_MAHAL_QUANTILE = 0.975      # Mahalanobis distance > this quantile → reject

# Temporal smoothing
SMOOTH_N_VOTES = 5              # Number of windows for majority vote

# =============================================================================
# Evaluation
# =============================================================================
# Target metrics
TARGET_MACRO_F1 = 0.90          # Macro F1 on merged-data rep-level CV

# =============================================================================
# Plotting
# =============================================================================
FIG_DPI = 150
FIG_WIDTH = 10
FIG_HEIGHT = 6

# =============================================================================
# Action label names (Chinese → English code)
# =============================================================================
ACTION_LABELS = {
    '接球': 'catch',
    '高位传球': 'pass_high',
    '防守': 'defense',
    '左右运球': 'dribble_left_right',
    '右手拍一次球': 'dribble_right_once',
    '投篮': 'shot',
    'Null': 'null',
}
ACTION_LABELS_INV = {v: k for k, v in ACTION_LABELS.items()}

# Family grouping (for analysis only)
FAMILIES = {
    'dribble_family': ['dribble_left_right', 'dribble_right_once'],
    'pass_catch_family': ['catch', 'pass_high'],
    'shot_family': ['shot'],
    'move_family': ['defense'],
}
