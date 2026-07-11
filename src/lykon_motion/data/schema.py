REQUIRED_COLUMNS = [
    "session_id", "player_id", "sensor_id", "side", "seq", "timestamp_us",
    "ax", "ay", "az", "gx", "gy", "gz", "mx", "my", "mz", "label",
]

OPTIONAL_COLUMNS = [
    "qw", "qx", "qy", "qz", "roll", "pitch", "yaw",
    "linear_acc_x", "linear_acc_y", "linear_acc_z",
    "uwb_x", "uwb_y", "uwb_z", "uwb_quality",
]

LABELS_V1 = ["idle", "dribble", "pass", "shot", "jump", "sprint"]
