import argparse
from pathlib import Path
import csv
import cv2
import mediapipe as mp

LANDMARK_NAMES = [
    "nose","left_eye_inner","left_eye","left_eye_outer",
    "right_eye_inner","right_eye","right_eye_outer",
    "left_ear","right_ear","mouth_left","mouth_right",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_pinky","right_pinky",
    "left_index","right_index","left_thumb","right_thumb",
    "left_hip","right_hip","left_knee","right_knee",
    "left_ankle","right_ankle","left_heel","right_heel",
    "left_foot_index","right_foot_index"
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument(
        "--output",
        default="data/processed/our_data/video_pose_keypoints.csv"
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(video_path)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    mp_pose = mp.solutions.pose

    fieldnames = [
        "frame_id", "timestamp_s", "fps",
        "landmark_id", "landmark_name",
        "x_norm", "y_norm", "z_rel", "visibility"
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose:
            frame_id = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = pose.process(rgb)
                timestamp_s = frame_id / fps

                if result.pose_landmarks:
                    for lid, lm in enumerate(result.pose_landmarks.landmark):
                        writer.writerow({
                            "frame_id": frame_id,
                            "timestamp_s": timestamp_s,
                            "fps": fps,
                            "landmark_id": lid,
                            "landmark_name": LANDMARK_NAMES[lid],
                            "x_norm": lm.x,
                            "y_norm": lm.y,
                            "z_rel": lm.z,
                            "visibility": lm.visibility,
                        })

                frame_id += 1

    cap.release()
    print("Saved:", output_path)
    print("注意：该视频若没有与IMU同步起始标记，目前只能做姿态参考，不能直接做毫秒级对齐。")

if __name__ == "__main__":
    main()
