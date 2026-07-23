import argparse, csv
from pathlib import Path
import cv2
import mediapipe as mp

NAMES=["nose","left_eye_inner","left_eye","left_eye_outer","right_eye_inner","right_eye","right_eye_outer","left_ear","right_ear","mouth_left","mouth_right","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_pinky","right_pinky","left_index","right_index","left_thumb","right_thumb","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle","left_heel","right_heel","left_foot_index","right_foot_index"]

p=argparse.ArgumentParser()
p.add_argument("--video",required=True)
p.add_argument("--out",default="data/processed/our_data/video_pose_world.csv")
a=p.parse_args()

video=Path(a.video); out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
if not video.exists(): raise FileNotFoundError(video)

cap=cv2.VideoCapture(str(video))
fps=cap.get(cv2.CAP_PROP_FPS) or 30.0
fields=["frame_id","timestamp_s","fps","landmark_id","landmark_name","x","y","z","visibility"]

with out.open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
    with mp.solutions.pose.Pose(model_complexity=2,smooth_landmarks=True,min_detection_confidence=.5,min_tracking_confidence=.5) as pose:
        frame_id=0
        while True:
            ok,frame=cap.read()
            if not ok: break
            r=pose.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            if r.pose_world_landmarks:
                for i,lm in enumerate(r.pose_world_landmarks.landmark):
                    w.writerow({"frame_id":frame_id,"timestamp_s":frame_id/fps,"fps":fps,"landmark_id":i,"landmark_name":NAMES[i],"x":lm.x,"y":lm.y,"z":lm.z,"visibility":lm.visibility})
            frame_id+=1
cap.release()
print("Saved:",out)
