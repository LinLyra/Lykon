import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

JOINTS=["nose","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle","left_heel","right_heel","left_foot_index","right_foot_index"]

def rz(x):
    x=np.asarray(x,float); m=np.nanmedian(x); mad=np.nanmedian(np.abs(x-m)); s=1.4826*mad
    if not np.isfinite(s) or s<1e-9: s=np.nanstd(x) or 1
    return (x-m)/s

def smooth(x,k=5): return np.convolve(x,np.ones(k)/k,mode="same")

def resample(x,n):
    return interp1d(np.linspace(0,1,len(x)),x,axis=0,fill_value="extrapolate")(np.linspace(0,1,n))

def xyz(w,j): return w[[f"x_{j}",f"y_{j}",f"z_{j}"]].to_numpy(float)

def speed(x,fps): return np.linalg.norm(np.diff(x,axis=0,prepend=x[[0]]),axis=1)*fps

p=argparse.ArgumentParser()
p.add_argument("--input",default="data/processed/our_data/video_pose_world.csv")
p.add_argument("--target-events",type=int,default=6)
p.add_argument("--out-dir",default="data/processed/our_data")
a=p.parse_args()

df=pd.read_csv(a.input)
sub=df[df.landmark_name.isin(JOINTS)]
w=sub.pivot_table(index=["frame_id","timestamp_s","fps"],columns="landmark_name",values=["x","y","z","visibility"],aggfunc="first")
w.columns=[f"{v}_{j}" for v,j in w.columns]; w=w.reset_index().sort_values("frame_id").reset_index(drop=True)
fps=float(w.fps.median())

rw, re, rh = xyz(w,"right_wrist"), xyz(w,"right_elbow"), xyz(w,"right_hip")
hip=(xyz(w,"left_hip")+rh)/2
score=smooth(.55*np.maximum(rz(speed(rw,fps)),0)+.25*np.maximum(rz(speed(re,fps)),0)+.20*np.maximum(rz(-np.gradient(hip[:,1])*fps),0),5)

best=None
for q in np.linspace(.95,.30,60):
    prom=max(float(np.quantile(score,q)-np.quantile(score,.25)),1e-6)
    peaks,props=find_peaks(score,distance=max(1,int(1.8*fps)),prominence=prom)
    item=(abs(len(peaks)-a.target_events),peaks,props)
    if best is None or item[0]<best[0]: best=item
peaks=best[1]
if len(peaks)>a.target_events:
    idx=np.argsort(best[2]["prominences"])[::-1][:a.target_events]; peaks=np.sort(peaks[idx])

segments=[]; rows=[]; nout=171
for n,pk in enumerate(peaks,1):
    s=pk-int(1.2*fps); e=pk+int(1.8*fps)
    if s<0 or e>=len(w): continue
    seg={}
    for j in JOINTS:
        pts=xyz(w.iloc[s:e+1],j)
        vis=w.iloc[s:e+1][f"visibility_{j}"].to_numpy(float)
        pts[vis<.35]=np.nan
        for d in range(3):
            v=pts[:,d]; ix=np.arange(len(v)); ok=np.isfinite(v)
            pts[:,d]=np.interp(ix,ix[ok],v[ok]) if ok.sum()>=2 else np.nan_to_num(v)
            pts[:,d]=smooth(pts[:,d],5)
        seg[j]=resample(pts,nout)
    root=(seg["left_hip"]+seg["right_hip"])/2
    for j in JOINTS: seg[j]-=root
    torso=np.median(np.linalg.norm(((seg["left_shoulder"]+seg["right_shoulder"])/2),axis=1))
    scale=(1.75*.29)/max(torso,1e-6)
    for j in JOINTS:
        v=seg[j]*scale
        seg[j]=np.c_[v[:,0],-v[:,2],-v[:,1]]
    segments.append(seg)
    rows.append({"video_event_id":f"V{n:02d}","peak_frame":int(w.loc[pk,"frame_id"]),"peak_time_s":float(w.loc[pk,"timestamp_s"])})

template={j:np.median(np.stack([s[j] for s in segments]),axis=0) for j in JOINTS}
out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
tensor=np.stack([template[j] for j in JOINTS],axis=1)
np.save(out/"free_throw_skeleton_template.npy",tensor)

data={"fps":50,"n_frames":nout,"joint_order":JOINTS,"frames":[]}
for i in range(nout):
    joints={j:template[j][i].round(6).tolist() for j in JOINTS}
    joints["hips"]=((template["left_hip"][i]+template["right_hip"][i])/2).round(6).tolist()
    data["frames"].append({"frame":i,"time_s":i/50,"joints":joints})
(out/"free_throw_skeleton_template.json").write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")
pd.DataFrame(rows).to_csv(out/"video_motion_segments.csv",index=False)
print("Valid segments:",len(segments))
print("Saved template:",out/"free_throw_skeleton_template.json")
