import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.interpolate import interp1d

OUT=Path("data/processed/our_data")
ANIM=OUT/"animation_json"; ANIM.mkdir(parents=True,exist_ok=True)

template=json.loads((OUT/"free_throw_skeleton_template.json").read_text(encoding="utf-8"))
release=pd.read_csv(OUT/"event_release_points.csv")
phases=pd.read_csv(OUT/"event_phase_segments.csv")
features=pd.read_csv(OUT/"shot_cycle_features.csv") if (OUT/"shot_cycle_features.csv").exists() else pd.DataFrame({"window_id":release.window_id})
table=release.merge(features,on="window_id",how="left")
joints_order=template["joint_order"]
source={j:np.array([f["joints"][j] for f in template["frames"]],float) for j in joints_order}
src_n=template["n_frames"]; src_release=int(src_n*.47)

def sample(v,idx): return interp1d(np.arange(len(v)),v,axis=0,fill_value="extrapolate")(idx)

def phase_at(g,i):
    m=g[(g.start_index<=i)&(g.end_index>=i)]
    return str(m.iloc[0].phase) if len(m) else "unknown"

def scale_col(col,lo,hi):
    if col not in table: return pd.Series(np.ones(len(table)))
    x=table[col].astype(float); a=x.quantile(.05); b=x.quantile(.95)
    return lo+((x-a)/max(b-a,1e-9)).clip(0,1)*(hi-lo)

table["s1"]=scale_col("n1_gyro_mag_max",.85,1.20)
table["s2"]=scale_col("n2_gyro_mag_max",.85,1.25)
table["s3"]=scale_col("n3_gyro_mag_max",.85,1.15)

manifest=[]
for _,r in table.iterrows():
    wid=str(r.window_id)
    g=phases[phases.window_id==wid]
    if g.empty: continue
    n=int(g.end_index.max())+1
    rel=int(r.release_index)
    t=np.arange(n); idx=np.zeros(n,float)
    before=t<=rel
    idx[before]=t[before]/max(rel,1)*src_release
    idx[~before]=src_release+(t[~before]-rel)/max(n-1-rel,1)*(src_n-1-src_release)
    seq={j:sample(source[j],idx) for j in joints_order}

    strength=.15
    rs,re,rw=seq["right_shoulder"],seq["right_elbow"],seq["right_wrist"]
    ls,le,lw=seq["left_shoulder"],seq["left_elbow"],seq["left_wrist"]
    re2=rs+(re-rs)*(1+strength*(float(r.s1)-1))
    rw2=re2+(rw-re)*(1+strength*(float(r.s2)-1))
    le2=ls+(le-ls)*(1+strength*(float(r.s3)-1))
    lw2=le2+(lw-le)*(1+strength*(float(r.s3)-1))
    seq["right_elbow"],seq["right_wrist"],seq["left_elbow"],seq["left_wrist"]=re2,rw2,le2,lw2

    data={"window_id":wid,"fps":50,"n_frames":n,"release_index":rel,"frames":[],"warning":"下肢来自视频动作模板，IMU控制节奏与上肢轻度变化。"}
    for i in range(n):
        js={j:seq[j][i].round(6).tolist() for j in joints_order}
        js["hips"]=((seq["left_hip"][i]+seq["right_hip"][i])/2).round(6).tolist()
        data["frames"].append({"frame":i,"time_s":i/50,"phase":phase_at(g,i),"joints":js})
    path=ANIM/f"{wid}.json"
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")
    manifest.append({"window_id":wid,"json_path":str(path),"n_frames":n,"release_index":rel})

pd.DataFrame(manifest).to_csv(OUT/"animation_manifest.csv",index=False)
print("Generated:",len(manifest))
print("Directory:",ANIM)
