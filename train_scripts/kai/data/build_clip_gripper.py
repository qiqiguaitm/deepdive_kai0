import glob, os, shutil, json
import numpy as np, pandas as pd
SB="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built"
S=f"{SB}/A_smooth800_dagger_all"; OUT=f"{SB}/A_smooth800_dagger_clip_all"
CAMS=["observation.images.top_head","observation.images.hand_left","observation.images.hand_right"]
GRIP=[6,13]
if os.path.exists(OUT): shutil.rmtree(OUT)
os.makedirs(f"{OUT}/data/chunk-000"); os.makedirs(f"{OUT}/meta")
for c in CAMS: os.makedirs(f"{OUT}/videos/chunk-000/{c}")
# meta copy (same episodes/frames/tasks)
for m in ["info.json","episodes.jsonl","episodes_stats.jsonl","tasks.jsonl"]:
    if os.path.exists(f"{S}/meta/{m}"): shutil.copy(f"{S}/meta/{m}", f"{OUT}/meta/{m}")
pqs=sorted(glob.glob(f"{S}/data/chunk-000/episode_*.parquet"))
n_clip=0; n_tot=0; n_outlier=0
for pq in pqs:
    df=pd.read_parquet(pq)
    a=np.stack(df['action'].to_numpy()).astype(np.float32)
    g=a[:,GRIP]
    n_tot+=g.size
    out_mask=g>0.1; n_outlier+=int(out_mask.sum()); g[out_mask]=0.1   # sanitize outliers
    clip_mask=g<=0.005; n_clip+=int(clip_mask.sum()); g[clip_mask]=0.0  # ≤5mm → 0
    a[:,GRIP]=g
    df['action']=list(a)
    df.to_parquet(f"{OUT}/data/chunk-000/{os.path.basename(pq)}", index=False)
    eid=int(os.path.basename(pq).split('_')[1].split('.')[0])
    for c in CAMS:
        src=f"{S}/videos/chunk-000/{c}/episode_{eid:06d}.mp4"
        os.symlink(os.path.realpath(src), f"{OUT}/videos/chunk-000/{c}/episode_{eid:06d}.mp4")
print(f"BUILT {OUT}: {len(pqs)} ep; gripper-action vals clipped ≤5mm→0: {n_clip}/{n_tot} ({100*n_clip/n_tot:.1f}%); outliers>0.1m clamped: {n_outlier}")
