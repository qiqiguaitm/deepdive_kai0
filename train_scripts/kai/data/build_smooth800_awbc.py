import glob, json, os, hashlib, shutil
import numpy as np, pandas as pd
SB="/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built"
BASE=f"{SB}/A_new_smooth_800/base"
AWBC=f"{SB}/A_smooth800_dagger_all_awbc"
OUT=f"{SB}/A_smooth800_awbc"
CAMS=["observation.images.top_head","observation.images.hand_left","observation.images.hand_right"]

def fp(pq):
    df=pd.read_parquet(pq, columns=['action'])
    a=np.stack(df['action'].to_numpy()).astype(np.float32)
    return (len(df), hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()[:12])

print("fingerprinting...", flush=True)
base=sorted(glob.glob(f"{BASE}/data/chunk-*/episode_*.parquet"))
awbc=sorted(glob.glob(f"{AWBC}/data/chunk-*/episode_*.parquet"))
bfp={}
for p in base: bfp.setdefault(fp(p),[]).append(p)
afp={}
for p in awbc: afp.setdefault(fp(p),[]).append(p)
matched=[(k, afp[k][0]) for k in bfp if k in afp and len(bfp[k])==1 and len(afp[k])==1]
old_idxs=sorted(int(os.path.basename(p).split('_')[1].split('.')[0]) for _,p in matched)
print(f"matched {len(old_idxs)} smooth800 eps", flush=True)

# load old meta indexed by old episode_index
oeps={json.loads(l)['episode_index']:json.loads(l) for l in open(f"{AWBC}/meta/episodes.jsonl")}
ostats={json.loads(l)['episode_index']:json.loads(l) for l in open(f"{AWBC}/meta/episodes_stats.jsonl")}

if os.path.exists(OUT): shutil.rmtree(OUT)
os.makedirs(f"{OUT}/data/chunk-000")
for c in CAMS: os.makedirs(f"{OUT}/videos/chunk-000/{c}")
os.makedirs(f"{OUT}/meta")

eps_out=[]; stats_out=[]; gidx=0; total=0
for new,old in enumerate(old_idxs):
    df=pd.read_parquet(f"{AWBC}/data/chunk-000/episode_{old:06d}.parquet")
    n=len(df)
    df=df.copy()
    df['episode_index']=np.int64(new)
    df['frame_index']=np.arange(n,dtype=np.int64)
    df['index']=np.arange(gidx,gidx+n,dtype=np.int64)
    df.to_parquet(f"{OUT}/data/chunk-000/episode_{new:06d}.parquet", index=False)
    for c in CAMS:
        src=f"{AWBC}/videos/chunk-000/{c}/episode_{old:06d}.mp4"
        tgt=os.path.realpath(src)  # resolve symlink to real file
        os.symlink(tgt, f"{OUT}/videos/chunk-000/{c}/episode_{new:06d}.mp4")
    er=dict(oeps[old]); er['episode_index']=new; eps_out.append(er)
    sr=dict(ostats[old]); sr['episode_index']=new; stats_out.append(sr)
    gidx+=n; total+=n

with open(f"{OUT}/meta/episodes.jsonl","w") as f:
    for e in eps_out: f.write(json.dumps(e)+"\n")
with open(f"{OUT}/meta/episodes_stats.jsonl","w") as f:
    for s in stats_out: f.write(json.dumps(s)+"\n")
shutil.copy(f"{AWBC}/meta/tasks.jsonl", f"{OUT}/meta/tasks.jsonl")
info=json.load(open(f"{AWBC}/meta/info.json"))
N=len(old_idxs)
info.update(total_episodes=N, total_frames=total, total_videos=N*3, chunks_size=N, splits={"train":f"0:{N}"})
json.dump(info, open(f"{OUT}/meta/info.json","w"), indent=2)
print(f"BUILT {OUT}: {N} ep, {total} frames", flush=True)
