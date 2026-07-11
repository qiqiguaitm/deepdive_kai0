import sys, os, json, numpy as np, cv2, av, time
from pathlib import Path
sys.path.insert(0, "/vePFS-North-E/vis_robot/workspace/deepdive_kai0/crave/src")
D = Path("/vePFS-North-E/vis_robot/workspace/deepdive_kai0")
ROOT = D / "temp/aloha_tasks"; OUT = D / "lmvla/crave/data"

def ep_lengths(dsdir):
    # episodes.jsonl: {"episode_index":.., "length":..} 或 meta/episodes/chunk*/file*.parquet(v3)
    ej = dsdir / "meta/episodes.jsonl"
    if ej.exists():
        rows = [json.loads(l) for l in open(ej)]
        if rows and "length" in rows[0]:
            return {int(r["episode_index"]): int(r["length"]) for r in rows}
    # fallback: 从 data parquet 的 episode_index 数
    import pandas as pd
    dfs = sorted((dsdir/"data").glob("chunk-*/*.parquet"))
    ei = np.concatenate([pd.read_parquet(p, columns=["episode_index"])["episode_index"].to_numpy() for p in dfs])
    u, c = np.unique(ei, return_counts=True); return {int(k): int(v) for k, v in zip(u, c)}

def worker(g, ds):
    from crave.encoders import load_encoder
    dsdir = ROOT / ds; enc = load_encoder("dinov3-base", device="cuda"); t0 = time.time()
    L = ep_lengths(dsdir); eps = sorted(L)
    vpath = dsdir / "videos/observation.images.cam_high/chunk-000/file-000.mp4"
    cont = av.open(str(vpath)); allf = []
    for fr in cont.decode(video=0):
        allf.append(cv2.resize(fr.to_ndarray(format="rgb24"), (224, 224)))
    cont.close()
    print(f"[g{g}] {ds}: {len(allf)} frames decoded, {len(eps)} eps (sum={sum(L.values())})", flush=True)
    E=[]; FR=[]; FE=[]; off=0
    for e in eps:
        n = L[e]; frames = allf[off:off+n]; off += n
        if len(frames) < 5: continue
        feats=[]
        for k in range(0, len(frames), 256):
            feats.append(np.asarray(enc.encode_pooled(np.stack(frames[k:k+256]))).astype(np.float16))
        fe = np.concatenate(feats); m = len(fe); E += [e]*m; FR += list(range(m)); FE.append(fe)
    Ea=np.array(E,np.int64); FRa=np.array(FR,np.int64); feat=np.concatenate(FE)
    o = OUT / f"{ds}_dinov3base"; o.mkdir(parents=True, exist_ok=True)
    np.savez(o/"index.npz", E=Ea, FR=FRa, T=(FRa/50.).astype(np.float32), n=np.int64(len(Ea)))
    np.savez(o/"shard_0.npz", gidx=np.arange(len(Ea),dtype=np.int64), feat=feat, valid=np.ones(len(Ea),bool))
    print(f"[g{g}] {ds} DONE {len(Ea)} frames ({(time.time()-t0)/60:.1f}min)", flush=True)

if __name__ == "__main__":
    worker(int(sys.argv[1]), sys.argv[2])
