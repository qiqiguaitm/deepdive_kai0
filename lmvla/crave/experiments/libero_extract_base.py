"""Extract DINOv3-base (facebook/dinov3-vitb16-pretrain-lvd1689m, 768D pooled) features for LIBERO-Long
(libero_10_no_noops_lerobot), building a bank identical in layout to crave/data/kai_dinov3base.

LIBERO libero_10 = LeRobot v2.1, per-ep mp4 + per-ep parquet, 20fps, main cam observation.images.image.
This is LaWM's in-distribution data (arena B). Output: lmvla/crave/data/libero10_dinov3base/{index,shard_0}.npz

Run (2 GPUs):
  cd lmvla
  CUDA_VISIBLE_DEVICES=0 srpo_py crave/experiments/libero_extract_base.py worker 0 &
  CUDA_VISIBLE_DEVICES=1 srpo_py crave/experiments/libero_extract_base.py worker 1 &
  wait; srpo_py crave/experiments/libero_extract_base.py merge
"""
import os, sys, json, time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]           # .../deepdive_kai0/lmvla
sys.path.insert(0, str(REPO / "crave/src"))
# suite via env: LIBERO_SUITE in {10,spatial,object,goal}; NSPLIT workers (1 = single-GPU-per-suite).
SUITE = os.environ.get("LIBERO_SUITE", "10")
LIB = Path(f"/vePFS/tim/workspace/LIBERO_fastwam/libero_{SUITE}_no_noops_lerobot")
CAM = "observation.images.image"; FPS = 20.0
# LIBERO videos are AV1 (cv2 can't decode); use the pre-decoded frame_cache npy (T,256,256,3 uint8, RGB).
FCACHE = LIB / "frame_cache/resize_256x256"
WORK = REPO / f"temp/libero{SUITE}_extract_base"; WORK.mkdir(parents=True, exist_ok=True)
NSPLIT = int(os.environ.get("NSPLIT", "2"))


def npy_path(e):
    return FCACHE / f"chunk-{e // 1000:03d}/{CAM}/episode_{e:06d}.npy"


def all_eps():
    npys = sorted((FCACHE / "chunk-000" / CAM).glob("episode_*.npy"))
    return sorted(int(p.stem.split("_")[1]) for p in npys)


def worker(g):
    from crave.encoders import load_encoder
    eps = all_eps()[g::NSPLIT]
    enc = load_encoder("dinov3-base", device="cuda"); t0 = time.time()
    E, FR, FE = [], [], []
    for ci, e in enumerate(eps):
        fp = npy_path(e)
        if not fp.exists():
            print(f"[g{g}] miss ep{e}", flush=True); continue
        arr = np.load(fp)                                    # (T,256,256,3) uint8 RGB
        frames = [cv2.resize(arr[i], (224, 224)) for i in range(len(arr))]
        if len(frames) < 5:
            continue
        feats = []
        for k in range(0, len(frames), 256):
            feats.append(np.asarray(enc.encode_pooled(np.stack(frames[k:k + 256]))).astype(np.float16))
        fe = np.concatenate(feats); n = len(fe)
        E += [e] * n; FR += list(range(n)); FE.append(fe)
        if ci % 25 == 0:
            print(f"[g{g}] {ci+1}/{len(eps)} ep{e} n{n} · {(time.time()-t0)/60:.1f}min", flush=True)
    np.savez(WORK / f"part_{g}.npz", E=np.array(E, np.int64), FR=np.array(FR, np.int64),
             feat=np.concatenate(FE))
    print(f"[g{g}] DONE {len(E)} frames ({(time.time()-t0)/60:.1f}min)", flush=True)


def merge():
    parts = sorted(WORK.glob("part_*.npz"))
    E = np.concatenate([np.load(p)["E"] for p in parts])
    FR = np.concatenate([np.load(p)["FR"] for p in parts])
    feat = np.concatenate([np.load(p)["feat"] for p in parts])
    o = np.lexsort((FR, E)); E, FR, feat = E[o], FR[o], feat[o]; n = len(E)
    out = REPO / f"crave/data/libero{SUITE}_dinov3base"; out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "index.npz", E=E.astype(np.int64), FR=FR.astype(np.int64),
             T=(FR / FPS).astype(np.float32), n=np.int64(n))
    np.savez(out / "shard_0.npz", gidx=np.arange(n, dtype=np.int64), feat=feat, valid=np.ones(n, bool))
    print(f"merged {len(np.unique(E))} eps / {n} frames -> crave/data/libero10_dinov3base/", flush=True)


if __name__ == "__main__":
    {"worker": lambda: worker(int(sys.argv[2])), "merge": merge}[sys.argv[1]]()
