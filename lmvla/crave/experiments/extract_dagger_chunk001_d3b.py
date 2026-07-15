#!/usr/bin/env python3
"""Extract DINOv3-base pooled features for chunk-001 dagger episodes.
Matches kai_extract_base.py extraction method (resize 224, encode_pooled, fp16).
Output: lmvla/crave/data/dagger_chunk001_dinov3base/{index.npz, shard_0.npz}
Run: PYTHONPATH=lmvla/crave/src python lmvla/crave/experiments/extract_dagger_chunk001_d3b.py [--limit N]
"""
import sys, json, time, numpy as np, cv2, av
from pathlib import Path

REPO = Path("/vePFS/tim/workspace/deepdive_kai0")
DAGGER = REPO / "kai0/data/Task_A/vis_dagger/v4"
OUT = REPO / "lmvla/crave/data/dagger_chunk001_dinov3base"
OUT.mkdir(parents=True, exist_ok=True)
LIMIT = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None

# Collect all chunk-001 episodes across dates
eps = []
for dt_dir in sorted(DAGGER.glob("2026-*-v4")):
    c1 = dt_dir / "data" / "chunk-001"
    if not c1.exists(): continue
    for pq in sorted(c1.glob("episode_*.parquet")):
        ep_idx = int(pq.stem.split("_")[1])
        # 不同日期 ep 号会冲突(每个日期都从 0 开始) → 编码: date_hash * 10000 + ep_idx
        # date_hash = day-of-month + month*100 (= 616 for 06-16, = 701 for 07-01)
        parts = dt_dir.name.split("-")
        dh = int(parts[1]) * 100 + int(parts[2])  # e.g., "06-16" → 616, "07-01" → 701
        global_ep = dh * 10000 + ep_idx
        vid = dt_dir / "videos" / "chunk-001" / "observation.images.top_head" / f"episode_{ep_idx:06d}.mp4"
        if vid.exists():
            eps.append((dt_dir.name, global_ep, vid, pq))
if LIMIT:
    eps = eps[:LIMIT]

print(f"{len(eps)} episodes with top_head video", flush=True)
if not eps:
    print("ERROR: no videos found. Run video sync first.", flush=True)
    sys.exit(1)

# Load encoder
from crave.encoders import load_encoder
enc = load_encoder("dinov3-base", device="cuda")
print("DINOv3-base loaded", flush=True)

E_all, FR_all, T_all, FEAT_all = [], [], [], []
t0 = time.time()
for i, (dt, ep, vid_path, pq_path) in enumerate(eps):
    # Decode video
    cap = av.open(str(vid_path))
    frames = []
    for f in cap.decode(video=0):
        frames.append(cv2.resize(cv2.cvtColor(f.to_ndarray(format="bgr24"), cv2.COLOR_BGR2RGB), (224, 224)))
    cap.close()
    n_frames = len(frames)
    if n_frames < 3:
        print(f"  [{i+1}/{len(eps)}] ep{ep} ({dt}): {n_frames} frames → SKIP", flush=True)
        continue

    # Batch encode
    feats = []
    bs = 256
    for k in range(0, n_frames, bs):
        batch = np.stack(frames[k:k+bs])
        feats.append(np.asarray(enc.encode_pooled(batch)).astype(np.float16))
    feat_ep = np.concatenate(feats)

    E_all.append(np.full(n_frames, ep, dtype=np.int64))
    FR_all.append(np.arange(n_frames, dtype=np.int64))
    T_all.append((np.arange(n_frames) / 30.0).astype(np.float32))
    FEAT_all.append(feat_ep)

    if (i + 1) % 50 == 0:
        el = time.time() - t0
        eta = el / (i + 1) * (len(eps) - i - 1) / 60
        print(f"  [{i+1}/{len(eps)}] {el/60:.0f}min, ~{eta:.0f}min left", flush=True)

# Concat and save
E = np.concatenate(E_all); FR = np.concatenate(FR_all)
T = np.concatenate(T_all); FEAT = np.concatenate(FEAT_all)
n = len(E)
np.savez(OUT / "index.npz", E=E, FR=FR, T=T, n=np.int64(n))
np.savez(OUT / "shard_0.npz", gidx=np.arange(n, dtype=np.int64), feat=FEAT, valid=np.ones(n, bool))
el = (time.time() - t0) / 60
print(f"DONE {len(eps)} eps / {n} frames in {el:.1f}min → {OUT}", flush=True)
print(f"  index: E unique={len(np.unique(E))}  FR range=[{FR.min()},{FR.max()}]  T range=[{T.min():.1f},{T.max():.1f}]", flush=True)
