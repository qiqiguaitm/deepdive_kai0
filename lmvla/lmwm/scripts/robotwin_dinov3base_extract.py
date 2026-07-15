#!/usr/bin/env python
"""抽 robotwin2.0 cam_high 的 DINOv3-base 池化特征 [N,768] —— 与 LIBERO 逐比特同编码路径。
源 = frame_cache_jpeg256(每 ep 一个 npz, 键 "0".."N-1" = 256px JPEG bytes), 避 av1 解码坑。
池化 = encode_grid[N,D,P,P].mean(spatial) —— 与 libero_dinov3base 消费方式(grid.mean(1))逐比特一致。
双卡并行: 两进程分别 --shard 0/1 --nshard 2 (CUDA_VISIBLE_DEVICES=0/1)。
用法: CUDA_VISIBLE_DEVICES=0 srpo python robotwin_dinov3base_extract.py --shard 0 --nshard 2
       ... --eps 0,1 --smoke   (冒烟)
Out: lmwm/data/robotwin_dinov3base/ep{e}.npz  key=pooled [N,768] fp16
"""
import os, sys, glob, argparse, time
import numpy as np, cv2

REPO = "/vePFS/tim/workspace/deepdive_kai0"
DS = f"{REPO}/lmvla/lawam/dataset/robotwin2.0"
CAM = "observation.images.cam_high"
CRAVE_SRC = f"{REPO}/lmvla/crave/src"
OUT = f"{REPO}/lmvla/lmwm/data/robotwin_dinov3base"

def build_ep_map():
    m = {}
    for p in glob.glob(f"{DS}/frame_cache_jpeg256/chunk-*/{CAM}/episode_*.npz"):
        e = int(os.path.basename(p).split("_")[1].split(".")[0]); m[e] = p
    return m

def decode_ep(path):
    d = np.load(path)
    n = len(d.files)
    out = []
    for i in range(n):
        img = cv2.imdecode(d[str(i)], cv2.IMREAD_COLOR)          # BGR uint8
        out.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--eps", default=None, help="逗号列表(仅冒烟/调试)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    epmap = build_ep_map()
    all_eps = sorted(epmap)
    if a.eps:
        eps = [int(x) for x in a.eps.split(",")]
    else:
        eps = all_eps[a.shard::a.nshard]
    print(f"[shard {a.shard}/{a.nshard}] {len(eps)}/{len(all_eps)} eps", flush=True)

    sys.path.insert(0, CRAVE_SRC)
    from crave.encoders import load_encoder
    import torch
    enc = load_encoder("dinov3-base", dtype="bf16")
    print(f"[enc] dinov3-base dim={enc.dim} loaded", flush=True)
    os.makedirs(a.out, exist_ok=True)

    t0 = time.time(); done = 0; nfr = 0
    for e in eps:
        op = os.path.join(a.out, f"ep{e}.npz")
        if os.path.exists(op) and not a.smoke:
            done += 1; continue                                   # 断点续抽
        frames = decode_ep(epmap[e]); n = len(frames)
        pooled = []
        with torch.no_grad():
            for i in range(0, n, a.bs):
                g = enc.encode_grid(frames[i:i+a.bs])             # [b,D,P,P]
                if hasattr(g, "detach"): g = g.detach().float().mean(dim=(2, 3)).cpu().numpy()
                else: g = np.asarray(g).mean(axis=(2, 3))
                pooled.append(g)
        pooled = np.concatenate(pooled).astype(np.float16)        # [N,768]
        if a.smoke:
            pf = pooled.astype(np.float32)
            print(f"  ep{e}: N={n} pooled={pooled.shape} mean={pf.mean():.3f} std={pf.std():.3f} "
                  f"norm~{np.linalg.norm(pf,axis=1).mean():.2f}", flush=True)
        else:
            np.savez_compressed(op, pooled=pooled)
        done += 1; nfr += n
        if done % 200 == 0:
            el = time.time()-t0; print(f"  [shard {a.shard}] {done}/{len(eps)} eps {nfr}f | {nfr/el:.0f} fr/s | ETA {(len(eps)-done)*el/max(done,1)/60:.0f}min", flush=True)
    print(f"[shard {a.shard}] DONE {done} eps {nfr}f ({time.time()-t0:.0f}s)", flush=True)
    print(f"SHARD_{a.shard}_DONE", flush=True)

if __name__ == "__main__":
    main()
