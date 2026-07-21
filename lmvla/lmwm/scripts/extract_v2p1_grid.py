#!/usr/bin/env python
"""extract_v2p1_grid.py — 从 pi05 训练用的 **v2.1 LIBERO_fastwam** 数据抽 grid 特征 (A1/A2 上游).

对齐纪律 (DESIGN §3): LMWM ckpt 训练特征来自 v3.0 merged, 但 pi05 训练走 v2.1 LIBERO_fastwam。
本脚本直接从 v2.1 每-episode mp4 (base 相机 observation.images.image) 逐帧解码 → DINOv3/So400m grid,
输出按 (suite, ep, frame) 索引, **天然对齐** pi05 训练样本 (v2.1 frame_index=0..N-1 = 解码顺序)。
产物喂 export_pi05_hint.py (--feat-root)。

v2.1 布局: <root>/<suite>_no_noops_lerobot/videos/chunk-000/observation.images.image/episode_NNNNNN.mp4
输出:     <out>/<suite>/ep{E}.npz  key "grid" [N,256,DIN] fp16 + "frame_index" [N]

用法 (集群 kai0/.venv, 有 pyav + crave 编码器 + GPU):
  python extract_v2p1_grid.py --encoder dinov3-base --suites libero_10 --out .../data/pi05_feat/libero_v2p1_dinov3base
  python extract_v2p1_grid.py --encoder so400m      --suites all         --out .../data/pi05_feat/libero_v2p1_so400m_grid
"""
import os, sys, argparse, glob
import numpy as np

FASTWAM = "/vePFS/tim/workspace/LIBERO_fastwam"           # cnsh; North-E 用 --root 覆盖
CAM = "observation.images.image"
CRAVE_SRC_CANDIDATES = [
    "/vePFS/tim/workspace/deepdive_kai0/lmvla/crave/src",
    "/vePFS-North-E/vis_robot/workspace/deepdive_kai0/lmvla/crave/src",
]
SUITES = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]


def decode_episode_mp4(mp4_path):
    """整段解码一个 v2.1 episode mp4 → list[HxWx3 rgb uint8] (顺序 = frame_index 0..N-1)."""
    import av
    cont = av.open(mp4_path)
    frames = [fr.to_ndarray(format="rgb24") for fr in cont.decode(video=0)]
    cont.close()
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=FASTWAM, help="LIBERO_fastwam 根 (cnsh/North-E 不同)")
    ap.add_argument("--suites", nargs="*", default=["all"])
    ap.add_argument("--encoder", default="dinov3-base", help="dinov3-base (A1,768) | so400m (A2,1152)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--eps", default="all", help="all | a,b,c | start:end (每 suite 内 episode 下标)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-existing", action="store_true", help="跳过已产出的 ep npz (resume)")
    ap.add_argument("--shard", default=None, help="i/N: 只处理全局 episode 序号 %% N == i 的 (多 GPU 并行)")
    args = ap.parse_args()

    shard_i, shard_n = (None, None)
    if args.shard:
        shard_i, shard_n = (int(x) for x in args.shard.split("/"))

    suites = SUITES if args.suites == ["all"] else args.suites

    src = next((p for p in CRAVE_SRC_CANDIDATES if os.path.isdir(p)), None)
    assert src, f"crave src 未找到: {CRAVE_SRC_CANDIDATES}"
    sys.path.insert(0, src)
    import torch  # noqa
    from crave.encoders import load_encoder
    enc = load_encoder(args.encoder, dtype="bf16")
    print(f"[enc] {args.encoder} dim={enc.dim} | root={args.root}", flush=True)

    gidx = -1  # 全局 episode 序号 (跨 suite), 用于 shard
    for suite in suites:
        vdir = os.path.join(args.root, f"{suite}_no_noops_lerobot", "videos", "chunk-000", CAM)
        mp4s = sorted(glob.glob(os.path.join(vdir, "episode_*.mp4")),
                      key=lambda p: int(os.path.basename(p)[8:-4]))
        assert mp4s, f"no mp4 under {vdir}"
        # episode 子集选择
        if args.eps == "all":
            sel = mp4s
        elif ":" in args.eps:
            a, b = args.eps.split(":"); sel = mp4s[int(a):int(b)]
        else:
            want = {int(x) for x in args.eps.split(",")}
            sel = [m for m in mp4s if int(os.path.basename(m)[8:-4]) in want]
        outdir = os.path.join(args.out, suite)
        os.makedirs(outdir, exist_ok=True)
        print(f"[{suite}] {len(sel)}/{len(mp4s)} episodes → {outdir}", flush=True)
        for mp4 in sel:
            E = int(os.path.basename(mp4)[8:-4])
            gidx += 1
            if shard_n is not None and gidx % shard_n != shard_i:
                continue
            outp = os.path.join(outdir, f"ep{E}.npz")
            if args.skip_existing and os.path.exists(outp):
                continue
            frames = decode_episode_mp4(mp4)
            n = len(frames)
            grids = enc.encode_grid(frames)               # [N, dim, P, P]
            g = grids.detach().cpu().float().numpy() if hasattr(grids, "detach") else np.asarray(grids)
            N, D, P, _ = g.shape
            g = g.transpose(0, 2, 3, 1).reshape(N, P * P, D)   # [N, 256, DIN]
            if args.smoke:
                gf = g.astype(np.float32)
                print(f"  {suite}/ep{E}: frames={n} grid={g.shape} mean={gf.mean():.3f} std={gf.std():.3f}", flush=True)
                break
            np.savez_compressed(outp, grid=g.astype(np.float16),
                                frame_index=np.arange(N, dtype=np.int64))
        if args.smoke:
            break
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
