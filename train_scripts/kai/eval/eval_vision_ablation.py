#!/usr/bin/env python3
"""Vision-ablation causal test: does a pi05 policy actually USE the camera?

For N real observations we run the policy on:
  (1) real images        -> pred_real_a, pred_real_b   (2 runs => flow-noise floor)
  (2) blacked images     -> pred_black                  (all cams = 0)
  (3) frozen images      -> pred_frozen                 (cams replaced by episode frame 0,
                                                          state kept at frame k => image/state mismatch)
We compare how much the predicted action chunk MOVES under image corruption vs the
pure flow-matching noise floor (same obs, different noise). A closed-loop policy's
output changes a lot when the image is corrupted (delta >> noise). An open-loop /
causally-confused policy ignores the image (delta ~= noise).

Metric per group: mean |Δ| over the chunk, and the key ratio
    SNR = delta_corrupt / noise_floor
SNR ~ 1  => ignores vision.   SNR >> 1 => uses vision.
Gripper dims (6,13) reported separately = the direct grasp-success channel.

Usage (run from repo root, JAX venv):
  CUDA_VISIBLE_DEVICES=3 kai0/.venv/bin/python train_scripts/kai/eval/eval_vision_ablation.py \
    --config <base_config_name> --ckpt <ckpt_dir> --asset-id <override_asset_id> \
    --val kai0/data/Task_A/self_built/A_new_pure_200_val --prompt "Flatten and fold the cloth." \
    --n-frames 24 --n-episodes 4
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

GRIP = [6, 13]
ARM = [i for i in range(14) if i not in GRIP]


def read_video(path: Path, n: int) -> np.ndarray:
    import av
    c = av.open(str(path)); s = c.streams.video[0]; s.thread_type = "AUTO"
    out = []
    for fr in c.decode(s):
        out.append(fr.to_ndarray(format="rgb24"))
        if len(out) >= n:
            break
    c.close()
    a = np.stack(out[:n], 0)
    if a.shape[0] < n:
        a = np.concatenate([a, np.repeat(a[-1:], n - a.shape[0], 0)], 0)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--asset-id", required=True, help="override_asset_id (norm stats subdir)")
    ap.add_argument("--val", required=True)
    ap.add_argument("--prompt", default="Flatten and fold the cloth.")
    ap.add_argument("--n-frames", type=int, default=24)
    ap.add_argument("--n-episodes", type=int, default=4)
    args = ap.parse_args()

    from openpi.policies import policy_config as pc
    from openpi.training import config as tc
    from openpi.training import checkpoints as ck

    ckpt = Path(args.ckpt).resolve()
    val = Path(args.val).resolve()
    train_cfg = tc.get_config(args.config)
    norm_stats = ck.load_norm_stats(ckpt / "assets", args.asset_id)
    print(f"[load] config={args.config} ckpt={ckpt.name} asset={args.asset_id}", flush=True)
    t0 = time.time()
    policy = pc.create_trained_policy(train_cfg, ckpt, norm_stats=norm_stats)
    print(f"[load] ready {time.time()-t0:.1f}s", flush=True)

    eps = [json.loads(l) for l in (val / "meta" / "episodes.jsonl").read_text().splitlines()]
    eps = eps[: args.n_episodes]
    cams = ("top_head", "hand_left", "hand_right")

    nf = {"floor": [], "black": [], "frozen": [], "motion": [],
          "floor_g": [], "black_g": [], "frozen_g": []}
    for ep in eps:
        ei, L = ep["episode_index"], ep["length"]
        df = pq.read_table(val / "data" / "chunk-000" / f"episode_{ei:06d}.parquet").to_pandas()
        state = np.stack([np.asarray(x) for x in df["observation.state"]]).astype(np.float32)
        vid = {c: read_video(val / "videos" / "chunk-000" / f"observation.images.{c}" / f"episode_{ei:06d}.mp4", L)
               for c in cams}
        ks = np.linspace(0, L - 1, args.n_frames).astype(int)
        for k in ks:
            real = {c: vid[c][k] for c in cams}
            black = {c: np.zeros_like(vid[c][k]) for c in cams}
            frozen = {c: vid[c][0] for c in cams}   # stale image, state stays at k
            st = state[k]

            def infer(imgs):
                return np.asarray(policy.infer({"images": imgs, "state": st, "prompt": args.prompt})["actions"])
            a = infer(real); b = infer(real)        # noise floor (same obs, diff noise)
            bl = infer(black); fz = infer(frozen)
            H = min(len(a), len(b), len(bl), len(fz))
            a, b, bl, fz = a[:H], b[:H], bl[:H], fz[:H]
            def md(x, y, idx): return float(np.mean(np.abs(x[:, idx] - y[:, idx])))
            nf["floor"].append(md(a, b, ARM)); nf["black"].append(md(a, bl, ARM)); nf["frozen"].append(md(a, fz, ARM))
            nf["floor_g"].append(md(a, b, GRIP)); nf["black_g"].append(md(a, bl, GRIP)); nf["frozen_g"].append(md(a, fz, GRIP))
            nf["motion"].append(float(np.mean(np.abs(a[:, ARM] - st[ARM]))))   # how far model wants to move
        print(f"  ep{ei}: done", flush=True)

    def m(key): return float(np.mean(nf[key]))
    floor, black, frozen, motion = m("floor"), m("black"), m("frozen"), m("motion")
    fg, bg, zg = m("floor_g"), m("black_g"), m("frozen_g")
    print("\n========== VISION ABLATION ==========")
    print(f"ckpt: {ckpt.name}")
    print(f"  ARM joints (rad):")
    print(f"    noise_floor (same obs, diff noise)   = {floor:.5f}")
    print(f"    Δ black-image  vs real               = {black:.5f}   SNR={black/max(floor,1e-9):.2f}x")
    print(f"    Δ frozen-image vs real               = {frozen:.5f}   SNR={frozen/max(floor,1e-9):.2f}x")
    print(f"    motion magnitude (|action-state|)    = {motion:.5f}")
    print(f"  GRIPPER dims [6,13] (grasp channel):")
    print(f"    noise_floor                          = {fg:.5f}")
    print(f"    Δ black-image                        = {bg:.5f}   SNR={bg/max(fg,1e-9):.2f}x")
    print(f"    Δ frozen-image                       = {zg:.5f}   SNR={zg/max(fg,1e-9):.2f}x")
    print(f"  => vision_sensitivity(arm) = Δblack/floor = {black/max(floor,1e-9):.2f}x  "
          f"(~1=ignores vision, >>1=uses vision)")


if __name__ == "__main__":
    main()
