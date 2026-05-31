#!/usr/bin/env python3
"""TAC vs non-TAC diagnostic on a held-out val split.

Reproduces the P1 (chunk-to-chunk continuity) / P2 (multi-sample noise variance)
methodology from docs/training/analysis/data_scale_vs_quality_vis_v2_full_vs_pure_200.md
plus a clean single-shot MAE control, so a TAC ckpt can be compared apples-to-apples
against a non-TAC baseline.

Why these metrics: TAC (training-time action conditioning) trains the model to predict
the chunk postfix given a clean prefix -> its design goal is *chunk self-consistency*,
not single-shot accuracy. Clean MAE is therefore the control (expect ~parity); the
discriminator is P1 (lower = smoother consecutive chunks = the thing that shows up as
real-machine oscillation). The buggy TAC v7 scored P1=0.067 (worse than baseline); this
script measures whether the bug-fixed tac_v2 actually improved P1.

Loads each ckpt via its <ckpt>/train_config.json sidecar (OPENPI_EXTRA_CONFIG), so no
edits to src/openpi/training/config.py are needed. No RTC swap is used — P1/P2/MAE all
use the plain (Pi0 / Pi0+tac) sample_actions with an injectable fixed noise.

Usage (run from kai0/):
  .venv/bin/python ../train_scripts/kai/eval/eval_tac_diagnostic.py \
      --ckpt /data1/DATA_IMP/checkpoints/ckpt_v0/pi05_flatten_fold_a_new_pure_200_tac_v2_step49999 \
      --val  data/Task_A/self_built/A_new_pure_200_val \
      --n-ep 5 --walk 30 --p2-stride 4 --p2-samples 8
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

CAMS = ("top_head", "hand_left", "hand_right")
HORIZONS = (1, 10, 25, 50)
ARMS = {"L": (0, 7), "R": (7, 14), "all": (0, 14)}


def read_video_frames(path: Path, n_frames: int) -> np.ndarray:
    import av
    container = av.open(str(path))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    out = []
    for frame in container.decode(stream):
        out.append(frame.to_ndarray(format="rgb24"))
        if len(out) >= n_frames:
            break
    container.close()
    arr = np.stack(out[:n_frames], axis=0)
    if arr.shape[0] < n_frames:
        arr = np.concatenate([arr, np.repeat(arr[-1:], n_frames - arr.shape[0], axis=0)], axis=0)
    return arr


def arm_mae(diff_abs: np.ndarray) -> dict:
    """diff_abs: (..., 14) absolute differences -> per-arm mean."""
    return {a: float(diff_abs[..., lo:hi].mean()) for a, (lo, hi) in ARMS.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="ckpt dir containing train_config.json sidecar")
    ap.add_argument("--val", required=True, help="LeRobot val repo dir (meta/, data/, videos/)")
    ap.add_argument("--prompt", default="Flatten and fold the cloth.")
    ap.add_argument("--n-ep", type=int, default=5, help="num val episodes to use")
    ap.add_argument("--walk", type=int, default=30, help="contiguous frames per ep for MAE+P1")
    ap.add_argument("--start-off", type=int, default=10, help="skip first N frames per ep")
    ap.add_argument("--p2-stride", type=int, default=4, help="P2 sampled every Nth walk frame")
    ap.add_argument("--p2-samples", type=int, default=8, help="P2 random samples per frame")
    ap.add_argument("--seed", type=int, default=0, help="fixed-noise seed")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ckpt = Path(args.ckpt).resolve()
    sidecar = ckpt / "train_config.json"
    spec = json.loads(sidecar.read_text())
    base_config = spec["base_config_name"]

    # Apply sidecar override at config import time.
    os.environ["OPENPI_EXTRA_CONFIG"] = str(sidecar)
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    train_cfg = _config.get_config(base_config)
    model_cfg = train_cfg.model
    ah = int(getattr(model_cfg, "action_horizon", 50))
    ad = int(getattr(model_cfg, "action_dim", 32))
    tac = bool(getattr(model_cfg, "tac_enabled", False))
    print(f"[load] ckpt={ckpt.name}  base_config={base_config}  tac_enabled={tac}  ah={ah} ad={ad}")

    t0 = time.time()
    policy = _policy_config.create_trained_policy(train_cfg, str(ckpt))
    print(f"[load] policy ready in {time.time()-t0:.1f}s")

    fixed_noise = np.random.default_rng(args.seed).standard_normal((ah, ad)).astype(np.float32)

    val = Path(args.val).resolve()
    episodes = [json.loads(l) for l in (val / "meta" / "episodes.jsonl").read_text().splitlines()]
    episodes = episodes[: args.n_ep]

    mae_acc = {h: [] for h in HORIZONS}           # clean single-shot MAE (fixed noise)
    p1_fixed = {a: [] for a in ARMS}              # consecutive chunk overlap |diff|, fixed noise
    p1_rand = {a: [] for a in ARMS}               # same, fresh random noise
    p2_std_means, p2_std_maxes = [], []           # multi-sample per-(step,dim) std

    def infer_chunk(obs, noise):
        return np.asarray(policy.infer(obs, noise=noise)["actions"])[:, :14]  # (ah,14)

    for ep in episodes:
        ei = ep["episode_index"]
        L = ep["length"]
        df = pq.read_table(val / "data" / "chunk-000" / f"episode_{ei:06d}.parquet").to_pandas()
        state = np.stack([np.asarray(x) for x in df["observation.state"]])  # (L,14)
        action = np.stack([np.asarray(x) for x in df["action"]])            # (L,14)
        cams = {c: read_video_frames(val / "videos" / "chunk-000" / f"observation.images.{c}" / f"episode_{ei:06d}.mp4", L) for c in CAMS}

        lo = args.start_off
        hi = min(lo + args.walk, L - max(HORIZONS) - 1)
        frames = list(range(lo, hi))
        chunks_fixed, chunks_rand = {}, {}
        tep = time.time()
        for k in frames:
            obs = {"images": {c: cams[c][k] for c in CAMS}, "state": state[k], "prompt": args.prompt}
            cf = infer_chunk(obs, fixed_noise)
            cr = infer_chunk(obs, None)  # advances policy rng -> fresh noise
            chunks_fixed[k] = cf
            chunks_rand[k] = cr
            # clean MAE (fixed-noise chunk vs GT), eval_val_action_mse convention
            for h in HORIZONS:
                gt = action[k + 1 : k + 1 + h]
                if gt.shape[0] == h:
                    mae_acc[h].append(float(np.abs(gt - cf[:h]).mean()))

        # P1: consecutive-frame chunk overlap |diff|
        for k in frames[:-1]:
            if k + 1 not in chunks_fixed:
                continue
            df_fix = np.abs(chunks_fixed[k][1:ah] - chunks_fixed[k + 1][0 : ah - 1])
            df_rnd = np.abs(chunks_rand[k][1:ah] - chunks_rand[k + 1][0 : ah - 1])
            for a, m in arm_mae(df_fix).items():
                p1_fixed[a].append(m)
            for a, m in arm_mae(df_rnd).items():
                p1_rand[a].append(m)

        # P2: multi-sample variance at strided frames
        for k in frames[:: args.p2_stride]:
            obs = {"images": {c: cams[c][k] for c in CAMS}, "state": state[k], "prompt": args.prompt}
            samples = np.stack([infer_chunk(obs, None) for _ in range(args.p2_samples)], axis=0)  # (N,ah,14)
            std = samples.std(axis=0)  # (ah,14)
            p2_std_means.append(float(std.mean()))
            p2_std_maxes.append(float(std.max()))
        print(f"  ep{ei:02d}  frames={len(frames)}  ({time.time()-tep:.0f}s)")

    summary = {
        "ckpt": str(ckpt),
        "base_config": base_config,
        "tac_enabled": tac,
        "n_episodes": len(episodes),
        "clean_mae": {h: float(np.mean(mae_acc[h])) for h in HORIZONS},
        "P1_fixed": {a: float(np.mean(p1_fixed[a])) for a in ARMS},
        "P1_random": {a: float(np.mean(p1_rand[a])) for a in ARMS},
        "P1_noise_contrib": {a: float(np.mean(p1_rand[a]) - np.mean(p1_fixed[a])) for a in ARMS},
        "P2_std_mean": float(np.mean(p2_std_means)),
        "P2_std_max": float(np.max(p2_std_maxes)),
    }
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))
    out = Path(args.out) if args.out else ckpt / "eval_tac_diagnostic.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
