"""Compare JAX vs V1Policy outputs on a real dataset frame.

End-to-end correctness test: feeds the same real obs (3 images + state + prompt)
into both pipelines and diffs the 50-step action chunks. Unlike numerical_compare.py
which uses random synthetic inputs, this exercises:
  - real images (decoded via av from training MP4 at fixed frame_idx)
  - real state vector from parquet
  - the actual production V1Policy + SentencepieceStateEncoder code path
    (not a stripped re-implementation)

Two-phase script (different venvs needed because JAX and V1's torch+triton can't
share a process cleanly):

  # 1. Capture a frame (any venv — only pyarrow + av needed)
  kai0/.venv_5090_trt/bin/python optimize/v1_triton/compare_jax_v1_real_obs.py frame \\
      --dataset-root /data1/DATA_IMP/KAI0/Task_A/base \\
      --date 2026-05-09-v2 \\
      --episode 0 \\
      --frame-idx 100 \\
      --out /tmp/frame.npz

  # 2. JAX inference (kai0 .venv)
  kai0/.venv/bin/python optimize/v1_triton/compare_jax_v1_real_obs.py jax \\
      --frame /tmp/frame.npz \\
      --ckpt /data1/DATA_IMP/checkpoints/task_a_new_pure_200_step49999 \\
      --base-config-name pi05_flatten_fold_a_new_pure_1200 \\
      --out /tmp/jax_chunk.npz

  # 3. V1 inference (kai0 .venv_5090_trt)
  kai0/.venv_5090_trt/bin/python optimize/v1_triton/compare_jax_v1_real_obs.py v1 \\
      --frame /tmp/frame.npz \\
      --pkl /data1/tim/workspace/deepdive_kai0/optimize/results/task_a_new_pure_200_v1_p200.pkl \\
      --norm-stats /data1/DATA_IMP/checkpoints/task_a_new_pure_200_step49999/assets/a_new_pure_200/norm_stats.json \\
      --tokenizer /data1/tim/workspace/deepdive_kai0/openpi_cache/big_vision/paligemma_tokenizer.model \\
      --out /tmp/v1_chunk.npz

  # 4. Compare
  python optimize/v1_triton/compare_jax_v1_real_obs.py compare \\
      --jax /tmp/jax_chunk.npz \\
      --v1 /tmp/v1_chunk.npz
"""
import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Phase frame: load 1 frame (state + 3 images + prompt) from LeRobot dataset
# ─────────────────────────────────────────────────────────────────────────────

def phase_frame(args):
    import pyarrow.parquet as pq

    root = Path(args.dataset_root) / args.date
    info_path = root / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)
    data_pattern = info["data_path"]   # "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    video_pattern = info["video_path"]  # "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"

    # Episode chunk is // 1000 of episode_index in LeRobot, but most datasets have <1000 eps → chunk 0
    ep_chunk = args.episode // 1000
    parquet_path = root / data_pattern.format(episode_chunk=ep_chunk, episode_index=args.episode)
    if not parquet_path.exists():
        raise FileNotFoundError(f"parquet missing: {parquet_path}")
    print(f"[frame] loading parquet: {parquet_path}")

    tbl = pq.read_table(parquet_path)
    df = tbl.to_pandas()
    n_frames = len(df)
    if args.frame_idx >= n_frames:
        raise ValueError(f"frame_idx={args.frame_idx} ≥ episode length={n_frames}")
    row = df.iloc[args.frame_idx]
    state = np.asarray(row["observation.state"], dtype=np.float32)
    action_gt = np.asarray(row.get("action", np.zeros_like(state)), dtype=np.float32)
    # Prompt from tasks.jsonl + episode's task_index
    task_index = int(row.get("task_index", 0))
    tasks_path = root / "meta" / "tasks.jsonl"
    prompt = "Flatten and fold the cloth"  # default
    if tasks_path.exists():
        with open(tasks_path) as f:
            for line in f:
                t = json.loads(line)
                if t.get("task_index") == task_index:
                    prompt = t["task"].rstrip(".")  # serve_policy_v1 cleans trailing dot
                    break
    print(f"[frame] state.shape={state.shape}, frame_idx={args.frame_idx}, prompt={prompt!r}")

    # Load 3 video frames at frame_idx via av
    import av
    images = {}
    for view_key in ("top_head", "hand_left", "hand_right"):
        mp4 = root / video_pattern.format(episode_chunk=ep_chunk, video_key=view_key, episode_index=args.episode)
        if not mp4.exists():
            raise FileNotFoundError(f"video missing: {mp4}")
        container = av.open(str(mp4))
        stream = container.streams.video[0]
        # av decode iteratively up to frame_idx (mp4 seek is keyframe-aligned; for accuracy do sequential)
        target = args.frame_idx
        img = None
        for i, packet in enumerate(container.demux(stream)):
            for frame in packet.decode():
                if frame.pts is None:
                    continue
                # Use stream's avg frame rate to compute pts→idx mapping
                # Simpler: decode in order, count
                if i == target:
                    img = frame.to_ndarray(format="rgb24")
                    break
            if img is not None:
                break
        # The above can miss if decoder is async; fall back to dump-all-then-pick
        if img is None:
            container.close()
            container = av.open(str(mp4))
            frames = []
            for frame in container.decode(video=0):
                frames.append(frame.to_ndarray(format="rgb24"))
                if len(frames) > target:
                    break
            if target >= len(frames):
                raise ValueError(f"video has {len(frames)} frames < target {target}")
            img = frames[target]
        container.close()
        images[view_key] = img
        print(f"[frame]   {view_key}: shape={img.shape}, dtype={img.dtype}, mp4={mp4.name}")

    np.savez(args.out,
             state=state,
             action_gt=action_gt,
             prompt=prompt,
             top_head=images["top_head"],
             hand_left=images["hand_left"],
             hand_right=images["hand_right"])
    print(f"[frame] saved → {args.out}")
    print(f"[frame] state: {state}")
    print(f"[frame] action_gt[0:5]: {action_gt[:5]}  (training-time action at this frame)")


# ─────────────────────────────────────────────────────────────────────────────
# Phase jax: load JAX policy, run on real frame
# ─────────────────────────────────────────────────────────────────────────────

def phase_jax(args):
    import jax
    sys.path.insert(0, "/home/tim/workspace/deepdive_kai0/kai0/src")
    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config

    data = np.load(args.frame, allow_pickle=True)
    state = data["state"]
    prompt = str(data["prompt"])
    obs = {
        "images": {
            "top_head": data["top_head"],
            "hand_left": data["hand_left"],
            "hand_right": data["hand_right"],
        },
        "state": state,
        "prompt": prompt,
    }
    print(f"[JAX] devices: {jax.devices()}")
    print(f"[JAX] loading config {args.base_config_name} ...")
    cfg = _config.get_config(args.base_config_name)
    print(f"[JAX] loading policy from {args.ckpt} ...")
    policy = _policy_config.create_trained_policy(cfg, args.ckpt)

    # Fixed noise so JAX vs V1 share noise → action diff only from preproc/model
    rng = np.random.RandomState(args.noise_seed)
    noise = rng.randn(50, 32).astype(np.float32)

    print(f"[JAX] running policy.infer(obs, noise=noise[seed={args.noise_seed}]) ...")
    out = policy.infer(obs, noise=noise)
    actions = np.array(out["actions"])  # (50, 14) unnormalized
    print(f"[JAX] action shape: {actions.shape}, dtype={actions.dtype}")
    print(f"[JAX] action[0]: {actions[0]}")
    print(f"[JAX] action[0] − state[:14]: {actions[0] - state[:14]}  (delta from current pose)")
    print(f"[JAX] action range over chunk: [{actions.min():.4f}, {actions.max():.4f}]")
    print(f"[JAX] |Δ(0→49)| L={np.linalg.norm(actions[-1][:6]-actions[0][:6]):.3f} R={np.linalg.norm(actions[-1][7:13]-actions[0][7:13]):.3f}")

    np.savez(args.out, actions=actions, noise=noise, prompt=prompt, state=state)
    print(f"[JAX] saved → {args.out}")


# ─────────────────────────────────────────────────────────────────────────────
# Phase v1: load V1Policy from serve_policy_v1, run on real frame
# ─────────────────────────────────────────────────────────────────────────────

def phase_v1(args):
    """Uses the production V1Policy (with the quantile fix applied)."""
    sys.path.insert(0, "/home/tim/workspace/deepdive_kai0/kai0/scripts")
    sys.path.insert(0, "/home/tim/workspace/deepdive_kai0/optimize/v1_triton")
    import torch
    import serve_policy_v1 as _serve
    _serve._ensure_imports()  # populate Pi05InferenceTuned etc. (lazy-loaded in main())
    from serve_policy_v1 import (
        V1Policy, SentencepieceStateEncoder, load_norm_stats, load_v1_inference,
    )

    data = np.load(args.frame, allow_pickle=True)
    state = data["state"]
    prompt = str(data["prompt"])
    images = {
        "top_head": data["top_head"],
        "hand_left": data["hand_left"],
        "hand_right": data["hand_right"],
    }

    print(f"[V1] loading V1 inference from {args.pkl} ...")
    v1_infer, embed_w = load_v1_inference(args.pkl, num_views=3, chunk_size=50)
    print(f"[V1] loading norm_stats from {args.norm_stats} ...")
    norm = load_norm_stats(args.norm_stats)
    s_stats = norm["state"]
    s_stats_sliced = {
        "mean": s_stats["mean"][:14], "std": s_stats["std"][:14],
        "q01": s_stats["q01"][:14] if s_stats["q01"] is not None else None,
        "q99": s_stats["q99"][:14] if s_stats["q99"] is not None else None,
    }
    state_encoder = SentencepieceStateEncoder(
        v1_infer,
        tokenizer_model_path=args.tokenizer,
        embedding_weight=embed_w,
        state_norm=s_stats_sliced,
    )
    policy = V1Policy(
        v1_infer,
        action_norm=norm["actions"],
        action_dim=14,
        state_encoder=state_encoder,
        default_prompt=prompt,
        image_keys=("top_head", "hand_left", "hand_right"),
    )

    # Override the policy's per-call noise to a fixed seed for parity with JAX
    rng = np.random.RandomState(args.noise_seed)
    noise = rng.randn(50, 32).astype(np.float32)
    # Hack: V1Policy generates its own noise inside infer(); we monkey-patch
    # the v1_infer.forward to use our fixed noise.
    fixed_noise_t = torch.from_numpy(noise).to(torch.bfloat16).cuda()
    orig_forward = v1_infer.forward
    def fixed_forward(input_image, input_noise=None):
        return orig_forward(input_image, fixed_noise_t)
    v1_infer.forward = fixed_forward

    obs = {
        "images": images,
        "state": state,
        "prompt": prompt,
    }
    print(f"[V1] running V1Policy.infer(obs) ...")
    out = policy.infer(obs)
    actions = np.asarray(out["actions"])  # (50, 14)
    print(f"[V1] action shape: {actions.shape}, dtype={actions.dtype}")
    print(f"[V1] action[0]: {actions[0]}")
    print(f"[V1] action[0] − state[:14]: {actions[0] - state[:14]}")
    print(f"[V1] action range over chunk: [{actions.min():.4f}, {actions.max():.4f}]")
    print(f"[V1] |Δ(0→49)| L={np.linalg.norm(actions[-1][:6]-actions[0][:6]):.3f} R={np.linalg.norm(actions[-1][7:13]-actions[0][7:13]):.3f}")
    print(f"[V1] policy_timing: {out.get('policy_timing', {})}")

    np.savez(args.out, actions=actions, noise=noise, prompt=prompt, state=state)
    print(f"[V1] saved → {args.out}")


# ─────────────────────────────────────────────────────────────────────────────
# Phase compare
# ─────────────────────────────────────────────────────────────────────────────

def phase_compare(args):
    j = np.load(args.jax, allow_pickle=True)
    v = np.load(args.v1, allow_pickle=True)
    a_j = j["actions"]
    a_v = v["actions"]
    if a_j.shape != a_v.shape:
        m = min(a_j.shape[-1], a_v.shape[-1])
        a_j = a_j[..., :m]; a_v = a_v[..., :m]

    diff = np.abs(a_j - a_v)
    print("=" * 70)
    print("JAX vs V1Policy — same real obs, same noise seed")
    print("=" * 70)
    print(f"  shape:       JAX={a_j.shape}, V1={a_v.shape}")
    print(f"  JAX  range:  [{a_j.min():+.4f}, {a_j.max():+.4f}]  |mean| = {np.abs(a_j).mean():.4f}")
    print(f"  V1   range:  [{a_v.min():+.4f}, {a_v.max():+.4f}]  |mean| = {np.abs(a_v).mean():.4f}")
    print(f"  Δ ratio:     V1 mean_abs / JAX mean_abs = {np.abs(a_v).mean()/max(1e-9, np.abs(a_j).mean()):.3f}")
    print(f"  max diff:    {diff.max():.4e}")
    print(f"  mean diff:   {diff.mean():.4e}")
    print(f"  median diff: {np.median(diff):.4e}")
    print()
    print("Per-dim MAE (0-5 left arm joints, 6 left gripper, 7-12 right joints, 13 right gripper):")
    per_dim = diff.mean(axis=0)
    for d, v_d in enumerate(per_dim):
        side = "L" if d < 7 else "R"
        jname = "grip" if d in (6, 13) else f"j{d % 7}"
        print(f"  dim {d:2d} ({side} {jname:4}): JAX[0]={a_j[0,d]:+.4f}  V1[0]={a_v[0,d]:+.4f}  MAE={v_d:.4f}")
    print()
    print("Δ over horizon (chunk start → end), per arm side:")
    for label, sl in [("L", slice(0,6)), ("R", slice(7,13))]:
        dj = np.linalg.norm(a_j[-1, sl] - a_j[0, sl])
        dv = np.linalg.norm(a_v[-1, sl] - a_v[0, sl])
        print(f"  {label}: JAX |Δ(0→49)| = {dj:.4f},  V1 = {dv:.4f}  (ratio = {dv/max(1e-9, dj):.3f})")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="phase", required=True)

    pf = sub.add_parser("frame")
    pf.add_argument("--dataset-root", required=True)
    pf.add_argument("--date", required=True)
    pf.add_argument("--episode", type=int, required=True)
    pf.add_argument("--frame-idx", type=int, default=100)
    pf.add_argument("--out", required=True)

    pj = sub.add_parser("jax")
    pj.add_argument("--frame", required=True)
    pj.add_argument("--ckpt", required=True)
    pj.add_argument("--base-config-name", required=True)
    pj.add_argument("--noise-seed", type=int, default=42)
    pj.add_argument("--out", required=True)

    pv = sub.add_parser("v1")
    pv.add_argument("--frame", required=True)
    pv.add_argument("--pkl", required=True)
    pv.add_argument("--norm-stats", required=True)
    pv.add_argument("--tokenizer", required=True)
    pv.add_argument("--noise-seed", type=int, default=42)
    pv.add_argument("--out", required=True)

    pc = sub.add_parser("compare")
    pc.add_argument("--jax", required=True)
    pc.add_argument("--v1", required=True)

    args = p.parse_args()
    {"frame": phase_frame, "jax": phase_jax, "v1": phase_v1, "compare": phase_compare}[args.phase](args)


if __name__ == "__main__":
    main()
