#!/usr/bin/env python3
"""Velocity-distribution fidelity: policy predicted-action speed vs dataset speed.

Research pilot (velocity_fidelity_plan): does a trained VLA/AWBC policy's produced
MOTION SPEED distribution match the demonstration dataset's? Freeze = the extreme
(speed→0). We test whether an offline, open-loop speed-distribution metric separates
a known-FROZEN ckpt from a known-GOOD ckpt — i.e. a pre-deploy freeze detector.

Open-loop protocol: for each val query frame, feed the observation with the DEPLOY
prompt ("...Advantage: positive") → model predicts a 50-step action chunk. Compare the
per-step arm speed WITHIN the predicted chunk against the GT future chunk at the same
state. Accumulate speed marginals; report distributional divergence vs GT.

One ckpt per invocation (parallelize across GPUs). Metrics saved to npz+json; aggregate
with --aggregate.

  CUDA_VISIBLE_DEVICES=0 .venv/bin/python train_scripts/kai/eval/velocity_fidelity.py \
      --config pi05_v4_awbc_plus_freshdagger \
      --ckpt checkpoints/pi05_v4_awbc_plus_freshdagger/pi05_v4_awbc_plus_freshdagger/49999 \
      --val data/Task_A/self_built/vis_v2_merged_val --tag frozen --n-frames 40
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np

ARM = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]   # 12 arm joints, exclude gripper 6/13
PROMPT_POSITIVE = "Flatten and fold the cloth. Advantage: positive"
CAMS = ("top_head", "hand_left", "hand_right")
H = 50


def load_val(val_root: str, n_frames: int):
    import pyarrow.parquet as pq
    import av
    vr = Path(val_root)
    samples = []
    for ep_path in sorted((vr / "data" / "chunk-000").glob("episode_*.parquet")):
        df = pq.read_table(ep_path).to_pandas()
        ep = int(ep_path.stem.split("_")[1])
        state = np.stack([np.asarray(x, np.float32) for x in df["observation.state"]])
        action = np.stack([np.asarray(x, np.float32) for x in df["action"]])
        L = len(state)
        q_idx = np.linspace(0, max(L - H - 1, 0), n_frames).astype(int)
        q_set = set(int(k) for k in q_idx)
        vids = {}
        for cam in CAMS:
            vp = vr / "videos" / "chunk-000" / f"observation.images.{cam}" / f"episode_{ep:06d}.mp4"
            c = av.open(str(vp)); c.streams.video[0].thread_type = "AUTO"
            picked = {}
            for i, f in enumerate(c.decode(video=0)):
                if i in q_set:
                    picked[i] = f.to_ndarray(format="rgb24")
                    if len(picked) == len(q_set):
                        break
            c.close()
            last = next(iter(sorted(picked, reverse=True)), None)
            if last is not None:
                for k in q_set - set(picked):
                    picked[k] = picked[last]
            vids[cam] = picked
        samples.append({"ep": ep, "L": L, "state": state, "action": action, "images": vids, "q": q_idx})
    if not samples:
        raise RuntimeError(f"no val samples in {val_root}")
    return samples


def speeds_of(chunk: np.ndarray) -> np.ndarray:
    """Per-step arm speed within an action chunk [T, D] (D>=13)."""
    return np.linalg.norm(np.diff(chunk[:, ARM], axis=0), axis=1)


def w1(a: np.ndarray, b: np.ndarray, n=2000) -> float:
    """1-D Wasserstein-1 via quantile integral (no scipy)."""
    qs = np.linspace(0, 1, n)
    return float(np.mean(np.abs(np.quantile(a, qs) - np.quantile(b, qs))))


def js(a: np.ndarray, b: np.ndarray, lo, hi, bins=80) -> float:
    edges = np.linspace(lo, hi, bins + 1)
    pa, _ = np.histogram(a, edges, density=True); pa = pa / pa.sum()
    pb, _ = np.histogram(b, edges, density=True); pb = pb / pb.sum()
    m = 0.5 * (pa + pb)
    def kl(p, q):
        mask = p > 0
        return float(np.sum(p[mask] * np.log(p[mask] / np.clip(q[mask], 1e-12, None))))
    return 0.5 * kl(pa, m) + 0.5 * kl(pb, m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config"); ap.add_argument("--ckpt"); ap.add_argument("--val")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--n-frames", type=int, default=40)
    ap.add_argument("--out-dir", default="docs/training/future_plans/plans/data/velocity_fidelity")
    ap.add_argument("--aggregate", nargs="*", help="tags to aggregate+compare (skips inference)")
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    if a.aggregate:
        rows = []
        for tag in a.aggregate:
            d = np.load(out / f"velfid_{tag}.npz")
            m = json.loads((out / f"velfid_{tag}.json").read_text())
            rows.append((tag, d["pred"], d["gt"], m))
        gt = rows[0][2]
        lo, hi = 0.0, float(np.percentile(np.concatenate([gt] + [r[1] for r in rows]), 99.5))
        print(f"\n{'tag':16s} {'pred_med':>9} {'gt_med':>9} {'med_ratio':>9} {'pred_p90':>9} "
              f"{'p90_ratio':>9} {'pred_static%':>12} {'gt_static%':>10} {'W1':>8} {'JS':>7}")
        for tag, pred, g, m in rows:
            print(f"{tag:16s} {np.median(pred):9.4f} {np.median(g):9.4f} "
                  f"{np.median(pred)/np.median(g):9.3f} {np.percentile(pred,90):9.4f} "
                  f"{np.percentile(pred,90)/np.percentile(g,90):9.3f} "
                  f"{100*np.mean(pred<0.02):12.1f} {100*np.mean(g<0.02):10.1f} "
                  f"{w1(pred,g):8.4f} {js(pred,g,lo,hi):7.4f}")
        (out / "velfid_summary.json").write_text(json.dumps(
            {tag: m for tag, _, _, m in rows}, indent=2))
        print(f"\nsummary → {out/'velfid_summary.json'}")
        return

    # ---- inference ----
    import jax
    from openpi.training import config as _config
    from openpi.policies import policy_config
    from openpi.shared import normalize as _normalize
    print(f"[{a.tag}] jax devices: {jax.devices()}", flush=True)
    cfg = _config.get_config(a.config)
    # ckpt step dir's assets/ is empty; norm_stats.json lives at the exp dir (parent).
    norm_dir = Path(a.ckpt).parent
    norm_stats = _normalize.load(norm_dir)
    print(f"[{a.tag}] norm_stats from {norm_dir} keys={list(norm_stats)}", flush=True)
    policy = policy_config.create_trained_policy(cfg, a.ckpt, norm_stats=norm_stats)
    samples = load_val(a.val, a.n_frames)
    print(f"[{a.tag}] {len(samples)} val eps, prompt='{PROMPT_POSITIVE}'", flush=True)

    pred_sp, gt_sp = [], []
    n_infer = 0
    for s in samples:
        for k in s["q"]:
            if k + 1 + H > s["L"]:
                continue
            obs = {"images": {c: s["images"][c][int(k)] for c in s["images"]},
                   "state": s["state"][int(k)], "prompt": PROMPT_POSITIVE}
            pred = np.asarray(policy.infer(obs)["actions"])   # [H, D]
            gt = s["action"][k + 1:k + 1 + H]
            hh = min(pred.shape[0], len(gt))
            pred_sp.append(speeds_of(pred[:hh]))
            gt_sp.append(speeds_of(gt[:hh]))
            n_infer += 1
        print(f"[{a.tag}] ep{s['ep']} done, infers={n_infer}", flush=True)

    pred = np.concatenate(pred_sp); gt = np.concatenate(gt_sp)
    lo, hi = 0.0, float(np.percentile(np.concatenate([pred, gt]), 99.5))
    metrics = {
        "tag": a.tag, "config": a.config, "ckpt": a.ckpt, "val": a.val,
        "n_infer": n_infer, "n_speed_samples": int(len(pred)),
        "pred_median": float(np.median(pred)), "gt_median": float(np.median(gt)),
        "median_ratio": float(np.median(pred) / np.median(gt)),
        "pred_p90": float(np.percentile(pred, 90)), "gt_p90": float(np.percentile(gt, 90)),
        "p90_ratio": float(np.percentile(pred, 90) / np.percentile(gt, 90)),
        "pred_static_pct": float(100 * np.mean(pred < 0.02)),
        "gt_static_pct": float(100 * np.mean(gt < 0.02)),
        "pred_mean": float(pred.mean()), "gt_mean": float(gt.mean()),
        "W1_pred_vs_gt": w1(pred, gt), "JS_pred_vs_gt": js(pred, gt, lo, hi),
    }
    np.savez(out / f"velfid_{a.tag}.npz", pred=pred, gt=gt)
    (out / f"velfid_{a.tag}.json").write_text(json.dumps(metrics, indent=2))
    print(f"[{a.tag}] DONE  median_ratio={metrics['median_ratio']:.3f}  "
          f"pred_static%={metrics['pred_static_pct']:.1f} (gt {metrics['gt_static_pct']:.1f})  "
          f"W1={metrics['W1_pred_vs_gt']:.4f}  JS={metrics['JS_pred_vs_gt']:.4f}", flush=True)
    print(f"[{a.tag}] → {out/f'velfid_{a.tag}.json'}", flush=True)


if __name__ == "__main__":
    main()
