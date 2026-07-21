#!/usr/bin/env python3
"""Probe #1 — advantage-conditioning velocity shift (the freeze mechanism, offline).

Open-loop velocity fidelity (velocity_fidelity.py) did NOT separate frozen vs good:
freeze is closed-loop, state-specific latching, invisible to a global teacher-forced
speed distribution. This probe tests the SPECIFIC mechanism offline: at the SAME state,
does conditioning on "Advantage: positive" (deploy) produce SLOWER motion than
"Advantage: negative"? A frozen policy should collapse to static under positive at
decision states; a good policy should be roughly prompt-invariant.

Per val query state, infer TWICE (positive, negative) → per-step arm speed of each chunk
+ per-state chunk-mean speed (paired). One ckpt per invocation (parallelize on GPUs).

  CUDA_VISIBLE_DEVICES=0 .venv/bin/python train_scripts/kai/eval/velocity_condition_shift.py \
      --config pi05_v4_awbc_plus_freshdagger \
      --ckpt checkpoints/pi05_v4_awbc_plus_freshdagger/pi05_v4_awbc_plus_freshdagger/49999 \
      --val data/Task_A/self_built/vis_v2_merged_val --tag frozen --n-frames 40
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from velocity_fidelity import load_val, speeds_of, w1, js, H  # noqa: E402

POS = "Flatten and fold the cloth. Advantage: positive"
NEG = "Flatten and fold the cloth. Advantage: negative"
STATIC = 0.02


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config"); ap.add_argument("--ckpt"); ap.add_argument("--val")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--n-frames", type=int, default=40)
    ap.add_argument("--out-dir", default="docs/training/future_plans/plans/data/velocity_fidelity")
    ap.add_argument("--aggregate", nargs="*")
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    if a.aggregate:
        print(f"\n{'tag':10s} {'med_pos':>8} {'med_neg':>8} {'pos/neg':>8} {'gt_med':>8} "
              f"{'pos/gt':>7} {'stat%_pos':>9} {'stat%_neg':>9} {'pos<neg_states%':>15} "
              f"{'pos→static%':>12}")
        summ = {}
        for tag in a.aggregate:
            d = np.load(out / f"cshift_{tag}.npz")
            m = json.loads((out / f"cshift_{tag}.json").read_text()); summ[tag] = m
            print(f"{tag:10s} {m['median_pos']:8.4f} {m['median_neg']:8.4f} "
                  f"{m['median_pos_over_neg']:8.3f} {m['gt_median']:8.4f} "
                  f"{m['median_pos_over_gt']:7.3f} {m['static_pct_pos']:9.1f} "
                  f"{m['static_pct_neg']:9.1f} {m['frac_states_pos_slower_pct']:15.1f} "
                  f"{m['pos_induced_static_pct']:12.1f}")
        (out / "cshift_summary.json").write_text(json.dumps(summ, indent=2))
        print(f"\nsummary → {out/'cshift_summary.json'}")
        return

    import jax
    from openpi.training import config as _config
    from openpi.policies import policy_config
    from openpi.shared import normalize as _normalize
    print(f"[{a.tag}] jax {jax.devices()}", flush=True)
    cfg = _config.get_config(a.config)
    norm_stats = _normalize.load(Path(a.ckpt).parent)
    policy = policy_config.create_trained_policy(cfg, a.ckpt, norm_stats=norm_stats)
    samples = load_val(a.val, a.n_frames)
    print(f"[{a.tag}] {len(samples)} val eps", flush=True)

    sp_pos, sp_neg, sp_gt = [], [], []          # per-step speeds (distribution)
    cm_pos, cm_neg = [], []                      # per-state chunk-mean speed (paired)
    n = 0
    for s in samples:
        for k in s["q"]:
            if k + 1 + H > s["L"]:
                continue
            imgs = {c: s["images"][c][int(k)] for c in s["images"]}
            st = s["state"][int(k)]
            pp = np.asarray(policy.infer({"images": imgs, "state": st, "prompt": POS})["actions"])
            pn = np.asarray(policy.infer({"images": imgs, "state": st, "prompt": NEG})["actions"])
            gt = s["action"][k + 1:k + 1 + H]
            hh = min(pp.shape[0], pn.shape[0], len(gt))
            vp, vn, vg = speeds_of(pp[:hh]), speeds_of(pn[:hh]), speeds_of(gt[:hh])
            sp_pos.append(vp); sp_neg.append(vn); sp_gt.append(vg)
            cm_pos.append(vp.mean()); cm_neg.append(vn.mean())
            n += 1
        print(f"[{a.tag}] ep{s['ep']} states={n}", flush=True)

    pos = np.concatenate(sp_pos); neg = np.concatenate(sp_neg); gt = np.concatenate(sp_gt)
    cmp_, cmn = np.array(cm_pos), np.array(cm_neg)
    lo, hi = 0.0, float(np.percentile(np.concatenate([pos, neg, gt]), 99.5))
    m = {
        "tag": a.tag, "config": a.config, "ckpt": a.ckpt, "n_states": int(n),
        "median_pos": float(np.median(pos)), "median_neg": float(np.median(neg)),
        "gt_median": float(np.median(gt)),
        "median_pos_over_neg": float(np.median(pos) / np.median(neg)),
        "median_pos_over_gt": float(np.median(pos) / np.median(gt)),
        "median_neg_over_gt": float(np.median(neg) / np.median(gt)),
        "static_pct_pos": float(100 * np.mean(pos < STATIC)),
        "static_pct_neg": float(100 * np.mean(neg < STATIC)),
        # paired per-state: does positive slow the arm vs negative at the SAME state?
        "frac_states_pos_slower_pct": float(100 * np.mean(cmp_ < cmn)),
        "mean_rel_slowdown_pos_vs_neg": float(np.mean((cmn - cmp_) / np.clip(cmn, 1e-6, None))),
        # positive-induced static: chunk static under pos but NOT under neg (freeze fingerprint)
        "pos_induced_static_pct": float(100 * np.mean((cmp_ < STATIC) & (cmn >= STATIC))),
        "W1_pos_vs_neg": w1(pos, neg), "JS_pos_vs_neg": js(pos, neg, lo, hi),
    }
    np.savez(out / f"cshift_{a.tag}.npz", pos=pos, neg=neg, gt=gt, cm_pos=cmp_, cm_neg=cmn)
    (out / f"cshift_{a.tag}.json").write_text(json.dumps(m, indent=2))
    print(f"[{a.tag}] DONE pos/neg={m['median_pos_over_neg']:.3f} "
          f"pos_slower_states%={m['frac_states_pos_slower_pct']:.1f} "
          f"pos_induced_static%={m['pos_induced_static_pct']:.1f}", flush=True)


if __name__ == "__main__":
    main()
