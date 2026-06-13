#!/usr/bin/env python3
"""Forward-dynamics (action-conditioned video generation) inference + Δaction perturbation test.

Phase-1 early-falsification gate (plan §3): given a GT first frame + a 14-D action chunk,
generate the future video in FORWARD_DYNAMICS mode and compare to the GT video. Then run the
**Δaction perturbation test** — the same first frame conditioned on three action variants:
  (a) GT action      — should best reconstruct the GT future
  (b) other-episode  — a different episode's action chunk (should diverge, different motion)
  (c) zero action    — null delta (should produce a near-static clip)
If (a) is NOT measurably better than (b)/(c) [ΔPSNR ≈ 0], the action pathway is being ignored —
a bug or an under-trained model. On the 300-step smoke checkpoint this primarily validates that
the FD pipeline runs end-to-end and produces non-noise video; quantitative action-following is
only meaningful after the 10k-step run.

Reuses eval_report.py building blocks (model load, quantile norm, concat_view). The action goes
through the SAME transform the training dataset applies: arm-joint DELTA vs the window's anchor
state + absolute grippers (_DELTA_MASK), then quantile-normalize with visrobot01.json (delta
stats), pad to max_action_dim. A built-in self-check compares this against what
WamFoldLeRobotDataset actually emits, so the eval-time action distribution provably matches train.

Usage:
  python fd_infer.py --export-dir <hf_dir> --n-episodes 3 --guidance 3.0 --num-steps 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

WAM_POLICY_EVAL = "/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/wam_fold_policy/eval"
DATA_ROOT = "/mnt/pfs/p46h4f/cosmos/deepdive_kai0/kai0/data/wam_fold_v1"
STATS = "/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/wam_fold_policy/data/stats/visrobot01.json"
TASK_TEXT = "Flatten and fold the cloth."
DOMAIN_NAME = "wam_fold"
# Mirror wam_fold_dataset.py: arm joints (idx 0-5,7-12) are delta vs anchor; grippers (6,13) absolute.
DELTA_MASK = np.array([True] * 6 + [False] + [True] * 6 + [False], dtype=bool)

sys.path.insert(0, WAM_POLICY_EVAL)


# ---------------------------------------------------------------- quantile norm (delta stats)
def _load_qstats(path):
    a = json.load(open(path))["global"]["action"]
    return np.asarray(a["q01"], np.float32), np.asarray(a["q99"], np.float32)


def _norm_quantile(raw, q01, q99):
    """norm = clamp(2*(a-q01)/(q99-q01)-1, -1, 1). Mirrors action_normalization.py:39."""
    denom = np.maximum(q99 - q01, 1e-8)
    return np.clip(2.0 * (raw - q01) / denom - 1.0, -1.0, 1.0)


# ---------------------------------------------------------------- metrics (mirror eval_i2v)
def video_metrics(pred_thwc_u8, gt_thwc_u8):
    """PSNR / SSIM over matched frames (uint8 [T,H,W,3]). Returns dict of floats."""
    import torch
    import torch.nn.functional as F

    T = min(pred_thwc_u8.shape[0], gt_thwc_u8.shape[0])
    p = torch.from_numpy(pred_thwc_u8[:T].astype(np.float32))
    g = torch.from_numpy(gt_thwc_u8[:T].astype(np.float32))
    # The model resizes/reflection-pads the 720x640 concat canvas to its 480-res target
    # (e.g. 560 wide), so pred and gt can differ in H/W. Resize pred -> gt grid before metrics.
    if p.shape[1:3] != g.shape[1:3]:
        p = F.interpolate(p.permute(0, 3, 1, 2), size=tuple(g.shape[1:3]),
                          mode="bilinear", align_corners=False).permute(0, 2, 3, 1).contiguous()
    mse = ((p - g) ** 2).mean(dim=(1, 2, 3)).clamp_min(1e-8)
    psnr = (10.0 * torch.log10((255.0**2) / mse)).mean().item()

    # SSIM via 11x11 Gaussian window on luma
    def luma(x):  # [T,H,W,3] -> [T,1,H,W]
        return (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]).unsqueeze(1)

    pl, gl = luma(p), luma(g)
    win = torch.ones(1, 1, 7, 7) / 49.0
    mu_p = F.conv2d(pl, win, padding=3)
    mu_g = F.conv2d(gl, win, padding=3)
    sp = F.conv2d(pl * pl, win, padding=3) - mu_p**2
    sg = F.conv2d(gl * gl, win, padding=3) - mu_g**2
    spg = F.conv2d(pl * gl, win, padding=3) - mu_p * mu_g
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ssim = (((2 * mu_p * mu_g + c1) * (2 * spg + c2)) / ((mu_p**2 + mu_g**2 + c1) * (sp + sg + c2))).mean().item()
    return {"psnr": round(psnr, 3), "ssim": round(ssim, 4)}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-dir", required=True)
    ap.add_argument("--out-dir", default="/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/wam_fold_wm_runs/reports/fd_eval")
    ap.add_argument("--n-episodes", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--resolution", default="480")
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--num-steps", type=int, default=8)
    ap.add_argument("--shift", type=float, default=5.0)
    ap.add_argument("--sampler", default="unipc")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-videos", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    import torch

    # ---- build val dataset (forward_dynamics; already applies arm-delta transform) ----
    sys.path.insert(0, "/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/packages/cosmos3")
    from cosmos_framework.data.vfm.action.datasets.wam_fold_dataset import WamFoldLeRobotDataset

    ds = WamFoldLeRobotDataset(rig="visrobot01", split="val", mode="forward_dynamics",
                               chunk_length=args.chunk, fps=float(args.fps))
    print(f"[ds] val windows: {len(ds)}", flush=True)

    q01, q99 = _load_qstats(STATS)

    # ---- self-check: our delta+norm vs what the dataset/packer emits (consistency anchor) ----
    s0 = ds[0]
    act_ds = s0["action"].numpy()  # [chunk,14] — dataset already applied arm-delta (raw units)
    act_norm_ours = _norm_quantile(act_ds, q01, q99)
    print(f"[self-check] dataset action[0,:3]={act_ds[0,:3].round(3)} "
          f"-> norm{act_norm_ours[0,:3].round(3)} (in[-1,1]: "
          f"{bool((np.abs(act_norm_ours)<=1.0+1e-5).all())})", flush=True)

    # ---- load model (reuse eval_report.CosmosFoldPolicy loader) ----
    from eval_report import CosmosFoldPolicy, _build_concat_view  # noqa: E402

    class _A:  # minimal args shim for CosmosFoldPolicy.__init__
        export_dir = args.export_dir
        out_dir = args.out_dir
        sampler = args.sampler
        stats_path = STATS
        guidance = args.guidance
        num_steps = args.num_steps
        shift = args.shift
        model_chunk = args.chunk
        fps_cond = args.fps
        resolution = args.resolution

    pol = CosmosFoldPolicy(_A(), device="cuda")
    model = pol.model
    ModelMode = pol._ModelMode
    build_batch = pol._build_action_batch
    max_ad = pol.max_action_dim

    def gen_fd(first_frame_chw01, action_norm_chunk):
        """first_frame [C,H,W] float[0,1]; action_norm [chunk,14] in[-1,1]. -> pred video [T,H,W,3] u8."""
        video = pol._video_from_frame(first_frame_chw01, args.chunk)  # [C,chunk+1,H,W] u8
        act = torch.zeros(args.chunk, max_ad, dtype=torch.float32)
        act[:, :14] = torch.from_numpy(action_norm_chunk).float()
        batch = build_batch(
            video=video, action=act, raw_action_dim=14, prompt=TASK_TEXT,
            view_point="concat_view", domain_name=DOMAIN_NAME,
            model_mode=ModelMode.FORWARD_DYNAMICS, action_chunk_size=args.chunk,
            fps=args.fps, resolution=str(args.resolution),
            input_video_key=model.input_video_key, batch_size=1, device="cuda",
        )
        pol._seed += 1
        with torch.inference_mode():
            samples = model.generate_samples_from_batch(
                batch, guidance=args.guidance, seed=[args.seed],
                num_steps=args.num_steps, shift=args.shift)
        return pol._decode_video(samples["vision"][0])  # [T,H,W,3] u8

    # ---- per-episode: GT vs Δaction perturbations ----
    rng = np.random.default_rng(args.seed)
    results = []
    n = min(args.n_episodes, len(ds))
    for i in range(n):
        s = ds[i]
        video_chw = s["video"].float() / 255.0 if s["video"].dtype == torch.uint8 else s["video"].float()
        # dataset video is [C,chunk+1,H,W] concat_view already; frame0 = conditioning, rest = GT future
        first = video_chw[:, 0]  # [C,H,W]
        gt_future = (video_chw.clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 3, 0).numpy()  # [T,H,W,3]

        act_gt = _norm_quantile(s["action"].numpy(), q01, q99)
        # other-episode action (different motion)
        j = int(rng.integers(0, len(ds)))
        while j == i:
            j = int(rng.integers(0, len(ds)))
        act_other = _norm_quantile(ds[j]["action"].numpy(), q01, q99)
        act_zero = np.zeros_like(act_gt)  # null delta

        m = {}
        for tag, a in [("gt", act_gt), ("other", act_other), ("zero", act_zero)]:
            pred = gen_fd(first, a)
            m[tag] = video_metrics(pred, gt_future)
            if args.save_videos:
                np.save(os.path.join(args.out_dir, f"ep{i}_{tag}_pred.npy"), pred)
        dpsnr_other = round(m["gt"]["psnr"] - m["other"]["psnr"], 3)
        dpsnr_zero = round(m["gt"]["psnr"] - m["zero"]["psnr"], 3)
        row = {"episode": i, "gt": m["gt"], "other": m["other"], "zero": m["zero"],
               "dPSNR_gt_minus_other": dpsnr_other, "dPSNR_gt_minus_zero": dpsnr_zero}
        results.append(row)
        print(f"[ep{i}] GT psnr={m['gt']['psnr']} | other={m['other']['psnr']} | zero={m['zero']['psnr']} "
              f"| ΔPSNR(gt-other)={dpsnr_other} ΔPSNR(gt-zero)={dpsnr_zero}", flush=True)

    # ---- aggregate ----
    if results:
        agg = {
            "n": len(results),
            "mean_gt_psnr": round(np.mean([r["gt"]["psnr"] for r in results]), 3),
            "mean_dPSNR_gt_minus_other": round(np.mean([r["dPSNR_gt_minus_other"] for r in results]), 3),
            "mean_dPSNR_gt_minus_zero": round(np.mean([r["dPSNR_gt_minus_zero"] for r in results]), 3),
        }
        verdict = ("ACTION-FOLLOWING DETECTED" if agg["mean_dPSNR_gt_minus_other"] > 1.0
                   else "WEAK/NO action-following (ΔPSNR≈0) — expected on 300-step smoke; "
                        "re-run after 10k-step training")
        out = {"args": vars(args), "aggregate": agg, "verdict": verdict, "per_episode": results}
        json.dump(out, open(os.path.join(args.out_dir, "fd_daction_report.json"), "w"), indent=2)
        print(f"\n=== AGG {json.dumps(agg)} ===\nVERDICT: {verdict}", flush=True)


if __name__ == "__main__":
    main()
