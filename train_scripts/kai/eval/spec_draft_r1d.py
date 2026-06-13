#!/usr/bin/env python3
"""R1-d: scaled draft distillation with disk-sharded cache + REAL acceptance eval.

Fixes the three gaps left by R1-c (``spec_draft_distill.py``):
  1. teacher = ZERO-noise full denoise (FLASH convention), not fixed random noise.
  2. disk-sharded safetensors cache (CPU RAM does not scale past a few hundred frames;
     here we stream one shard at a time during training).
  3. eval measures the REAL verify-from-draft ``accepted_prefix_len`` through the full
     ``SpeculativeSampler.sample_from_prefix`` (draft -> K-way denoise verify -> radius
     -> gripper gate -> fallback), NOT the draft-vs-teacher proxy shortcut R1-c used.

All ADDITIVE: loads the frozen pi05 model once, never mutates it; trains only the new
DraftChunkHead; writes the cache + draft to their own files.

Phases (A is skipped when --reuse-cache and a complete manifest already exists):
  A. CACHE  -> sharded safetensors on disk + manifest.json (zero-noise teacher target)
  B. TRAIN  -> stream shards, regress DraftChunkHead, warm-started from VLM layer 0
  C. EVAL   -> real accepted_prefix_len distribution on the episode-disjoint holdout

Run (GPU, patched venv):
  CUDA_VISIBLE_DEVICES=3 kai0/.venv_5090/bin/python \
    train_scripts/kai/eval/spec_draft_r1d.py \
    --config pi05_pytorch_a_new_pure_200 \
    --ckpt /data1/DATA_IMP/checkpoints/ckpt_others/pytorch_pure200_step50000 \
    --asset-id a_new_pure_200 \
    --val /data1/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/A_new_pure_200_val \
    --cache-dir /data1/tmp/spec_cache_r1d_pure200 \
    --train-eps 16 --holdout-eps 4 --frames-per-ep 100 --holdout-frames-per-ep 60 \
    --epochs 300 --out /tmp/draft_r1d_pure200.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors.torch import load_file as _load_safetensors
from safetensors.torch import save_file as _save_safetensors
import torch


def read_video(path: Path, n: int) -> np.ndarray:
    import av

    c = av.open(str(path))
    s = c.streams.video[0]
    s.thread_type = "AUTO"
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


# --------------------------------------------------------------------------- cache


def _build_cache(args, policy, sampler, eps_split):
    """Phase A: encode sampled frames -> zero-noise teacher -> sharded safetensors."""
    import jax

    from openpi.models import model as _model

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    device = sampler_device(sampler)
    cams = ("top_head", "hand_left", "hand_right")
    val = Path(args.val).resolve()
    ah, ad = sampler.action_horizon, sampler.action_dim

    def build_obs(k, vid, state):
        imgs = {c: vid[c][k] for c in cams}
        obs = {"images": imgs, "state": state[k], "prompt": args.prompt}
        inputs = policy._input_transform(obs)  # noqa: SLF001
        inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(device)[None, ...], inputs)
        return _model.Observation.from_dict(inputs)

    manifest = {"config": args.config, "ckpt": str(Path(args.ckpt).resolve()), "teacher_noise_mode": "zero",
                "chunk_m": ah, "out_dim": ad, "shard_size": args.shard_size, "splits": {}}
    zero_noise = torch.zeros((1, ah, ad), device=device, dtype=torch.float32)

    with torch.no_grad():
        for split, eps, fpe in eps_split:
            buf, shard_id, n_total = [], 0, 0
            shard_paths = []

            def _flush(buf, shard_id, split=split, shard_paths=shard_paths):
                if not buf:
                    return shard_id
                stacked = {k: torch.stack([b[k] for b in buf], 0).contiguous() for k in buf[0]}
                fname = f"{split}_shard{shard_id:04d}.safetensors"
                _save_safetensors(stacked, str(cache_dir / fname))
                shard_paths.append({"path": fname, "num_samples": len(buf)})
                return shard_id + 1

            for ix, ep in enumerate(eps):
                ei, n_frames = ep["episode_index"], ep["length"]
                tbl = __import__("pyarrow.parquet", fromlist=["x"]).read_table(
                    val / "data" / "chunk-000" / f"episode_{ei:06d}.parquet"
                ).to_pandas()
                state = np.stack([np.asarray(x) for x in tbl["observation.state"]]).astype(np.float32)
                vid = {c: read_video(
                    val / "videos" / "chunk-000" / f"observation.images.{c}" / f"episode_{ei:06d}.mp4", n_frames)
                    for c in cams}
                ks = np.linspace(0, n_frames - 1, fpe).astype(int)
                for k in ks:
                    obs = build_obs(int(k), vid, state)
                    pe, ppad, patt, st = sampler._embed_prefix(obs)  # noqa: SLF001
                    pkv = sampler._prefill_kv(pe, ppad, patt)  # noqa: SLF001
                    teacher = sampler._full_denoise(st, ppad, pkv, zero_noise)  # noqa: SLF001
                    buf.append({
                        "prefix_embs": pe.squeeze(0).to(torch.float16).cpu(),
                        "prefix_pad": ppad.squeeze(0).bool().cpu(),
                        "prefix_att": patt.squeeze(0).bool().cpu(),
                        "robot_state": st.squeeze(0).to(torch.float16).cpu(),
                        "target": teacher.squeeze(0).to(torch.float16).cpu(),
                        "episode_index": torch.tensor(int(ei), dtype=torch.int64),
                        "frame_index": torch.tensor(int(k), dtype=torch.int64),
                    })
                    n_total += 1
                    if len(buf) >= args.shard_size:
                        shard_id = _flush(buf, shard_id)
                        buf = []
                print(f"  [{split}] cached ep{ei} ({ix + 1}/{len(eps)})  n={n_total}", flush=True)
            shard_id = _flush(buf, shard_id)
            manifest["splits"][split] = {"shards": shard_paths, "num_samples": n_total}

    # feature (prefix hidden) dim from any written shard
    first_shard = next((sh["path"] for sp in manifest["splits"].values() for sh in sp["shards"]), None)
    manifest["feature_dim"] = _peek_feat(cache_dir, first_shard) if first_shard else 0
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[cache] wrote manifest -> {cache_dir / 'manifest.json'}")
    return manifest


def _peek_feat(cache_dir, fname):
    t = _load_safetensors(str(Path(cache_dir) / fname))
    return int(t["prefix_embs"].shape[-1])


def sampler_device(sampler):
    return next(sampler.model.parameters()).device


def _iter_shards(cache_dir, split_manifest):
    for sh in split_manifest["shards"]:
        yield _load_safetensors(str(Path(cache_dir) / sh["path"]))


# --------------------------------------------------------------------------- train


def _train_draft(args, sampler, manifest):
    """Phase B: stream train shards, regress DraftChunkHead (best-not-last)."""
    cache_dir = Path(args.cache_dir)
    model = sampler.model
    device = sampler_device(sampler)
    ah, ad = sampler.action_horizon, sampler.action_dim
    img_dim = int(manifest["feature_dim"])

    from openpi.models_pytorch.draft import DraftChunkHead

    vlm_lm = model.paligemma_with_expert.paligemma.language_model
    draft = DraftChunkHead(
        img_dim=img_dim, chunk_m=ah, out_dim=ad, use_state_token=False, gemma_config=vlm_lm.config,
    ).to(device=device, dtype=torch.float32)
    draft.init_from_vlm_layer(vlm_lm.layers[0])
    draft.train()
    opt = torch.optim.Adam(draft.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    step_w = torch.tensor([0.97 ** i for i in range(ah)], device=device).view(1, ah, 1)

    train_man = manifest["splits"]["train"]
    n_train = int(train_man["num_samples"])
    best_loss, best_state = float("inf"), None
    rng = np.random.default_rng(0)
    for ep_i in range(args.epochs):
        shards = list(train_man["shards"])
        rng.shuffle(shards)
        tot, seen = 0.0, 0
        for sh in shards:
            t = _load_safetensors(str(cache_dir / sh["path"]))
            pe_all = t["prefix_embs"].to(torch.float32)
            ppad_all, patt_all = t["prefix_pad"], t["prefix_att"]
            tgt_all = t["target"].to(torch.float32)
            n = pe_all.shape[0]
            order = rng.permutation(n)
            for b0 in range(0, n, args.batch):
                bi = order[b0 : b0 + args.batch]
                pe = pe_all[bi].to(device)
                ppad = ppad_all[bi].to(device)
                patt = patt_all[bi].to(device)
                tgt = tgt_all[bi].to(device)
                pred = draft(prefix_embs=pe, prefix_pad_masks=ppad, prefix_att_masks=patt)
                loss = (torch.nn.functional.huber_loss(pred, tgt, reduction="none", delta=0.1) * step_w).mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(draft.parameters(), max_norm=1.0)
                opt.step()
                tot += loss.item() * len(bi)
                seen += len(bi)
        sched.step()
        epoch_loss = tot / max(1, seen)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = {k: v.detach().clone() for k, v in draft.state_dict().items()}
        if ep_i % 25 == 0 or ep_i == args.epochs - 1:
            print(f"  epoch {ep_i:4d}  train_huber={epoch_loss:.5f}  (best={best_loss:.5f})  n={n_train}", flush=True)

    if best_state is not None:
        draft.load_state_dict(best_state)
        print(f"[best] restored draft @ train_huber={best_loss:.5f}")
    draft.eval()
    torch.save({"state_dict": draft.state_dict(), "img_dim": img_dim, "chunk_m": ah, "out_dim": ad,
                "config": args.config, "ckpt": str(Path(args.ckpt).resolve()), "best_train_huber": best_loss},
               args.out)
    print(f"[save] draft -> {args.out}")
    return draft, best_loss


# --------------------------------------------------------------------------- eval


def _eval_real_acceptance(args, sampler, draft, manifest):
    """Phase C: real verify-from-draft accepted_prefix_len on holdout shards."""
    cache_dir = Path(args.cache_dir)
    device = sampler_device(sampler)
    mdtype = next(sampler.model.parameters()).dtype
    ah = sampler.action_horizon
    sampler.draft = draft.to(mdtype)

    torch.manual_seed(0)  # reproducible verify noise
    hold_man = manifest["splits"]["holdout"]
    accs, dists, falls, gstops = [], [], 0, 0
    with torch.no_grad():
        for t in _iter_shards(cache_dir, hold_man):
            n = t["prefix_embs"].shape[0]
            for i in range(n):
                pe = t["prefix_embs"][i : i + 1].to(device, mdtype)
                ppad = t["prefix_pad"][i : i + 1].to(device)
                patt = t["prefix_att"][i : i + 1].to(device)
                st = t["robot_state"][i : i + 1].to(device, mdtype)
                noise = None
                if args.verify_noise == "zero":
                    noise = torch.zeros((1, ah, sampler.action_dim), device=device, dtype=torch.float32)
                out = sampler.sample_from_prefix(pe, ppad, patt, st, noise=noise, last_gripper=None)
                accs.append(int(out["accepted_prefix_len"].item()))
                dists.append(float(out["radius_dist"].min(dim=1).values.mean().item()))
                falls += int(bool(out["used_full_fallback"]))
                gstops += int(bool(out["gripper_verify_stop"].any().item()))
    accs = np.asarray(accs)
    print("\n========== R1-d REAL-ACCEPTANCE RESULT (holdout, verify-from-draft) ==========")
    print(f"  frames={len(accs)}  verify_noise={args.verify_noise}  tau={sampler.args.tau_radius}  "
          f"eval_h={sampler.args.max_exec_steps}/{ah}")
    print(f"  accepted_prefix_len: mean={accs.mean():.1f}  median={np.median(accs):.0f}  "
          f"p25={np.percentile(accs, 25):.0f}  p75={np.percentile(accs, 75):.0f}  max={accs.max()}")
    print(f"  zero-accept frames={int((accs <= 0).sum())}/{len(accs)}  full-fallback={falls}/{len(accs)}  "
          f"gripper-verify-stop={gstops}/{len(accs)}")
    print(f"  mean min-over-K radius (eval window)={np.mean(dists):.4f}")
    return accs


# --------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--asset-id", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out", default="/tmp/draft_r1d.pt")
    ap.add_argument("--prompt", default="Flatten and fold the cloth.")
    ap.add_argument("--train-eps", type=int, default=16)
    ap.add_argument("--holdout-eps", type=int, default=4)
    ap.add_argument("--frames-per-ep", type=int, default=100)
    ap.add_argument("--holdout-frames-per-ep", type=int, default=60)
    ap.add_argument("--shard-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--tau", type=float, default=0.3)
    ap.add_argument("--teacher-steps", type=int, default=10)
    ap.add_argument("--verify-noise", choices=["random", "zero"], default="random")
    ap.add_argument("--reuse-cache", action="store_true")
    args = ap.parse_args()

    from openpi.models_pytorch.spec_pi0_pytorch import SpecArgs
    from openpi.models_pytorch.spec_pi0_pytorch import SpeculativeSampler
    from openpi.policies import policy_config as pc
    from openpi.training import checkpoints as ck
    from openpi.training import config as tc

    ckpt = Path(args.ckpt).resolve()
    train_cfg = tc.get_config(args.config)
    norm_stats = ck.load_norm_stats(ckpt / "assets", args.asset_id)
    policy = pc.create_trained_policy(train_cfg, ckpt, norm_stats=norm_stats)
    model = policy._model  # noqa: SLF001
    ah, ad = int(model.config.action_horizon), int(model.config.action_dim)
    print(f"[model] pi05={getattr(model, 'pi05', '?')} H={ah} action_dim={ad} "
          f"dtype={next(model.parameters()).dtype}")

    spec_args = SpecArgs(chunk_m=ah, tau_radius=args.tau, max_exec_steps=ah, full_num_steps=args.teacher_steps)
    sampler = SpeculativeSampler(model, None, spec_args)

    eps = [json.loads(line) for line in (Path(args.val).resolve() / "meta" / "episodes.jsonl").read_text().splitlines()]
    hold_eps = eps[: args.holdout_eps]
    train_eps = eps[args.holdout_eps : args.holdout_eps + args.train_eps]
    print(f"[data] train_eps={len(train_eps)} holdout_eps={len(hold_eps)} "
          f"(episode-disjoint) frames/ep train={args.frames_per_ep} holdout={args.holdout_frames_per_ep}")

    manifest_path = Path(args.cache_dir) / "manifest.json"
    if args.reuse_cache and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        print(f"[cache] reusing existing cache at {args.cache_dir} "
              f"(train={manifest['splits']['train']['num_samples']} "
              f"holdout={manifest['splits']['holdout']['num_samples']})")
    else:
        manifest = _build_cache(args, policy, sampler, [
            ("holdout", hold_eps, args.holdout_frames_per_ep),
            ("train", train_eps, args.frames_per_ep),
        ])

    draft, _ = _train_draft(args, sampler, manifest)
    _eval_real_acceptance(args, sampler, draft, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
