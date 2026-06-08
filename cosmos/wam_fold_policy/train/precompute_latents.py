#!/usr/bin/env python
"""Offline WAN-VAE latent precompute for wam_fold cross-rig training.

The online VAE encode costs ~2.35 s/step (~34% of a 6.9 s step, measured via
timer/encoding). It is deterministic per clip (seed-independent: video decode +
spatial pad are fixed per window index), so we encode every clip ONCE here and
cache the RAW latent (encode output of the padded video, BEFORE
_remove_padding_from_latent — train-time crops it via image_size, same as online).

Parity is preserved by reusing the EXACT training components:
  * WamFoldLeRobotDataset  (same clip selection, same cache_key)
  * ActionDataPacker.sft_process_sample → ActionTransformPipeline (same resize/pad)
  * Wan2pt2VAEInterface via the same tokenizer config (same VAE weights/dtype)
Only the 1-line normalize (`stack([video]).to(dev,dtype)/127.5 - 1`) is replicated,
matching OmniMoTModel._normalize_video_databatch_inplace exactly.

Shardable for parallel runs (e.g. b1/b2): --shard i --nshards N processes window
indices i, i+N, i+2N, ...  Resumable: existing <cache_key>.pt files are skipped.

Usage:
  WAN_VAE_PATH=... python precompute_latents.py --rig visrobot01 \
      --cache-dir <dir> --shard 0 --nshards 2 [--limit 20] [--verify]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

# The experiment module wires WamFoldLeRobotDataset + ActionDataPacker identically
# to training; import them so preprocessing is guaranteed to match.
from cosmos_framework.configs.base.experiment.action.posttrain_config.wam_fold_nano import (
    ActionDataPacker,
)
from cosmos_framework.data.vfm.action.datasets.wam_fold_dataset import WamFoldLeRobotDataset
from cosmos_framework.configs.base.defaults.tokenizer import Wan2pt2VAEConfig
from cosmos_framework.utils.lazy_config import instantiate as lazy_instantiate


def build_vae(device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
    """Instantiate the SAME VAE the model uses.

    Start from Wan2pt2VAEConfig (carries _target_) and overlay the model's exact
    tokenizer params (chunk_duration / encode_chunk_frames / encode_exact_durations /
    use_streaming_encode) so the encode bucketing is bit-identical to training. The
    underlying WanVAE loads straight to its default CUDA device, so no .to() is needed
    (the interface isn't an nn.Module). reset_dtype() is a documented no-op.
    """
    cfg = {**Wan2pt2VAEConfig}
    try:
        from cosmos_framework.configs.base.experiment.sft.models.nano_model_config import (
            NANO_MODEL_CONFIG,
        )
        for k, v in NANO_MODEL_CONFIG["tokenizer"].items():
            if k != "_target_":
                cfg[k] = v  # exact training tokenizer params (incl. encode_chunk_frames etc.)
    except Exception as e:  # pragma: no cover
        print(f"[precompute] WARN could not merge NANO tokenizer params ({e}); using defaults", flush=True)
    vae_path = os.environ.get("WAN_VAE_PATH")
    if vae_path:
        cfg["vae_path"] = vae_path  # mirror toml vae_path="${oc.env:WAN_VAE_PATH}"
    cfg.pop("bucket_name", None)  # local-path mode (vae_path is absolute)
    vae = lazy_instantiate(cfg)
    if hasattr(vae, "reset_dtype"):
        vae.reset_dtype()  # OmniMoTModel does this at init (omni_mot_model.py:156); no-op here
    return vae


@torch.no_grad()
def encode_one(vae, video_uint8: torch.Tensor, device: str, dtype: torch.dtype) -> torch.Tensor:
    """Replicate OmniMoTModel: stack→[1,C,T,H,W], /127.5-1, encode, .contiguous().float()."""
    state = torch.stack([video_uint8]).to(device=device, dtype=dtype) / 127.5 - 1.0
    return vae.encode(state).contiguous().float().cpu()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rig", required=True, choices=["visrobot01", "kairobot01"])
    ap.add_argument("--split", default=None, help="visrobot01 → 'train'; kairobot01 → None")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--chunk-length", type=int, default=16)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--resolution", default="480")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--limit", type=int, default=0, help="0 = all; >0 = first N (small validation)")
    ap.add_argument("--verify", action="store_true",
                    help="re-encode an existing cached key and assert allclose (parity self-check)")
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    os.makedirs(args.cache_dir, exist_ok=True)

    split = args.split
    if split is None and args.rig == "visrobot01":
        split = "train"

    ds = WamFoldLeRobotDataset(rig=args.rig, split=split, mode="policy",
                               chunk_length=args.chunk_length, fps=args.fps)
    packer = ActionDataPacker(tokenizer_config=None, resolution=args.resolution,
                              latent_cache_dir=None)  # None → pipeline only, no latent load
    vae = build_vae("cuda", dtype)

    n = len(ds)
    idxs = list(range(args.shard, n, args.nshards))
    if args.limit > 0:
        idxs = idxs[: args.limit]
    print(f"[precompute] rig={args.rig} split={split} total_windows={n} "
          f"shard={args.shard}/{args.nshards} → {len(idxs)} windows  dtype={args.dtype} "
          f"cache={args.cache_dir}", flush=True)

    done = skipped = 0
    t0 = time.time()
    for i, idx in enumerate(idxs):
        item = ds[idx]
        item = packer.sft_process_sample(item)
        key = item["cache_key"]
        out = os.path.join(args.cache_dir, key + ".pt")
        video = item["video"]  # uint8 [C,T,H,W] (padded by the pipeline)

        if args.verify and os.path.exists(out):
            cached = torch.load(out, map_location="cpu")
            fresh = encode_one(vae, video, "cuda", dtype)
            ok = (cached.shape == fresh.shape) and torch.allclose(cached, fresh, atol=1e-3, rtol=1e-3)
            md = (cached - fresh).abs().max().item() if cached.shape == fresh.shape else float("nan")
            print(f"[verify] {key} shape={tuple(fresh.shape)} allclose={ok} max|Δ|={md:.2e}", flush=True)
            if not ok:
                print("[verify] PARITY FAIL", flush=True); return 2
            continue

        if os.path.exists(out):
            skipped += 1
            continue
        latent = encode_one(vae, video, "cuda", dtype)
        tmp = out + ".tmp"
        torch.save(latent, tmp)
        os.replace(tmp, out)  # atomic → safe under parallel shards / resume
        done += 1
        if i < 3 or i % 200 == 0:
            r = (time.time() - t0) / max(done, 1)
            print(f"[precompute] {i+1}/{len(idxs)} {key} latent={tuple(latent.shape)} "
                  f"{r:.3f}s/clip done={done} skip={skipped}", flush=True)

    print(f"[precompute] DONE shard={args.shard}/{args.nshards} encoded={done} skipped={skipped} "
          f"elapsed={time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
