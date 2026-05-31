#!/usr/bin/env python3
"""Faithful TAC prefix-conditioned eval (correct action-space handling).

TAC trains the model to denoise the chunk POSTFIX given a CLEAN prefix at per-token
time=0 (pi0.py compute_loss). This script feeds a clean GT prefix in the MODEL's
internal action space and measures postfix MAE vs GT, comparing it to the standard
no-prefix prediction. If TAC learned its objective, conditioning on a clean prefix
should cut the TAC model's postfix MAE far more than the baseline's.

Mechanics (all via the policy's own proven jitted path + transforms, so no manual
forward / normalization — avoids action-space mismatch):
  - prefix in model space: inject "actions"=raw_GT into pol._input_transform.
  - conditioned sample: pol._sample_actions(..., tac_prefix=<model-space GT>, tac_delay=d)
    (additive, default-off kwargs added to Pi0.sample_actions — non-TAC path unchanged).
  - compare in raw space via pol._output_transform.

Usage (from kai0/):
  .venv/bin/python ../train_scripts/kai/eval/eval_tac_faithful.py \
    --ckpt <dir-with-train_config.json> --val data/Task_A/self_built/A_new_pure_200_val \
    --n-ep 4 --n-frames 10 --delays 8,16
"""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

CAMS = ("top_head", "hand_left", "hand_right")


def read_video_frames(path: Path, n: int) -> np.ndarray:
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
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--prompt", default="Flatten and fold the cloth.")
    ap.add_argument("--n-ep", type=int, default=4)
    ap.add_argument("--n-frames", type=int, default=10)
    ap.add_argument("--delays", default="8,16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    delays = [int(x) for x in args.delays.split(",")]

    ckpt = Path(args.ckpt).resolve()
    spec = json.loads((ckpt / "train_config.json").read_text())
    base_config = spec["base_config_name"]
    os.environ["OPENPI_EXTRA_CONFIG"] = str(ckpt / "train_config.json")

    import jax, jax.numpy as jnp
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config
    from openpi.models import model as _model

    train_cfg = _config.get_config(base_config)
    ah = int(getattr(train_cfg.model, "action_horizon", 50))
    ad = int(getattr(train_cfg.model, "action_dim", 32))
    tac = bool(getattr(train_cfg.model, "tac_enabled", False))
    print(f"[load] {ckpt.name} base={base_config} tac={tac} ah={ah} ad={ad}")
    pol = _policy_config.create_trained_policy(train_cfg, str(ckpt))

    fixed_noise = jnp.asarray(np.random.default_rng(args.seed).standard_normal((1, ah, ad)).astype(np.float32))
    rng = jax.random.key(args.seed)

    def to_raw(state_b, model_out):
        outs = {"state": np.asarray(state_b), "actions": np.asarray(model_out)}
        return np.asarray(pol._output_transform(outs)["actions"])[:, :14]

    val = Path(args.val).resolve()
    eps = [json.loads(l) for l in (val / "meta" / "episodes.jsonl").read_text().splitlines()][: args.n_ep]

    res = {d: {"cond": [], "nopfx": []} for d in delays}
    for ep in eps:
        ei, L = ep["episode_index"], ep["length"]
        df = pq.read_table(val / "data" / "chunk-000" / f"episode_{ei:06d}.parquet").to_pandas()
        action = np.stack([np.asarray(x) for x in df["action"]])
        state = np.stack([np.asarray(x) for x in df["observation.state"]])
        cams = {c: read_video_frames(val / "videos" / "chunk-000" / f"observation.images.{c}" / f"episode_{ei:06d}.mp4", L) for c in CAMS}
        qs = np.linspace(0, L - ah - 1, args.n_frames).astype(int)
        tep = time.time()
        for k in qs:
            gt = action[k : k + ah]  # (ah,14) raw
            obs = {"images": {c: cams[c][k] for c in CAMS}, "state": state[k], "prompt": args.prompt}
            # observation (no actions) for sampling
            inpb = jax.tree.map(lambda x: jnp.asarray(x)[None], pol._input_transform(jax.tree.map(lambda x: x, obs)))
            obsv = _model.Observation.from_dict(inpb)
            state_b = inpb["state"][0]
            # model-space prefix via injecting raw GT actions into the input transform
            obsA = dict(obs); obsA["actions"] = gt.astype(np.float32)
            prefix_ms = jnp.asarray(pol._input_transform(jax.tree.map(lambda x: x, obsA))["actions"])[None]  # (1,ah,ad)
            # no-prefix standard chunk (fixed noise)
            nop_raw = to_raw(state_b, pol._sample_actions(rng, obsv, noise=fixed_noise)[0])
            for d in delays:
                cond_raw = to_raw(state_b, pol._sample_actions(rng, obsv, noise=fixed_noise, tac_prefix=prefix_ms, tac_delay=d)[0])
                res[d]["nopfx"].append(float(np.abs(nop_raw[d:ah] - gt[d:ah]).mean()))
                res[d]["cond"].append(float(np.abs(cond_raw[d:ah] - gt[d:ah]).mean()))
        print(f"  ep{ei:02d} frames={len(qs)} ({time.time()-tep:.0f}s)")

    summary = {"ckpt": str(ckpt), "base_config": base_config, "tac_enabled": tac, "n_ep": len(eps), "postfix_mae_raw": {}}
    for d in delays:
        nop = float(np.mean(res[d]["nopfx"])); cond = float(np.mean(res[d]["cond"]))
        summary["postfix_mae_raw"][str(d)] = {
            "no_prefix": nop, "clean_prefix_cond": cond,
            "improvement_pct": float((nop - cond) / nop * 100.0)}
    print("\n=== summary ==="); print(json.dumps(summary, indent=2))
    out = Path(args.out) if args.out else ckpt / "eval_tac_faithful.json"
    out.write_text(json.dumps(summary, indent=2)); print(f"saved -> {out}")


if __name__ == "__main__":
    main()
