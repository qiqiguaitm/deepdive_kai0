"""Pack a trained checkpoint into a self-contained inference bundle.

Output layout (Type A flat per `kai0/checkpoints/README.md`):

  <out_dir>/
  ├── _CHECKPOINT_METADATA      (copied from <ckpt_step>/)
  ├── assets/<asset_id>/
  │   └── norm_stats.json       (copied from --norm_stats)
  ├── params/                   (copied from <ckpt_step>/params/)
  └── train_config.json         {"base_config_name": ..., "override_asset_id": ...}

The JSON sidecar is consumed by sim01's `start_autonomy_from_ckpt.sh` (which sets
OPENPI_EXTRA_CONFIG=<json>) so the bundle runs without editing
src/openpi/training/config.py per-experiment. Pre-req: sim01 must already have
the base_config_name entry in its config.py.

Usage:
  python pack_inference_ckpt.py \\
    --config_name pi05_pick_place_box_kai0_unfreeze_20k_v2 \\
    --ckpt_step  /home/tim/workspace/.../19999 \\
    --norm_stats /home/tim/.../v2_aligned_train/norm_stats.json \\
    --asset_id   task_p_v2_aligned \\
    --out_dir    /tmp/ckpt_pack/task_p_v2_aligned_step19999
"""

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config_name", required=True)
    p.add_argument("--ckpt_step", required=True, type=Path)
    p.add_argument("--norm_stats", required=True, type=Path)
    p.add_argument("--asset_id", required=True)
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--with_train_state", action="store_true")
    args = p.parse_args()

    src = args.ckpt_step
    if not (src / "params").is_dir():
        raise SystemExit(f"missing {src}/params")
    if not (src / "_CHECKPOINT_METADATA").is_file():
        raise SystemExit(f"missing {src}/_CHECKPOINT_METADATA")
    if not args.norm_stats.is_file():
        raise SystemExit(f"missing norm_stats: {args.norm_stats}")

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    shutil.copy(src / "_CHECKPOINT_METADATA", out / "_CHECKPOINT_METADATA")
    print(f"[pack] _CHECKPOINT_METADATA → {out}/_CHECKPOINT_METADATA")

    if (out / "params").exists():
        shutil.rmtree(out / "params")
    shutil.copytree(src / "params", out / "params")
    sz_gb = sum(f.stat().st_size for f in (out / "params").rglob("*") if f.is_file()) / 1024**3
    print(f"[pack] params/ → {out}/params/  ({sz_gb:.1f} GB)")

    asset_dir = out / "assets" / args.asset_id
    asset_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.norm_stats, asset_dir / "norm_stats.json")
    print(f"[pack] norm_stats.json → {asset_dir}/norm_stats.json")

    if args.with_train_state and (src / "train_state").is_dir():
        if (out / "train_state").exists():
            shutil.rmtree(out / "train_state")
        shutil.copytree(src / "train_state", out / "train_state")
        print(f"[pack] train_state/ → {out}/train_state/")

    sidecar = {"base_config_name": args.config_name, "override_asset_id": args.asset_id}
    (out / "train_config.json").write_text(json.dumps(sidecar, indent=2) + "\n")
    print(f"[pack] train_config.json → {out}/train_config.json")

    print(f"\n[pack] DONE: {out}")
    print(f"[pack] tar:    cd {out.parent} && tar -cf {out.name}.tar {out.name}/")
    print(f"[pack] launch: ./start_scripts/kai/start_autonomy_from_ckpt.sh {out}")


if __name__ == "__main__":
    main()
