#!/usr/bin/env python3
"""Aggregate eval_i2v per-clip jsonl results into the Phase-0 realism baseline.

Reads wam_fold_policy/eval_i2v/results/<Model>_<shard>.jsonl (fields: model, ep, cam,
horizon, anchor, gen_s, psnr, ssim, temporal_absdiff_ratio) and writes baseline.json
with mean/median PSNR+SSIM per (model, cam, horizon) plus pooled per-model rows.

These are ZERO-SHOT I2V numbers (no action conditioning) — the realism floor every
posttrained FD checkpoint must beat (plan §4 gate: val PSNR > I2V baseline + 2 dB).
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

RESULTS = Path("/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/wam_fold_policy/eval_i2v/results")
OUT = Path("/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/wam_fold_wm_runs/reports/baseline.json")


def main() -> None:
    rows = []
    for f in sorted(RESULTS.glob("*.jsonl")):
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        raise SystemExit(f"no jsonl rows under {RESULTS}")

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if "psnr" not in r:
            continue
        groups[(r["model"], r["cam"], r["horizon"])].append(r)

    def agg(items: list[dict]) -> dict:
        psnr = [x["psnr"] for x in items]
        ssim = [x["ssim"] for x in items]
        return {
            "n": len(items),
            "psnr_mean": round(statistics.mean(psnr), 3),
            "psnr_median": round(statistics.median(psnr), 3),
            "ssim_mean": round(statistics.mean(ssim), 4),
            "ssim_median": round(statistics.median(ssim), 4),
        }

    detail = {f"{m}|{c}|{h}": agg(v) for (m, c, h), v in sorted(groups.items())}

    pooled: dict[str, list[dict]] = defaultdict(list)
    for (m, _c, h), v in groups.items():
        pooled[f"{m}|all_cams|{h}"].extend(v)
    summary = {k: agg(v) for k, v in sorted(pooled.items())}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "source": str(RESULTS),
                "protocol": "zero-shot I2V, GT first frame, vs GT video (per-clip PSNR/SSIM)",
                "summary_per_model_horizon": summary,
                "detail_per_model_cam_horizon": detail,
            },
            indent=2,
        )
    )
    print(f"wrote {OUT}")
    print("\n=== summary (model | horizon -> mean PSNR / mean SSIM, n) ===")
    for k, v in summary.items():
        print(f"{k:45s} PSNR {v['psnr_mean']:6.2f}  SSIM {v['ssim_mean']:.4f}  (n={v['n']})")


if __name__ == "__main__":
    main()
