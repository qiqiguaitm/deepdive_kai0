#!/usr/bin/env python3
"""Continuous training-curve monitor for the wam_fold cosmos run (no GPU).

Parses the offline-wandb datastore every --interval seconds and renders:
  - loss curve         : train/loss
  - action-loss curve  : train@2_detail/flow_matching_loss_action
  - video-loss curve   : train@2_detail/flow_matching_loss_vision
  - eval curve (if any): action_mae / video metric per checkpoint from eval_curve.csv
into <out>/curves.{png,html} + a small JSON, so progress is visible without GPUs.

Wandb history items arrive via nested_key (not key); _step is the x-axis.
"""
from __future__ import annotations
import argparse, glob, json, os, time, csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KEYS = {
    "loss":   "train/loss",
    "action": "train@2_detail/flow_matching_loss_action",
    "video":  "train@2_detail/flow_matching_loss_vision",
}


def read_series(wandb_file):
    from wandb.sdk.internal import datastore
    from wandb.proto import wandb_internal_pb2 as pb
    ds = datastore.DataStore(); ds.open_for_scan(wandb_file)
    series = {k: {} for k in KEYS}  # name -> {step: val}
    cur_step = None
    while True:
        try:
            d = ds.scan_data()
        except Exception:
            break
        if d is None:
            break
        r = pb.Record()
        try:
            r.ParseFromString(d)
        except Exception:
            continue
        if r.WhichOneof("record_type") != "history":
            continue
        row = {}
        for it in r.history.item:
            k = it.key or "/".join(it.nested_key)
            row[k] = it.value_json
        step = row.get("_step")
        if step is None:
            continue
        try:
            step = int(float(step))
        except Exception:
            continue
        for name, key in KEYS.items():
            if key in row:
                try:
                    series[name][step] = float(row[key])
                except Exception:
                    pass
    return series


def read_eval_curve(path):
    if not os.path.isfile(path):
        return []
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                rows.append({k: (float(v) if v not in ("", None) else None) for k, v in row.items()})
            except Exception:
                pass
    return rows


def render(series, evalrows, out, latest):
    os.makedirs(out, exist_ok=True)
    npan = 3 + (1 if evalrows else 0)
    fig, axes = plt.subplots(npan, 1, figsize=(10, 3.2 * npan), squeeze=False)
    axes = axes[:, 0]
    titles = {"loss": "Total loss (train/loss)",
              "action": "Action flow-matching loss",
              "video": "Video flow-matching loss"}
    for ax, name in zip(axes[:3], ["loss", "action", "video"]):
        s = series[name]
        if s:
            xs = sorted(s)
            ys = [s[x] for x in xs]
            ax.plot(xs, ys, lw=1.0)
            # smoothed
            if len(ys) > 20:
                w = max(1, len(ys) // 100)
                sm = [sum(ys[max(0, i - w):i + w + 1]) / len(ys[max(0, i - w):i + w + 1]) for i in range(len(ys))]
                ax.plot(xs, sm, lw=2.0, alpha=0.7)
            ax.set_title(f"{titles[name]}  (latest={ys[-1]:.4f} @ step {xs[-1]})")
        else:
            ax.set_title(f"{titles[name]}  (no data yet)")
        ax.set_xlabel("step"); ax.grid(alpha=0.3)
    if evalrows:
        ax = axes[3]
        xs = [r["iter"] for r in evalrows if r.get("iter") is not None]
        for col, lbl in [("action_mae", "eval action_mae"), ("video_psnr", "eval video_psnr"),
                         ("mae@48", "eval mae@48")]:
            ys = [(r.get(col)) for r in evalrows if r.get("iter") is not None]
            if any(v is not None for v in ys):
                ax.plot(xs, ys, "o-", label=lbl, lw=1.5)
        ax.set_title("Eval metrics per checkpoint"); ax.set_xlabel("step")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    png = os.path.join(out, "curves.png")
    fig.savefig(png, dpi=90); plt.close(fig)
    # html
    html = f"""<!doctype html><meta charset=utf-8><title>wam_fold training curves</title>
<body style="font-family:sans-serif;max-width:1100px;margin:auto">
<h2>wam_fold cosmos3 — training curves</h2>
<p>updated: {latest} | loss steps: {len(series['loss'])} | eval checkpoints: {len(evalrows)}</p>
<img src="curves.png" style="width:100%">
</body>"""
    open(os.path.join(out, "curves.html"), "w").write(html)
    summary = {name: (sorted(s)[-1], s[sorted(s)[-1]]) if s else None for name, s in series.items()}
    json.dump({"updated": latest, "latest_step_val": summary, "n_eval": len(evalrows)},
              open(os.path.join(out, "curves.json"), "w"), indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wandb_glob", default="/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/wam_fold_policy_runs/"
                    "train_out_4n8g/cosmos3/action/wam_fold_nano/wandb/offline-run-*/run-*.wandb")
    ap.add_argument("--out", default="/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/wam_fold_policy_runs/reports/curves")
    ap.add_argument("--eval_curve", default="/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/wam_fold_policy_runs/reports/eval_curve.csv")
    ap.add_argument("--interval", type=int, default=600)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    while True:
        files = sorted(glob.glob(a.wandb_glob), key=os.path.getmtime)
        if files:
            try:
                series = read_series(files[-1])
                evalrows = read_eval_curve(a.eval_curve)
                summ = render(series, evalrows, a.out, time.strftime("%Y-%m-%d %H:%M:%S"))
                print(f"[curve] {time.strftime('%H:%M:%S')} loss_steps={len(series['loss'])} "
                      f"latest={summ.get('loss')} eval_ckpts={len(evalrows)} -> {a.out}/curves.html", flush=True)
            except Exception as e:
                print(f"[curve] err: {type(e).__name__}: {e}", flush=True)
        else:
            print(f"[curve] no wandb file yet @ {time.strftime('%H:%M:%S')}", flush=True)
        if a.once:
            break
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
