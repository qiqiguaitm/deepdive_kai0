#!/usr/bin/env python3
"""Task E 4-experiment progress dashboard.

Shows:
  - live training step / rate / ETA from log tail
  - loss curve snapshots at key steps (from wandb offline LevelDB)
  - inline-eval val MAE @1/@10/@25/@50 per saved ckpt
  - GPU util / VRAM + RAM snapshot

Usage:
  scripts/check_task_e_progress.py               # one-shot
  scripts/check_task_e_progress.py --watch 30    # refresh every 30s
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENV_SITE = PROJECT_ROOT / "kai0/.venv/lib/python3.11/site-packages"
if VENV_SITE.exists():
    sys.path.insert(0, str(VENV_SITE))

LOGS_DIR = PROJECT_ROOT / "logs"
WANDB_DIR = PROJECT_ROOT / "kai0/wandb"

EXPERIMENTS = [
    # (label,  exp_name,         gpu, pretty)
    ("v3", "v3_kai0_base",    1, "kai0+base"),
    ("v4", "v4_pi05_aug",     2, "pi05+aug"),
    ("v5", "v5_kai0_aug",     3, "kai0+aug"),
    ("v8", "v8_pi05_mirror",  0, "pi05+mirror"),
]

LOSS_STEPS = [100, 500, 1000, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 14999]


def latest_wandb_run(exp_name: str) -> Path | None:
    """Most recent wandb offline-run-* dir matching --exp_name=<exp_name>."""
    runs = sorted(WANDB_DIR.glob("offline-run-*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for d in runs:
        meta = d / "files" / "wandb-metadata.json"
        if not meta.exists():
            continue
        try:
            args = json.loads(meta.read_text()).get("args", [])
        except Exception:
            continue
        if f"--exp_name={exp_name}" in args:
            return d
    return None


def scan_wandb_history(wandb_file: Path) -> dict[int, dict]:
    """step -> {key: value} for every history record."""
    from wandb.sdk.internal.datastore import DataStore
    from wandb.proto import wandb_internal_pb2
    ds = DataStore()
    ds.open_for_scan(str(wandb_file))
    out: dict[int, dict] = {}
    while True:
        try:
            data = ds.scan_data()
        except AssertionError:
            break
        if data is None:
            break
        pb = wandb_internal_pb2.Record()
        pb.ParseFromString(data)
        if pb.WhichOneof("record_type") != "history":
            continue
        d = {}
        for it in pb.history.item:
            k = it.nested_key[0] if it.nested_key else it.key
            try:
                d[k] = json.loads(it.value_json)
            except Exception:
                pass
        s = d.get("_step")
        if s is not None:
            out[s] = d
    return out


PROGRESS_RE = re.compile(
    r"Progress on: ([\d.]+)k?it/([\d.]+)k?it rate:([\d.]+)it/s remaining:([\d:]+)"
)


def tail_progress(log_path: Path) -> dict | None:
    if not log_path.exists():
        return None
    try:
        out = subprocess.check_output(
            ["tac", str(log_path)], stderr=subprocess.DEVNULL, text=True
        )
    except Exception:
        try:
            out = log_path.read_text()[-20000:][::-1]
            out = "\n".join(reversed(log_path.read_text().splitlines()[-500:]))
        except Exception:
            return None
    for line in out.splitlines()[:5000]:
        m = PROGRESS_RE.search(line)
        if m:
            cur = float(m.group(1))
            total = float(m.group(2))
            if "kit/" in line:
                cur *= 1000
                total *= 1000
            return {"step": int(cur), "total": int(total), "rate": float(m.group(3)), "eta": m.group(4)}
    return None


INLINE_EVAL_RE = re.compile(
    r"\[inline-eval\] step=(\d+)\s+MAE@1=([\d.]+)\s+@10=([\d.]+)\s+@25=([\d.]+)\s+@50=([\d.]+)"
)


def inline_eval_records(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    out = []
    for line in log_path.read_text().splitlines():
        m = INLINE_EVAL_RE.search(line)
        if m:
            out.append({
                "step": int(m.group(1)),
                "mae_1": float(m.group(2)),
                "mae_10": float(m.group(3)),
                "mae_25": float(m.group(4)),
                "mae_50": float(m.group(5)),
            })
    return out


def gpu_ram_snapshot() -> tuple[list[dict], dict]:
    """Return per-GPU info list and a single RAM dict."""
    gpus = []
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw",
             "--format=csv,noheader,nounits"], text=True
        )
        for line in out.strip().splitlines():
            idx, util, used, total, power = [x.strip() for x in line.split(",")]
            gpus.append({
                "idx": int(idx), "util": int(util), "mem_used": int(float(used)),
                "mem_total": int(float(total)), "power": float(power),
            })
    except Exception:
        pass
    ram = {"total": 0, "used": 0, "avail": 0}
    try:
        out = subprocess.check_output(["free", "-g"], text=True)
        row = out.splitlines()[1].split()
        ram = {"total": int(row[1]), "used": int(row[2]), "avail": int(row[-1])}
    except Exception:
        pass
    return gpus, ram


# --- rendering ---

def fmt_loss_cell(v):
    if v is None:
        return "    -"
    return f"{v:.4f}"


def render(args) -> str:
    lines = []
    gpus, ram = gpu_ram_snapshot()
    now = time.strftime("%H:%M:%S")

    lines.append(f"══════ Task E 4-experiment dashboard @ {now} ══════")
    # GPU / RAM
    gpu_str = "  ".join(
        f"GPU{g['idx']}:{g['util']:>3}% {g['mem_used']:>5}/{g['mem_total']:>5}MB {g['power']:>4.0f}W"
        for g in gpus
    )
    lines.append(gpu_str)
    lines.append(f"RAM: used={ram['used']}/{ram['total']}GB  avail={ram['avail']}GB")
    lines.append("")

    # Progress table
    lines.append(f"{'exp':<4} {'GPU':>3} {'cfg':<14} {'step':>7}/total {'rate':>8} {'ETA':>10}")
    lines.append("-" * 60)
    progress_map = {}
    for label, exp, gpu, pretty in EXPERIMENTS:
        log = LOGS_DIR / f"train_{exp}.log"
        p = tail_progress(log)
        progress_map[label] = p
        if p:
            lines.append(
                f"{label:<4} {gpu:>3} {pretty:<14} {p['step']:>7}/{p['total']:<5} "
                f"{p['rate']:>6.2f}it/s {p['eta']:>10}"
            )
        else:
            lines.append(f"{label:<4} {gpu:>3} {pretty:<14} {'?':>7}  no-log-or-progress-yet")
    lines.append("")

    # Loss table
    lines.append("── loss ──")
    header = f"{'step':>6} | " + " | ".join(f"{lbl} {p:<11}" for lbl, _, _, p in EXPERIMENTS)
    lines.append(header)
    lines.append("-" * len(header))
    tables = {}
    for label, exp, _, _ in EXPERIMENTS:
        d = latest_wandb_run(exp)
        if d is None:
            tables[label] = {}
            continue
        try:
            wf = next(d.glob("*.wandb"))
            tables[label] = scan_wandb_history(wf)
        except Exception as e:
            tables[label] = {}
    for s in LOSS_STEPS:
        row = [f"{s:>6}"]
        any_present = False
        for label, _, _, _ in EXPERIMENTS:
            rec = tables[label].get(s)
            v = rec.get("loss") if rec else None
            if v is not None:
                any_present = True
            row.append(f"   {fmt_loss_cell(v):<11}")
        if any_present:
            lines.append(" | ".join(row))
    lines.append("")

    # Inline eval table
    lines.append("── val/mae_1 (inline eval at save_interval) ──")
    evals = {label: inline_eval_records(LOGS_DIR / f"train_{exp}.log") for label, exp, _, _ in EXPERIMENTS}
    all_steps = sorted({e["step"] for rs in evals.values() for e in rs})
    if not all_steps:
        lines.append("  (no inline-eval results yet — first fires at step=save_interval)")
    else:
        header2 = f"{'step':>6} | " + " | ".join(f"{lbl} {p:<11}" for lbl, _, _, p in EXPERIMENTS)
        lines.append(header2)
        lines.append("-" * len(header2))
        best = {label: min((e["mae_1"] for e in evals[label]), default=None) for label, _, _, _ in EXPERIMENTS}
        for s in all_steps:
            row = [f"{s:>6}"]
            for label, _, _, _ in EXPERIMENTS:
                hit = next((e for e in evals[label] if e["step"] == s), None)
                v = hit["mae_1"] if hit else None
                row.append(f"   {fmt_loss_cell(v):<11}")
            lines.append(" | ".join(row))
        # best row
        row_best = [f"{'best':>6}"]
        for label, _, _, _ in EXPERIMENTS:
            v = best[label]
            row_best.append(f"   {fmt_loss_cell(v):<11}")
        lines.append("-" * len(header2))
        lines.append(" | ".join(row_best))
        lines.append("")
        # full-metric summary for latest step across all runs
        latest = all_steps[-1]
        lines.append(f"── full MAE @ step={latest} (@1 / @10 / @25 / @50) ──")
        for label, exp, _, pretty in EXPERIMENTS:
            hit = next((e for e in evals[label] if e["step"] == latest), None)
            if hit:
                lines.append(
                    f"  {label} {pretty:<14} @1={hit['mae_1']:.4f}  @10={hit['mae_10']:.4f}  "
                    f"@25={hit['mae_25']:.4f}  @50={hit['mae_50']:.4f}"
                )
            else:
                lines.append(f"  {label} {pretty:<14} (no eval at step {latest})")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", type=int, default=0,
                    help="refresh every N seconds; 0 = one-shot (default)")
    args = ap.parse_args()
    if args.watch <= 0:
        print(render(args))
        return
    try:
        while True:
            os.system("clear")
            print(render(args))
            print(f"\n[watch every {args.watch}s — Ctrl-C to quit]")
            time.sleep(args.watch)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
