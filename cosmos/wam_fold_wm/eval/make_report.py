#!/usr/bin/env python3
"""Generate a self-contained report.html for one wam_fold_wm checkpoint eval.

Content:
  • Summary cards  — GT PSNR, ΔPSNR(real−null), ΔPSNR(real−other), verdict
  • Training loss chart  — per-iteration mean loss (avg over 8 ranks) + EMA smoother
  • Eval metrics history chart  — GT PSNR + ΔPSNR across all checkpoints so far
  • Per-episode table  — PSNR/SSIM for gt/other/null per episode
  • Episode comparison videos  — <video> players: GT | Real Action | Null Action
  • Summary mosaic video  — all episodes in one file

All assets (CSS, JS, SVG charts) are inline; videos are referenced by relative path.
No internet access required.

Usage:
  python make_report.py \\
    --iter 500 \\
    --report-dir  $RUNS/reports/fd_eval/iter500 \\
    --log-file    $RUNS/train_out_5n8g/train_node0.log \\
    --eval-history $RUNS/eval_results.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from datetime import datetime
from pathlib import Path

# ─────────────────────────── log parser ────────────────────────────────────

def parse_loss_log(log_path: str) -> dict[int, float]:
    """Return {iteration: mean_loss_over_ranks} from the training log."""
    pattern = re.compile(r'\[RANK\s+\d+\]\s+Iteration\s+(\d+):.+?Loss:\s+([\d.]+)')
    buckets: dict[int, list[float]] = {}
    try:
        with open(log_path, errors="replace") as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    it, loss = int(m.group(1)), float(m.group(2))
                    buckets.setdefault(it, []).append(loss)
    except FileNotFoundError:
        return {}
    return {it: sum(vs) / len(vs) for it, vs in sorted(buckets.items())}


def ema_smooth(data: list[tuple[int, float]], alpha: float = 0.05) -> list[tuple[int, float]]:
    """Exponential moving average (alpha = weight on new value)."""
    if not data:
        return []
    smoothed = [data[0]]
    for x, y in data[1:]:
        s = alpha * y + (1 - alpha) * smoothed[-1][1]
        smoothed.append((x, s))
    return smoothed


# ─────────────────────────── SVG chart ─────────────────────────────────────

def _fmt(v: float) -> str:
    if abs(v) >= 10:
        return f"{v:.1f}"
    if abs(v) >= 1:
        return f"{v:.2f}"
    return f"{v:.3f}"


def svg_chart(
    series: list[tuple[str, str, list[tuple[float, float]]]],  # (name, color, [(x,y)…])
    *,
    width: int = 640,
    height: int = 260,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    y_min: float | None = None,
    y_max: float | None = None,
    n_xticks: int = 6,
    n_yticks: int = 5,
    dot_r: int = 3,
) -> str:
    """Return an inline SVG string for a multi-series line chart."""
    PAD_L, PAD_R, PAD_T, PAD_B = 56, 24, 30, 46

    all_x = [x for _, _, pts in series for x, _ in pts]
    all_y = [y for _, _, pts in series for _, y in pts]
    if not all_x:
        return f'<svg width="{width}" height="{height}"><text x="50%" y="50%" text-anchor="middle" fill="#888">no data</text></svg>'

    x0, x1 = min(all_x), max(all_x)
    y0 = y_min if y_min is not None else min(all_y)
    y1 = y_max if y_max is not None else max(all_y)
    if x0 == x1:
        x0, x1 = x0 - 1, x1 + 1
    if y0 == y1:
        y0, y1 = y0 - 0.05, y1 + 0.05

    W = width - PAD_L - PAD_R
    H = height - PAD_T - PAD_B

    def px(x: float) -> float:
        return PAD_L + (x - x0) / (x1 - x0) * W

    def py(y: float) -> float:
        return PAD_T + H - (y - y0) / (y1 - y0) * H

    lines: list[str] = []

    # background
    lines.append(f'<rect width="{width}" height="{height}" fill="#1e1e2e" rx="6"/>')

    # title
    if title:
        lines.append(f'<text x="{width//2}" y="18" text-anchor="middle" '
                     f'fill="#cdd6f4" font-size="13" font-weight="bold">{title}</text>')

    # guard against degenerate single-point case
    n_xticks = max(n_xticks, 2)
    n_yticks = max(n_yticks, 2)

    # grid + x-ticks
    xtick_vals = [x0 + i * (x1 - x0) / (n_xticks - 1) for i in range(n_xticks)]
    for xv in xtick_vals:
        xp = px(xv)
        lines.append(f'<line x1="{xp:.1f}" y1="{PAD_T}" x2="{xp:.1f}" y2="{PAD_T+H}" '
                     f'stroke="#313244" stroke-width="1"/>')
        lines.append(f'<text x="{xp:.1f}" y="{PAD_T+H+14}" text-anchor="middle" '
                     f'fill="#6c7086" font-size="10">{_fmt(xv)}</text>')

    # y-ticks
    ytick_vals = [y0 + i * (y1 - y0) / (n_yticks - 1) for i in range(n_yticks)]
    for yv in ytick_vals:
        yp = py(yv)
        lines.append(f'<line x1="{PAD_L}" y1="{yp:.1f}" x2="{PAD_L+W}" y2="{yp:.1f}" '
                     f'stroke="#313244" stroke-width="1"/>')
        lines.append(f'<text x="{PAD_L-6}" y="{yp+4:.1f}" text-anchor="end" '
                     f'fill="#6c7086" font-size="10">{_fmt(yv)}</text>')

    # axes
    lines.append(f'<line x1="{PAD_L}" y1="{PAD_T}" x2="{PAD_L}" y2="{PAD_T+H}" '
                 f'stroke="#585b70" stroke-width="1.5"/>')
    lines.append(f'<line x1="{PAD_L}" y1="{PAD_T+H}" x2="{PAD_L+W}" y2="{PAD_T+H}" '
                 f'stroke="#585b70" stroke-width="1.5"/>')

    # axis labels
    if xlabel:
        lines.append(f'<text x="{PAD_L + W//2}" y="{height-4}" text-anchor="middle" '
                     f'fill="#6c7086" font-size="11">{xlabel}</text>')
    if ylabel:
        cx, cy = 12, PAD_T + H // 2
        lines.append(f'<text x="{cx}" y="{cy}" text-anchor="middle" '
                     f'fill="#6c7086" font-size="11" '
                     f'transform="rotate(-90 {cx} {cy})">{ylabel}</text>')

    # series
    legend_x = PAD_L + 8
    for idx, (name, color, pts) in enumerate(series):
        if not pts:
            continue
        polypts = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in pts)
        lines.append(f'<polyline points="{polypts}" fill="none" stroke="{color}" '
                     f'stroke-width="1.8" stroke-linejoin="round" stroke-linecap="round" opacity="0.8"/>')
        # dots
        if len(pts) <= 60:
            for x, y in pts:
                lines.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="{dot_r}" '
                             f'fill="{color}" opacity="0.9"/>')
        # legend
        ly = PAD_T + 6 + idx * 16
        lines.append(f'<line x1="{legend_x}" y1="{ly}" x2="{legend_x+18}" y2="{ly}" '
                     f'stroke="{color}" stroke-width="2"/>')
        lines.append(f'<text x="{legend_x+22}" y="{ly+4}" fill="#cdd6f4" font-size="10">{name}</text>')

    svg_body = "\n  ".join(lines)
    return (f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" '
            f'style="font-family:monospace">\n  {svg_body}\n</svg>')


# ─────────────────────────── HTML builder ──────────────────────────────────

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #11111b; color: #cdd6f4; font-family: 'Segoe UI', system-ui, sans-serif;
       font-size: 14px; padding: 20px; }
h1 { font-size: 22px; color: #89b4fa; margin-bottom: 4px; }
h2 { font-size: 15px; color: #89dceb; margin: 22px 0 10px; border-bottom: 1px solid #313244;
     padding-bottom: 6px; }
.meta { color: #6c7086; font-size: 12px; margin-bottom: 20px; }
.cards { display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 20px; }
.card { background: #1e1e2e; border: 1px solid #313244; border-radius: 8px;
        padding: 14px 20px; min-width: 160px; }
.card .label { font-size: 11px; color: #6c7086; text-transform: uppercase; letter-spacing:.05em; }
.card .value { font-size: 26px; font-weight: bold; margin-top: 4px; }
.card .value.good  { color: #a6e3a1; }
.card .value.warn  { color: #f9e2af; }
.card .value.bad   { color: #f38ba8; }
.card .value.neutral { color: #89b4fa; }
.charts { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.chart-box { background: #1e1e2e; border: 1px solid #313244; border-radius: 8px; padding: 10px; }
table { width: 100%; border-collapse: collapse; margin-top: 6px; }
th { background: #313244; color: #89b4fa; font-size: 12px; padding: 7px 10px;
     text-align: left; font-weight: 600; }
td { padding: 6px 10px; border-bottom: 1px solid #1e1e2e; font-size: 13px; }
tr:nth-child(even) td { background: #181825; }
.verdict { padding: 12px 16px; border-radius: 8px; margin-bottom: 20px;
           font-size: 13px; font-weight: 600; }
.verdict.good { background: #1c2a1c; border: 1px solid #40a02b; color: #a6e3a1; }
.verdict.warn { background: #2a251c; border: 1px solid #df8e1d; color: #f9e2af; }
.videos { display: flex; flex-direction: column; gap: 24px; }
.ep-block { background: #1e1e2e; border: 1px solid #313244; border-radius: 8px; padding: 14px; }
.ep-block h3 { font-size: 13px; color: #cba6f7; margin-bottom: 10px; }
video { display: block; width: 100%; max-width: 1100px; border-radius: 6px;
        background: #000; margin-top: 6px; }
.summary-video video { max-width: 100%; }
.psnr-pos { color: #a6e3a1; }
.psnr-neg { color: #f38ba8; }
.psnr-neu { color: #89b4fa; }
@media (max-width: 900px) { .charts { grid-template-columns: 1fr; } }
"""

def _card(label: str, value: str, cls: str = "neutral") -> str:
    return (f'<div class="card"><div class="label">{label}</div>'
            f'<div class="value {cls}">{value}</div></div>')


def _delta_class(v: float) -> str:
    if v > 1.0:   return "good"
    if v > 0.0:   return "warn"
    return "bad"


def _psnr_fmt(v: float) -> str:
    return f"{v:.2f} dB"


def generate_report(
    iter_num: int,
    report_dir: str,
    log_file: str,
    eval_history_file: str,
) -> str:
    """Build and write report.html; returns the output path."""
    report_dir = Path(report_dir)
    out_path = report_dir / "report.html"
    rpt_json = report_dir / "fd_daction_report.json"

    # ── load data ──────────────────────────────────────────────────────────
    rpt: dict = {}
    if rpt_json.exists():
        rpt = json.loads(rpt_json.read_text())

    agg   = rpt.get("aggregate", {})
    eps   = rpt.get("per_episode", [])
    verdict_txt = rpt.get("verdict", "—")
    args_rpt = rpt.get("args", {})

    loss_data = parse_loss_log(log_file)
    loss_pts  = [(it, v) for it, v in sorted(loss_data.items())]
    loss_ema  = ema_smooth(loss_pts, alpha=0.05)

    history: list[dict] = []
    try:
        hist_path = Path(eval_history_file)
        if hist_path.exists():
            for line in hist_path.read_text().splitlines():
                line = line.strip()
                if line:
                    history.append(json.loads(line))
    except Exception:
        pass

    # ── summary cards ──────────────────────────────────────────────────────
    gt_psnr  = agg.get("mean_gt_psnr", 0.0)
    dp_null  = agg.get("mean_dPSNR_gt_minus_zero", 0.0)
    dp_other = agg.get("mean_dPSNR_gt_minus_other", 0.0)
    mean_ssim = (sum(e.get("gt", {}).get("ssim", 0.0) for e in eps) / len(eps)) if eps else 0.0
    step_s = args_rpt.get("num_steps", "?")
    n_ep   = agg.get("n", 0)

    cards_html = "<div class='cards'>"
    cards_html += _card("GT PSNR (mean)", f"{gt_psnr:.2f} dB", "neutral")
    cards_html += _card("ΔPSNR real−null", f"{dp_null:+.2f} dB", _delta_class(dp_null))
    cards_html += _card("ΔPSNR real−other", f"{dp_other:+.2f} dB", _delta_class(dp_other))
    cards_html += _card("GT SSIM (mean)", f"{mean_ssim:.4f}", "neutral")
    cards_html += _card("Eval episodes", str(n_ep), "neutral")
    cards_html += _card("Diffusion steps", str(step_s), "neutral")
    if loss_pts:
        last_loss = loss_pts[-1][1]
        last_iter = loss_pts[-1][0]
        cards_html += _card(f"Train loss @ iter {last_iter}", f"{last_loss:.4f}", "neutral")
    cards_html += "</div>"

    verdict_cls = "good" if dp_other > 1.0 else "warn"
    verdict_html = f'<div class="verdict {verdict_cls}">🔍 {verdict_txt}</div>'

    # ── charts ─────────────────────────────────────────────────────────────
    # 1. training loss
    loss_series = [
        ("raw loss", "#585b70", loss_pts),
        ("EMA(α=0.05)", "#89b4fa", loss_ema),
    ]
    loss_svg = svg_chart(
        loss_series,
        width=620, height=260,
        title="Training Loss",
        xlabel="iteration", ylabel="loss",
        n_xticks=7, n_yticks=5,
        dot_r=0,
    )

    # 2. eval metrics history
    if history:
        hist_iters  = [h["iter"] for h in history]
        psnr_series = list(zip(hist_iters, [h.get("mean_gt_psnr", 0) for h in history]))
        dpnull_s    = list(zip(hist_iters, [h.get("mean_dPSNR_gt_minus_zero", 0) for h in history]))
        dpother_s   = list(zip(hist_iters, [h.get("mean_dPSNR_gt_minus_other", 0) for h in history]))
        metrics_series = [
            ("GT PSNR",       "#89b4fa", psnr_series),
            ("ΔPSNR real−null",  "#a6e3a1", dpnull_s),
            ("ΔPSNR real−other", "#f9e2af", dpother_s),
        ]
    else:
        metrics_series = []
    metrics_svg = svg_chart(
        metrics_series,
        width=620, height=260,
        title="Eval Metrics History",
        xlabel="checkpoint iter", ylabel="dB",
        n_xticks=max(min(len(history), 7), 2), n_yticks=5,
        dot_r=4,
    )

    charts_html = (
        "<div class='charts'>"
        f"<div class='chart-box'>{loss_svg}</div>"
        f"<div class='chart-box'>{metrics_svg}</div>"
        "</div>"
    )

    # ── per-episode table ──────────────────────────────────────────────────
    table_rows = ""
    for e in eps:
        ep_i  = e.get("episode", "?")
        gtp   = e.get("gt",    {}).get("psnr", 0)
        gts   = e.get("gt",    {}).get("ssim", 0)
        othp  = e.get("other", {}).get("psnr", 0)
        nulp  = e.get("zero",  {}).get("psnr", 0)
        dp_o  = e.get("dPSNR_gt_minus_other", 0)
        dp_z  = e.get("dPSNR_gt_minus_zero",  0)

        def signed(v: float) -> str:
            cls = "psnr-pos" if v > 0.5 else ("psnr-neg" if v < -0.5 else "psnr-neu")
            return f'<span class="{cls}">{v:+.3f}</span>'

        table_rows += (
            f"<tr><td>{ep_i}</td>"
            f"<td>{gtp:.3f}</td><td>{gts:.4f}</td>"
            f"<td>{othp:.3f}</td><td>{nulp:.3f}</td>"
            f"<td>{signed(dp_o)}</td><td>{signed(dp_z)}</td></tr>"
        )

    table_html = """
<table>
<thead><tr>
  <th>Episode</th>
  <th>GT PSNR</th><th>GT SSIM</th>
  <th>Other PSNR</th><th>Null PSNR</th>
  <th>ΔPSNR (real−other)</th><th>ΔPSNR (real−null)</th>
</tr></thead>
<tbody>""" + table_rows + "</tbody></table>"

    # ── videos ─────────────────────────────────────────────────────────────
    ep_videos_html = "<div class='videos'>"
    iter_tag = f"iter{iter_num:07d}"
    for e in eps:
        ep_i = e.get("episode", 0)
        gtp  = e.get("gt",   {}).get("psnr", 0)
        nulp = e.get("zero", {}).get("psnr", 0)
        dp   = e.get("dPSNR_gt_minus_zero", 0)
        dp_cls = "psnr-pos" if dp > 0.5 else ("psnr-neg" if dp < 0 else "psnr-neu")
        vid_name = f"{iter_tag}_ep{ep_i:02d}_compare.mp4"
        vid_path = report_dir / vid_name
        if vid_path.exists():
            ep_videos_html += (
                f"<div class='ep-block'>"
                f"<h3>Episode {ep_i} — GT PSNR {gtp:.2f} dB | Null PSNR {nulp:.2f} dB | "
                f"ΔPSNR <span class='{dp_cls}'>{dp:+.2f} dB</span></h3>"
                f"<video src='{vid_name}' controls preload='metadata'></video>"
                f"</div>"
            )
        else:
            ep_videos_html += (
                f"<div class='ep-block'>"
                f"<h3>Episode {ep_i} (video not found: {vid_name})</h3>"
                f"</div>"
            )
    ep_videos_html += "</div>"

    summary_vid = f"{iter_tag}_summary.mp4"
    if (report_dir / summary_vid).exists():
        summary_html = (
            f"<div class='summary-video'>"
            f"<video src='{summary_vid}' controls preload='metadata'></video>"
            f"</div>"
        )
    else:
        summary_html = "<p style='color:#6c7086'>Summary video not found.</p>"

    # ── assemble HTML ──────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>WM Eval — iter {iter_num:,}</title>
<style>{_CSS}</style>
</head>
<body>
<h1>wam_fold_wm_nano — Eval Report</h1>
<div class="meta">iter {iter_num:,} &nbsp;|&nbsp; generated {ts} &nbsp;|&nbsp;
  log: <code>{log_file}</code></div>

{verdict_html}

{cards_html}

<h2>Training Loss &amp; Eval Metrics History</h2>
{charts_html}

<h2>Per-Episode Results</h2>
{table_html}

<h2>Episode Comparison Videos &nbsp;<span style="font-size:12px;color:#6c7086;font-weight:normal">
  [ GT future &nbsp;|&nbsp; Real Action &nbsp;|&nbsp; Null Action ]</span></h2>
{ep_videos_html}

<h2>Summary Video (all episodes)</h2>
{summary_html}

</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")
    print(f"[report] {out_path}  ({out_path.stat().st_size // 1024} KB)", flush=True)
    return str(out_path)


# ─────────────────────────── CLI ───────────────────────────────────────────

def main():
    RUNS = "/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/wam_fold_wm_runs"
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", type=int, required=True)
    ap.add_argument("--report-dir",   default=None,
                    help="Directory with fd_daction_report.json and video files. "
                         "Default: $RUNS/reports/fd_eval/iter<N>")
    ap.add_argument("--log-file",     default=f"{RUNS}/train_out_5n8g/train_node0.log")
    ap.add_argument("--eval-history", default=f"{RUNS}/eval_results.jsonl")
    args = ap.parse_args()

    if args.report_dir is None:
        args.report_dir = f"{RUNS}/reports/fd_eval/iter{args.iter}"

    generate_report(
        iter_num=args.iter,
        report_dir=args.report_dir,
        log_file=args.log_file,
        eval_history_file=args.eval_history,
    )


if __name__ == "__main__":
    main()
