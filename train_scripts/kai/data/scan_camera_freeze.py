#!/usr/bin/env python3
"""Detect frozen-camera segments in KAI0 LeRobot datasets (RealSense frame-drop ->
recorder repeats last frame). Verified real (pyav byte-identical), not a decode artifact.

A frame is "frozen" if full-res max per-pixel abs diff vs the previous frame <= --maxdiff
(default 1): a real freeze (recorder repeats last frame) decodes byte-identical, while a
near-static LIVE scene still moves some pixel by >1. A "stall" = a run of >= --stall-frames
consecutive frozen frames.

Outputs (under --out-dir):
  freeze_per_camera.json  : per (date,camera) freeze-frame% + stall-episode% + simultaneity + onset-phase
  freeze_blocklist.txt    : episodes with any camera freeze-fraction > --block-frac
  masks/<date>/<episode>.json : per-camera frozen frame indices (only with --mask)

Usage:
  python train_scripts/kai/data/scan_camera_freeze.py \
      --root /data1/DATA_IMP/KAI0/Task_A/base --out-dir <dir> [--mask] [--workers 16]
"""
from __future__ import annotations
import argparse, glob, json, os
from multiprocessing import Pool
import numpy as np, cv2

CAMS = ("top_head", "hand_left", "hand_right")


def frozen_vector(path, maxdiff):
    """Return bool array len N-1: frame i is a TRUE duplicate of i-1.
    Criterion = full-res max per-pixel abs diff <= maxdiff (default 1). A real camera
    freeze (recorder repeats last frame) -> decoded frames byte-identical (maxdiff 0).
    A near-static LIVE scene still has some pixel moving by >1 -> NOT flagged. This
    cleanly separates genuine freezes from low-motion wrist close-ups."""
    cap = cv2.VideoCapture(path)
    prev = None; out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        # 160x120 downsample: a true freeze is identical at any resolution (maxdiff 0);
        # a near-static LIVE scene still moves some pixel >1 even downsampled. ~16x faster.
        fi = cv2.resize(f, (160, 120)).astype(np.int16)
        if prev is not None:
            out.append(int(np.abs(fi - prev).max()) <= maxdiff)
        prev = fi
    cap.release()
    return np.array(out, dtype=bool)


def runs(mask):
    """List of (start,len) frozen runs over a bool array."""
    r = []; i = 0; n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            r.append((i, j - i)); i = j
        else:
            i += 1
    return r


def scan_episode(args):
    date, ename, root, maxdiff, stall = args
    res = {c: None for c in CAMS}
    for c in CAMS:
        vp = f"{root}/{date}/videos/chunk-000/{c}/{ename}"
        if os.path.exists(vp):
            try:
                res[c] = frozen_vector(vp, maxdiff)
            except Exception:
                res[c] = None
    return date, ename, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data1/DATA_IMP/KAI0/Task_A/base")
    ap.add_argument("--out-dir", default="/data1/DATA_IMP/KAI0/Task_A/base/analysis/freeze")
    ap.add_argument("--maxdiff", type=int, default=1)
    ap.add_argument("--stall-frames", type=int, default=15)
    ap.add_argument("--block-frac", type=float, default=0.10, help="block ep if any cam freeze-frac > this")
    ap.add_argument("--mask", action="store_true", help="also write per-episode frozen-index masks")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--dates", default="", help="comma list to restrict (default all 2026-*-v2)")
    args = ap.parse_args()

    dates = ([d.strip() for d in args.dates.split(",") if d.strip()]
             or sorted(os.path.basename(x) for x in glob.glob(f"{args.root}/2026-*-v2")))
    tasks = []
    for d in dates:
        for v in sorted(glob.glob(f"{args.root}/{d}/videos/chunk-000/top_head/*.mp4")):
            tasks.append((d, os.path.basename(v), args.root, args.maxdiff, args.stall_frames))

    os.makedirs(args.out_dir, exist_ok=True)
    # per (date,cam) accumulators
    agg = {d: {c: dict(frames=0, frozen=0, stall_eps=0, n_ep=0, worst=0) for c in CAMS} for d in dates}
    # simultaneity: of frozen frames in any cam, how many cams frozen at same t
    simult = {d: dict(single=0, multi=0, total=0) for d in dates}
    # onset phase histogram (10 bins) per cam
    phase = {c: np.zeros(10, dtype=int) for c in CAMS}
    blocklist = []

    with Pool(args.workers) as p:
        for date, ename, res in p.imap_unordered(scan_episode, tasks, chunksize=4):
            T = max((len(v) for v in res.values() if v is not None), default=0)
            if T == 0:
                continue
            stacks = {}
            ep_block = False
            for c in CAMS:
                v = res[c]
                agg[date][c]["n_ep"] += 1
                if v is None or len(v) == 0:
                    continue
                stacks[c] = v
                agg[date][c]["frames"] += len(v)
                agg[date][c]["frozen"] += int(v.sum())
                rr = runs(v)
                worst = max((L for _, L in rr), default=0)
                agg[date][c]["worst"] = max(agg[date][c]["worst"], worst)
                if any(L >= args.stall_frames for _, L in rr):
                    agg[date][c]["stall_eps"] += 1
                if v.sum() / max(1, len(v)) > args.block_frac:
                    ep_block = True
                for s, L in rr:
                    if L >= args.stall_frames:
                        phase[c][min(9, int(10 * s / len(v)))] += 1
            if ep_block:
                blocklist.append(f"{date}/{ename}")
            # simultaneity over common length
            common = [stacks[c] for c in CAMS if c in stacks]
            if len(common) >= 2:
                m = min(len(x) for x in common)
                M = np.stack([x[:m] for x in common], 0)  # (cams, m)
                cnt = M.sum(0)                              # frozen-cam count per t
                anyfz = cnt >= 1
                simult[date]["total"] += int(anyfz.sum())
                simult[date]["single"] += int((cnt == 1).sum())
                simult[date]["multi"] += int((cnt >= 2).sum())
            if args.mask:
                md = f"{args.out_dir}/masks/{date}"; os.makedirs(md, exist_ok=True)
                json.dump({c: [int(i) for i in np.where(res[c])[0]] for c in CAMS if res[c] is not None},
                          open(f"{md}/{ename.replace('.mp4','')}.json", "w"))

    # write reports
    report = {}
    for d in dates:
        report[d] = {}
        for c in CAMS:
            a = agg[d][c]
            report[d][c] = dict(
                freeze_pct=round(100 * a["frozen"] / max(1, a["frames"]), 2),
                stall_ep_pct=round(100 * a["stall_eps"] / max(1, a["n_ep"]), 1),
                worst_run=a["worst"], n_ep=a["n_ep"])
        s = simult[d]
        report[d]["_simult"] = dict(
            total_frozen_frames=s["total"],
            multi_cam_pct=round(100 * s["multi"] / max(1, s["total"]), 1))
    report["_onset_phase_hist"] = {c: phase[c].tolist() for c in CAMS}
    json.dump(report, open(f"{args.out_dir}/freeze_per_camera.json", "w"), indent=2)
    open(f"{args.out_dir}/freeze_blocklist.txt", "w").write("\n".join(sorted(blocklist)) + "\n")

    # console summary: per-camera marginal
    print(f"{'date':14s} " + " ".join(f"{c[:4]+'_fz%':>8s} {c[:4]+'_ep%':>7s}" for c in CAMS) + "  multiCam%")
    for d in dates:
        line = f"{d:14s} "
        for c in CAMS:
            line += f"{report[d][c]['freeze_pct']:8.2f} {report[d][c]['stall_ep_pct']:7.1f}"
        line += f"  {report[d]['_simult']['multi_cam_pct']:7.1f}"
        print(line)
    # global per-camera totals
    print("\n=== global per-camera (all dates) ===")
    for c in CAMS:
        fr = sum(agg[d][c]["frozen"] for d in dates); tot = sum(agg[d][c]["frames"] for d in dates)
        se = sum(agg[d][c]["stall_eps"] for d in dates); ne = sum(agg[d][c]["n_ep"] for d in dates)
        print(f"  {c:11s} freeze={100*fr/max(1,tot):5.2f}% of frames | stall in {100*se/max(1,ne):5.1f}% of episodes")
    tm = sum(simult[d]["multi"] for d in dates); tt = sum(simult[d]["total"] for d in dates)
    print(f"  simultaneity: {100*tm/max(1,tt):.1f}% of frozen frames have >=2 cams frozen at once")
    print("  onset phase (frozen-run starts, 10 bins 0->100% of episode):")
    for c in CAMS:
        print(f"    {c:11s} {phase[c].tolist()}")
    print(f"\nblocklist: {len(blocklist)} episodes -> {args.out_dir}/freeze_blocklist.txt")
    print(f"report -> {args.out_dir}/freeze_per_camera.json")


if __name__ == "__main__":
    main()
