#!/usr/bin/env python3
"""Generate v4 datasets from v3: front/tail-trimmed v3 data + gripper dims (6,13)
affine-remapped onto the canonical real-robot range [0, 0.07]m (0-70mm), clamped.
Other dims untouched. Gripper remap does not touch pixels, so the real video+depth bytes
are MOVED into v4 (the canonical location) and v3 is left holding relative symlinks into v4
("v4-real" convention, 2026-06-28). meta copied. Idempotent: a v3 entry already symlinked
into v4 is skipped, never moved.

Anchors (per gripper dim, GLOBAL over all in-scope v3 frames):
  observation.state : own [q01,q99] -> [0,0.07]   (state q01 ≈ true mechanical zero)
  action            : reuses the STATE affine      (default; --action-own-anchor reverts)

  WHY action reuses state: the master/leader gripper has a negative encoder zero-offset
  (raw q01 ≈ -2.5mm L / -4.8mm R), so its own q01 sits BELOW the real closed grasp;
  anchoring 0 there leaves "fully closed" ~2.3mm(L)/3.9mm(R) open -> policy can't close
  tight (observed on pi05_v4_awbc, 2026-06-28). State q01 ≈ true zero and shares the
  master's open extent (q99 identical), so the STATE affine maps true-closed -> 0 and
  co-registers action↔state. The original (broken) v4 used per-key own-q01 anchoring.

Layout: .../Task_*/<subset>/v3/<date>-v3  ->  .../Task_*/<subset>/v4/<date>-v4

Usage:
  python3 train_scripts/kai/data/make_v4_gripper_remap.py            # convert all KAI0 v3
  python3 train_scripts/kai/data/make_v4_gripper_remap.py --only base   # only paths containing 'base'
  python3 train_scripts/kai/data/make_v4_gripper_remap.py --dry-run     # compute params + plan, no write
"""
import argparse
import glob
import os
import shutil
import subprocess

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = "/data1/DATA_IMP/KAI0"
DIMS = [6, 13]
LO, HI = 0.0, 0.07  # canonical real-robot gripper range (meters) = 0-70mm
ACTION_OWN_ANCHOR = False  # default: action gripper anchored to STATE affine (see compute_global_params)


def find_v3_dirs():
    out = subprocess.check_output(["find", ROOT, "-type", "d", "-name", "20*-v3"]).decode().split()
    return sorted(out)


def _sa(t):
    s = np.array(t.column("observation.state").to_pylist(), dtype=np.float64)
    a = np.array(t.column("action").to_pylist(), dtype=np.float64)
    return s, a


def compute_global_params(dirs):
    """One global affine per (key, dim) from [q01,q99] over ALL in-scope v3 frames."""
    acc = {("state", d): [] for d in DIMS} | {("action", d): [] for d in DIMS}
    for d in dirs:
        for f in sorted(glob.glob(f"{d}/data/chunk-*/episode_*.parquet")):
            s, a = _sa(pq.read_table(f, columns=["observation.state", "action"]))
            for dim in DIMS:
                acc[("state", dim)].append(s[:, dim])
                acc[("action", dim)].append(a[:, dim])
    params = {}
    for k, vals in acc.items():
        v = np.concatenate(vals)
        q01, q99 = float(np.quantile(v, 0.01)), float(np.quantile(v, 0.99))
        span = q99 - q01
        a = (HI - LO) / span if span > 1e-9 else 1.0
        b = LO - a * q01
        params[k] = (q01, q99, a, b)
    # ---- ACTION gripper anchored to STATE's affine (per dim) ----
    # The master/leader gripper (action) has an encoder zero-offset: its mechanical
    # full-close reads NEGATIVE (raw q01 ≈ -2.5mm L / -4.8mm R), so the master's own
    # q01 sits in a negative noise tail BELOW the actual closed grasp. Anchoring 0 to
    # that q01 leaves the typical "fully closed" master value ~2.3mm(L)/~3.9mm(R) ABOVE
    # 0 → policy learns to command the gripper a few mm open → never closes tight.
    # The follower/slave (state) q01 ≈ true mechanical zero and its open extent (q99)
    # is identical to the master's, so reuse the STATE affine for the ACTION channel:
    # this co-registers action↔state and maps true-closed → 0. clamp() in remap()
    # folds the master's negative over-squeeze tail onto 0 (= full close). Disable with
    # --action-own-anchor to restore the legacy per-key anchoring.
    if not ACTION_OWN_ANCHOR:
        for dim in DIMS:
            params[("action", dim)] = params[("state", dim)]
    return params


def remap(arr, key, params):
    """arr (T,14) -> affine-remap gripper DIMS, clamp [LO,HI]; other dims untouched."""
    arr = arr.copy()
    for dim in DIMS:
        _, _, a, b = params[(key, dim)]
        arr[:, dim] = np.clip(a * arr[:, dim] + b, LO, HI)
    return arr


def v4_path(d):
    parts = d.split("/")
    assert parts[-2] == "v3", f"unexpected layout: {d}"
    parts[-2] = "v4"
    parts[-1] = parts[-1].replace("-v3", "-v4")
    return "/".join(parts)


def convert_dataset(d, params):
    v4 = v4_path(d)
    pq_files = sorted(glob.glob(f"{d}/data/chunk-*/episode_*.parquet"))
    vids = [p for p in glob.glob(f"{d}/videos/**/*", recursive=True) if os.path.isfile(p)]
    # v4-real convention (2026-06-28): the canonical real video/depth bytes live under v4; v3
    # holds relative symlinks into v4. The video loop below MOVES each real v3 file into v4 and
    # replaces it with a symlink. A v3 entry that is already a symlink (re-run / prior build) is
    # skipped — so this is idempotent and can never delete real bytes (the old symlink-based
    # build could, and did; see the 2026-06-28 incident).
    print(f"  {d.replace(ROOT + '/', '')} -> v4  ({len(pq_files)} eps, {len(vids)} media)")
    # ---- data: remap gripper dims, preserve schema ----
    for f in pq_files:
        dst = os.path.join(v4, os.path.relpath(f, d))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        t = pq.read_table(f)
        s, a = _sa(t)
        s2 = remap(s, "state", params).astype(np.float32)
        a2 = remap(a, "action", params).astype(np.float32)
        st = t.schema.field("observation.state").type
        at = t.schema.field("action").type
        t = t.set_column(t.schema.get_field_index("observation.state"), "observation.state",
                         pa.array([r.tolist() for r in s2], type=st))
        t = t.set_column(t.schema.get_field_index("action"), "action",
                         pa.array([r.tolist() for r in a2], type=at))
        pq.write_table(t, dst)
    # ---- videos + depth: MOVE real bytes into v4, leave v3 as a relative symlink -> v4 ----
    for vf in vids:
        if os.path.islink(vf):
            continue                                       # already flipped (v3 -> v4); skip
        dst = os.path.join(v4, os.path.relpath(vf, d))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.lexists(dst):
            os.remove(dst)                                 # stale v4 entry (old symlink / partial)
        shutil.move(vf, dst)                               # real v3 bytes -> v4
        os.symlink(os.path.relpath(dst, os.path.dirname(vf)), vf)   # v3 -> v4 symlink
    # closing assertion: v4 must hold the real file and v3 must resolve to it
    for vf in vids:
        dst = os.path.join(v4, os.path.relpath(vf, d))
        if not (os.path.isfile(dst) and not os.path.islink(dst)):
            raise SystemExit(f"ABORT {v4}: expected real file at {dst} after v4-real build")
        if not os.path.exists(vf):
            raise SystemExit(f"ABORT {d}: v3 symlink broken after flip: {vf}")
    # ---- meta: copy as-is (shapes unchanged) ----
    os.makedirs(os.path.join(v4, "meta"), exist_ok=True)
    for m in glob.glob(f"{d}/meta/*"):
        shutil.copy2(m, os.path.join(v4, "meta", os.path.basename(m)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="only convert v3 dirs whose path contains this substring")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--action-own-anchor", action="store_true",
                    help="legacy: anchor ACTION gripper to its OWN q01 (master encoder-offset noise tail) "
                         "instead of STATE's affine; leaves closed-mode ~3mm open. Do NOT use for real-robot deploy.")
    args = ap.parse_args()

    global ACTION_OWN_ANCHOR
    ACTION_OWN_ANCHOR = args.action_own_anchor
    print(f"action gripper anchor: {'OWN q01 (legacy)' if ACTION_OWN_ANCHOR else 'STATE affine (closed->0)'}")

    dirs = find_v3_dirs()
    print(f"in-scope v3 dirs: {len(dirs)}")
    print("computing GLOBAL [q01,q99] -> [0,0.07] params over ALL in-scope v3 ...")
    params = compute_global_params(dirs)
    for (key, dim), (q01, q99, a, b) in params.items():
        print(f"  {key}[{dim}]: q01={q01:.5f} q99={q99:.5f} -> a={a:.4f} b={b:+.5f}  "
              f"(q01->{a * q01 + b:.4f}, q99->{a * q99 + b:.4f})")

    todo = [d for d in dirs if args.only in d]
    print(f"\nconverting {len(todo)} dataset(s):")
    if args.dry_run:
        for d in todo:
            print(f"  [dry] {d.replace(ROOT + '/', '')} -> {v4_path(d).replace(ROOT + '/', '')}")
        return
    for d in todo:
        convert_dataset(d, params)
    print("\ndone")


if __name__ == "__main__":
    main()
