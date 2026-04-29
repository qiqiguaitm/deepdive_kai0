#!/usr/bin/env python3
"""Nightly TOS batch sync: sim01 → TOS → gf1 (fuse) → /vePFS (shared gpfs).

Why: gf0 direct rsync over WAN is ~2 MB/s, can't keep up with recording's data
     production rate. TOS multi-part upload reaches ~85 MB/s, gf1 fuse-mounted
     /transfer-shanghai can extract tar in-place to shared gpfs. One nightly pass
     cleans all backlog.

Flow per (task, subset, date) (v2 layout):
  1. tar -C DATA_ROOT task/subset/date → /tmp/<tarname>.tar (uncompressed;
     video/zarr already compressed).
  2. Upload /tmp/<tarname>.tar → TOS bucket "transfer-shanghai" key "KAI0/<tarname>.tar"
     (16 workers × 64 MB multi-part, ~85 MB/s).
  3. ssh gf1: tar xf /transfer-shanghai/KAI0/<tarname>.tar -C /vePFS/visrobot01/KAI0/
     (gf1 has TOS bucket mounted as /transfer-shanghai via hpvs_fs; extract target
     is shared gpfs with gf0, so gf0 sees the data immediately).
  4. Delete TOS object (free bucket storage).
  5. Delete local tar.

CLI:
  python nightly_tos_batch.py                       # all subsets modified in last 26 h
  python nightly_tos_batch.py --since-hours 48      # last 48 h
  python nightly_tos_batch.py --task Task_A         # only this task
  python nightly_tos_batch.py --dry-run             # list plan, no I/O

Cron (run this once to install):
  (crontab -l 2>/dev/null; echo '30 0 * * * /data1/miniconda3/bin/python /data1/tim/workspace/deepdive_kai0/web/data_manager/backend/tools/nightly_tos_batch.py >> /data1/tim/workspace/deepdive_kai0/web/data_manager/logs/nightly_tos.log 2>&1') | crontab -
Fires at 00:30 local (within the night bwlimit-released window). Logs to web/data_manager/logs/nightly_tos.log.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# --- purge proxy before tos import (intra-region path, proxy breaks handshake) ---
_PROXY_KEYS = ["http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY",
               "ftp_proxy", "FTP_PROXY", "all_proxy", "ALL_PROXY",
               "no_proxy", "NO_PROXY"]
for _k in _PROXY_KEYS:
    os.environ.pop(_k, None)

import tos
from tos import exceptions as tos_exc

# Credentials are loaded from env — NEVER hard-code (set in web/data_manager/.env).
AK = os.environ.get("KAI0_TOS_AK", "")
SK = os.environ.get("KAI0_TOS_SK", "")
ENDPOINT = os.environ.get("KAI0_TOS_ENDPOINT", "tos-cn-shanghai.volces.com")
REGION = os.environ.get("KAI0_TOS_REGION", "cn-shanghai")
BUCKET = os.environ.get("KAI0_TOS_BUCKET", "transfer-shanghai")
if not (AK and SK):
    import sys
    print("[ERROR] KAI0_TOS_AK / KAI0_TOS_SK not set. Source web/data_manager/.env first.", file=sys.stderr)
    sys.exit(2)
TOS_PREFIX = "KAI0/"

DATA_ROOT = Path(os.environ.get("KAI0_DATA_ROOT", "/data1/DATA_IMP/KAI0"))
GF1_SSH_CMD = ("ssh -p 11111 -o BatchMode=yes "
               "-o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 "
               "tim@14.103.44.161")
GF_TRANSFER_DIR = "/transfer-shanghai/KAI0"   # gf1 fuse mount of bucket
GF_GPFS_ROOT = "/vePFS/visrobot01/KAI0"        # shared gpfs both gf0 and gf1 see


# ------------------------------ helpers -----------------------------------
def log(msg: str, *, level: str = "INFO") -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{ts} {level:<5} {msg}", flush=True)


def find_modified_subsets(since_hours: float, only_task: str | None) -> list[tuple[str, str, str, Path]]:
    """Return (task, date, subset, abs_path) for each subset dir modified recently.
    'Modified' = any file inside newer than threshold."""
    threshold = time.time() - since_hours * 3600
    out: list[tuple[str, str, str, Path]] = []
    if not DATA_ROOT.is_dir():
        log(f"DATA_ROOT missing: {DATA_ROOT}", level="ERROR")
        return out
    # v2 layout: <task>/<subset>/<YYYY-MM-DD>
    def _looks_like_date(name: str) -> bool:
        return len(name) == 10 and name[4] == "-" and name[7] == "-"
    for task_dir in sorted(DATA_ROOT.iterdir()):
        if not task_dir.is_dir():
            continue
        if only_task and task_dir.name != only_task:
            continue
        # skip obvious non-task dirs
        if task_dir.name in ("ckpt_downloads", "task_e_parts"):
            continue
        for sub_dir in sorted(task_dir.iterdir()):
            if not sub_dir.is_dir():
                continue
            for date_dir in sorted(sub_dir.iterdir()):
                if not date_dir.is_dir():
                    continue
                if not _looks_like_date(date_dir.name):
                    continue
                # has recent activity?
                newest = 0.0
                for p in date_dir.rglob("*"):
                    try:
                        m = p.stat().st_mtime
                        if m > newest:
                            newest = m
                            if newest > threshold:
                                break
                    except OSError:
                        continue
                if newest > threshold:
                    out.append((task_dir.name, date_dir.name, sub_dir.name, date_dir))
    return out


def tar_subset(src_subset: Path, tar_path: Path) -> float:
    """Pack subset dir → tar_path (uncompressed). Return seconds."""
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    # -C DATA_ROOT include task/subset/date relpath → extracted dst/Task_X/<subset>/YYYY-MM-DD/.
    # Archive members start with 'Task_X/<subset>/YYYY-MM-DD/...' (v2 layout).
    rel = src_subset.relative_to(DATA_ROOT)
    t0 = time.time()
    log(f"tar cf {tar_path} -C {DATA_ROOT} {rel}")
    # --ignore-failed-read: skip files that disappeared during tar (live recording)
    cmd = ["tar", "cf", str(tar_path), "--ignore-failed-read",
           "-C", str(DATA_ROOT), str(rel)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    dt = time.time() - t0
    if p.returncode != 0 and p.returncode != 1:  # rc=1 = tar saw "file changed", ok
        log(f"tar FAILED rc={p.returncode}: {p.stderr[-500:]}", level="ERROR")
        raise RuntimeError(f"tar failed rc={p.returncode}")
    size = tar_path.stat().st_size
    log(f"tar ok in {dt:.1f}s  size={size/1e9:.2f} GB  rate={size/1e6/max(dt,0.01):.1f} MB/s")
    return dt


def tos_upload(client: tos.TosClientV2, tar_path: Path, key: str) -> float:
    t0 = time.time()
    log(f"TOS upload → {key}")
    client.upload_file(BUCKET, key, str(tar_path), task_num=16, part_size=64 * 1024 * 1024)
    dt = time.time() - t0
    size = tar_path.stat().st_size
    log(f"upload ok in {dt:.1f}s  {size/1e6/max(dt,0.01):.1f} MB/s")
    return dt


def gf1_extract(tos_key: str) -> float:
    """ssh gf1: tar xf /transfer-shanghai/<key> -C /vePFS/visrobot01/KAI0/
    Then verify by du."""
    remote_tar = f"{GF_TRANSFER_DIR}/{tos_key.split('/', 1)[1]}"  # strip KAI0/ prefix
    # Actually the key is "KAI0/<name>.tar"; fuse path is /transfer-shanghai/KAI0/<name>.tar
    remote_tar = f"/transfer-shanghai/{tos_key}"
    cmd = (
        f"set -e; mkdir -p {GF_GPFS_ROOT}; "
        f"tar xf {remote_tar} -C {GF_GPFS_ROOT} --overwrite; "
        f"echo EXTRACT_OK"
    )
    t0 = time.time()
    log(f"gf1 extract: {remote_tar} → {GF_GPFS_ROOT}")
    p = subprocess.run(GF1_SSH_CMD.split() + [cmd],
                       capture_output=True, text=True, timeout=3600)
    dt = time.time() - t0
    if p.returncode != 0 or "EXTRACT_OK" not in p.stdout:
        log(f"gf1 extract FAILED rc={p.returncode}: {p.stderr[-500:]}", level="ERROR")
        raise RuntimeError("gf1 extract failed")
    log(f"extract ok in {dt:.1f}s")
    return dt


def tos_delete(client: tos.TosClientV2, key: str) -> None:
    try:
        client.delete_object(BUCKET, key)
        log(f"deleted TOS object {key}")
    except tos_exc.TosServerError as e:
        log(f"TOS delete failed: {e}", level="WARN")


def process_one(client: tos.TosClientV2, task: str, date: str, subset: str,
                src: Path, tmp_dir: Path) -> dict:
    tar_name = f"batch_{task}_{date}_{subset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tar"
    tar_path = tmp_dir / tar_name
    tos_key = f"{TOS_PREFIX}{tar_name}"
    stats = {"task": task, "date": date, "subset": subset, "ok": False}
    try:
        stats["tar_s"] = tar_subset(src, tar_path)
        stats["upload_s"] = tos_upload(client, tar_path, tos_key)
        stats["extract_s"] = gf1_extract(tos_key)
        tos_delete(client, tos_key)
        stats["ok"] = True
    except Exception as e:
        log(f"batch {task}/{date}/{subset} FAILED: {e}", level="ERROR")
        # best-effort cleanup of TOS object on failure
        try:
            client.head_object(BUCKET, tos_key)
            tos_delete(client, tos_key)
        except Exception:
            pass
    finally:
        tar_path.unlink(missing_ok=True)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--since-hours", type=float, default=26.0,
                    help="process subsets with any file modified within last N hours (default 26)")
    ap.add_argument("--task", default=None, help="only this task (default: all)")
    ap.add_argument("--tmp", default="/data1", help="directory for temp tar (default /data1)")
    ap.add_argument("--dry-run", action="store_true", help="list plan, don't do I/O")
    args = ap.parse_args()

    subsets = find_modified_subsets(args.since_hours, args.task)
    log(f"plan: {len(subsets)} subset(s) modified in last {args.since_hours}h"
        + (f" (task={args.task})" if args.task else ""))
    for (t, d, s, p) in subsets:
        log(f"  {t}/{d}/{s}")
    if args.dry_run or not subsets:
        return 0

    tmp_dir = Path(args.tmp)
    client = tos.TosClientV2(AK, SK, ENDPOINT, REGION)
    results = []
    for (t, d, s, p) in subsets:
        r = process_one(client, t, d, s, p, tmp_dir)
        results.append(r)
    ok = sum(1 for r in results if r["ok"])
    log(f"DONE. {ok}/{len(results)} subsets synced")
    for r in results:
        if r["ok"]:
            log(f"  ok  {r['task']}/{r['date']}/{r['subset']}  "
                f"tar={r.get('tar_s',0):.0f}s up={r.get('upload_s',0):.0f}s "
                f"ext={r.get('extract_s',0):.0f}s")
        else:
            log(f"  FAIL {r['task']}/{r['date']}/{r['subset']}", level="ERROR")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
