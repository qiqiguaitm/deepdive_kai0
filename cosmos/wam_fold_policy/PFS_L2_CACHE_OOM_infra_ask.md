# PFS-L2 / RapidFS client cache → node OOM (infra request)

## Problem
Training jobs that stream many files from the PFS-L2 mount get **node-OOM-killed (exitCode 137)**
after a few hundred steps. The PFS-L2 client (RapidFS) caches every read in **host RAM as
unreclaimable `used` memory** and never evicts within a run, climbing ~1 GB/step until the node's
1 TB fills.

## Evidence
- `free -g` on the node: **`used` climbs to ~890 GB** while `buff/cache` stays ~100 GB
  (so it is NOT normal reclaimable page cache — `echo 1 > drop_caches` does nothing).
- The training **process-tree RSS is only ~110 GB** (psutil), so the other ~700–780 GB is held
  **outside the pod's processes** — i.e. by the PFS-L2/RapidFS client, host-level.
- Growth ≈ the per-step raw-data read volume (video + parquet, ~1 GB/step). Cache **does not
  plateau** (passed 650 GB, OOM ~900 GB).
- A fresh `aihc job create` starts at ~30 GB `used` → **the cache is per-node and resets when the
  job lands on a fresh node**, which is why restart-from-checkpoint works (but is slow).

## Mount details
- type: **pfsl2**, mount target id: **`mt-zSSaab`**
- host path: `/pfs/visdata`  → pod mountPath: `/mnt/pfs/p46h4f`  (sourcePath `/visdata`)
- datasource options: `sizeLimit: 0` (uncapped), `medium: ""`, `pfsL1ClusterPort: 8888`
- node: 1005 GB RAM, 8× A100-80G, 1 pod/node

## Ask (either one fixes it permanently, for all jobs on this mount)
1. **Size-cap the RapidFS/PFS-L2 client read cache** for `mt-zSSaab` (e.g. ≤ 300–400 GB) so it
   evicts LRU and can never exhaust node RAM; **or**
2. **Back the read cache with node-local disk (NVMe)** instead of RAM (disk cache medium) — keeps
   full read throughput with zero RAM pressure.

## Questions to confirm scope
- Is RapidFS enabled on `mt-zSSaab`? What is the current cache **medium** (memory vs disk) and
  **size policy**?
- Is the cache cap configurable **per mount target** or **per job** (e.g. does the datasource
  `sizeLimit` field actually bound the read cache, or only the writable volume size)?
- Is there an LRU eviction high-water mark, and can it be set below node RAM?

## Current workaround (in use)
Auto-resume-cycle: each pod-life trains ~700 steps until node-OOM, then resubmits onto a fresh
node (cache reset) and resumes from the latest checkpoint. Correct but ~13 h for a 5000-step run
due to per-restart recompile. A capped/disk-backed cache would let a single job run to completion.
