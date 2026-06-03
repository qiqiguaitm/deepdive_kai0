"""WAM 数据集全面体检 —— LeRobot v2.1 (wam_fold_v1 visrobot01/kairobot01)。

对每个数据集做以下硬检查(任一失败 → 退出码 1),适合 build/repair 后跑一遍确认可训练:

  1. meta 一致性     info.json 的 total_episodes/total_frames/total_videos/fps 与磁盘实际吻合;
                     tasks.jsonl 存在且 task_index 0 覆盖;episodes.jsonl 行数一致。
  2. parquet 完整性   每个 episode parquet 可读、含必需列、state/action 维度==EXPECT_DIM。
  3. 帧/索引连续性    每集 frame_index==0..N-1;episode_index 列==文件号;
                     全局 index across 集为 0..total_frames-1 连续单调。
  4. 时间戳规整       集内相邻 Δt 与 1/fps 偏差 <= tolerance_s(默认 1e-3)。
  5. 无尖刺           |observation.state|、|action| <= abs_threshold(默认 10;piper 关节~3.2/夹爪~0.1)。
  6. state/action 有限 无 NaN/Inf。
  7. 视频齐全         每集每个相机视图(info.features 里 dtype==video 的 key)对应 mp4 存在;
                     --check-video-decode 时再抽样解码首帧。
  8. episodes_stats   meta/episodes_stats.jsonl 存在,每个 episode_index 一行,覆盖 state+action。

用法:
  python -m scripts.wam_pipeline.validate_dataset <root> [<root> ...] \
      [--abs-threshold 10] [--tolerance-s 1e-3] [--expect-dim 14] [--num-views N] \
      [--check-video-decode] [--sample-video-decode 20]
  # 不传 root 时默认体检 ../kai0/data/wam_fold_v1/{visrobot01,kairobot01}
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq

REQUIRED_COLS = ["observation.state", "action", "timestamp", "frame_index", "episode_index", "index", "task_index"]


def _load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def validate(root, abs_threshold, tolerance_s, expect_dim, num_views, check_decode, sample_decode):
    name = os.path.basename(root.rstrip("/"))
    print(f"\n================ {name} ================")
    fails = []  # (check, detail)

    # ---- meta presence ----
    info_p = os.path.join(root, "meta", "info.json")
    eps_p = os.path.join(root, "meta", "episodes.jsonl")
    tasks_p = os.path.join(root, "meta", "tasks.jsonl")
    es_p = os.path.join(root, "meta", "episodes_stats.jsonl")
    for p in (info_p, eps_p, tasks_p):
        if not os.path.exists(p):
            fails.append(("meta", f"missing {p}"))
    if not os.path.exists(info_p):
        print("  FATAL: no info.json"); return fails + [("fatal", root)]
    info = json.load(open(info_p))
    fps = int(info["fps"])
    chunks_size = int(info.get("chunks_size", 1000))
    video_keys = [k for k, v in info.get("features", {}).items() if v.get("dtype") == "video"]
    eps = _load_jsonl(eps_p) if os.path.exists(eps_p) else []
    eplen = {int(e["episode_index"]): int(e["length"]) for e in eps}

    files = sorted(glob.glob(os.path.join(root, "data", "chunk-*", "episode_*.parquet")))
    file_idx = [int(os.path.basename(f)[8:14]) for f in files]

    # ---- (1) meta consistency: counts ----
    if info.get("total_episodes") != len(files):
        fails.append(("meta", f"total_episodes={info.get('total_episodes')} != parquet={len(files)}"))
    if len(eps) != len(files):
        fails.append(("meta", f"episodes.jsonl={len(eps)} != parquet={len(files)}"))
    if file_idx != list(range(len(files))):
        fails.append(("meta", "parquet episode numbering not contiguous 0..N-1"))
    tasks = _load_jsonl(tasks_p) if os.path.exists(tasks_p) else []
    if not any(int(t.get("task_index", -1)) == 0 for t in tasks):
        fails.append(("meta", "tasks.jsonl missing task_index 0"))
    if num_views and video_keys and len(video_keys) != num_views:
        fails.append(("meta", f"video views={len(video_keys)} != --num-views {num_views}"))

    # ---- scan parquet ----
    gmax = 0.0; nframes = 0; global_index_ok = True; next_index = 0
    n_spike = n_naninf = n_ts = n_fi = n_epidx = n_lenmis = n_dim = n_unreadable = n_vid_missing = 0
    smin = smax = amin = amax = None
    spike_ex = []
    for f, idx in zip(files, file_idx):
        try:
            t = pq.read_table(f)
        except Exception as e:
            n_unreadable += 1; fails.append(("parquet", f"ep{idx} unreadable: {e}")); continue
        cols = t.column_names
        miss = [c for c in REQUIRED_COLS if c not in cols]
        if miss:
            fails.append(("parquet", f"ep{idx} missing cols {miss}")); continue
        S = np.stack(t.column("observation.state").to_pylist()).astype(np.float64)
        A = np.stack(t.column("action").to_pylist()).astype(np.float64)
        n = len(S); nframes += n
        if S.shape[1] != expect_dim or A.shape[1] != expect_dim:
            n_dim += 1
        if eplen.get(idx) != n:
            n_lenmis += 1
        # finite + spikes
        for col, M in (("state", S), ("action", A)):
            if not np.isfinite(M).all():
                n_naninf += 1
            mx = float(np.abs(np.nan_to_num(M, nan=0, posinf=1e30, neginf=-1e30)).max())
            gmax = max(gmax, mx)
            if mx > abs_threshold:
                n_spike += 1
                if len(spike_ex) < 8:
                    spike_ex.append((idx, col, round(mx, 2)))
        smin = S.min(0) if smin is None else np.minimum(smin, S.min(0)); smax = S.max(0) if smax is None else np.maximum(smax, S.max(0))
        amin = A.min(0) if amin is None else np.minimum(amin, A.min(0)); amax = A.max(0) if amax is None else np.maximum(amax, A.max(0))
        # continuity
        fi = np.asarray(t.column("frame_index").to_pylist())
        if not np.array_equal(fi, np.arange(n)):
            n_fi += 1
        epc = np.asarray(t.column("episode_index").to_pylist())
        if not (epc == idx).all():
            n_epidx += 1
        gi = np.asarray(t.column("index").to_pylist())
        if not np.array_equal(gi, np.arange(next_index, next_index + n)):
            global_index_ok = False
        next_index += n
        # timestamp grid
        ts = np.asarray(t.column("timestamp").to_pylist(), dtype=np.float64)
        if n > 1 and float(np.abs(np.diff(ts) - 1.0 / fps).max()) > tolerance_s:
            n_ts += 1
        # videos
        chunk = idx // chunks_size
        for vk in video_keys:
            vp = os.path.join(root, info["video_path"].format(episode_chunk=chunk, video_key=vk, episode_index=idx))
            if not os.path.isfile(vp):
                n_vid_missing += 1
                if n_vid_missing <= 5:
                    fails.append(("video", f"missing {os.path.relpath(vp, root)}"))

    # tally hard checks
    if nframes != info.get("total_frames"):
        fails.append(("meta", f"sum frames={nframes} != info.total_frames={info.get('total_frames')}"))
    exp_videos = len(files) * len(video_keys)
    if info.get("total_videos") not in (None, exp_videos):
        fails.append(("meta", f"total_videos={info.get('total_videos')} != episodes*views={exp_videos}"))
    if n_dim: fails.append(("parquet", f"{n_dim} eps with state/action dim != {expect_dim}"))
    if n_unreadable: pass  # already recorded
    if n_lenmis: fails.append(("continuity", f"{n_lenmis} eps length(jsonl) != parquet rows"))
    if n_fi: fails.append(("continuity", f"{n_fi} eps frame_index != 0..N-1"))
    if n_epidx: fails.append(("continuity", f"{n_epidx} eps episode_index col != file idx"))
    if not global_index_ok: fails.append(("continuity", "global `index` not contiguous 0..total_frames-1"))
    if n_ts: fails.append(("timestamp", f"{n_ts} eps with |Δt-1/fps|>{tolerance_s}"))
    if n_spike: fails.append(("spike", f"{n_spike} (ep,col) with |value|>{abs_threshold}: {spike_ex}"))
    if n_naninf: fails.append(("finite", f"{n_naninf} (ep,col) with NaN/Inf"))
    if n_vid_missing: fails.append(("video", f"{n_vid_missing} missing video files total"))

    # episodes_stats coverage
    if not os.path.exists(es_p):
        fails.append(("episodes_stats", "missing meta/episodes_stats.jsonl"))
    else:
        es = _load_jsonl(es_p)
        es_idx = {int(r["episode_index"]) for r in es}
        if es_idx != set(file_idx):
            fails.append(("episodes_stats", f"coverage mismatch: rows={len(es)} vs episodes={len(files)}"))
        else:
            sample = es[0].get("stats", es[0])
            for col in ("observation.state", "action"):
                if col not in sample:
                    fails.append(("episodes_stats", f"row missing stats for {col}"))

    # optional decode
    if check_decode and files:
        try:
            import av
            step = max(1, len(files) // max(1, sample_decode))
            dec_fail = 0
            for f, idx in list(zip(files, file_idx))[::step]:
                chunk = idx // chunks_size
                for vk in video_keys:
                    vp = os.path.join(root, info["video_path"].format(episode_chunk=chunk, video_key=vk, episode_index=idx))
                    if not os.path.isfile(vp):
                        continue
                    try:
                        c = av.open(vp); frame = next(c.decode(video=0)); _ = frame.to_ndarray(format="rgb24"); c.close()
                    except Exception as e:
                        dec_fail += 1
                        if dec_fail <= 5:
                            fails.append(("video-decode", f"ep{idx} {vk}: {e}"))
            if dec_fail:
                fails.append(("video-decode", f"{dec_fail} sampled videos failed to decode"))
        except ImportError:
            print("  (pyav not available, skipping decode check)")

    # ---- report ----
    np.set_printoptions(precision=3, suppress=True)
    print(f"  robot_type={info.get('robot_type')} fps={fps} views={video_keys}")
    print(f"  episodes={len(files)} frames={nframes} global|max|(state/action)={gmax:.3f}")
    if smin is not None:
        print(f"  state  min={smin}")
        print(f"  state  max={smax}")
        print(f"  action min={amin}")
        print(f"  action max={amax}")
    t5 = len(glob.glob(os.path.join(root, "t5_embedding", "*.pt")))
    print(f"  t5_embedding pt files={t5} (info only)")
    if fails:
        print(f"  RESULT: FAIL ({len(fails)} issues)")
        for c, d in fails:
            print(f"    [{c}] {d}")
    else:
        print("  RESULT: PASS — all checks green")
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="*", help="dataset roots; default vis+kai under wam_fold_v1")
    ap.add_argument("--abs-threshold", type=float, default=10.0)
    ap.add_argument("--tolerance-s", type=float, default=1e-3)
    ap.add_argument("--expect-dim", type=int, default=14)
    ap.add_argument("--num-views", type=int, default=3)
    ap.add_argument("--check-video-decode", action="store_true")
    ap.add_argument("--sample-video-decode", type=int, default=20)
    args = ap.parse_args()

    roots = args.roots
    if not roots:
        base = os.environ.get("GWP_DATA", "../kai0/data/wam_fold_v1")
        roots = [f"{base}/visrobot01", f"{base}/kairobot01"]

    total_fail = 0
    for r in roots:
        fails = validate(r, args.abs_threshold, args.tolerance_s, args.expect_dim,
                         args.num_views, args.check_video_decode, args.sample_video_decode)
        total_fail += len(fails)
    print(f"\n==== OVERALL: {'ALL PASS' if total_fail == 0 else f'{total_fail} ISSUES'} ====")
    sys.exit(1 if total_fail else 0)


if __name__ == "__main__":
    main()
