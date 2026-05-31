#!/usr/bin/env python3
"""真机 "走一步退一步" 往复现象的离线归因诊断。

链路已确认: 模型输出 14D absolute joint chunk → integrate(裁前+overlap平滑) →
pop → jump保护 → EMA → gripper_offset → publish (absolute 直发, 无 delta 累加)。
所以往复若存在, 根因在模型 chunk 本身; 本脚本量化之。

三个指标 (per ckpt, fixed-noise 主 + random 对照):
  A. chunk 内往复 — 单个 50 步 chunk 内, 非夹爪关节 net/gross 比 (1=单调前进,
     →0=来回) + 平均方向反转次数。
  B. 模拟执行轨迹往复 — 沿时间线每 stride 帧 replan, 拼接每个 chunk 前 stride 步
     成 "实际会执行的轨迹", 算其 net/gross。与 GT 同段 net/gross 对照 (GT 人示范
     应接近单调)。这是 "走一步退一步" 最直接的离线代理。
  C. chunk 间衔接回跳 — 相邻两次 replan, 新 chunk 起始运动方向 vs 旧 chunk 在
     衔接点的运动方向, 反向比例 (高=每次 replan 都掉头)。

用法 (kai0/):
  .venv/bin/python ../train_scripts/kai/eval/eval_oscillation_diag.py \
    --ckpt /data1/DATA_IMP/checkpoints/ckpt_v0/A_0423_0527_mixed1_step20000 \
    --val data/Task_A/self_built/A_new_pure_200_val --n-ep 4 --stride 10
"""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

CAMS = ("top_head", "hand_left", "hand_right")
JIDX = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]  # 12 non-gripper joints


def read_video_frames(path: Path, n: int) -> np.ndarray:
    import av
    c = av.open(str(path)); s = c.streams.video[0]; s.thread_type = "AUTO"
    out = []
    for fr in c.decode(s):
        out.append(fr.to_ndarray(format="rgb24"))
        if len(out) >= n:
            break
    c.close()
    a = np.stack(out[:n], 0)
    if a.shape[0] < n:
        a = np.concatenate([a, np.repeat(a[-1:], n - a.shape[0], 0)], 0)
    return a


def net_gross(traj: np.ndarray) -> float:
    """traj (T,14) -> mean over non-gripper joints of |end-start| / pathlength."""
    t = traj[:, JIDX]
    net = np.abs(t[-1] - t[0])
    gross = np.sum(np.abs(np.diff(t, axis=0)), axis=0)
    return float(np.mean(net / (gross + 1e-6)))


def reversals(chunk: np.ndarray) -> float:
    """avg direction-reversals per joint across a chunk (T,14)."""
    d = np.diff(chunk[:, JIDX], axis=0)
    sgn = np.sign(d)
    sgn[sgn == 0] = 1
    chg = np.sum(np.abs(np.diff(sgn, axis=0)) > 0, axis=0)  # per joint
    return float(np.mean(chg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--prompt", default="Flatten and fold the cloth.")
    ap.add_argument("--n-ep", type=int, default=4)
    ap.add_argument("--stride", type=int, default=10, help="replan cadence in steps (3Hz@30Hz=10)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ckpt = Path(args.ckpt).resolve()
    spec = json.loads((ckpt / "train_config.json").read_text())
    os.environ["OPENPI_EXTRA_CONFIG"] = str(ckpt / "train_config.json")
    import jax  # noqa
    from openpi.policies import policy_config as _pc
    from openpi.training import config as _cfg
    train_cfg = _cfg.get_config(spec["base_config_name"])
    ah = int(getattr(train_cfg.model, "action_horizon", 50))
    ad = int(getattr(train_cfg.model, "action_dim", 32))
    print(f"[load] {ckpt.name} ah={ah} ad={ad}")
    pol = _pc.create_trained_policy(train_cfg, str(ckpt))
    fixed = np.random.default_rng(args.seed).standard_normal((ah, ad)).astype(np.float32)

    val = Path(args.val).resolve()
    eps = [json.loads(l) for l in (val / "meta" / "episodes.jsonl").read_text().splitlines()][: args.n_ep]

    agg = {k: [] for k in ("chunk_ng_fixed", "chunk_rev_fixed", "chunk_ng_rand",
                           "exec_ng_fixed", "exec_ng_rand", "gt_ng", "reversal_rate", "noise_chunk_mae")}
    for ep in eps:
        ei, L = ep["episode_index"], ep["length"]
        df = pq.read_table(val / "data" / "chunk-000" / f"episode_{ei:06d}.parquet").to_pandas()
        action = np.stack([np.asarray(x) for x in df["action"]])
        state = np.stack([np.asarray(x) for x in df["observation.state"]])
        cams = {c: read_video_frames(val / "videos" / "chunk-000" / f"observation.images.{c}" / f"episode_{ei:06d}.mp4", L) for c in CAMS}
        qs = list(range(0, L - ah - 1, args.stride))
        if len(qs) < 2:
            continue
        exec_f, exec_r = [], []
        chunks_f = []
        t0 = time.time()
        for k in qs:
            obs = {"images": {c: cams[c][k] for c in CAMS}, "state": state[k], "prompt": args.prompt}
            cf = np.asarray(pol.infer(obs, noise=fixed)["actions"])[:, :14]
            cr = np.asarray(pol.infer(obs, noise=None)["actions"])[:, :14]
            chunks_f.append(cf)
            agg["chunk_ng_fixed"].append(net_gross(cf))
            agg["chunk_rev_fixed"].append(reversals(cf))
            agg["chunk_ng_rand"].append(net_gross(cr))
            agg["noise_chunk_mae"].append(float(np.abs(cf - cr).mean()))
            exec_f.append(cf[: args.stride])
            exec_r.append(cr[: args.stride])
        exec_f = np.concatenate(exec_f, 0)
        exec_r = np.concatenate(exec_r, 0)
        gt_seg = action[qs[0] : qs[0] + len(exec_f)]
        agg["exec_ng_fixed"].append(net_gross(exec_f))
        agg["exec_ng_rand"].append(net_gross(exec_r))
        agg["gt_ng"].append(net_gross(gt_seg))
        # C: chunk-to-chunk seam reversal
        rev = 0; tot = 0
        for i in range(len(chunks_f) - 1):
            old = chunks_f[i]; new = chunks_f[i + 1]
            dir_old = old[args.stride, JIDX] - old[args.stride - 1, JIDX]
            dir_new = new[1, JIDX] - new[0, JIDX]
            rev += int(np.sum(np.sign(dir_old) != np.sign(dir_new)))
            tot += len(JIDX)
        if tot:
            agg["reversal_rate"].append(rev / tot)
        print(f"  ep{ei:02d} q={len(qs)} exec_ng_fix={agg['exec_ng_fixed'][-1]:.3f} "
              f"gt_ng={agg['gt_ng'][-1]:.3f} chunk_ng={np.mean(agg['chunk_ng_fixed'][-len(qs):]):.3f} ({time.time()-t0:.0f}s)")

    summary = {"ckpt": str(ckpt), "stride": args.stride, "n_ep": len(eps)}
    for k, v in agg.items():
        summary[k] = float(np.mean(v)) if v else None
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))
    print("\n读法: exec_ng_fixed 远低于 gt_ng => 模拟执行轨迹往复 (走一步退一步);")
    print("      chunk_ng_fixed 低 => 单 chunk 内部就往复; reversal_rate 高 => 每次 replan 掉头;")
    print("      noise_chunk_mae 大 + chunk_ng_rand<<chunk_ng_fixed => noise multi-modal collapse")
    out = Path(args.out) if args.out else ckpt / "eval_oscillation_diag.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
