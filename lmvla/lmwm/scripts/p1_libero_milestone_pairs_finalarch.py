#!/usr/bin/env python
"""P1 终版架构版: LIBERO 每任务 milestone 训练对(替代退化的 p1_libero_milestone_pairs.py)。

对齐 CRAVE 终版(final_architecture.md): img(DINOv3-base pooled,PCA128) ⊕ proprio(state,L2,1:1)
  → BayesianGMM(自适应K) + per-mode coverage≥0.5 + 中位数值 → 双锚 Viterbi 分段
  → 建对: next-seg medoid 目标 + 末段 self-loop(向终止 medoid 收敛, 不丢末段)。
这修 P1 退化(argmin+cummax+丢末段)导致的 task6 类弥散子任务欠分割。

输出: lmwm/data/libero_milestone_finalarch/pairs.npz
  keys 与旧版一致(provider 兼容): cur_ep, cur_fi, tgt_fi, cur_ms, pair_task
  cur_fi/tgt_fi = 特征帧索引(stride2 空间, 0..N-1), 与 provider 的 i=frame//2 对齐。
用法: srpo python p1_libero_milestone_pairs_finalarch.py [--maxtask N] [--smoke]
"""
import os, sys, glob, argparse
import numpy as np, pandas as pd
from sklearn.mixture import BayesianGaussianMixture
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d

ROOT = "/home/tim/workspace/deepdive_kai0/lmvla/lawam/dataset/libero_merged_no_noops_20hz"
FEAT = "/home/tim/workspace/deepdive_kai0/lmvla/lmwm/data/libero_dinov3base"
OUT  = "/home/tim/workspace/deepdive_kai0/lmvla/lmwm/data/libero_milestone_finalarch"
MIN_COV = 0.50; LAM = 16.0

def l2(x): return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)

def mode_split(Tc, nbins=30):
    """谷分裂 + 每段中位数 (faithful, 支持双峰; final_architecture §2.7/gen_final_v3)."""
    h, ed = np.histogram(Tc, bins=nbins, range=(0, 1)); h = h.astype(float) / (h.sum() + 1e-9)
    hs = gaussian_filter1d(h, 1.2); c = (ed[:-1] + ed[1:]) / 2
    peaks = [i for i in range(nbins) if hs[i] >= hs[max(0, i-1)] and hs[i] >= hs[min(nbins-1, i+1)] and hs[i] >= 0.10 * hs.max()]
    merged = []
    for p in peaks:
        if merged and abs(c[p] - c[merged[-1]]) < 0.10:
            if hs[p] > hs[merged[-1]]: merged[-1] = p
        else: merged.append(p)
    final = [merged[0]] if merged else [int(np.argmax(hs))]
    for p in merged[1:]:
        valley = hs[final[-1]:p+1].min()
        if valley < 0.6 * min(hs[final[-1]], hs[p]): final.append(p)
        elif hs[p] > hs[final[-1]]: final[-1] = p
    if len(final) <= 1:
        return [(float(np.median(Tc)), np.ones(len(Tc), bool))]
    cuts = [c[a + int(np.argmin(hs[a:b+1]))] for a, b in zip(final[:-1], final[1:])]
    edges = [0.0] + cuts + [1.0]; out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        msk = (Tc >= lo) & (Tc < hi)
        if msk.sum() >= 5: out.append((float(np.median(Tc[msk])), msk))
    return out if out else [(float(np.median(Tc)), np.ones(len(Tc), bool))]

def dual_anchor_states(Jq, Ctgt, vals, s_anchor, e_anchor):
    """双锚 Viterbi(gen_final_v3.vit): 发射目标=起点锚+milestones+终点锚; 强制首=0末=顶.
    返回每帧的 state 索引(0=start_anchor, 1..M=milestones, M+1=end_anchor)."""
    C = np.concatenate([s_anchor[None], Ctgt, e_anchor[None]], 0)         # [M+2, D]
    v = np.concatenate([[0.0], vals, [1.0]])                             # 每 state 的值
    order = np.argsort(v); C = C[order]; v = v[order]                    # 按值排序
    nb = len(v); pen = LAM * np.abs(v[:, None] - v[None])
    em = np.linalg.norm(Jq[:, None] - C[None], axis=2)                  # [T, nb]
    cost = np.full(nb, 1e9); cost[0] = em[0, 0]; BP = np.zeros((len(Jq), nb), int)
    for j in range(1, len(Jq)):
        tr = cost[None, :] + pen; kk = tr.argmin(1); cost = em[j] + tr[np.arange(nb), kk]; BP[j] = kk
    cost[nb - 1] -= 2; s = int(cost.argmin()); path = np.zeros(len(Jq), int); path[-1] = s
    for j in range(len(Jq) - 2, -1, -1): s = BP[j + 1][s]; path[j] = s
    return path, C                                                       # path 索引进(排序后)C

def main():
    global FEAT, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--maxtask", type=int, default=0)
    ap.add_argument("--only_task", type=int, default=-1, help="只跑某个 merged task_index(调试)")
    ap.add_argument("--smoke", action="store_true")
    # [2026-07-19] 跨空间预实验: 允许换特征目录(如 libero_qwen3vl = VLA 自身编码器空间)。
    #   ⚠️ 传入的特征必须与 libero_dinov3base 同为 **stride=2**, 否则第 ~105 行的
    #      state_ep[e][::2][:n] 对齐会静默错位(实测 dinov3 为 107 帧 = 214/2)。
    ap.add_argument("--feat", default=FEAT, help="特征目录(默认 DINOv3-base)")
    ap.add_argument("--out", default=OUT, help="pairs.npz 输出目录")
    args = ap.parse_args()
    FEAT, OUT = args.feat, args.out
    print(f"[cfg] FEAT={FEAT}\n[cfg] OUT={OUT}", flush=True)

    dpar = sorted(glob.glob(f"{ROOT}/data/**/*.parquet", recursive=True))
    ep2task = pd.concat([pd.read_parquet(p, columns=["episode_index", "task_index"]) for p in dpar]) \
        .groupby("episode_index")["task_index"].first().to_dict()
    # proprio: 每 episode state
    want = set(e for e in ep2task if os.path.exists(f"{FEAT}/ep{e}.npz"))
    if args.only_task >= 0:
        want = set(e for e in want if ep2task[e] == args.only_task)
    state_ep = {}
    for p in dpar:
        df = pd.read_parquet(p, columns=["episode_index", "frame_index", "observation.state"])
        df = df[df.episode_index.isin(want)]
        for e, g in df.groupby("episode_index"):
            state_ep[e] = np.stack(g.sort_values("frame_index")["observation.state"].to_numpy())
    from collections import defaultdict
    task_eps = defaultdict(list)
    for e in want: task_eps[ep2task[e]].append(e)
    tasks = sorted(task_eps)
    if args.only_task >= 0: tasks = [args.only_task]
    if args.maxtask: tasks = tasks[:args.maxtask]
    print(f"[load] {len(want)} eps, {len(tasks)} tasks", flush=True)

    cur_ep, cur_fi, tgt_fi, cur_ms, pair_task = [], [], [], [], []
    Ms = []; tailfracs = []
    for tk in tasks:
        teps = [e for e in task_eps[tk] if len(state_ep.get(e, [])) > 0]
        if len(teps) < 5: continue
        gd = {e: np.load(f"{FEAT}/ep{e}.npz")["grid"].astype(np.float32).mean(1) for e in teps}
        # 特征: img128 ⊕ proprio, 1:1
        IMG = np.concatenate([gd[e] for e in teps]); n_per = [(e, len(gd[e])) for e in teps]
        pca = PCA(128, random_state=0).fit(IMG); img128 = l2(pca.transform(IMG))
        ST = []
        for e in teps:
            n = len(gd[e]); st = state_ep[e][::2][:n]
            if len(st) < n: st = np.concatenate([st, np.repeat(st[-1:], n - len(st), 0)])
            ST.append(st)
        ST = np.concatenate(ST).astype(np.float32)
        SMU, SSD = ST.mean(0), ST.std(0) + 1e-8
        joint = np.concatenate([img128, l2((ST - SMU) / SSD)], 1); Jn = l2(joint)
        T = np.concatenate([np.linspace(0, 1, len(gd[e])) for e in teps])
        Ev = np.concatenate([np.full(len(gd[e]), e) for e in teps]); NC = len(teps)
        # BayesianGMM 过聚类
        bg = BayesianGaussianMixture(n_components=40, covariance_type="diag", weight_concentration_prior=1e-2,
                                     n_init=1, max_iter=150, random_state=0).fit(Jn)
        labs = bg.predict(Jn)
        cand = np.array([Jn[labs == k].mean(0) for k in range(40) if (labs == k).sum() >= 20], np.float32)
        if len(cand) == 0: continue
        assign = np.linalg.norm(Jn[:, None] - cand[None], axis=2).argmin(1)
        # per-candidate mode-split + per-mode coverage
        targets = []
        for ki in range(len(cand)):
            mk = assign == ki
            if mk.sum() < 20: continue
            for mv, sub in mode_split(T[mk]):
                cov = len(set(Ev[mk][sub].tolist())) / NC
                if cov >= MIN_COV: targets.append((float(np.median(T[mk][sub])), cand[ki]))
        if not targets: continue
        targets.sort(key=lambda t: t[0])
        vals = np.array([t[0] for t in targets]); Ctgt = np.array([t[1] for t in targets], np.float32)
        Ms.append(len(vals))
        # 端点锚(joint 空间, 全task首/末3帧均值)
        s_anchor = np.mean([Jn[np.cumsum([0]+[len(gd[e]) for e in teps])[i]:][:3].mean(0) for i, e in enumerate(teps)], 0)
        e_anchor = np.mean([Jn[np.cumsum([len(gd[e]) for e in teps])[i]-3:np.cumsum([len(gd[e]) for e in teps])[i]].mean(0) for i, e in enumerate(teps)], 0)
        s_anchor = l2(s_anchor); e_anchor = l2(e_anchor)
        # 逐 episode: 双锚 Viterbi 分段 → next-seg medoid + self-loop
        ptr = 0
        for (e, n) in n_per:
            Jq = Jn[ptr:ptr + n]; ptr += n
            path, Csorted = dual_anchor_states(Jq, Ctgt, vals, s_anchor, e_anchor)
            ch = np.where(np.diff(path) != 0)[0] + 1
            st = np.concatenate([[0], ch]); en = np.concatenate([ch, [n]])
            seg_med = []; seg_state = []
            for s0, e0 in zip(st, en):
                sidx = path[s0]
                med = s0 + int(np.linalg.norm(Jq[s0:e0] - Csorted[sidx], axis=1).argmin())  # 段内离质心最近帧
                seg_med.append(med); seg_state.append(sidx)
            for i in range(len(seg_state) - 1):                          # 每段每帧 → 下一段 medoid
                for fpos in range(st[i], en[i]):
                    cur_ep.append(e); cur_fi.append(fpos); tgt_fi.append(seg_med[i + 1]); cur_ms.append(seg_state[i]); pair_task.append(tk)
            li = len(seg_state) - 1                                       # 末段 self-loop → 自身 medoid
            for fpos in range(st[li], en[li]):
                cur_ep.append(e); cur_fi.append(fpos); tgt_fi.append(seg_med[li]); cur_ms.append(seg_state[li]); pair_task.append(tk)
            tailfracs.append((n - st[li]) / n)
        print(f"  task{tk}: {len(teps)}ep M={len(vals)} 末段占比中位(截至)~{np.median(tailfracs):.2f}", flush=True)

    cur_ep = np.array(cur_ep); cur_fi = np.array(cur_fi); tgt_fi = np.array(tgt_fi)
    cur_ms = np.array(cur_ms); pair_task = np.array(pair_task)
    print(f"[pairs] {len(cur_ep)} 对, {len(set(pair_task.tolist()))} 任务, per-task M 中位={int(np.median(Ms))}, 末段占比中位={np.median(tailfracs):.3f}", flush=True)
    if args.smoke:
        print("[smoke] 不落盘", flush=True); return
    os.makedirs(OUT, exist_ok=True)
    np.savez(f"{OUT}/pairs.npz", cur_ep=cur_ep, cur_fi=cur_fi, tgt_fi=tgt_fi, cur_ms=cur_ms, pair_task=pair_task)
    print(f"[save] {OUT}/pairs.npz", flush=True); print("DONE", flush=True)

if __name__ == "__main__":
    main()
