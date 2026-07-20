#!/usr/bin/env python
"""时序之战: UR-VC 的 τ=0.3 时间带在 变速/长回退 场景失效, 时间无关检索不失效。
唯一变量 = 时间带(τ=0.3 vs 无带), 其余(per-ep 1-NN, 平均匹配帧时间标签)完全同 UR-VC。
场景(在真实 kai0 特征上做受控改造, 奇数位 ep 被改, 偶数位保持原样当"正常示范库"):
  S1 变速:   前 35% 帧每帧×5(开局慢 5 倍) → 归一化时间与真进度错位最高 0.38 > τ
  S2 长回退: 在 75% 处插入 5%-45% 段的复制(模拟失败重做, 40% 长) → 重做帧错位 0.37-0.49 > τ
GT: 改造后逐帧携带原 stage_progress_gt(S2 重做段 GT 真回退)。
判据: 被改 ep 的 corr(med) + 错位>τ 帧上的 mean|ĝ-gt|; 回退帧的回退捕获率(ĝ 相对拼接点下降)。
"""
import glob, os, sys
import numpy as np, pandas as pd

FEAT = "/vePFS/tim/workspace/deepdive_kai0/lmvla/lmwm/data/dino_sub20"
GT   = "/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/kai0_advantage/data/chunk-000"
STRIDE, N_EP, TAU = 20, 150, 0.30

rng = np.random.default_rng(0)
files = sorted(glob.glob(f"{FEAT}/ep*.npz"), key=lambda p: int(os.path.basename(p)[2:-4]))
sel = sorted(rng.choice(len(files), min(N_EP, len(files)), replace=False))
featL, gtL = [], []
for k in sel:
    epi = int(os.path.basename(files[k])[2:-4])
    pq = f"{GT}/episode_{epi:06d}.parquet"
    if not os.path.exists(pq): continue
    g = np.load(files[k])["grid"].astype(np.float32).mean(1)
    g /= (np.linalg.norm(g, axis=1, keepdims=True) + 1e-9)
    df = pd.read_parquet(pq, columns=["stage_progress_gt"])
    idx = np.minimum(np.arange(len(g)) * STRIDE, len(df) - 1)
    featL.append(g); gtL.append(df["stage_progress_gt"].values[idx].astype(np.float32))
ne = len(featL)

def build(scn):
    """返回 feats, gts, wall(归一化时间), mod(是否被改ep), special(逐帧: 错位>τ 或 重做段)"""
    F, T, W, MOD, SP = [], [], [], [], []
    for i,(f,t) in enumerate(zip(featL, gtL)):
        n = len(f)
        if i % 2 == 1 and scn == "S1":       # 变速: 前35%帧 ×5
            k = int(n*0.35); ridx = np.concatenate([np.repeat(np.arange(k),5), np.arange(k,n)])
            f2, t2 = f[ridx], t[ridx]; mod = True
            redo = np.zeros(len(ridx), bool)
        elif i % 2 == 1 and scn == "S2":     # 长回退: 75%处插入 5%-45% 复制段
            a,b,c = int(n*0.05), int(n*0.45), int(n*0.75)
            ridx = np.concatenate([np.arange(c), np.arange(a,b), np.arange(c,n)])
            f2, t2 = f[ridx], t[ridx]; mod = True
            redo = np.zeros(len(ridx), bool); redo[c:c+(b-a)] = True
        else:
            f2, t2, mod, redo = f, t, False, np.zeros(n, bool)
        w = np.arange(len(f2), dtype=np.float32)/max(len(f2)-1,1)
        # special = 错位>τ(用 原时间标签近似真进度: |wall - 原相对位置|)
        orig_pos = (ridx if mod else np.arange(n)).astype(np.float32)/max(n-1,1)
        sp = np.abs(w - orig_pos) > TAU
        F.append(f2); T.append(t2); W.append(w); MOD.append(mod); SP.append(sp | redo)
    return F, T, W, MOD, SP

def urvc(F, W, tau):
    """UR-VC 检索: 每帧对每条其他 ep 在 |Δwall|<=tau 内取 cos 最优 1 帧, 平均其 wall 时间标签"""
    preds = []
    for a in range(ne):
        acc = np.zeros(len(F[a]), np.float64); cnt = np.zeros(len(F[a]), np.int32)
        for b in range(ne):
            if a == b: continue
            C = F[a] @ F[b].T
            M = np.abs(W[a][:, None] - W[b][None, :]) <= tau
            C = np.where(M, C, -2.0); j = C.argmax(1)
            ok = C[np.arange(len(j)), j] > -1.5
            acc[ok] += W[b][j[ok]]; cnt[ok] += 1
        preds.append(np.where(cnt>0, acc/np.maximum(cnt,1), W[a]).astype(np.float32))
    return preds

for scn in ["S1", "S2"]:
    F, T, W, MOD, SP = build(scn)
    rows = {"时间标签(wall)": W, "UR-VC τ=0.3": urvc(F, W, TAU), "无带检索 τ=∞": urvc(F, W, 10.0)}
    name = "S1 变速(前35%慢5x)" if scn=="S1" else "S2 长回退(75%处重做40%)"
    print(f"\n=== {name} | 被改ep={sum(MOD)}/{ne} ===")
    print(f"{'方法':<16} {'corr被改ep':>10} {'err@错位帧':>10}" + (" {:>10}".format("回退捕获率") if scn=="S2" else ""))
    for nm, P in rows.items():
        cs = [np.corrcoef(P[i], T[i])[0,1] for i in range(ne) if MOD[i] and np.std(T[i])>1e-6]
        errs = np.concatenate([np.abs(P[i]-T[i])[SP[i]] for i in range(ne) if SP[i].any()])
        extra = ""
        if scn == "S2":
            # 回退捕获: 重做段内 ĝ 比拼接点前值下降(>0.05)的帧占比
            caps = []
            for i in range(ne):
                if not MOD[i]: continue
                r = np.where(SP[i])[0]
                if len(r)==0: continue
                pre = P[i][r[0]-1]
                caps.append((P[i][r] < pre - 0.05).mean())
            extra = f" {np.mean(caps):>10.1%}"
        print(f"{nm:<16} {np.median(cs):>10.3f} {np.mean(errs):>10.3f}" + extra)
print("\n判读: τ=0.3 在两场景 corr/err 应显著劣于 无带检索; S2 回退捕获率 τ 带应接近 0(带外检索不到真匹配).")
