#!/usr/bin/env python
"""跨特征空间的 milestone 边界重合度分析(DINOv3 vs Qwen3-VL 视觉塔)。

回答: "世界模型换到 VLA 自身特征空间后, CRAVE 发现的子目标边界是否改变?"
  · 重合度 ≈ 随机基线  ⇒ 空间决定了"什么算子目标", 跨空间迁移是实质改动 → 值得投入完整验证
  · 重合度 ≫ 随机基线  ⇒ 空间选择基本不影响分段 → "空间隔离"痛点应降级/删除

★ 公平性前提(必须满足, 否则结论不可归因):
  两侧 pairs.npz 必须由**同一批 episode、同一 stride(=2)、同一 CRAVE 脚本**产出。
  仅特征来源不同。本脚本会硬校验 episode 集合一致, 不一致直接报错退出。

★ 随机基线的必要性: 边界较密时, 两组无关边界也会有可观的偶然匹配率。
  故对每个 episode 做边界位置置换(保持边界个数与 episode 长度), 给出 chance-level。

用法: python compare_milestone_spaces.py --a <dinoPairs>/pairs.npz --b <qwenPairs>/pairs.npz
"""
import argparse, numpy as np
from collections import defaultdict


def boundaries_by_ep(pairs_path):
    """从 pairs.npz 还原每个 episode 的 milestone 边界(cur_ms 变化处的帧索引)。"""
    p = np.load(pairs_path)
    ep, fi, ms = p["cur_ep"], p["cur_fi"], p["cur_ms"]
    out, length = {}, {}
    for e in np.unique(ep):
        m = ep == e
        f, s = fi[m], ms[m]
        o = np.argsort(f); f, s = f[o], s[o]
        b = f[1:][s[1:] != s[:-1]]                  # 段变化点 = 边界
        out[int(e)] = np.unique(b)
        length[int(e)] = int(f.max()) + 1
    return out, length


def match_rate(ba, bb, tol):
    """双向匹配的 F1(贪心一对一, |Δframe| <= tol 视为同一边界)。"""
    if len(ba) == 0 and len(bb) == 0: return 1.0, 0
    if len(ba) == 0 or len(bb) == 0: return 0.0, 0
    used = np.zeros(len(bb), bool); hit = 0
    for x in ba:
        d = np.abs(bb - x); d[used] = 10**9
        j = int(d.argmin())
        if d[j] <= tol: used[j] = True; hit += 1
    prec = hit / len(ba); rec = hit / len(bb)
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    return f1, hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="空间A的 pairs.npz (DINOv3)")
    ap.add_argument("--b", required=True, help="空间B的 pairs.npz (Qwen3-VL)")
    ap.add_argument("--na", default="DINOv3"); ap.add_argument("--nb", default="Qwen3-VL")
    ap.add_argument("--tols", default="1,2,3", help="容差(特征帧)")
    ap.add_argument("--nperm", type=int, default=200)
    a = ap.parse_args()

    A, LA = boundaries_by_ep(a.a)
    B, LB = boundaries_by_ep(a.b)
    ea, eb = set(A), set(B)
    common = sorted(ea & eb)
    print(f"[集合] {a.na}: {len(ea)} ep | {a.nb}: {len(eb)} ep | 交集: {len(common)}")
    if len(common) == 0:
        raise SystemExit("❌ 无公共 episode, 无法比较")
    only = (ea ^ eb)
    if only:
        print(f"⚠️  非对称 episode {len(only)} 个 → 两侧不是同一批数据, **结论不可归因于特征空间**。")
        print(f"    仅在一侧出现的前 10 个: {sorted(only)[:10]}")

    # 段数对比
    ma = np.array([len(A[e]) + 1 for e in common], float)
    mb = np.array([len(B[e]) + 1 for e in common], float)
    print(f"\n[段数 M/episode] {a.na}: 均值 {ma.mean():.2f} 中位 {np.median(ma):.0f} | "
          f"{a.nb}: 均值 {mb.mean():.2f} 中位 {np.median(mb):.0f}")

    rng = np.random.default_rng(0)
    print(f"\n{'容差':>4} | {'实测F1':>8} | {'随机基线':>8} | {'提升':>7} | 判读")
    print("-" * 60)
    for tol in [int(x) for x in a.tols.split(",")]:
        f1s, chance = [], []
        for e in common:
            f1, _ = match_rate(A[e], B[e], tol); f1s.append(f1)
            n = max(LA[e], LB[e])
            cs = []
            for _ in range(a.nperm):
                rb = rng.choice(np.arange(1, n), size=min(len(B[e]), max(1, n - 1)), replace=False)
                cs.append(match_rate(A[e], np.sort(rb), tol)[0])
            chance.append(float(np.mean(cs)))
        f1m, chm = float(np.mean(f1s)), float(np.mean(chance))
        ratio = f1m / chm if chm > 1e-9 else float("inf")
        verdict = ("重合度高 → 空间不重要" if ratio >= 2.0 and f1m >= 0.5
                   else "接近随机 → 空间决定分段" if ratio < 1.3
                   else "中等")
        print(f"{tol:>4} | {f1m:>8.3f} | {chm:>8.3f} | {ratio:>6.2f}× | {verdict}")

    print("\n[判据] F1 ≥ 0.5 且 ≥2× 随机 ⇒ 两空间发现的是同一批子目标, 跨空间迁移收益存疑;")
    print("       F1 < 1.3× 随机     ⇒ 两空间发现的子目标基本无关, 空间选择是实质设计变量。")


if __name__ == "__main__":
    main()
