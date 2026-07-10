#!/usr/bin/env python
"""诊断:成功数据下 greedy ≠ max-product,分歧有两个成分(混叠 + 行为熵)。

在 LMWM 循环转移图(kai0_base DINOv3-H,37 milestone,3055 成功 ep)上量化:
① greedy 边概率 ~0.21(非确定性前向链)+ 每态 ~14.5 有效分支;
② 50% 的 milestone 转移是横向/后退(尽管数据 100% 成功);
③ greedy vs max-product 仅 43% 一致(16/37);
④ rollout 的 "milestone +1"(实际下一态 / 主导前进后继)更贴合 greedy;
⑤ 两成分分解:全后继 14.5 分支 →(删后退)→ 仅前进 7.8 分支;后者=行为熵(不可修);
⑥ 反例:相位唯一 + 全成功,greedy≠max-product(绕路稀释)。

渲染 6 联诊断图 → crave/docs/visualization/milestone_policy/greedy_vs_maxprod_diag.png
Run: /home/tim/miniconda3/envs/srpo/bin/python crave/experiments/greedy_vs_maxprod_diag.py
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from crave.render import setup_mpl

plt = setup_mpl()

REPO = Path(os.environ.get("REPO", "/home/tim/workspace/deepdive_kai0"))
GRAPH = REPO / "lmwm/data/recurrence_graphs/kai0base_dinov3h/recurrence_graph.npz"
OUT = REPO / "crave/docs/visualization/milestone_policy"
OUT.mkdir(parents=True, exist_ok=True)


def eff_branches_mean(rows):
    e = []
    for r in rows:
        s = r.sum()
        if s <= 0:
            continue
        p = r[r > 0] / s
        e.append(np.exp(-(p * np.log(p)).sum()))
    return float(np.mean(e)), np.array(e)


def main():
    z = np.load(GRAPH)
    C = z["transition_counts"].astype(np.float64)
    P = z["transition_probs"].astype(np.float64)
    g = z["greedy_next"]
    m = z["max_product_next"]
    pord = z["pord"].astype(np.float64)
    n = len(g)
    tot = C.sum()

    gprob = P[np.arange(n), g]
    ent = np.array([-(P[i][P[i] > 0] * np.log(P[i][P[i] > 0])).sum() for i in range(n)])
    allb, _ = eff_branches_mean(C)
    fwd = pord[None, :] > pord[:, None]
    Cf = C * fwd
    fwdb, fwd_e = eff_branches_mean(Cf)
    fwd_mass = Cf.sum() / tot
    agree = (g == m)
    hit_g = sum(C[i, g[i]] for i in range(n)) / tot
    hit_m = sum(C[i, m[i]] for i in range(n)) / tot
    fmode = np.array([Cf[i].argmax() if Cf[i].sum() > 0 else -1 for i in range(n)])
    fm_g = np.mean([g[i] == fmode[i] for i in range(n) if fmode[i] >= 0])
    fm_m = np.mean([m[i] == fmode[i] for i in range(n) if fmode[i] >= 0])

    fig, axs = plt.subplots(2, 3, figsize=(18.5, 10))
    A, B, E, Cc, D, F = axs.ravel()

    # ① greedy 边概率直方图
    A.hist(gprob, bins=np.linspace(0, 1, 21), color="#5b8def", edgecolor="#204080")
    A.axvline(gprob.mean(), color="#d62728", lw=2, label=f"均值 {gprob.mean():.2f}")
    A.axvline(1.0, color="#2ca02c", lw=2, ls="--", label="确定性链应 ≈1.0")
    A.set_xlim(0, 1.02)
    A.set_xlabel("greedy 边概率  max_j P(next=j | cur=i)")
    A.set_ylabel("milestone 数")
    A.set_title(f"① 不是确定性前向链\n每态众数仅 ~0.21,行熵 {ent.mean():.2f} nats ≈ {allb:.0f} 分支", fontsize=11)
    A.legend(fontsize=9)
    A.grid(alpha=.25)

    # ② 前进 vs 后退 转移质量
    B.bar(["前进 (Pord↑)", "横向 / 后退"], [fwd_mass, 1 - fwd_mass],
          color=["#2ca02c", "#d62728"], edgecolor="#333")
    for x, v in enumerate([fwd_mass, 1 - fwd_mass]):
        B.text(x, v + .01, f"{v:.0%}", ha="center", fontsize=13, fontweight="bold")
    B.set_ylim(0, 0.65)
    B.set_ylabel("占全部 milestone 转移质量")
    B.set_title("② 数据 100% 成功,仍有一半转移横向/后退\n= 视觉簇混叠(同 milestone 现于多阶段)", fontsize=11)
    B.grid(alpha=.25, axis="y")

    # ⑤ 两成分分解:全后继 vs 仅前进 有效分支
    E.bar(["全部后继", "仅前进方向"], [allb, fwdb], color=["#9467bd", "#ff7f0e"], edgecolor="#333")
    for x, v in enumerate([allb, fwdb]):
        E.text(x, v + .2, f"{v:.1f}", ha="center", fontsize=13, fontweight="bold")
    E.annotate("", xy=(1, fwdb + .5), xytext=(0, allb - .5),
               arrowprops=dict(arrowstyle="->", color="#666", lw=1.5))
    E.text(0.5, (allb + fwdb) / 2 + 1.2, "删掉后退边\n= 消混叠成分(可修)", ha="center", fontsize=9, color="#444")
    E.set_ylim(0, allb * 1.18)
    E.set_ylabel("平均有效分支数")
    E.set_title("⑤ 两成分:混叠(可修)+ 行为熵(不可修)\n剥掉所有后退边,前进方向仍有 7.8 分支 = 操作员多模态/次优", fontsize=11)
    E.grid(alpha=.25, axis="y")

    # ③ greedy vs max-product 目标 Pord
    Cc.plot([0, 1], [0, 1], color="#999", ls="--", lw=1, label="停留 (y=x)")
    Cc.scatter(pord, pord[g], s=55, color="#5b8def", label="greedy 下一态", zorder=3)
    Cc.scatter(pord, pord[m], s=55, marker="x", color="#ff7f0e", label="max-product 下一态", zorder=3, linewidths=2)
    for i in np.where(~agree)[0]:
        Cc.plot([pord[i], pord[i]], [pord[g[i]], pord[m[i]]], color="#cccccc", lw=.8, zorder=1)
    Cc.set_xlabel("当前 milestone 的 progress (Pord)")
    Cc.set_ylabel("预测下一 milestone 的 Pord")
    Cc.set_title(f"③ greedy vs max-product 仅 {agree.sum()}/{n} = {agree.mean():.0%} 一致\n灰线=分歧态({(~agree).sum()} 个);对角线上方=前进", fontsize=11)
    Cc.legend(fontsize=9, loc="upper left")
    Cc.grid(alpha=.25)

    # ④ milestone +1 更贴合谁
    x = np.arange(2)
    w = 0.36
    D.bar(x - w / 2, [hit_g, fm_g], w, color="#5b8def", edgecolor="#204080", label="greedy")
    D.bar(x + w / 2, [hit_m, fm_m], w, color="#ff7f0e", edgecolor="#803f00", label="max-product")
    for xi, (a_, b_) in enumerate([(hit_g, hit_m), (fm_g, fm_m)]):
        D.text(xi - w / 2, a_ + .008, f"{a_:.2f}", ha="center", fontsize=11, fontweight="bold")
        D.text(xi + w / 2, b_ + .008, f"{b_:.2f}", ha="center", fontsize=11, fontweight="bold")
    D.set_xticks(x)
    D.set_xticklabels(["匹配实际下一态\n(计数加权)", "匹配主导前进后继\n(milestone +1)"])
    D.set_ylim(0, 0.62)
    D.set_ylabel("匹配率")
    D.set_title("④ rollout 的 “milestone +1” 更贴合 greedy\ngreedy = 下一态最大似然众数;max-product = 奔终点规划步", fontsize=11)
    D.legend(fontsize=10)
    D.grid(alpha=.25, axis="y")

    # ⑥ 反例:相位唯一 + 全成功,greedy≠max-product
    F.axis("off")
    F.set_xlim(0, 4.2)
    F.set_ylim(-0.15, 1.05)
    nodes = {"A": (0.25, 0.5), "B": (1.5, 0.85), "G": (3.6, 0.5),
             "C": (1.5, 0.18), "D1": (2.6, 0.30), "D2": (2.6, 0.03)}
    for k, (xx, yy) in nodes.items():
        F.add_patch(plt.Circle((xx, yy), 0.11, fc="#eef", ec="#333", zorder=3))
        F.text(xx, yy, k, ha="center", va="center", fontsize=10, fontweight="bold", zorder=4)

    def edge(a, b, lab, color="#999", lw=1.4):
        x0, y0 = nodes[a]; x1, y1 = nodes[b]
        F.annotate("", xy=(x1, y1), xytext=(x0, y0),
                   arrowprops=dict(arrowstyle="->", color=color, lw=lw, shrinkA=13, shrinkB=13), zorder=2)
        F.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.045, lab, ha="center", fontsize=8.5, color=color)
    # greedy first step = A->C (red, most common); max-product path = A->B->G (green)
    edge("A", "B", "0.45", color="#2ca02c", lw=2.6)
    edge("B", "G", "1.0", color="#2ca02c", lw=2.6)
    edge("A", "C", "0.55", color="#d62728", lw=2.6)
    edge("C", "D1", "0.6", color="#d62728")
    edge("C", "D2", "0.4", color="#d62728")
    edge("D1", "G", "1.0", color="#bbb")
    edge("D2", "G", "1.0", color="#bbb")
    F.text(2.1, 1.0, "反例:相位唯一 + 全部成功", ha="center", fontsize=11, fontweight="bold")
    F.text(2.1, -0.1,
           "greedy(A)=C (最常见第一步 0.55)  ≠  max-product(A)=B\n"
           "经B整路=0.45 > 经C最优路=0.55·0.6=0.33(绕路被下游分支稀释)",
           ha="center", fontsize=8.8, color="#333")

    fig.suptitle("成功数据下 greedy ≠ max-product:混叠(可修) + 行为熵(不可修)两成分 —— kai0_base DINOv3-H(37 milestone / 3055 成功 ep)",
                 fontsize=13.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUT / "greedy_vs_maxprod_diag.png"
    fig.savefig(out, dpi=125)
    print(f"SAVED {out}")
    print(f"greedy_prob mean={gprob.mean():.3f} | all_branch={allb:.2f} | fwd_branch={fwdb:.2f} | "
          f"fwd_mass={fwd_mass:.3f} | agree={agree.sum()}/{n} | hit_g={hit_g:.3f} hit_m={hit_m:.3f} | "
          f"fmode_g={fm_g:.3f} fmode_m={fm_m:.3f} | frac_fwd_multi={(fwd_e>=2).mean():.2f}")


if __name__ == "__main__":
    main()
