#!/usr/bin/env python
"""受控对比: 单预测器(直接回归目标 latent) vs 预测器+生成器 —— 证明为何最终用后者。

回答用户问题: "为什么不用单预测器根据当前帧直接预测 milestone+1 代表帧 latent,
而用 预测器+生成器?" 历史(v1 UnifiedLMWM → v2 twomodel)只有失效诊断,无干净同空间 A/B。
本脚本在**同一 So400m(pi05) 空间、同一 pairs、同一 held-out 口径**下补这张表。

三臂(容量匹配: 都用同一 conv backbone hid=512 nblk=4):
  A  单模型直接回归      : f(g_t) -> ĝ, loss = smooth_l1(ĝ, g_f)         [= v1 风格, 无 code 无 lift]
  A+ 单模型 + lift       : 同 A, 加反持久项 lift·relu(cos(ĝ,g_t)-cos(ĝ,g_f))  [隔离: 光加 lift 够吗?]
  B  预测器 + 生成器      : InverseEnc teacher + MilestoneGenerator(code) + MDN 预测器 + lift  [当前架构]

推理口径统一(都只看 g_t, 不看未来):
  A/A+ : ĝ = f(g_t)
  B    : ĝ = gen(g_t, predm.deploy_mean(gist_t))     [MDN 取 mode, 不看未来]
  另报 B-oracle = gen(g_t, teacher_z)  [看未来, 上界参照]

判据: recon_cos(ĝ,g_f) [越高越好] · persist=cos(g_t,g_f) [基线] · lift=recon−persist
      ★ 持久坍缩证据: copy_cos=cos(ĝ,g_t) [越接近 persist 说明越"原地不动"]
用法: CUDA_VISIBLE_DEVICES=1 python exp_single_vs_twomodel.py --steps 4000
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from p1_train_lmwm_libero import InverseEnc, MilestoneGenerator, MilestonePredictorGrid, cosr, load_grid  # noqa: E402
import p1_train_lmwm_libero as B  # noqa: E402


class DirectRegressor(nn.Module):
    """单模型: g_t -> ĝ 直接回归。与 MilestoneGenerator 同 backbone, 但无 code 调制(纯残差 conv)。

    容量对齐: proj + nblk 个残差块 + out, 与生成器逐层同形, 仅去掉 AdaLN 的 mod 头
    (少 code_dim*nblk*3*hid 个参数, 生成器那部分本就 zero-init 起步)。
    """
    def __init__(self, din, hid=512, nblk=4):
        super().__init__()
        self.proj = nn.Conv2d(din, hid, 3, 1, 1)
        self.gn = nn.ModuleList([nn.GroupNorm(8, hid) for _ in range(nblk)])
        self.blk = nn.ModuleList([nn.Sequential(nn.Conv2d(hid, hid, 3, 1, 1), nn.GELU(),
                                                nn.Conv2d(hid, hid, 3, 1, 1)) for _ in range(nblk)])
        self.out = nn.Conv2d(hid, din, 3, 1, 1)
        self.nblk = nblk

    def forward(self, gt):
        h = self.proj(gt)
        for i in range(self.nblk):
            h = h + self.blk[i](self.gn[i](h))
        return self.out(h)


def cn(a, b):
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8)


def mdn_mode(prd, grid):
    """MilestonePredictorGrid 无 deploy_mean: 取最高权重分量的 mu 作部署 code(不看未来)。"""
    logit, mu, _ = prd(grid)                        # grid 输入(conv 内部下采样)
    k = logit.argmax(1)
    return mu[torch.arange(len(mu), device=mu.device), k]


def mdn_component_means(prd, grid):
    """返回 K 个分量均值 [K, B, C] —— 多模态分支提议(best-of-K 用, 不看未来)。"""
    _, mu, _ = prd(grid)                            # mu: [B, K, C]
    return mu.permute(1, 0, 2)                       # [K, B, C]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default=str(HERE.parents[1] / "lmwm/data/libero_so400m_grid"))
    ap.add_argument("--pairs", default=str(HERE.parents[1] / "lmwm/data/libero_so400m_rvalley/pairs.npz"))
    ap.add_argument("--din", type=int, default=1152)
    ap.add_argument("--pgrid", type=int, default=16)
    ap.add_argument("--code_dim", type=int, default=32)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--lift_w", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE.parents[1] / "lmwm/outputs/exp_single_vs_twomodel.json"))
    a = ap.parse_args()
    dev = "cuda"
    B.FEAT, B.DIN, B.PGRID = a.feat, a.din, a.pgrid
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    import glob
    import os
    have = {int(os.path.basename(f)[2:-4]) for f in glob.glob(f"{a.feat}/ep*.npz")}
    P = np.load(a.pairs)
    ce, cf, tf = P["cur_ep"], P["cur_fi"], P["tgt_fi"]
    keep = np.isin(ce, list(have))
    ce, cf, tf = ce[keep], cf[keep], tf[keep]
    # held-out by episode 80/20
    eps = np.unique(ce); rng = np.random.default_rng(a.seed); rng.shuffle(eps)
    val_ep = set(eps[:max(1, len(eps) // 5)].tolist())
    tr = np.array([i for i in range(len(ce)) if ce[i] not in val_ep])
    va = np.array([i for i in range(len(ce)) if ce[i] in val_ep])
    print(f"[data] {len(have)} ep grid; pairs tr={len(tr)} va={len(va)} (val {len(val_ep)} ep)", flush=True)

    cache = {}

    def fetch(idx):
        gt = np.stack([load_grid(cache, int(ce[i]))[int(cf[i])] for i in idx])
        gf = np.stack([load_grid(cache, int(ce[i]))[int(tf[i])] for i in idx])
        return torch.from_numpy(gt).to(dev), torch.from_numpy(gf).to(dev)

    def train_direct(with_lift):
        m = DirectRegressor(a.din).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=2e-4, weight_decay=1e-5)
        for s in range(a.steps):
            gt, gf = fetch(tr[np.random.randint(0, len(tr), a.bs)])
            gh = m(gt)
            loss = F.smooth_l1_loss(gh, gf)
            if with_lift:
                loss = loss + a.lift_w * torch.relu(cosr(gh.flatten(1), gt.flatten(1))
                                                    - cosr(gh.flatten(1), gf.flatten(1))).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval(); return m

    def train_twomodel():
        inv = InverseEnc(a.din, a.code_dim).to(dev)
        gen = MilestoneGenerator(a.din, a.code_dim).to(dev)
        prd = MilestonePredictorGrid(a.din, a.code_dim, a.K).to(dev)
        o1 = torch.optim.AdamW(list(inv.parameters()) + list(gen.parameters()), lr=2e-4, weight_decay=1e-5)
        o2 = torch.optim.AdamW(prd.parameters(), lr=2e-4, weight_decay=1e-5)
        for s in range(a.steps):
            gt, gf = fetch(tr[np.random.randint(0, len(tr), a.bs)])
            z = inv(gt, gf); gh = gen(gt, z)
            lift = torch.relu(cosr(gh.flatten(1), gt.flatten(1)) - cosr(gh.flatten(1), gf.flatten(1))).mean()
            l1 = F.smooth_l1_loss(gh, gf) + a.lift_w * lift
            o1.zero_grad(); l1.backward(retain_graph=True); o1.step()
            l2 = prd.nll(gt, z.detach()); o2.zero_grad(); l2.backward(); o2.step()   # grid 输入
        inv.eval(); gen.eval(); prd.eval(); return inv, gen, prd

    print("[train] A 单模型直接回归 ...", flush=True); mA = train_direct(False)
    print("[train] A+ 单模型+lift ...", flush=True); mAL = train_direct(True)
    print("[train] B 预测器+生成器 ...", flush=True); inv, gen, prd = train_twomodel()

    # ---- held-out eval, 统一只看 g_t ----
    recon = {k: [] for k in ["A", "A+", "B_deploy", "B_bestofK", "B_oracle"]}
    copyc = {k: [] for k in recon}
    persist = []
    with torch.no_grad():
        for s in range(0, len(va), 128):
            gt, gf = fetch(va[s:s + 128])
            gtn, gfn = gt.cpu().numpy().reshape(len(gt), -1), gf.cpu().numpy().reshape(len(gt), -1)
            persist.append(cn(gtn, gfn))
            # best-of-K: K 个分量均值各生成一张, 逐样本取 recon 最高(多模态提议命中)
            comp = mdn_component_means(prd, gt)                     # [K,B,C]
            bk_r = None
            for k in range(comp.shape[0]):
                gh = gen(gt, comp[k]); ghn = gh.cpu().numpy().reshape(len(gh), -1)
                r = cn(ghn, gfn); c = cn(ghn, gtn)
                if bk_r is None:
                    bk_r, bk_c = r.copy(), c.copy()
                else:
                    take = r > bk_r; bk_r[take] = r[take]; bk_c[take] = c[take]
            recon["B_bestofK"].append(bk_r); copyc["B_bestofK"].append(bk_c)
            for name, gh in [("A", mA(gt)), ("A+", mAL(gt)),
                             ("B_deploy", gen(gt, mdn_mode(prd, gt))),
                             ("B_oracle", gen(gt, inv(gt, gf)))]:
                ghn = gh.cpu().numpy().reshape(len(gh), -1)
                recon[name].append(cn(ghn, gfn)); copyc[name].append(cn(ghn, gtn))
    pm = float(np.concatenate(persist).mean())
    res = {"persist": round(pm, 4), "n_val": int(len(va)), "n_ep_grid": len(have),
           "steps": a.steps, "arms": {}}
    for k in recon:
        rc = float(np.concatenate(recon[k]).mean()); cc = float(np.concatenate(copyc[k]).mean())
        res["arms"][k] = {"recon_cos": round(rc, 4), "lift": round(rc - pm, 4), "copy_cos": round(cc, 4)}
    print("\n" + "=" * 68)
    print(f"{'臂':<12}{'recon_cos':>11}{'lift':>9}{'copy_cos':>10}  说明")
    print("-" * 68)
    lab = {"A": "单模型直接回归", "A+": "单模型+lift", "B_deploy": "预测器+生成器(mode)",
           "B_bestofK": "两模型(best-of-K多模态)", "B_oracle": "两模型(oracle上界)"}
    for k in ["A", "A+", "B_deploy", "B_bestofK", "B_oracle"]:
        d = res["arms"][k]
        print(f"{k:<12}{d['recon_cos']:>11.4f}{d['lift']:>+9.4f}{d['copy_cos']:>10.4f}  {lab[k]}")
    print("-" * 68)
    print(f"{'persist':<12}{pm:>11.4f}{'—':>9}{'1.0000':>10}  当前帧==目标(原地不动基线)")
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"\n[save] {a.out}", flush=True)


if __name__ == "__main__":
    main()
