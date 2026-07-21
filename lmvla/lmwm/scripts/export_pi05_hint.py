#!/usr/bin/env python
"""export_pi05_hint.py — pi05 × LMWM 两侧唯一接口产物 (PLAN_pi05_lmwm_sameencoder §2.3).

把已训好的 LMWM 模型对**逐帧 patch-grid 特征**跑推理, 产出子目标 hint 向量 ĝ_next,
存成按 (suite, episode_index, frame_index) 索引的 npz, 供 pi05 训练 obs.lmwm_hint 消费。

## 本文件 = 推理核 (dataset-independent)
输入 = 逐帧 grid 特征 G[N, DIN, 16, 16] (DINOv3-base 768D / So400m 1152D);
输出 = hint[N, DIN] (单发) 或 [N, K, DIN] (best-of-K)。LMWM 模型在特征空间工作, 与数据集来源无关。

## ⚠️ 对齐纪律 (DESIGN §3, agent 勘查):
LMWM ckpt 训练用的特征来自 v3.0 libero_merged_no_noops_20hz; 而 pi05 训练走 v2.1 LIBERO_fastwam。
两者 episode_index 体系不同 → **不能跨库互查**。**正确做法 (Option A)**: 特征也从 **pi05 训练用的 v2.1 数据**
重抽 (上游 stage: extract_v2p1_grid.py, 用 pyav 解码 + DINOv3/So400m), 使 (suite,ep,frame) 天然对齐。
本脚本消费那份 v2.1 grid 特征缓存 (--feat-root), 不做跨库映射。

## 用法
  # (A1 DINOv3)  载 v2.1 grid 缓存 → hint
  python export_pi05_hint.py --ckpt .../lmwm_libero_rvalley/lmwm.pt \
      --feat-root .../data/pi05_feat/libero_v2p1_dinov3base --out .../data/pi05_hint/libero_dino --K 1
  # (A2 So400m)  --ckpt .../lmwm_libero_so400m/lmwm.pt --feat-root .../libero_v2p1_so400m_grid ...

特征缓存布局 (由上游 extract 产): <feat-root>/<suite>/ep{E}.npz, key "grid" shape [N, 256, DIN],
可选 key "frame_index" [N] (缺省则用 range(N))。
"""
import os, sys, glob, json, argparse
import numpy as np
import torch

# LMWM 模型类 (自包含定义在 p1_train_lmwm_libero.py)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p1_train_lmwm_libero import MilestoneGenerator, MilestonePredictorGrid  # noqa: E402

PGRID = 16  # 16×16 = 256 patch tokens


def load_lmwm(ckpt_path, device="cpu"):
    """载 LMWM ckpt → (gen, prd, din, code_dim). 只需 gen (生成 ĝ_next) + prd (MDN deploy 头)。"""
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    din, cd = int(ck["din"]), int(ck["code_dim"])
    K_prd = ck["prd"]["pi.weight"].shape[0]  # MDN 分量数 (训练时 K, 通常 4)
    gen = MilestoneGenerator(din, cd).to(device).eval()
    gen.load_state_dict(ck["gen"])
    prd = MilestonePredictorGrid(din, cd, K_prd).to(device).eval()
    prd.load_state_dict(ck["prd"])
    return gen, prd, din, cd, K_prd


@torch.no_grad()
def grid_to_hint(gen, prd, G, K=1):
    """G[B, DIN, 16, 16] → hint. K=1: 单发 (argmax-π code); K>1: best-of-K (前 K 个 MDN 分量).
    返回 [B, DIN] (K=1) 或 [B, K, DIN] (K>1)。hint = 池化后的 ĝ_next grid。"""
    logit, mu, ls = prd(G)  # logit[B,Kp], mu[B,Kp,C]
    if K == 1:
        code = mu[torch.arange(len(G)), logit.argmax(1)]      # [B,C] argmax-π 分量均值
        gnext = gen(G, code)                                   # [B,DIN,16,16]
        return gnext.mean((2, 3))                              # [B,DIN]
    # best-of-K: 取 π 最大的前 K 个分量各生成一次
    order = logit.argsort(dim=1, descending=True)[:, :K]       # [B,K]
    hints = []
    for k in range(K):
        code_k = mu[torch.arange(len(G)), order[:, k]]         # [B,C]
        hints.append(gen(G, code_k).mean((2, 3)))              # [B,DIN]
    return torch.stack(hints, dim=1)                           # [B,K,DIN]


def _load_grid_npz(path, din):
    """<feat>/ep{E}.npz key 'grid' [N,256,DIN] → G[N,DIN,16,16] + frame_index[N]."""
    z = np.load(path)
    g = z["grid"].astype(np.float32)                           # [N,256,DIN]
    assert g.shape[1] == PGRID * PGRID and g.shape[2] == din, \
        f"grid shape {g.shape} != [N,{PGRID*PGRID},{din}] (ckpt din vs feature din 不匹配?)"
    G = g.reshape(len(g), PGRID, PGRID, din).transpose(0, 3, 1, 2)  # [N,DIN,16,16]
    fi = z["frame_index"].astype(np.int64) if "frame_index" in z.files else np.arange(len(g), dtype=np.int64)
    return G, fi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="LMWM lmwm.pt (含 gen/prd/din/code_dim)")
    ap.add_argument("--feat-root", required=True, help="v2.1 grid 特征根: <root>/<suite>/ep{E}.npz")
    ap.add_argument("--out", required=True, help="输出目录, 存 hint.npz + _env.json")
    ap.add_argument("--K", type=int, default=1, help="1=单发; >1=best-of-K")
    ap.add_argument("--suites", nargs="*", default=None, help="子集 suite (缺省=feat-root 下全部子目录)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    gen, prd, din, cd, Kp = load_lmwm(args.ckpt, args.device)
    print(f"[lmwm] loaded din={din} code_dim={cd} MDN-K={Kp} | export K={args.K} device={args.device}")

    suites = args.suites or sorted(
        d for d in os.listdir(args.feat_root) if os.path.isdir(os.path.join(args.feat_root, d))
    )
    os.makedirs(args.out, exist_ok=True)
    # 索引数组 (与 hint 平行): suite / episode_index / frame_index
    all_suite, all_ep, all_fi, all_hint = [], [], [], []
    for suite in suites:
        eps = sorted(glob.glob(os.path.join(args.feat_root, suite, "ep*.npz")),
                     key=lambda p: int(os.path.basename(p)[2:-4]))
        for ep_path in eps:
            E = int(os.path.basename(ep_path)[2:-4])
            G, fi = _load_grid_npz(ep_path, din)
            hs = []
            for i in range(0, len(G), args.batch):
                Gb = torch.from_numpy(G[i:i + args.batch]).to(args.device)
                hs.append(grid_to_hint(gen, prd, Gb, args.K).cpu().numpy())
            h = np.concatenate(hs, 0).astype(np.float16)       # [N,DIN] 或 [N,K,DIN]
            all_suite += [suite] * len(h); all_ep += [E] * len(h)
            all_fi.append(fi); all_hint.append(h)
        print(f"[{suite}] {len(eps)} eps done")

    hint = np.concatenate(all_hint, 0)
    out_npz = os.path.join(args.out, "hint.npz")
    np.savez_compressed(
        out_npz,
        suite=np.array(all_suite), episode_index=np.array(all_ep, np.int64),
        frame_index=np.concatenate(all_fi).astype(np.int64), hint=hint,
    )
    env = {"ckpt": args.ckpt, "feat_root": args.feat_root, "din": din, "code_dim": cd,
           "K": args.K, "n_frames": int(len(hint)), "hint_shape": list(hint.shape),
           "torch": torch.__version__, "note": "hint = pooled ĝ_next; index=(suite,episode_index,frame_index)"}
    json.dump(env, open(os.path.join(args.out, "_env.json"), "w"), indent=2, ensure_ascii=False)
    print(f"[done] {len(hint)} frames → {out_npz}  hint{hint.shape} din={din}")


if __name__ == "__main__":
    main()
