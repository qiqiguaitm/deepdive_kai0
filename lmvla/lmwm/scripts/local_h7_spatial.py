#!/usr/bin/env python
"""H7: is the LaWM advantage from SPATIAL tokens, or just transformer capacity?

Same LAMEncoder multi-head fed either (A) the real 256 patch tokens, or (B) the
mean-pooled feature broadcast to 256 identical tokens (same params/token-count, NO
spatial variation). If A >> B, the gain is spatial information. Local grid cache subset.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE.parent))
from lmwm.data import split_indices  # noqa: E402
from train_lawm_gf3 import LaWMMulti, build_extra  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="lmwm/data/crave_sequences/kai0base_dinov3h_frame2proto/pairs_next_unique_augin.npz")
    ap.add_argument("--graph_npz", default="lmwm/data/recurrence_graphs/kai0base_dinov3h/recurrence_graph.npz")
    ap.add_argument("--grid_glob", default="lmwm/data/grid_cache/*.npz")
    ap.add_argument("--n_shards", type=int, default=5)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--out", default="lmwm/outputs/local_h7/summary.json", type=Path)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    shards = sorted(glob.glob(args.grid_glob))[:args.n_shards]
    grids, rows = [], []
    for f in shards:
        z = np.load(f); grids.append(z["grid"]); rows.append(z["rows"])
    grid = np.concatenate(grids); rows = np.concatenate(rows)          # rows contiguous 0..M
    M = len(rows); print(f"loaded {M} grids from {len(shards)} shards", flush=True)

    z = np.load(args.pairs)
    proto_tbl = np.load(args.graph_npz)["prototype_table"].astype(np.float32)
    num_m = int(z["future_milestone"].max()) + 1
    extra_all = build_extra(z, proto_tbl)
    y_all = z["future_milestone"].astype(np.int64)
    med_all = z["next_medoid"].astype(np.float32); ok = np.linalg.norm(med_all, axis=1) > 1e-6
    med_all = med_all / (np.linalg.norm(med_all, axis=1, keepdims=True) + 1e-8)
    n = len(z["current_milestone"])
    ti, vi = split_indices(z, n, 0.2, 2026, torch.device("cpu"), "episode")
    inset = np.zeros(n, bool); inset[rows] = True
    tr = np.array([r for r in ti.numpy() if inset[r] and ok[r]])
    va = np.array([r for r in vi.numpy() if inset[r] and ok[r]])
    print(f"train {len(tr)} val {len(va)}", flush=True)

    r2l = -np.ones(n, np.int64); r2l[rows] = np.arange(M)               # global row -> local grid idx
    G = torch.from_numpy(grid)                                          # [M,256,1280] fp16
    Ex = torch.from_numpy(extra_all); Y = torch.from_numpy(y_all); Md = torch.from_numpy(med_all)
    trans = np.load(args.graph_npz)["transition_probs"].astype(np.float64); trans = trans / trans.sum(1, keepdims=True).clip(1e-12)
    prior_v = trans[z["current_milestone"][va].astype(np.int64)]
    yv, mv = y_all[va], med_all[va]

    def fetch(rows_b, broadcast):
        g = G[torch.from_numpy(r2l[rows_b])].to(dev).float()           # [b,256,1280]
        if broadcast:
            g = g.mean(1, keepdim=True).expand(-1, 256, -1).contiguous()
        return g

    def run(broadcast):
        torch.manual_seed(0)
        model = LaWMMulti(num_m, ctx=768, layers=6).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 400))
        for s in range(args.steps):
            bi = tr[np.random.randint(0, len(tr), 128)]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg, pr = model(fetch(bi, broadcast), Ex[bi].to(dev))
                per = F.cross_entropy(lg, Y[bi].to(dev), reduction="none")
                k = max(1, int(0.1 * len(per)))
                loss = 0.5 * per.mean() + 0.5 * torch.topk(per, k).values.mean() + 5.0 * (1 - (pr * Md[bi].to(dev)).sum(-1)).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        model.eval()
        ps, cs = [], []
        with torch.no_grad():
            for s in range(0, len(va), 256):
                b = va[s:s + 256]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg, pr = model(fetch(b, broadcast), Ex[b].to(dev))
                ps.append(F.softmax(lg.float(), -1).cpu().numpy()); cs.append(F.normalize(pr.float(), dim=-1).cpu().numpy())
        p = np.concatenate(ps); c = np.concatenate(cs)
        lp = (1 - args.lam) * np.log(np.clip(p, 1e-12, 1)) + args.lam * np.log(np.clip(prior_v, 1e-12, 1))
        lp -= lp.max(1, keepdims=True); pf = np.exp(lp); pf /= pf.sum(1, keepdims=True)
        per = -np.log(np.clip(pf[np.arange(len(yv)), yv], 1e-12, 1)); rank = (np.argsort(-pf, 1) == yv[:, None]).argmax(1)
        return {"top1": round(float((rank == 0).mean()), 4), "top5": round(float((rank < 5).mean()), 4),
                "nll": round(float(per.mean()), 4), "subgoal_cos": round(float((c * mv).sum(1).mean()), 4)}

    res = {"n_train": len(tr), "n_val": len(va),
           "A_real_grid_spatial": run(False), "B_broadcast_pooled_noSpatial": run(True)}
    for k, v in res.items():
        if isinstance(v, dict):
            print(f"{k:30s} {v}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
