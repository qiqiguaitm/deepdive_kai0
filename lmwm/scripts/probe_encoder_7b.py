#!/usr/bin/env python
"""Encoder-side lever probe: does a BIGGER encoder (DINOv3-7B int8, 4096-D) raise
the next-milestone prediction ceiling vs DINOv3-H (1280-D)? And does CRAVE-style
proprio fusion (z-score+L2, equal weight) beat raw state append?

Labels (milestone clustering / real next-unique target) stay defined by the H-based
clustering -- the task is fixed; only the INPUT representation changes. Held-out
episodes; both kNN (k=50) and a learned MLP probe. Reports mean NLL + top1 AND the
per-sample NLL tail (std, p90) since the goal is mean AND variance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_cache(feature_dir: Path, dim: int):
    idx = np.load(feature_dir / "index.npz")
    e, fr, n = idx["E"].astype(np.int64), idx["FR"].astype(np.int64), int(idx["n"])
    feat = np.zeros((n, dim), dtype=np.float16)
    valid = np.zeros(n, dtype=bool)
    for shard in sorted(feature_dir.glob("shard_*.npz")):
        z = np.load(shard)
        gi = z["gidx"].astype(np.int64)
        feat[gi] = z["feat"]; valid[gi] |= z["valid"].astype(bool)
    fv = feat[valid].astype(np.float32)
    fv /= np.linalg.norm(fv, axis=1, keepdims=True) + 1e-8
    return e[valid], fr[valid], fv


class Probe(nn.Module):
    def __init__(self, in_dim, num_m, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.LayerNorm(hidden),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.LayerNorm(hidden),
                                 nn.Linear(hidden, num_m))

    def forward(self, x): return self.net(x)


def mlp_eval(Xtr, ytr, Xva, yva, num_m, dev, steps=2000, bs=4096):
    torch.manual_seed(0)
    m = Probe(Xtr.shape[1], num_m).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-5)
    Xtr = torch.from_numpy(Xtr).to(dev); ytr = torch.from_numpy(ytr).to(dev)
    Xva = torch.from_numpy(Xva).to(dev); yva = torch.from_numpy(yva).to(dev)
    for _ in range(steps):
        bi = torch.randint(0, len(Xtr), (bs,), device=dev)
        loss = F.cross_entropy(m(Xtr[bi]), ytr[bi]); opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        lg = m(Xva)
        per = F.cross_entropy(lg, yva, reduction="none").cpu().numpy()
        top1 = float((lg.argmax(1) == yva).float().mean())
    return {"top1": round(top1, 4), "nll": round(float(per.mean()), 4),
            "nll_std": round(float(per.std()), 4), "nll_p90": round(float(np.percentile(per, 90)), 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h_dir", default="temp/crave_full_dinov3h", type=Path)
    ap.add_argument("--b7_dir", default="temp/crave_full_dinov37bint8", type=Path)
    ap.add_argument("--dataset_root", default="kai0/data/Task_A/kai0_base", type=Path)
    ap.add_argument("--graph_npz", default="lmwm/data/recurrence_graphs/kai0base_dinov3h/recurrence_graph.npz")
    ap.add_argument("--out", default="lmwm/outputs/ceiling_diag/encoder_7b.json", type=Path)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    proto = np.load(args.graph_npz)["prototype_table"].astype(np.float32)
    num_m = len(proto)
    Eh, FRh, FH = load_cache(args.h_dir, 1280)
    E7, FR7, F7 = load_cache(args.b7_dir, 4096)
    # align: build (episode, FR) -> row map for 7B
    key7 = {(int(e), int(f)): i for i, (e, f) in enumerate(zip(E7, FR7))}

    # stage transitions defined on H clustering (fixed task)
    assign = np.empty(len(FH), dtype=np.int64)
    for i in range(0, len(FH), 32768):
        assign[i:i + 32768] = (FH[i:i + 32768] @ proto.T).argmax(1)
    rows = []  # (gidx_h, gidx_7 or -1, next_m, ep)
    for ep in np.unique(Eh):
        loc = np.where(Eh == ep)[0]; order = loc[np.argsort(FRh[loc])]
        seq = assign[order]
        ch = np.where(np.diff(seq) != 0)[0] + 1
        st = np.concatenate([[0], ch]); en = np.concatenate([ch, [len(seq)]])
        ms = [int(seq[s]) for s in st]
        for i in range(len(ms) - 1):
            gh = int(order[en[i] - 1])
            g7 = key7.get((int(ep), int(FRh[gh])), -1)
            rows.append((gh, g7, ms[i + 1], int(ep)))
    rows = [r for r in rows if r[1] >= 0]
    gh = np.array([r[0] for r in rows]); g7 = np.array([r[1] for r in rows])
    nxt = np.array([r[2] for r in rows]).astype(np.int64); ep_a = np.array([r[3] for r in rows])
    print(f"{len(rows)} transitions with aligned 7B features", flush=True)

    # state from parquet
    cs = int(json.loads((args.dataset_root / "meta/info.json").read_text())["chunks_size"])
    state = np.zeros((len(rows), 14), np.float32)
    frs = FRh[gh]
    for ep in np.unique(ep_a):
        pq = args.dataset_root / f"data/chunk-{ep // cs:03d}/episode_{ep:06d}.parquet"
        df = pd.read_parquet(pq, columns=["observation.state"])
        arr = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
        m = ep_a == ep
        state[m] = arr[np.clip(frs[m], 0, len(arr) - 1)]
    state_z = (state - state.mean(0)) / (state.std(0) + 1e-6)
    state_crave = state_z / (np.linalg.norm(state_z, axis=1, keepdims=True) + 1e-8)  # z-score + L2 (CRAVE style)

    rng = np.random.default_rng(args.seed); eps = np.unique(Eh); rng.shuffle(eps)
    val = set(eps[:max(1, int(round(len(eps) * 0.2)))].tolist())
    iv = np.array([e in val for e in ep_a]); itr = ~iv

    XH = FH[gh]; X7 = F7[g7]
    variants = {
        "H_1280": XH,
        "7B_4096": X7,
        "H+state_raw_append": np.concatenate([XH, state_z], 1),
        "H+state_crave_l2": np.concatenate([XH, state_crave], 1),
        "7B+state_crave_l2": np.concatenate([X7, state_crave], 1),
        "H+7B_concat": np.concatenate([XH, X7], 1),
    }
    res = {}
    for name, X in variants.items():
        r = mlp_eval(X[itr].astype(np.float32), nxt[itr], X[iv].astype(np.float32), nxt[iv], num_m, dev)
        res[name] = r
        print(f"{name:22s} top1={r['top1']:.4f} nll={r['nll']:.4f} nll_std={r['nll_std']:.3f} p90={r['nll_p90']:.3f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"n_transitions": len(rows), "variants": res}, indent=2), encoding="utf-8")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
