#!/usr/bin/env python
"""L4: distill the 9-member fused ensemble (teacher) into ONE student model.

The teacher = big3+mixed6 ensemble-averaged probs, FUSED with the graph prior
(lambda=0.3), plus averaged subgoal proto. The student (single UnifiedLMWM) is
trained on TRAIN-split inputs to match teacher soft-probs (KL) + teacher proto
(cosine), absorbing both the ensemble AND the graph prior into ONE forward pass
-> deployable for VLA at 1x cost. Eval on VAL vs the teacher.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lmwm.data import split_indices  # noqa: E402
from lmwm.models import UnifiedLMWM  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_mean_variance import load_model, forward_all, discrete_stats, proto_stats  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="lmwm/data/crave_sequences/kai0base_dinov3h_frame2proto/pairs_next_unique_augin.npz")
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lam", type=float, default=0.3)
    ap.add_argument("--out", default="lmwm/outputs/lever_distill/summary.json", type=Path)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    z = np.load(args.pairs)
    n = len(z["current_milestone"])
    ti, vi = split_indices(z, n, 0.2, 2026, torch.device("cpu"), "episode")
    ti, vi = ti.numpy(), vi.numpy()
    X = z["current"].astype(np.float32); din = X.shape[1]
    y = z["future_milestone"].astype(np.int64)
    cur_m = z["current_milestone"].astype(np.int64)
    med = z["next_medoid"].astype(np.float32)
    ok = np.linalg.norm(med, axis=1) > 1e-6
    med = med / (np.linalg.norm(med, axis=1, keepdims=True) + 1e-8)

    # ---- teacher: 9-member ensemble, fused with graph prior ----
    paths = (sorted(glob.glob("lmwm/checkpoints/stage3_augin_big/*/best.pt"))
             + sorted(glob.glob("lmwm/checkpoints/stage3_augin_ens/*/best.pt"))
             + sorted(glob.glob("lmwm/checkpoints/stage3_augin/*/best.pt"))[-1:]
             + sorted(glob.glob("lmwm/checkpoints/stage3_augin_tail/*cecvar*/best.pt")))
    print(f"teacher = {len(paths)} members", flush=True)
    models = [load_model(p, dev)[0] for p in paths]
    probs, protos = forward_all(models, X, dev)
    g = np.load("lmwm/data/recurrence_graphs/kai0base_dinov3h/recurrence_graph.npz")
    trans = g["transition_probs"].astype(np.float64); trans = trans / trans.sum(1, keepdims=True).clip(1e-12)
    prior = trans[cur_m]
    lp = (1 - args.lam) * np.log(np.clip(probs, 1e-12, 1)) + args.lam * np.log(np.clip(prior, 1e-12, 1))
    lp -= lp.max(1, keepdims=True); tp = np.exp(lp); tp = tp / tp.sum(1, keepdims=True)  # teacher fused probs
    num_m = tp.shape[1]

    Tp = torch.from_numpy(tp.astype(np.float32)); Tg = torch.from_numpy(protos.astype(np.float32))
    Xt = torch.from_numpy(X)
    tset = ti

    student = UnifiedLMWM(din, med.shape[1], num_m, args.hidden, args.depth).to(dev)
    opt = torch.optim.AdamW(student.parameters(), lr=5e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 300))
    for s in range(args.steps):
        bi = tset[torch.randint(0, len(tset), (2048,)).numpy()]
        xb = Xt[bi].to(dev); tpb = Tp[bi].to(dev); tgb = Tg[bi].to(dev)
        out = student(xb)
        logs = F.log_softmax(out["greedy_logits"], -1)
        kl = F.kl_div(logs, tpb, reduction="batchmean")
        pcos = (1.0 - (out["greedy_proto"] * tgb).sum(-1)).mean()
        loss = kl + 5.0 * pcos
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()

    # ---- eval student vs teacher on VAL ----
    student.eval()
    ps, prs = [], []
    with torch.no_grad():
        for s in range(0, len(vi), 8192):
            b = vi[s:s + 8192]
            out = student(Xt[b].to(dev))
            ps.append(F.softmax(out["greedy_logits"], -1).cpu().numpy()); prs.append(out["greedy_proto"].cpu().numpy())
    sp = np.concatenate(ps); sg = np.concatenate(prs)
    sg = sg / (np.linalg.norm(sg, axis=1, keepdims=True) + 1e-8)
    yv = y[vi]; okv = ok[vi]; medv = med[vi]

    res = {"student": f"{args.hidden}x{args.depth}", "n_members_teacher": len(paths),
           "student_params_M": round(sum(p.numel() for p in student.parameters()) / 1e6, 3),
           "teacher_fused": {"discrete": discrete_stats(tp[vi], yv), "subgoal": proto_stats(protos[vi][okv], medv[okv])},
           "student_distilled": {"discrete": discrete_stats(sp, yv), "subgoal": proto_stats(sg[okv], medv[okv])}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    ck_dir = Path("lmwm/checkpoints/stage3_distilled"); ck_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": student.state_dict(),
                "config": {"model": {"hidden_dim": args.hidden, "depth": args.depth},
                           "graph_npz": "lmwm/data/recurrence_graphs/kai0base_dinov3h/recurrence_graph.npz",
                           "note": "distilled from big3+mixed6 fused teacher; graph prior baked in"},
                "meta": {"input_dim": din, "teacher_members": len(paths), "eval": res}},
               ck_dir / "student.pt")
    print("saved student ->", ck_dir / "student.pt")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
