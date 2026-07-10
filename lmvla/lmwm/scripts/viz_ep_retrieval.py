#!/usr/bin/env python
"""Sharp subgoal via RETRIEVAL: LMWM predicts a real-frame latent, so instead of
synthesizing (blurry pooled decode) we retrieve the nearest real frame to the
predicted latent. Compares, per stage transition:
  current | pred decoded (blurry) | pred -> nearest real frame (sharp) | true medoid real frame
Retrieval excludes the query episode. Reports retrieved-milestone match to the true
next milestone and cosine.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from train_dinov3h_decoder import PooledDecoder, load_features, l2  # noqa: E402
from lmwm.models import UnifiedLMWM  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--decoder", default="lmwm/checkpoints/dinov3h_decoder/dec.pt", type=Path)
    ap.add_argument("--model", default=None, type=Path)
    ap.add_argument("--feature_dir", default="temp/crave_full_dinov3h", type=Path)
    ap.add_argument("--dataset_root", default="kai0/data/Task_A/kai0_base", type=Path)
    ap.add_argument("--graph_npz", default="lmwm/data/recurrence_graphs/kai0base_dinov3h/recurrence_graph.npz")
    ap.add_argument("--camera", default="observation.images.top_head")
    ap.add_argument("--out_dir", default="lmwm/docs/assets", type=Path)
    ap.add_argument("--max_rows", type=int, default=10)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    if args.model is None:
        args.model = sorted(Path("lmwm/checkpoints/stage3_realfuture_medoid").glob("*/best.pt"))[-1]
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ck = torch.load(args.decoder, map_location="cpu")
    R = int(ck["res"]); dec = PooledDecoder(din=int(ck["din"]), res=R).to(dev); dec.load_state_dict(ck["model"]); dec.eval()

    def decode(lat):
        x = torch.from_numpy(l2(np.atleast_2d(lat).astype(np.float32))).to(dev)
        with torch.no_grad():
            o = dec(x).cpu().numpy()
        return np.clip((o.transpose(0, 2, 3, 1) + 1) * 127.5, 0, 255).astype(np.uint8)

    g = np.load(args.graph_npz); proto = g["prototype_table"].astype(np.float32); num_m = len(proto)
    mk = torch.load(args.model, map_location="cpu"); meta = mk.get("meta", {}); mc = mk["config"]["model"]
    model = UnifiedLMWM(int(meta.get("input_dim", 1280)), int(meta.get("latent_dim", 1280)), num_m,
                        int(mc.get("hidden_dim", 512)), int(mc.get("depth", 2))).to(dev)
    model.load_state_dict(mk["model"]); model.eval()

    E, FR, F = load_features(args.feature_dir)
    Fn = l2(F.astype(np.float32))
    assign = np.array([(Fn[i:i+1] @ proto.T).argmax() for i in range(0, 0)])  # placeholder
    Fn_gpu = torch.from_numpy(Fn).to(dev)
    not_ep = torch.from_numpy((E != args.episode)).to(dev)

    loc = np.where(E == args.episode)[0]; order = loc[np.argsort(FR[loc])]
    seq = (Fn[order] @ proto.T).argmax(1)
    ch = np.where(np.diff(seq) != 0)[0] + 1
    st = np.concatenate([[0], ch]); en = np.concatenate([ch, [len(seq)]])
    stages = []
    for s, e in zip(st, en):
        m = int(seq[s]); sub = order[s:e]
        stages.append({"m": m, "rep_g": int(order[e - 1]), "med_g": int(sub[(Fn[sub] @ proto[m]).argmax()])})

    cs = int(json.loads((args.dataset_root / "meta/info.json").read_text())["chunks_size"])
    caps: dict[int, cv2.VideoCapture] = {}

    def frame(gidx):
        ep = int(E[gidx])
        if ep not in caps:
            caps[ep] = cv2.VideoCapture(str(args.dataset_root / f"videos/chunk-{ep // cs:03d}/{args.camera}/episode_{ep:06d}.mp4"))
        caps[ep].set(cv2.CAP_PROP_POS_FRAMES, int(FR[gidx]))
        ok, im = caps[ep].read()
        return cv2.resize(im[:, :, ::-1], (R, R)) if ok else np.zeros((R, R, 3), np.uint8)

    rows = []
    for i in range(len(stages) - 1):
        cur, nxt = stages[i], stages[i + 1]
        with torch.no_grad():
            pred = model(torch.from_numpy(F[cur["rep_g"]][None].astype(np.float32)).to(dev))["greedy_proto"][0]
            sim = Fn_gpu @ pred
            sim = sim.masked_fill(~not_ep, -2.0)  # exclude query episode
            j = int(sim.argmax().item())
        cos_med = float((l2(pred.cpu().numpy()[None])[0]) @ Fn[nxt["med_g"]])
        rows.append({"cur": frame(cur["rep_g"]), "dec": decode(pred.cpu().numpy())[0],
                     "retr": frame(j), "med": frame(nxt["med_g"]),
                     "cur_m": cur["m"], "next_m": nxt["m"], "retr_m": int((Fn[j:j+1] @ proto.T).argmax()),
                     "cos_med": cos_med})

    match = np.mean([r["retr_m"] == r["next_m"] for r in rows])
    for c in caps.values():
        c.release()

    show = rows if len(rows) <= args.max_rows else [rows[i] for i in np.linspace(0, len(rows) - 1, args.max_rows).astype(int)]
    titles = ["current frame", "pred decoded (blurry)", "pred -> nearest REAL frame", "true medoid (real)"]
    fig, axes = plt.subplots(len(show), 4, figsize=(4 * 2.1, len(show) * 2.2))
    if len(show) == 1:
        axes = axes[None, :]
    for ri, r in enumerate(show):
        for ci, im in enumerate([r["cur"], r["dec"], r["retr"], r["med"]]):
            a = axes[ri, ci]; a.imshow(im); a.set_xticks([]); a.set_yticks([])
            if ri == 0:
                a.set_title(titles[ci], fontsize=8)
        ok = "OK" if r["retr_m"] == r["next_m"] else f"x(m{r['retr_m']})"
        axes[ri, 0].set_ylabel(f"m{r['cur_m']}->m{r['next_m']}\ncos={r['cos_med']:.2f} {ok}", fontsize=7)
        for s in axes[ri, 2].spines.values():
            s.set_color("#2ca02c" if r["retr_m"] == r["next_m"] else "#d62728"); s.set_linewidth(2.5)
    fig.suptitle(f"ep{args.episode}: sharp subgoal via retrieval (nearest real frame) | retrieved-milestone match={match:.0%}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = args.out_dir / f"ep{args.episode}_subgoal_retrieval.png"
    fig.savefig(out, dpi=115); plt.close(fig)
    print(f"saved {out}")
    print(f"ep{args.episode}: {len(rows)} transitions, retrieved-milestone==true-next: {match:.3f}, mean cos={np.mean([r['cos_med'] for r in rows]):.4f}")


if __name__ == "__main__":
    main()
