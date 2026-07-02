#!/usr/bin/env python
"""Visualize, for one episode, LMWM's predicted next-stage subgoal latent decoded
into an image vs the true stage's most-similar real frame (episode medoid).

Per stage transition (row), columns:
  current real frame | pred subgoal latent -> decoded | true next-medoid latent -> decoded | true next-medoid REAL frame
Each row annotated with cos(pred_latent, true_next_medoid_latent). Also prints the
per-episode mean cosine gap.
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
    ap.add_argument("--model", default=None, type=Path, help="medoid-trained LMWM checkpoint (best.pt)")
    ap.add_argument("--feature_dir", default="temp/crave_full_dinov3h", type=Path)
    ap.add_argument("--dataset_root", default="kai0/data/Task_A/kai0_base", type=Path)
    ap.add_argument("--graph_npz", default="lmwm/data/recurrence_graphs/kai0base_dinov3h/recurrence_graph.npz")
    ap.add_argument("--camera", default="observation.images.top_head")
    ap.add_argument("--out_dir", default="lmwm/docs/assets", type=Path)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    if args.model is None:
        cands = sorted(Path("lmwm/checkpoints/stage3_realfuture_medoid").glob("*/best.pt"))
        args.model = cands[-1]
    device = torch.device(args.device if (args.device != "cpu" and torch.cuda.is_available()) else "cpu")

    ck = torch.load(args.decoder, map_location="cpu")
    R = int(ck["res"]); dec = PooledDecoder(din=int(ck["din"]), res=R).to(device)
    dec.load_state_dict(ck["model"]); dec.eval()

    def decode(lat):
        x = torch.from_numpy(l2(np.atleast_2d(lat).astype(np.float32))).to(device)
        with torch.no_grad():
            o = dec(x).cpu().numpy()
        return np.clip((o.transpose(0, 2, 3, 1) + 1) * 127.5, 0, 255).astype(np.uint8)

    g = np.load(args.graph_npz); proto = g["prototype_table"].astype(np.float32); num_m = len(proto)
    mk = torch.load(args.model, map_location="cpu"); meta = mk.get("meta", {})
    in_dim = int(meta.get("input_dim", 1280)); mc = mk["config"]["model"]
    model = UnifiedLMWM(in_dim, int(meta.get("latent_dim", 1280)), num_m,
                        int(mc.get("hidden_dim", 512)), int(mc.get("depth", 2))).to(device)
    model.load_state_dict(mk["model"]); model.eval()

    E, FR, F = load_features(args.feature_dir)
    Fn = l2(F.astype(np.float32))
    loc = np.where(E == args.episode)[0]
    order = loc[np.argsort(FR[loc])]
    seq = np.array([(Fn[i] @ proto.T).argmax() for i in order])
    ch = np.where(np.diff(seq) != 0)[0] + 1
    st = np.concatenate([[0], ch]); en = np.concatenate([ch, [len(seq)]])
    stages = []
    for s, e in zip(st, en):
        m = int(seq[s]); sub_g = order[s:e]
        med_local = sub_g[(Fn[sub_g] @ proto[m]).argmax()]
        stages.append({"m": m, "rep_g": int(order[e - 1]), "med_g": int(med_local)})

    chunks_size = int(json.loads((args.dataset_root / "meta/info.json").read_text())["chunks_size"])
    cap = cv2.VideoCapture(str(args.dataset_root / f"videos/chunk-{args.episode // chunks_size:03d}/{args.camera}/episode_{args.episode:06d}.mp4"))

    def frame(gidx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(FR[gidx]))
        ok, im = cap.read()
        return cv2.resize(im[:, :, ::-1], (R, R)) if ok else np.zeros((R, R, 3), np.uint8)

    rows = []
    for i in range(len(stages) - 1):
        cur = stages[i]; nxt = stages[i + 1]
        with torch.no_grad():
            pred = model(torch.from_numpy(F[cur["rep_g"]][None].astype(np.float32)).to(device))["greedy_proto"].cpu().numpy()[0]
        true_med_lat = Fn[nxt["med_g"]]
        cos = float(l2(pred[None])[0] @ true_med_lat)
        rows.append({
            "cur_img": frame(cur["rep_g"]),
            "pred_dec": decode(pred)[0],
            "med_dec": decode(true_med_lat)[0],
            "med_real": frame(nxt["med_g"]),
            "cur_m": cur["m"], "next_m": nxt["m"], "cos": cos,
        })
    cap.release()

    coss = [r["cos"] for r in rows]
    all_rows = rows
    max_rows = 10
    if len(rows) > max_rows:
        sel = np.linspace(0, len(rows) - 1, max_rows).astype(int)
        rows = [rows[i] for i in sel]
    ncol = 4
    titles = ["current frame", "pred subgoal (decoded)", "true medoid (decoded)", "true medoid (real frame)"]
    fig, axes = plt.subplots(len(rows), ncol, figsize=(ncol * 2.1, len(rows) * 2.2))
    if len(rows) == 1:
        axes = axes[None, :]
    for ri, r in enumerate(rows):
        imgs = [r["cur_img"], r["pred_dec"], r["med_dec"], r["med_real"]]
        for ci in range(ncol):
            a = axes[ri, ci]; a.imshow(imgs[ci]); a.set_xticks([]); a.set_yticks([])
            if ri == 0:
                a.set_title(titles[ci], fontsize=8)
        axes[ri, 0].set_ylabel(f"m{r['cur_m']}->m{r['next_m']}\ncos={r['cos']:.3f}", fontsize=7)
    split = "held-out" if True else "train"
    fig.suptitle(f"ep{args.episode} ({split}): LMWM predicted subgoal decoded vs true next-stage medoid | mean cos={np.mean(coss):.3f}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = args.out_dir / f"ep{args.episode}_subgoal_decode.png"
    fig.savefig(out, dpi=115); plt.close(fig)
    print(f"saved {out}")
    print(f"ep{args.episode}: {len(rows)} transitions, mean cos(pred, true_medoid)={np.mean(coss):.4f}, "
          f"min={np.min(coss):.4f}, max={np.max(coss):.4f}")


if __name__ == "__main__":
    main()
