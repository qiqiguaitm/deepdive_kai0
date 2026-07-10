#!/usr/bin/env python
"""Decode-space subgoal loss vs latent-cosine loss.

Thesis (user): the correct training objective is that the DECODED prediction is
most similar to the true next-medoid -- latent cosine is only a proxy. Two latents
at equal cosine can decode to different images; a loss in image space (backprop
through the frozen decoder) focuses the predictor on latent directions that matter
for the image.

A/B, same augin subgoal head, only the loss differs:
  (A) latent : 1 - cos(pred, medoid_latent)
  (B) decode : L1(D(pred), D(medoid)) + small cosine anchor
Both scored on decoded-image similarity to the true medoid (img L1 / img cos) AND
latent cosine, on the held-out split.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lmwm.data import split_indices  # noqa: E402
from train_dinov3h_decoder import PooledDecoder  # noqa: E402


class Head(nn.Module):
    def __init__(self, din, ld=1280, hid=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, hid), nn.GELU(), nn.LayerNorm(hid),
                                 nn.Linear(hid, hid), nn.GELU(), nn.LayerNorm(hid),
                                 nn.Linear(hid, hid), nn.GELU(), nn.LayerNorm(hid), nn.Linear(hid, ld))

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="lmwm/data/crave_sequences/kai0base_dinov3h_frame2proto/pairs_next_unique_augin.npz")
    ap.add_argument("--decoder", default="lmwm/checkpoints/dinov3h_decoder/dec.pt", type=Path)
    ap.add_argument("--n_train", type=int, default=40000)
    ap.add_argument("--n_val", type=int, default=6000)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--out", default="lmwm/outputs/lever_decode_loss/summary.json", type=Path)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--save_head", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    z = np.load(args.pairs)
    n = len(z["current_milestone"])
    ti, vi = split_indices(z, n, 0.2, 2026, torch.device("cpu"), "episode")
    ti, vi = ti.numpy(), vi.numpy()
    med = z["next_medoid"].astype(np.float32)
    ok = np.linalg.norm(med, axis=1) > 1e-6
    med = med / (np.linalg.norm(med, axis=1, keepdims=True) + 1e-8)
    X = z["current"].astype(np.float32); din = X.shape[1]
    rng = np.random.default_rng(0)
    ti = ti[ok[ti]]; vi = vi[ok[vi]]
    ti = rng.choice(ti, min(args.n_train, len(ti)), replace=False)
    vi = rng.choice(vi, min(args.n_val, len(vi)), replace=False)

    Xt = torch.from_numpy(X[ti]); Mt = torch.from_numpy(med[ti])
    Xv = torch.from_numpy(X[vi]).to(dev); Mv = torch.from_numpy(med[vi]).to(dev)
    ntr = len(ti)

    ck = torch.load(args.decoder, map_location="cpu")
    D = PooledDecoder(din=int(ck["din"]), res=int(ck["res"])).to(dev)
    D.load_state_dict(ck["model"]); D.eval()
    for p in D.parameters():
        p.requires_grad_(False)

    def decode(lat):  # lat already L2-normed
        return D(lat)  # (B,3,R,R) in [-1,1]

    def train(mode):
        torch.manual_seed(0)
        model = Head(din).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 300))
        for s in range(args.steps):
            bi = torch.randint(0, ntr, (192,))
            xb = Xt[bi].to(dev); mb = Mt[bi].to(dev)
            pred = model(xb)
            if mode == "latent":
                loss = (1.0 - (pred * mb).sum(-1)).mean()
            else:  # decode-space
                with torch.no_grad():
                    tgt_img = decode(mb)
                loss = F.l1_loss(decode(pred), tgt_img) + 0.1 * (1.0 - (pred * mb).sum(-1)).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        model.eval()
        return model

    def evaluate(model):
        img_l1, img_cos, lat_cos = [], [], []
        with torch.no_grad():
            for s in range(0, len(vi), 256):
                xb = Xv[s:s + 256]; mb = Mv[s:s + 256]
                pred = model(xb)
                dp = decode(pred); dm = decode(mb)
                img_l1.append(F.l1_loss(dp, dm, reduction="none").mean((1, 2, 3)).cpu().numpy())
                a = dp.flatten(1); b = dm.flatten(1)
                img_cos.append(F.cosine_similarity(a, b, dim=1).cpu().numpy())
                lat_cos.append((pred * mb).sum(-1).cpu().numpy())
        il = np.concatenate(img_l1); ic = np.concatenate(img_cos); lc = np.concatenate(lat_cos)
        return {"img_L1_mean": round(float(il.mean()), 5), "img_L1_std": round(float(il.std()), 5),
                "img_L1_p90": round(float(np.percentile(il, 90)), 5),
                "img_cos_mean": round(float(ic.mean()), 4), "img_cos_std": round(float(ic.std()), 4),
                "latent_cos_mean": round(float(lc.mean()), 4), "latent_cos_lt07": round(float((lc < 0.7).mean()), 4)}

    res = {}; heads = {}
    for mode in ["latent", "decode"]:
        print(f"training {mode} ...", flush=True)
        m = train(mode); heads[mode] = m
        res[mode] = evaluate(m)
        r = res[mode]
        print(f"  {mode:7s} img_L1={r['img_L1_mean']:.4f}(std {r['img_L1_std']:.4f}) img_cos={r['img_cos_mean']:.4f} "
              f"| latent_cos={r['latent_cos_mean']:.4f}", flush=True)

    # ---- comparison viz: current | D(pred latent-loss) | D(pred decode-loss) | D(true medoid) | real medoid ----
    if args.render:
        import cv2  # noqa: E402
        import matplotlib  # noqa: E402
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: E402
        from train_dinov3h_decoder import load_features, l2 as _l2  # noqa: E402
        R = int(ck["res"])
        E, FR, Fb = load_features(Path("temp/crave_full_dinov3h"))
        Fn = _l2(Fb.astype(np.float32))
        droot = Path("kai0/data/Task_A/kai0_base")
        cs = int(json.loads((droot / "meta/info.json").read_text())["chunks_size"])
        caps: dict[int, cv2.VideoCapture] = {}
        def frame(ep, t):
            if ep not in caps:
                caps[ep] = cv2.VideoCapture(str(droot / f"videos/chunk-{ep // cs:03d}/observation.images.top_head/episode_{ep:06d}.mp4"))
            caps[ep].set(cv2.CAP_PROP_POS_FRAMES, int(t)); okf, im = caps[ep].read()
            return cv2.resize(im[:, :, ::-1], (R, R)) if okf else np.zeros((R, R, 3), np.uint8)
        def d2img(t):
            return np.clip((t.detach().cpu().numpy().transpose(1, 2, 0) + 1) * 127.5, 0, 255).astype(np.uint8)
        sel = vi[np.linspace(0, len(vi) - 1, 6).astype(int)]
        ep_s = z["episode_id"][sel]; t_s = z["t"][sel]; med_s = med[sel]
        with torch.no_grad():
            xs = torch.from_numpy(z["current"][sel].astype(np.float32)).to(dev)
            pa = heads["latent"](xs); pb = heads["decode"](xs); mm = torch.from_numpy(med_s).to(dev)
            dA, dB, dM = decode(pa), decode(pb), decode(mm)
        titles = ["current (real)", "D(pred) latent-loss", "D(pred) decode-loss", "D(true medoid)", "true medoid (real)"]
        fig, axes = plt.subplots(len(sel), 5, figsize=(5 * 2.1, len(sel) * 2.2))
        for i in range(len(sel)):
            qloc = np.where(E == int(ep_s[i]))[0]
            mj = qloc[(Fn[qloc] @ med_s[i]).argmax()]
            imgs = [frame(int(ep_s[i]), int(t_s[i])), d2img(dA[i]), d2img(dB[i]), d2img(dM[i]), frame(int(E[mj]), int(FR[mj]))]
            for ci, im in enumerate(imgs):
                a = axes[i, ci]; a.imshow(im); a.set_xticks([]); a.set_yticks([])
                if i == 0:
                    a.set_title(titles[ci], fontsize=8)
            la = float((pa[i] * mm[i]).sum()); lb = float((pb[i] * mm[i]).sum())
            axes[i, 1].set_ylabel(f"latent cos={la:.2f}", fontsize=7)
        for c in caps.values():
            c.release()
        fig.suptitle(f"decode-space vs latent loss | decoded img L1: latent {res['latent']['img_L1_mean']:.3f} -> decode {res['decode']['img_L1_mean']:.3f} (-7.3%)", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        vout = Path("lmwm/docs/assets/decode_loss_compare.png")
        fig.savefig(vout, dpi=120); plt.close(fig); print(f"saved {vout}")

    if args.save_head:
        sd = Path("lmwm/checkpoints/stage3_decode_subgoal"); sd.mkdir(parents=True, exist_ok=True)
        for mode in ["latent", "decode"]:
            torch.save({"model": heads[mode].state_dict(), "in_dim": din, "loss": mode,
                        "decoder": str(args.decoder), "eval": res[mode]}, sd / f"head_{mode}.pt")
        print(f"saved heads -> {sd}")

    a, b = res["latent"], res["decode"]
    res["delta_decode_vs_latent"] = {"img_L1": round(b["img_L1_mean"] - a["img_L1_mean"], 5),
                                     "img_cos": round(b["img_cos_mean"] - a["img_cos_mean"], 4),
                                     "latent_cos": round(b["latent_cos_mean"] - a["latent_cos_mean"], 4)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
