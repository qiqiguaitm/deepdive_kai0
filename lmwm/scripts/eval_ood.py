#!/usr/bin/env python
"""OOD evaluation of a trained two-model milestone predictor on a DIFFERENT dataset (e.g. vis_base).
Loads a kai0-trained ckpt, builds pairs on the OOD feature bank via Viterbi over the SAME kai0 37
prototypes (cross-domain milestone assignment), and reports the same metrics as train_ablation's eval
block (oracle / deploy / persistence / best-of-N / identity top-N / value-forward). All OOD episodes
are treated as the val set. Answers: does the cluster-center CE anchor (center_w) help or hurt OOD?
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "crave/src"))
from train_lawm_patch import load_index, read_imgs  # noqa: E402
from train_ablation import build_pairs_abl, topn_hit  # noqa: E402
from train_twomodel_v2 import MilestoneGenerator, MilestonePredictor, MilestonePredictorGrid, PI05_NPZ, PI05_NPZ_GF3  # noqa: E402
from _siglip_bigvision import SiglipBigVision  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--feature_dir", required=True, type=Path)          # OOD DINOv3-H bank (visbase)
    ap.add_argument("--dataset_root", required=True, type=Path)         # OOD frames (visbase)
    ap.add_argument("--camera", default="observation.images.top_head")
    ap.add_argument("--mode", default="seglast")
    ap.add_argument("--bestof", type=int, default=8)
    ap.add_argument("--tag", default="ood")
    ap.add_argument("--pi05_npz", default="")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    npz = args.pi05_npz or (PI05_NPZ if Path(PI05_NPZ).exists() else PI05_NPZ_GF3)

    # --- kai0 prototypes (SAME as training) for cross-domain milestone assignment ---
    rg = np.load(REPO / "lmwm/data/recurrence_graphs/kai0base_dinov3h/recurrence_graph.npz")
    proto = rg["prototype_table"].astype(np.float32); pord = rg["pord"].astype(np.float32)
    protoL = proto / (np.linalg.norm(proto, axis=1, keepdims=True) + 1e-8)
    Mn = len(proto)

    E, FR, Fn = load_index(args.feature_dir)
    all_eps = set(np.unique(E).tolist())                                # ALL OOD eps -> val
    _tr, va = build_pairs_abl(E, FR, Fn, proto, protoL, pord, args.mode, all_eps, 2026)
    uniq = sorted(set([p[0] for p in va] + [p[1] for p in va])); u2k = {g: k for k, g in enumerate(uniq)}
    print(f"[{args.tag}] OOD val pairs={len(va)} uniq frames={len(uniq)} from {len(all_eps)} eps", flush=True)

    imgs, _ = read_imgs(args.dataset_root, args.camera, E, FR, np.array(uniq), 224, 128)
    enc = SiglipBigVision(npz, device=dev)
    grids = enc.encode_grid(imgs, bs=32); din = grids.shape[1]

    tm = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    gmu, gsd, cd, K = tm["gmu"], tm["gsd"], tm["code_dim"], tm["K"]
    pin = tm.get("pred_input", "gist")
    GZnp = ((grids - gmu) / gsd).astype(np.float32); del grids                 # kai0-normalized (true OOD)
    gist = torch.from_numpy(GZnp.mean((2, 3)))
    GZ = torch.from_numpy(GZnp).half(); del GZnp

    from train_lawm_patch import InverseEnc
    fwd = MilestoneGenerator(din, cd).to(dev); fwd.load_state_dict(tm["fwd"]); fwd.eval()
    predm = (MilestonePredictorGrid if pin == "grid" else MilestonePredictor)(din, cd, K).to(dev)
    predm.load_state_dict(tm["predm"]); predm.eval()
    inv = InverseEnc(din, cd).to(dev); inv.load_state_dict(tm["inv"]); inv.eval()

    vaa = np.array([u2k[p[0]] for p in va]); vab = np.array([u2k[p[1]] for p in va])
    vbn = np.array([p[3] for p in va]); vcm = np.array([p[2] for p in va])

    # SigLIP milestone prototypes (identity retrieval) from OOD gists by kai0 milestone id
    msid = (Fn[np.array(uniq)] @ proto.T).argmax(1); gnp = gist.numpy()
    sp = np.stack([gnp[msid == m].mean(0) if (msid == m).any() else np.zeros(din, np.float32) for m in range(Mn)])
    spL = sp / (np.linalg.norm(sp, axis=1, keepdims=True) + 1e-8)

    def cn(a, b): return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8)
    f = lambda x: (x.detach().cpu().numpy() * gsd + gmu).reshape(len(x), -1)
    co, cd_, cb, cp = [], [], [], []
    with torch.no_grad():
        for s in range(0, len(vaa), 128):
            Gc = GZ[vaa[s:s + 128]].float().to(dev); Gf = GZ[vab[s:s + 128]].float().to(dev)
            gc = gist[vaa[s:s + 128]].to(dev); pc = Gc if pin == "grid" else gc
            gtr = f(Gf); zdep = predm.deploy_mean(pc)
            co.append(cn(f(fwd(Gc, inv(Gc, Gf))), gtr))
            cd_.append(cn(f(fwd(Gc, zdep)), gtr)); cp.append(cn(f(Gc), gtr))
            best = None
            for zi in predm.sample(pc, args.bestof):
                ck = cn(f(fwd(Gc, zi)), gtr); best = ck if best is None else np.maximum(best, ck)
            cb.append(best)
        idpred = []
        for s in range(0, len(vaa), 128):
            Gc = GZ[vaa[s:s + 128]].float().to(dev); gc = gist[vaa[s:s + 128]].to(dev)
            pc = Gc if pin == "grid" else gc
            idpred.append(fwd(Gc, predm.deploy_mean(pc)).mean((2, 3)).cpu().numpy())
        idpred = np.concatenate(idpred); idpred /= (np.linalg.norm(idpred, axis=1, keepdims=True) + 1e-8)
        idn = topn_hit(idpred @ spL.T, vbn)
        pred_ms = (idpred @ spL.T).argmax(1)
        value_fwd = float((pord[pred_ms] > pord[vcm]).mean())
        value_true = float((pord[vbn] > pord[vcm]).mean())

    res = {"tag": args.tag, "ckpt": str(args.ckpt), "center_w": tm.get("center_w"), "pred_input": pin,
           "n_val": len(va), "oracle_grid_cos": round(float(np.concatenate(co).mean()), 4),
           "deploy_grid_cos": round(float(np.concatenate(cd_).mean()), 4),
           "bestof%d" % args.bestof: round(float(np.concatenate(cb).mean()), 4),
           "persistence": round(float(np.concatenate(cp).mean()), 4),
           "lift_deploy": round(float(np.concatenate(cd_).mean() - np.concatenate(cp).mean()), 4),
           "identity_topN": idn, "value_forward_frac": round(value_fwd, 4),
           "target_value_forward_ref": round(value_true, 4)}
    outp = REPO / f"lmwm/outputs/ood/{args.tag}.json"; outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    main()
