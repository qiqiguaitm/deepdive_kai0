"""深度诊断 MAE 偏高根因(一次跑完 3 个探针,小样本单卡):
  1) 逐维 MAE(@1/@48)— 对照 stay-baseline 逐维,看误差集中在哪些关节;
  2) NFE sweep(10 vs 30 denoise steps)— 采样步数是否是瓶颈;
  3) best-of-4 seeds — 同窗口采 4 个样本取逐窗口最优 mae@48;若显著低于单样本
     → 误差主体是多模态模式错配(采样到不同合法未来),不是能力不足。
用法:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/wam_pipeline/_diag_deep.py \
    --ckpt <transformer_dir> --stats_path assets_visrobot01/norm_stats_vis_abs.json --n 24
"""
import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from scripts.wam_pipeline.eval_watch import EpisodeFrameCache, _hwc_to_chw01, build_window_indices  # noqa: E402

VK = ["observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist"]
HOR = [1, 10, 24, 48]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--stats_path", required=True)
    ap.add_argument("--model_id", default="../checkpoints/Wan2.2-TI2V-5B-Diffusers")
    ap.add_argument("--val_root", default="../kai0/data/wam_fold_v1/visrobot01_val")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--nfes", default="10,30")
    args = ap.parse_args()
    dev, dt = "cuda", torch.bfloat16

    from diffusers.models import AutoencoderKLWan
    from giga_datasets import load_dataset

    from world_action_model.models.transformer_wa_casual import CasualWorldActionTransformer
    from world_action_model.pipeline.utils import (
        add_state_to_action, build_ref_image, denormalize_action, extract_normalization_tensors,
        load_stats, load_t5_embedding_from_pkl, normalize_state, resolve_delta_mask,
    )
    from world_action_model.pipeline.wa_pipeline import WAPipeline

    ds = load_dataset([dict(_class_name="LeRobotDataset", data_path=args.val_root, delta_info={"action": 48},
                            skip_video_decoding=True, embodiment="visrobot01", tolerance_s=1e-3)])
    fc = EpisodeFrameCache(args.val_root, VK, 2)
    stats = load_stats(args.stats_path)
    norm = extract_normalization_tensors(stats, device=dev, state_dim=14, action_dim=14)
    mask = torch.tensor(resolve_delta_mask(stats, 14).tolist(), device=dev, dtype=torch.bool)
    t5 = load_t5_embedding_from_pkl(f"{args.val_root}/t5_embedding/episode_000000.pt", target_len=64).to(dev, torch.float32)
    idx, _, info = build_window_indices(args.val_root, "exec", 16, 48, 16)
    sel = [idx[i] for i in np.unique(np.linspace(0, len(idx) - 1, args.n).astype(int))]

    vae = AutoencoderKLWan.from_pretrained(args.model_id, subfolder="vae", torch_dtype=dt)
    tf = CasualWorldActionTransformer.from_pretrained(args.ckpt).to(dt)
    lookahead = bool(getattr(tf.config, "action_attends_video", False))
    pipe = WAPipeline.from_pretrained(args.model_id, vae=vae, transformer=tf, torch_dtype=dt).to(dev)
    print(f"[diag] ckpt={args.ckpt} lookahead={lookahead} n={len(sel)} seeds={args.seeds}", flush=True)

    def infer(gi, nfe, seed):
        d = ds[int(gi)]; ep, f = info[int(gi)]; fr = fc.get(ep)
        ref = build_ref_image(images={k: _hwc_to_chw01(fr[k][f]) for k in VK}, dst_size=(768, 192), crop_mode="center")
        st = d["observation.state"].float().unsqueeze(0).to(dev)
        ns = normalize_state(st, norm, mode="zscore").to(dev, dt)
        g = torch.Generator(device=dev).manual_seed(seed)
        with torch.no_grad():
            _, act = pipe(height=192, width=768, action_chunk=48, state=ns, num_frames=5,
                          guidance_scale=0.0, num_inference_steps=nfe, image=ref,
                          action_only=not lookahead, generator=g,
                          return_dict=False, prompt_embeds=t5.unsqueeze(0).to(dev, torch.float32))
        pa = add_state_to_action(denormalize_action(act[0].float(), norm, mode="zscore"),
                                 st[0].float().to(act.device), action_chunk=48, mask=mask).cpu().numpy()
        gt = d["action"].float().numpy()[:, :14]
        L = min(len(pa), len(gt))
        return np.abs(pa[:L] - gt[:L]), st[0].cpu().numpy()[:14], gt[:L]

    nfes = [int(x) for x in args.nfes.split(",")]
    # ---- probe 1+2: per-dim + NFE(seed0) ----
    for nfe in nfes:
        ae1 = np.zeros(14); ae48 = np.zeros(14); agg = {h: 0.0 for h in HOR}; nw = 0
        for gi in sel:
            ae, _, _ = infer(gi, nfe, 0)
            ae1 += ae[0]; ae48 += ae[min(47, len(ae) - 1)]
            for h in HOR:
                if h <= len(ae): agg[h] += ae[h - 1].mean()
            nw += 1
        print(f"[NFE={nfe}] mae " + " ".join(f"@{h}={agg[h]/nw:.4f}" for h in HOR), flush=True)
        if nfe == nfes[0]:
            print("  per-dim mae@48:", np.round(ae48 / nw, 3).tolist(), flush=True)
    # ---- probe 3: best-of-N(NFE=10) + stay 同窗对照 ----
    single = []; best = []; stay48 = []
    for gi in sel:
        m48s = []
        for s in range(args.seeds):
            ae, st_np, gt = infer(gi, nfes[0], s)
            h = min(47, len(ae) - 1)
            m48s.append(ae[h].mean())
            if s == 0:
                stay48.append(np.abs(st_np[None, :] - gt)[h].mean())
        single.append(m48s[0]); best.append(min(m48s))
    print(f"[best-of-{args.seeds}] mae@48 single={np.mean(single):.4f} best={np.mean(best):.4f} "
          f"(drop {1 - np.mean(best)/np.mean(single):.1%}) | stay@48(same windows)={np.mean(stay48):.4f}", flush=True)


if __name__ == "__main__":
    main()
