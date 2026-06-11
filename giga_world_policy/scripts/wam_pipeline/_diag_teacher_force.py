"""Teacher-forcing 探针:把去噪循环中的视频 latents 每步替换为「加噪 GT 未来视频」,
隔离回答:lookahead 模型的 action 解码到底从未来视频里拿到了多少信息?
  - TF 比 normal 显著降 mae → 世界模型预测质量是瓶颈(值得做 X-WAM 式耦合/更好视频);
  - TF ≈ normal → action 没在用视频(lookahead 通路无效);
  - 对照组 = 非 lookahead ckpt(action 不 attend 视频),TF 必须≈normal,否则探针有 bug。
实现:callback_on_step_end 把 latents 覆写为 sigma_{i+1}*noise0 + (1-sigma_{i+1})*gt_lat
(flow-matching 前向加噪,noise0 每窗口固定),与训练分布一致。
用法:
  CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python scripts/wam_pipeline/_diag_teacher_force.py \
    --ckpt <transformer_dir> --stats_path assets_visrobot01/norm_stats_vis_abs.json --n 16
"""
import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from scripts.wam_pipeline.eval_watch import EpisodeFrameCache, _hwc_to_chw01, build_window_indices  # noqa: E402

VK = ["observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist"]
HOR = [10, 24, 48]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--stats_path", required=True)
    ap.add_argument("--model_id", default="../checkpoints/Wan2.2-TI2V-5B-Diffusers")
    ap.add_argument("--val_root", default="../kai0/data/wam_fold_v1/visrobot01_val")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--nfe", type=int, default=10)
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
    lat_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(dev, torch.float32)
    lat_std = torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(dev, torch.float32)
    print(f"[tf-probe] ckpt={args.ckpt} lookahead={lookahead} n={len(sel)} nfe={args.nfe}", flush=True)

    def gt_latents(ep_frames, f, Lf):
        offs = [0, 12, 24, 36, 48]
        imgs = [build_ref_image(images={k: _hwc_to_chw01(ep_frames[k][min(f + o, Lf - 1)]) for k in VK},
                                dst_size=(768, 192), crop_mode="center") for o in offs]
        x = torch.stack([pipe.video_processor.preprocess(im, height=192, width=768)[0] for im in imgs], dim=1)  # (C,T,H,W)
        with torch.no_grad():
            z = vae.encode(x.unsqueeze(0).to(dev, dt)).latent_dist.mode().float()
        return (z - lat_mean) / lat_std  # pipeline 内部 latent 空间(decode 的逆)

    def infer(gi, teacher_force):
        d = ds[int(gi)]; ep, f = info[int(gi)]; fr = fc.get(ep); Lf = fr[VK[0]].shape[0]
        ref = build_ref_image(images={k: _hwc_to_chw01(fr[k][f]) for k in VK}, dst_size=(768, 192), crop_mode="center")
        st = d["observation.state"].float().unsqueeze(0).to(dev)
        ns = normalize_state(st, norm, mode="zscore").to(dev, dt)
        g = torch.Generator(device=dev).manual_seed(0)
        cb = None
        if teacher_force:
            gt_lat = gt_latents(fr, f, Lf)
            noise0 = torch.randn(gt_lat.shape, generator=torch.Generator(device=dev).manual_seed(1), device=dev)

            def cb(p, i, t, kw):
                sig = p.scheduler.sigmas[min(i + 1, len(p.scheduler.sigmas) - 1)].to(dev)
                kw["latents"] = (sig * noise0 + (1 - sig) * gt_lat).to(kw["latents"].dtype)
                return kw
        with torch.no_grad():
            _, act = pipe(height=192, width=768, action_chunk=48, state=ns, num_frames=5,
                          guidance_scale=0.0, num_inference_steps=args.nfe, image=ref, action_only=False,
                          generator=g, callback_on_step_end=cb,
                          return_dict=False, prompt_embeds=t5.unsqueeze(0).to(dev, torch.float32))
        pa = add_state_to_action(denormalize_action(act[0].float(), norm, mode="zscore"),
                                 st[0].float().to(act.device), action_chunk=48, mask=mask).cpu().numpy()
        gt = d["action"].float().numpy()[:, :14]
        L = min(len(pa), len(gt))
        return np.abs(pa[:L] - gt[:L])

    for tag, tfc in [("normal", False), ("teacher", True)]:
        agg = {h: [] for h in HOR}
        for gi in sel:
            ae = infer(gi, tfc)
            for h in HOR:
                if h <= len(ae): agg[h].append(ae[h - 1].mean())
        print(f"[{tag:7s}] mae " + " ".join(f"@{h}={np.mean(agg[h]):.4f}" for h in HOR), flush=True)


if __name__ == "__main__":
    main()
