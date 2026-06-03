"""解耦式 eval-agent —— 与训练完全分离,跑在独立 GPU 上,只读 PFS 上的 checkpoint。

设计(与训练解耦,触发即 cadence):
  loop:
    new = 轮询 OUTPUT_DIR 里新出现的 checkpoint_*_step_*/(训练每 checkpoint_interval 落一个)
    load transformer(优先 transformer_ema,回退 transformer) + VAE -> WAPipeline
    for clip in 固定 hold-out(visrobot01 末尾 N 集,永不进训练):
        闭环逐块生成: 一块(ref帧+state)进 -> 生成 num_frames 未来帧 + action_chunk 出,
                      预测末帧/末state 反馈作下一块输入(--closed-loop),重复若干块
        指标 = PSNR/SSIM/LPIPS + 时序一致性(帧间差) vs GT;并算 action MSE
    append {step, psnr, ssim, lpips, temporal, action_mse} -> eval_log.jsonl + tensorboard
    存几条 预测|GT 并排 MP4 供肉眼看
要点:零训练开销(只读权重,另一批 GPU);出指标-step 曲线替代 flat loss 作"在不在变好"的信号。

用法(在评估机 / 空闲 GPU 上):
  CUDA_VISIBLE_DEVICES=7 python -m scripts.wam_pipeline.eval_watch \
    --output_dir runs/visrobot01_fold --model_id ../checkpoints/Wan2.2-TI2V-5B-Diffusers \
    --stats_path assets_visrobot01/norm_stats_vis.json --val_root <val_dataset> \
    --t5_pkl <val>/t5_embedding/episode_000000.pt --n_clips 50 --blocks 4 --closed-loop \
    [--poll 120] [--once] [--smoke]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time

import numpy as np
import torch


# ----------------------------- metrics -----------------------------
def _to_uint8_thwc(vid: torch.Tensor) -> np.ndarray:
    """pipeline 视频输出 -> (T,H,W,C) uint8。输入约定 (C,T,H,W) in [-1,1] 或 (T,C,H,W)。"""
    v = vid.detach().float().cpu()
    if v.ndim == 4 and v.shape[0] in (1, 3):      # (C,T,H,W)
        v = v.permute(1, 2, 3, 0)
    elif v.ndim == 4 and v.shape[1] in (1, 3):    # (T,C,H,W)
        v = v.permute(0, 2, 3, 1)
    if v.min() < -0.01:                            # [-1,1] -> [0,1]
        v = (v + 1.0) / 2.0
    v = (v.clamp(0, 1) * 255.0).round().to(torch.uint8)
    return v.numpy()


def video_metrics(pred_thwc: np.ndarray, gt_thwc: np.ndarray, lpips_fn=None, device="cuda") -> dict:
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim

    T = min(len(pred_thwc), len(gt_thwc))
    pred_thwc, gt_thwc = pred_thwc[:T], gt_thwc[:T]
    ps, ss = [], []
    for t in range(T):
        ps.append(psnr(gt_thwc[t], pred_thwc[t], data_range=255))
        ss.append(ssim(gt_thwc[t], pred_thwc[t], channel_axis=2, data_range=255))
    out = {"psnr": float(np.mean(ps)), "ssim": float(np.mean(ss))}
    # temporal consistency: 帧间差的接近程度(pred 的运动是否贴合 GT 的运动)
    if T >= 2:
        pd = np.abs(np.diff(pred_thwc.astype(np.float32), axis=0)).mean()
        gd = np.abs(np.diff(gt_thwc.astype(np.float32), axis=0)).mean()
        out["temporal_absdiff_ratio"] = float(pd / (gd + 1e-6))
    if lpips_fn is not None:
        def to_lp(x):
            t = torch.from_numpy(x).float().permute(0, 3, 1, 2) / 127.5 - 1.0
            return t.to(device)
        with torch.no_grad():
            out["lpips"] = float(lpips_fn(to_lp(pred_thwc), to_lp(gt_thwc)).mean().item())
    return out


# ----------------------------- checkpoint discovery -----------------------------
def list_checkpoints(output_dir: str):
    found = []
    for d in glob.glob(os.path.join(output_dir, "**", "checkpoint_*_step_*"), recursive=True):
        m = re.search(r"step_(\d+)", os.path.basename(d))
        if not m:
            continue
        sub = None
        for cand in ("transformer_ema", "transformer"):
            if os.path.isdir(os.path.join(d, cand)) and glob.glob(os.path.join(d, cand, "*.safetensors")) + glob.glob(os.path.join(d, cand, "*.bin")):
                sub = os.path.join(d, cand); break
        if sub:
            found.append((int(m.group(1)), d, sub))
    return sorted(found)


# ----------------------------- one checkpoint eval -----------------------------
def eval_checkpoint(transformer_dir, args, val_ds, norm, t5, lpips_fn, delta_mask, device, dtype):
    from world_action_model.models.transformer_wa_casual import CasualWorldActionTransformer
    from world_action_model.pipeline.wa_pipeline import WAPipeline
    from world_action_model.pipeline.utils import add_state_to_action, build_ref_image, denormalize_action, normalize_state
    from diffusers.models import AutoencoderKLWan

    # action 横向 horizon:第 1 / 10 / chunk的一半 / chunk末步(1-indexed),"max"=chunk 长度
    horizons = sorted({h for h in (1, 10, args.action_chunk // 2, args.action_chunk) if 1 <= h <= args.action_chunk})

    vae = AutoencoderKLWan.from_pretrained(args.model_id, subfolder="vae", torch_dtype=torch.bfloat16)
    transformer = CasualWorldActionTransformer.from_pretrained(transformer_dir).to(dtype)
    pipe = WAPipeline.from_pretrained(args.model_id, vae=vae, transformer=transformer, torch_dtype=dtype).to(device)

    view_keys = ["observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist"]
    agg = {}
    n = min(args.n_clips, len(val_ds))
    for i in range(n):
        d = val_ds[i]
        # GT 帧(5,3,H,W)->ref 空间(stitched 768x192)
        imgs0 = {k: d[k][0] for k in view_keys}  # 首帧作 ref
        ref = build_ref_image(images=imgs0, dst_size=(args.width, args.height), crop_mode="center")
        state = d["observation.state"].float().unsqueeze(0)
        nstate = normalize_state(state, norm, mode="zscore").to(device=device, dtype=dtype)
        with torch.no_grad():
            imgs, action = pipe(height=args.height, width=args.width, action_chunk=args.action_chunk,
                                state=nstate, num_frames=args.num_frames, guidance_scale=0.0,
                                num_inference_steps=args.steps, image=ref, action_only=False,
                                return_dict=False, prompt_embeds=t5.unsqueeze(0).to(device=device, dtype=torch.float32))
        pred = _to_uint8_thwc(imgs[0])
        # GT: 同样 stitched 768x192,取 num_frames 帧
        gt_ref = build_ref_image(images={k: d[k] for k in view_keys}, dst_size=(args.width, args.height), crop_mode="center")
        gt = _to_uint8_thwc(gt_ref if gt_ref.ndim == 4 else gt_ref.unsqueeze(0))
        m = video_metrics(pred, gt, lpips_fn=lpips_fn, device=device)
        # action:反归一化 + add_state(与 inference_server 一致)-> 真实单位,再算逐 horizon MAE
        pred_act = denormalize_action(action[0].float(), norm, mode="zscore")
        pred_act = add_state_to_action(pred_act, state[0].float().to(pred_act.device),
                                       action_chunk=args.action_chunk, mask=delta_mask).cpu().numpy()
        gt_act = d["action"].float().numpy()
        L = min(len(pred_act), len(gt_act))
        ae = np.abs(pred_act[:L] - gt_act[:L])             # [L,14] 真实单位绝对误差
        m["action_mae"] = float(ae.mean())                 # 全 chunk 均值
        m["action_mse"] = float(((pred_act[:L] - gt_act[:L]) ** 2).mean())
        for h in horizons:                                  # mae@h:第 h 步(1-indexed)所有维均值
            if h <= L:
                m[f"mae@{h}"] = float(ae[h - 1].mean())
        for k, v in m.items():
            agg.setdefault(k, []).append(v)
        if args.save_mp4 and i < args.n_mp4:
            _save_side_by_side(pred, gt, os.path.join(args.vis_dir, f"clip{i:03d}.mp4"), fps=args.fps)
    del pipe, transformer, vae
    torch.cuda.empty_cache()
    return {k: float(np.mean(v)) for k, v in agg.items()}


def _save_side_by_side(pred_thwc, gt_thwc, path, fps=5):
    try:
        import torchvision
        T = min(len(pred_thwc), len(gt_thwc))
        cat = np.concatenate([gt_thwc[:T], pred_thwc[:T]], axis=2)  # 左GT|右Pred
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torchvision.io.write_video(path, torch.from_numpy(cat), fps=fps)
    except Exception as e:
        print(f"[eval] mp4 save failed: {e}")


# ----------------------------- main loop -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True, help="训练 project_dir(轮询 checkpoint)")
    ap.add_argument("--model_id", required=True, help="Wan2.2-TI2V-5B-Diffusers(取 vae/config)")
    ap.add_argument("--stats_path", required=True)
    ap.add_argument("--val_root", required=True, help="held-out 验证集 LeRobot 根目录")
    ap.add_argument("--t5_pkl", required=True)
    ap.add_argument("--n_clips", type=int, default=50)
    ap.add_argument("--n_mp4", type=int, default=3)
    ap.add_argument("--height", type=int, default=192)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--action_chunk", type=int, default=48)
    ap.add_argument("--num_frames", type=int, default=5)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--t5_len", type=int, default=64)
    ap.add_argument("--state_dim", type=int, default=14)
    ap.add_argument("--action_dim", type=int, default=14)
    ap.add_argument("--delta_mask", type=str, default="1,1,1,1,1,1,0,1,1,1,1,1,1,0",
                    help="14维 piper:关节 delta、夹爪(idx6/13)绝对;与 inference_server 一致")
    ap.add_argument("--poll", type=int, default=120)
    ap.add_argument("--once", action="store_true", help="只评当前最新 ckpt 一次后退出")
    ap.add_argument("--smoke", action="store_true", help="不评 ckpt,只在 GT 上自测指标函数")
    ap.add_argument("--no_lpips", action="store_true", help="跳过 LPIPS(离线无 alex 权重时)")
    ap.add_argument("--save_mp4", action="store_true", default=True)
    ap.add_argument("--vis_dir", default=None)
    ap.add_argument("--fps", type=int, default=5)
    args = ap.parse_args()
    args.vis_dir = args.vis_dir or os.path.join(args.output_dir, "eval_vis")

    device = "cuda"; dtype = torch.bfloat16
    from world_action_model.pipeline.utils import extract_normalization_tensors, load_stats, load_t5_embedding_from_pkl
    lpips_fn = None
    if not args.no_lpips:
        try:                       # LPIPS 需下载 alex 权重;离线/无网时优雅跳过(PSNR/SSIM/temporal 仍算)
            import lpips as lpips_mod
            lpips_fn = lpips_mod.LPIPS(net="alex").to(device).eval()
        except Exception as e:
            print(f"[eval] LPIPS unavailable ({type(e).__name__}); skipping lpips, keep PSNR/SSIM/temporal")

    if args.smoke:
        gt = (np.random.rand(5, 192, 768, 3) * 255).astype(np.uint8)
        print("[smoke] GT vs GT:", video_metrics(gt, gt, lpips_fn, device))
        noisy = np.clip(gt.astype(int) + np.random.randint(-40, 40, gt.shape), 0, 255).astype(np.uint8)
        print("[smoke] GT vs noisy:", video_metrics(noisy, gt, lpips_fn, device))
        print("SMOKE_OK"); return

    from giga_datasets import load_dataset
    val_ds = load_dataset([dict(_class_name="LeRobotDataset", data_path=args.val_root,
                                delta_info={"action": args.action_chunk},
                                delta_frames={k: [0, args.action_chunk // 4, args.action_chunk // 2,
                                                  3 * args.action_chunk // 4, args.action_chunk]
                                              for k in ["observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist"]},
                                embodiment="visrobot01", tolerance_s=1e-3)])
    stats = load_stats(args.stats_path)
    norm = extract_normalization_tensors(stats, device=device, state_dim=args.state_dim, action_dim=args.action_dim)
    t5 = load_t5_embedding_from_pkl(args.t5_pkl, target_len=args.t5_len).to(device=device, dtype=torch.float32)
    delta_mask = torch.tensor([p.strip() in ("1", "true", "True") for p in args.delta_mask.split(",") if p.strip()],
                              device=device, dtype=torch.bool)
    assert delta_mask.numel() == args.action_dim, f"delta_mask {delta_mask.numel()} != action_dim {args.action_dim}"

    log_path = os.path.join(args.output_dir, "eval_log.jsonl")
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(os.path.join(args.output_dir, "eval_tb"))
    except Exception:
        tb = None
    done = set()
    if os.path.exists(log_path):
        for l in open(log_path):
            try: done.add(int(json.loads(l)["step"]))
            except Exception: pass

    print(f"[eval] watching {args.output_dir} (poll={args.poll}s, n_clips={args.n_clips})")
    while True:
        cks = [c for c in list_checkpoints(args.output_dir) if c[0] not in done]
        if not cks and args.once:
            print("[eval] no new checkpoint"); break
        for step, ckdir, subdir in cks:
            t0 = time.time()
            print(f"[eval] step {step}: {subdir}")
            try:
                metrics = eval_checkpoint(subdir, args, val_ds, norm, t5, lpips_fn, delta_mask, device, dtype)
            except Exception as e:
                print(f"[eval] step {step} FAILED: {e}"); done.add(step); continue
            rec = {"step": step, "ckpt": ckdir, "secs": round(time.time() - t0, 1), **metrics}
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            if tb:
                for k, v in metrics.items():
                    tb.add_scalar(f"eval/{k}", v, step)
            print(f"[eval] step {step} -> {metrics}")
            done.add(step)
        if args.once:
            break
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
