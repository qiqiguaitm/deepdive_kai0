"""Model-soup (uniform weight averaging) as a poor-man's EMA probe.

Goal: 临时验证 "PyTorch 比 JAX 差是因为缺 EMA" 假说 (见
docs/training/history/experiments/task_a_new_pure_200_new_norm_results.md §8.4).

EMA(decay=0.9999) 的有效平均窗口 ≈ 1/(1-0.9999) = 10000 步, 所以对最后 ~10k 步的
ckpt 做均匀平均 (40000..50000, 每 2000 一个, 共 6 个) 近似 EMA 尾部。

输出一个 souped ckpt 目录 (复制参考 ckpt 的 assets/metadata, 只替换 model.safetensors),
之后用 eval_val_action_mse.py 评估, 与 plain step 50000 对比 MAE@{1,10,25,50}。

Usage:
  python train_scripts/kai/eval/model_soup_ema_probe.py \
    --ckpt-root kai0/checkpoints/pi05_pytorch_a_new_pure_200/A_mirror200_pi05_pytorch \
    --steps 40000,42000,44000,46000,48000,50000 \
    --ref-step 50000 \
    --out kai0/checkpoints/pi05_pytorch_a_new_pure_200/A_mirror200_pi05_pytorch/soup_40k_50k
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import safetensors.torch
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", required=True, help="dir containing <step>/ subdirs")
    ap.add_argument("--steps", required=True, help="comma-separated step list to average")
    ap.add_argument("--ref-step", required=True, help="step whose dir is copied as the template (assets/metadata)")
    ap.add_argument("--out", required=True, help="output souped ckpt dir")
    args = ap.parse_args()

    root = Path(args.ckpt_root)
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    out = Path(args.out)

    safet_paths = [root / s / "model.safetensors" for s in steps]
    for p in safet_paths:
        if not p.exists():
            raise FileNotFoundError(f"missing {p}")
    print(f"Averaging {len(steps)} ckpts: {steps}")

    # Accumulate in float32 for numerical stability.
    acc: dict[str, torch.Tensor] = {}
    dtypes: dict[str, torch.dtype] = {}
    n = len(safet_paths)
    for i, p in enumerate(safet_paths):
        sd = safetensors.torch.load_file(str(p))
        for k, v in sd.items():
            if i == 0:
                dtypes[k] = v.dtype
            if v.is_floating_point():
                acc[k] = (v.to(torch.float32) if k not in acc else acc[k] + v.to(torch.float32))
            else:
                # non-float (int/bool buffers): keep from the LAST ckpt (ref), not averaged
                acc[k] = v.clone()
        del sd
        print(f"  loaded {p}")

    souped = {}
    for k, v in acc.items():
        if dtypes[k].is_floating_point:
            souped[k] = (v / n).to(dtypes[k])
        else:
            souped[k] = v  # last-ckpt value, original dtype

    # Build output dir from the reference ckpt (copy everything, then overwrite model.safetensors).
    ref = root / args.ref_step
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(ref, out)
    # remove optimizer.pt to save space (not needed for eval)
    opt = out / "optimizer.pt"
    if opt.exists():
        opt.unlink()
    safetensors.torch.save_file(souped, str(out / "model.safetensors"))
    print(f"\nSouped ckpt written -> {out}")
    print("Next: eval with eval_val_action_mse.py --ckpt-path", out)


if __name__ == "__main__":
    main()
