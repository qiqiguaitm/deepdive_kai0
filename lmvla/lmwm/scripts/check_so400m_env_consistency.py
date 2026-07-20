#!/usr/bin/env python
"""核查 So400m patch-mean 特征是否跨环境一致(srpo tf4.57.6 vs kai0/.venv tf5.13.1)。

动机: ENV_SELECTION_RULES §1 未给 SigLIP/So400m 抽取指定环境; 而本会话已被 transformers
版本差异坑过(DINOv3 模块嵌套变化 -> LAM 204 key 不匹配)。若两环境产出不一致,
则 kai0_aligned_urvc(UR-VC 可比性资产)与新抽的 LIBERO 特征不可混用。

用法(两个环境各跑一次, 再比对):
  <python> check_so400m_env_consistency.py --out /tmp/so400m_env_<tag>.npy
  python check_so400m_env_consistency.py --compare /tmp/so400m_env_a.npy /tmp/so400m_env_b.npy
"""
import argparse
import os

import numpy as np

SO400M = "/vePFS/tim/workspace/deepdive_kai0/lmvla/lmwm/data/hf_so400m"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--compare", nargs=2, default=None)
    a = ap.parse_args()

    if a.compare:
        x, y = (np.load(p) for p in a.compare)
        d = np.abs(x - y)
        cos = float((x * y).sum(1).mean() / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)).mean())
        print(f"shape {x.shape} vs {y.shape}")
        print(f"max|Δ| = {d.max():.3e}   mean|Δ| = {d.mean():.3e}   逐行 cos = {cos:.8f}")
        print("判定:", "✅ 比特级/数值级一致, 环境无关" if d.max() < 1e-2
              else "⚠️ 存在差异 —— 两环境特征不可混用")
        return

    import torch
    from transformers import AutoModel, AutoProcessor
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    import transformers
    print(f"transformers {transformers.__version__}  torch {torch.__version__}", flush=True)

    rng = np.random.default_rng(0)                       # 固定合成输入, 排除数据读取差异
    imgs = [rng.integers(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(8)]
    proc = AutoProcessor.from_pretrained(SO400M)
    mdl = AutoModel.from_pretrained(SO400M, torch_dtype=torch.float32).cuda().eval()   # fp32 排除 bf16 噪声
    px = proc(images=[i for i in imgs], return_tensors="pt")["pixel_values"].to("cuda")
    with torch.no_grad():
        h = mdl.vision_model(pixel_values=px).last_hidden_state
    f = h.mean(1).float().cpu().numpy()
    print(f"feat {f.shape} mean={f.mean():.6f} std={f.std():.6f}", flush=True)
    if a.out:
        np.save(a.out, f)
        print(f"[save] {a.out}")


if __name__ == "__main__":
    main()
