#!/usr/bin/env python
"""B线(P3 前哨): 抽 LIBERO primary 相机的 **SigLIP-So400m patch-token 均值** 特征(= pi05 真 token 空间)。

★ 只取 patch token 层(vision_model.last_hidden_state 的 token 均值), **不取**文本对齐 pooled 头
  (get_image_features)—— 后者已被证会杀死 r 场的读法②(边界), 两代 SigLIP 一致。见 r 场三读法结论 §2。
  这也正是 pi05 实际喂给 LLM 的那一层。

输出格式与 p1_libero_dinov3base_extract.py / p1_libero_qwen3vl_extract.py 一致
(npz, key='grid', [N, tokens, D], fp16), 下游 CRAVE/LMWM 脚本零改动复用。
帧加载与 qwen3vl 版逐字同构 → 保证两个空间抽的是同一批帧。

用法: python p1_libero_so400m_extract.py --smoke --per-task 1
      python p1_libero_so400m_extract.py --eps all
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

REPO = "/vePFS/tim/workspace/deepdive_kai0"
ROOT = f"{REPO}/lmvla/lawam/dataset/libero_merged_no_noops_20hz"
CAM = "observation.images.image"
SO400M = f"{REPO}/lmvla/lmwm/data/hf_so400m"          # aria2c 本地目录(download_methods.md §5)

from p1_libero_qwen3vl_extract import load_episode_frames  # noqa: E402  帧加载完全复用, 保证同批帧


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", default="", help="all | a,b,c | start:end (与 --per-task 二选一)")
    ap.add_argument("--per-task", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--pool", default="mean", choices=["mean", "grid"],
                    help="mean: patch token 均值 [N,1,D](r 场用); grid: 全 token [N,T,D](生成器用)")
    ap.add_argument("--out", default=f"{REPO}/lmvla/lmwm/data/libero_so400m")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    epm = pd.read_parquet(sorted(glob.glob(f"{ROOT}/meta/episodes/**/*.parquet", recursive=True))[0])
    # ★ v3.0 一个 parquet 含多个 episode, 必须按 episode_index 分组取 task_index(取 [0] 会全判成同一 task)
    dpar = sorted(glob.glob(f"{ROOT}/data/**/*.parquet", recursive=True))
    ep2task = pd.concat([pd.read_parquet(p, columns=["episode_index", "task_index"]) for p in dpar]) \
        .groupby("episode_index")["task_index"].first().to_dict()

    if a.per_task > 0:
        by = {}
        for e in sorted(ep2task):
            ti = int(ep2task[e]); by.setdefault(ti, [])
            if len(by[ti]) < a.per_task:
                by[ti].append(int(e))
        eps = sorted(x for v in by.values() for x in v)
        print(f"[eps] 每任务{a.per_task}个 → {len(eps)} ep, 覆盖 {len(by)} 个 task", flush=True)
    else:
        allep = list(epm["episode_index"])
        if a.eps == "all":
            eps = allep
        elif ":" in a.eps:
            s, e = a.eps.split(":"); eps = allep[int(s):int(e)]
        else:
            eps = [int(x) for x in a.eps.split(",")]
        print(f"[eps] {len(eps)} ep", flush=True)

    proc = AutoProcessor.from_pretrained(SO400M)
    mdl = AutoModel.from_pretrained(SO400M, torch_dtype=torch.bfloat16).cuda().eval()
    print(f"[enc] SigLIP-So400m vision tower loaded, pool={a.pool} (patch token 层, 非文本对齐头)", flush=True)

    os.makedirs(a.out, exist_ok=True)
    for idx, e in enumerate(eps):
        outf = os.path.join(a.out, f"ep{e}.npz")
        if os.path.exists(outf) and not a.smoke:
            continue                                        # 断点续跑
        row = epm[epm["episode_index"] == e].iloc[0]
        frames = load_episode_frames(ROOT, CAM, row, stride=a.stride)
        if not frames:
            print(f"  ! ep{e} 无帧, 跳过", flush=True); continue
        outs = []
        for i in range(0, len(frames), a.batch):
            chunk = [Image.fromarray(f) for f in frames[i:i + a.batch]]
            px = proc(images=chunk, return_tensors="pt")["pixel_values"].to("cuda", torch.bfloat16)
            with torch.no_grad():
                h = mdl.vision_model(pixel_values=px).last_hidden_state   # [B, T, D] patch tokens
            h = h.float().cpu().numpy()
            outs.append(h.mean(1, keepdims=True) if a.pool == "mean" else h)
        g = np.concatenate(outs, 0)
        if a.smoke:
            print(f"  ep{e}: frames={len(frames)} grid={g.shape} (期望 [~{row['length']}, "
                  f"{1 if a.pool=='mean' else 'T'}, D]) mean={g.mean():.3f} std={g.std():.3f}", flush=True)
        else:
            np.savez_compressed(outf, grid=g.astype(np.float16))
            if idx % 10 == 0:
                print(f"  [{idx+1}/{len(eps)}] ep{e}: {g.shape}", flush=True)
    print("EXTRACT_DONE", flush=True)


if __name__ == "__main__":
    main()
