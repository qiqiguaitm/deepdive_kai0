#!/usr/bin/env python
"""抽 LIBERO primary 相机的 **Qwen3-VL 视觉塔** grid 特征(= VLA 自身编码器空间)。

用途: 跨空间预实验 —— 与 libero_dinov3base(DINOv3-vitb16)对照, 检验
"世界模型换到 VLA 自身特征空间后, CRAVE 发现的 milestone 边界是否改变"。
重合度高 ⇒ 空间选择不重要; 显著分歧 ⇒ 空间决定了"什么算子目标"。

★ 取的是 `visual(...).pooler_output`(= merger 输出, [tokens, 2048]),
  这正是**送进 LLM 的那一层**; 而 `last_hidden_state` 是 merge 前的 1024 维,
  不是 VLA 真正消费的表示。用后者会让"共用 VLA 编码器空间"这个论证失真。

输出格式与 p1_libero_dinov3base_extract.py 完全一致(npz, key='grid', [N, tokens, D], fp16),
以便下游 CRAVE 脚本零改动复用。Qwen3-VL: 256x256 输入 → patch16 → 16x16=256 → spatial_merge2 → 64 token/帧。

用法: python p1_libero_qwen3vl_extract.py --per-task 5            (每任务5个ep, 预实验)
      python p1_libero_qwen3vl_extract.py --eps all               (全量)
"""
import os, sys, argparse, glob
import numpy as np, pandas as pd

REPO = "/vePFS/tim/workspace/deepdive_kai0"
ROOT = f"{REPO}/lmvla/lawam/dataset/libero_merged_no_noops_20hz"
CAM = "observation.images.image"
QWEN = f"{REPO}/lmvla/lawam/results/Checkpoints/qwen3_weights"
IMG_HW = 256   # 与 starVLA 训练时的 image_resolution 一致(train_libero.yaml: image_resolution 256)


def load_episode_frames(root, cam, ep_row, stride=1):
    """与 dinov3 版逐字同构, 保证两个空间抽的是**同一批帧**。"""
    import av
    fi = int(ep_row[f"videos/{cam}/file_index"]); ci = int(ep_row[f"videos/{cam}/chunk_index"])
    t0 = float(ep_row[f"videos/{cam}/from_timestamp"]); t1 = float(ep_row[f"videos/{cam}/to_timestamp"])
    mp4 = os.path.join(root, "videos", cam, f"chunk-{ci:03d}", f"file-{fi:03d}.mp4")
    cont = av.open(mp4); stream = cont.streams.video[0]
    try:
        cont.seek(max(0, int((t0 - 0.5) / stream.time_base)), stream=stream, any_frame=False, backward=True)
    except Exception:
        pass
    frames = []; k = 0
    for fr in cont.decode(video=0):
        ts = float(fr.pts * stream.time_base)
        if ts < t0 - 1e-4: continue
        if ts > t1 - 1e-4: break
        if k % stride == 0: frames.append(fr.to_ndarray(format="rgb24"))
        k += 1
    cont.close()
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", default="", help="all | a,b,c | start:end (与 --per-task 二选一)")
    ap.add_argument("--per-task", type=int, default=0, help=">0 时每个 task 取前 N 个 ep")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--batch", type=int, default=32, help="每次前向的帧数")
    ap.add_argument("--out", default=f"{REPO}/lmvla/lmwm/data/libero_qwen3vl")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from PIL import Image

    epm = pd.read_parquet(sorted(glob.glob(f"{ROOT}/meta/episodes/**/*.parquet", recursive=True))[0])

    # episode -> task_index。★ v3.0 的**一个 parquet 文件含多个 episode**,
    #   取 task_index[0] 会把所有 ep 判成同一个 task(2026-07-19 踩过)。
    #   必须按 episode_index 分组 —— 与 p1_libero_milestone_pairs_finalarch.py 同法。
    dpar = sorted(glob.glob(f"{ROOT}/data/**/*.parquet", recursive=True))
    ep2task = pd.concat([pd.read_parquet(p, columns=["episode_index", "task_index"]) for p in dpar]) \
        .groupby("episode_index")["task_index"].first().to_dict()

    if a.per_task > 0:
        by = {}
        for e in sorted(ep2task):
            ti = int(ep2task[e]); by.setdefault(ti, [])
            if len(by[ti]) < a.per_task: by[ti].append(int(e))
        eps = sorted(x for v in by.values() for x in v)
        print(f"[eps] 每任务{a.per_task}个 → {len(eps)} ep, 覆盖 {len(by)} 个 task", flush=True)
    else:
        allep = list(epm["episode_index"])
        if a.eps == "all": eps = allep
        elif ":" in a.eps: s, e = a.eps.split(":"); eps = allep[int(s):int(e)]
        else: eps = [int(x) for x in a.eps.split(",")]
        print(f"[eps] {len(eps)} ep", flush=True)

    proc = AutoProcessor.from_pretrained(QWEN)
    model = AutoModelForImageTextToText.from_pretrained(QWEN, dtype=torch.bfloat16, device_map="cuda:0")
    vis = model.model.visual
    vis.eval()
    print("[enc] Qwen3-VL visual tower loaded (取 pooler_output = merger 输出, VLA 实际消费层)", flush=True)

    os.makedirs(a.out, exist_ok=True)
    for idx, e in enumerate(eps):
        outf = os.path.join(a.out, f"ep{e}.npz")
        if os.path.exists(outf) and not a.smoke:
            continue                                    # 断点续跑
        row = epm[epm["episode_index"] == e].iloc[0]
        frames = load_episode_frames(ROOT, CAM, row, stride=a.stride)
        if not frames:
            print(f"  ! ep{e} 无帧, 跳过", flush=True); continue
        outs = []
        for i in range(0, len(frames), a.batch):
            chunk = [Image.fromarray(f).resize((IMG_HW, IMG_HW), Image.BILINEAR) for f in frames[i:i + a.batch]]
            ip = proc.image_processor(images=chunk, return_tensors="pt")
            with torch.no_grad():
                o = vis(ip["pixel_values"].to("cuda:0", torch.bfloat16),
                        grid_thw=ip["image_grid_thw"].to("cuda:0")).pooler_output
            outs.append(o.float().cpu().numpy().reshape(len(chunk), -1, o.shape[-1]))
        g = np.concatenate(outs, 0)                     # [N, 64, 2048]
        if a.smoke:
            print(f"  ep{e}: frames={len(frames)} grid={g.shape} (期望 [~{row['length']}, 64, 2048]) "
                  f"mean={g.mean():.3f} std={g.std():.3f}", flush=True)
        else:
            np.savez_compressed(outf, grid=g.astype(np.float16))
            if idx % 10 == 0:
                print(f"  [{idx+1}/{len(eps)}] ep{e}: {g.shape}", flush=True)
    print("EXTRACT_DONE", flush=True)


if __name__ == "__main__":
    main()
