"""离线预抽 VAE 视频 latent(根治训练数据瓶颈)。分片 + DataLoader 并行解码 + 批量 VAE 编码。

严格对齐 forward_vae:images(b,t,c,h,w)->rearrange(b c t h w)->vae.encode().latent_dist.mode()->(x-mean)*std。
本数据 transform 确定性:max_ref_frames=1→ref=帧0;480x640→192x256 等比例→裁剪 randint(0,0)=0 无随机。
缓存:{root}/vae_latent/episode_{idx:06d}.pt = {stride,starts,visual[N,C,T,h,w] bf16, ref[N,C,1,h,w] bf16}

多 GPU/多节点:每 (node,gpu) 起一个进程,--shard k --num-shards K 按 episode 取模分片;DataLoader 在
进程内用 --workers 并行解码。用法见 run_precompute_aihc.sh。
"""
import argparse, json, os
import torch
from einops import rearrange
from torch.utils.data import Dataset, DataLoader
from diffusers.models import AutoencoderKLWan
from world_action_model.transformers.wa_transforms_lerobot import WATransformsLerobot
from giga_datasets import load_dataset

DATA = os.environ.get("GWP_DATA", "../kai0/data/wam_fold_v1")
CKPT = os.environ.get("WAN_DIFFUSERS", "../checkpoints/Wan2.2-TI2V-5B-Diffusers")
NORM = "./assets_visrobot01/norm_stats_vis.json"
# [ACWM unify] 可用环境变量覆盖相机序/时序窗/输出目录(默认与原行为一致,向后兼容):
#   GWP_VIEW_KEYS=cam1,cam2,cam3  统一相机序(俯视,左腕,右腕);visrobot 用 top_head,hand_left,hand_right
#   GWP_OFFS=0,4,8,...,48         时序采样偏移;更密(如 range(0,49,4)=13帧)→ 更长 latent 窗(T_lat=4)→ 支持 K>1 history
#   GWP_OUT_SUBDIR=vae_latent_uni 输出子目录(避免覆盖已有 vae_latent)
VIEW_KEYS = os.environ.get(
    "GWP_VIEW_KEYS",
    "observation.images.cam_high,observation.images.cam_left_wrist,observation.images.cam_right_wrist",
).split(",")
NUM_FRAMES = 48
if os.environ.get("GWP_OFFS"):
    OFFS = [int(x) for x in os.environ["GWP_OFFS"].split(",")]
else:
    OFFS = [0, NUM_FRAMES // 4, NUM_FRAMES // 2, 3 * NUM_FRAMES // 4, NUM_FRAMES]


class WindowDS(Dataset):
    """枚举本分片各 episode 的 strided 窗口;__getitem__ 解码+transform 返回 images/ref(并行)。"""
    def __init__(self, root, emb_id, stride, gstart, windows):
        self.windows = windows  # list of (epi, fstart, gidx)
        self.tr = WATransformsLerobot(robotype_to_embed_id={"visrobot01": 0, "kairobot01": 1}, dst_size=(256, 192),
                                      num_frames=NUM_FRAMES, is_train=True, norm_path=[NORM, NORM], model_action_dim=14,
                                      num_views=3, t5_len=64, view_keys=VIEW_KEYS,
                                      image_cfg=dict(mask_generator=dict(max_ref_frames=1, start=1, factor=4)))
        self.ds = load_dataset([dict(_class_name="LeRobotDataset", data_path=root, delta_info={"action": NUM_FRAMES},
                                     delta_frames={k: OFFS for k in VIEW_KEYS}, embodiment=emb_id, tolerance_s=1e-3)])

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        epi, f, gidx = self.windows[i]
        o = self.tr(self.ds[gidx])
        return epi, f, o["images"].float(), o["ref_images"][:1].float()  # (t,c,h,w),(1,c,h,w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-episodes", type=int, default=0)
    args = ap.parse_args()
    emb_id = "visrobot01" if "vis" in args.emb else "kairobot01"
    root = f"{DATA}/{args.emb}"
    out_dir = f"{root}/{os.environ.get('GWP_OUT_SUBDIR', 'vae_latent')}"; os.makedirs(out_dir, exist_ok=True)
    dev, dt = "cuda", torch.bfloat16

    eps = [json.loads(l) for l in open(f"{root}/meta/episodes.jsonl") if l.strip()]
    gstart, acc = {}, 0
    for e in eps:
        gstart[int(e["episode_index"])] = acc; acc += int(e["length"])
    # 本分片负责的 episode(按序号取模),并跳过已完成的
    my_eps = [(int(e["episode_index"]), int(e["length"])) for j, e in enumerate(eps) if j % args.num_shards == args.shard]
    if args.max_episodes:
        my_eps = my_eps[: args.max_episodes]
    my_eps = [(i, L) for i, L in my_eps if not os.path.exists(f"{out_dir}/episode_{i:06d}.pt")]
    if not my_eps:
        print(f"DONE shard{args.shard} ({args.emb}): nothing to do", flush=True); return
    windows = []
    expected = {}
    for i, L in my_eps:
        ws = list(range(0, max(1, L - NUM_FRAMES + 1), args.stride))
        expected[i] = len(ws)
        for f in ws:
            windows.append((i, f, gstart[i] + f))
    print(f"[{args.emb} shard{args.shard}] {len(my_eps)} eps, {len(windows)} windows", flush=True)

    vae = AutoencoderKLWan.from_pretrained(CKPT, subfolder="vae", torch_dtype=dt).to(dev).eval()
    lm = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(dev, dt)
    ls = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(dev, dt)

    @torch.no_grad()
    def enc(x):  # x:(b,t,c,h,w) -> (b,C,T,h,w)
        x = rearrange(x.to(dev, dt), "b t c h w -> b c t h w")
        z = vae.encode(x).latent_dist.mode()
        return ((z - lm) * ls).to(torch.bfloat16).cpu()

    dl = DataLoader(WindowDS(root, emb_id, args.stride, gstart, windows), batch_size=args.batch,
                    num_workers=args.workers, collate_fn=lambda b: b)  # list of (epi,f,images,ref)
    buf = {}  # epi -> {"starts":[], "visual":[], "ref":[]}
    n = 0; saved = 0
    for batch in dl:
        imgs = torch.stack([x[2] for x in batch]); refs = torch.stack([x[3] for x in batch])
        vz = enc(imgs); rz = enc(refs)
        for k, (epi, f, _, _) in enumerate(batch):
            b = buf.setdefault(epi, {"starts": [], "visual": [], "ref": []})
            b["starts"].append(f); b["visual"].append(vz[k]); b["ref"].append(rz[k])
            if len(b["starts"]) >= expected[epi]:   # 该 episode 全部窗口完成 → 立即落盘并释放内存
                tmp = f"{out_dir}/episode_{epi:06d}.pt.tmp"
                torch.save({"stride": args.stride, "starts": b["starts"],
                            "visual": torch.stack(b["visual"]), "ref": torch.stack(b["ref"])}, tmp)
                os.replace(tmp, f"{out_dir}/episode_{epi:06d}.pt")   # 原子落盘
                del buf[epi]; saved += 1
        n += len(batch)
        if n % 400 == 0:
            print(f"[{args.emb} shard{args.shard}] {n}/{len(windows)} windows, {saved} eps saved", flush=True)
    # 兜底:保存任何残留(理论上 expected 触发后应为空)
    for epi, b in list(buf.items()):
        torch.save({"stride": args.stride, "starts": b["starts"],
                    "visual": torch.stack(b["visual"]), "ref": torch.stack(b["ref"])},
                   f"{out_dir}/episode_{epi:06d}.pt"); saved += 1
    print(f"DONE shard{args.shard} ({args.emb}): {saved} eps, {n} windows -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
