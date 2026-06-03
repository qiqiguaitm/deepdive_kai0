"""Episode-grouped sampler for cached-latent training.

为什么需要:VAE latent 按 episode 文件缓存(~63MB/集)。若按窗口全局 shuffle,每个样本可能命中
不同 episode 文件 → 每样本读 63MB(比解 mp4 还慢)。本 sampler 把同一 episode 的窗口连续产出
(episode 顺序每 epoch 打乱、集内窗口打乱),配合 dataset 的 episode 文件 LRU(size 小即可)→
每集只读一次、~0.25MB/样本摊销。只产出**已缓存** episode 的窗口(跳过未抽 latent 的集)。

仅枚举 (sub-dataset, episode) 的 strided 窗口全局索引;ConcatDataset 的全局索引会被 _get_data
路由回正确子集与帧。
"""
import json
import math
import os

import numpy as np
import torch


class LatentEpisodeSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size=None, stride=4, num_frames=48,
                 shuffle=True, seed=6666, infinite=True):
        self.shuffle = shuffle
        self.seed = int(seed)
        self.epoch = 0
        self.infinite = infinite
        subs = dataset.datasets if hasattr(dataset, "datasets") else [dataset]
        self.groups = []  # 每项 = 某 (sub,episode) 的窗口全局索引 np.array
        offset = 0
        for sub in subs:
            sub.open()
            n = len(sub)
            root = sub.data_path
            latent_dir = getattr(sub, "latent_dir", None) or os.path.join(root, "vae_latent")
            eps = [json.loads(l) for l in open(os.path.join(root, "meta", "episodes.jsonl")) if l.strip()]
            gs = 0
            for e in eps:
                ei = int(e["episode_index"]); L = int(e["length"])
                if os.path.exists(os.path.join(latent_dir, f"episode_{ei:06d}.pt")):
                    starts = range(0, max(1, L - num_frames + 1), stride)
                    self.groups.append(np.fromiter((offset + gs + s for s in starts), dtype=np.int64))
                gs += L
            offset += n
        self.data_size = int(sum(len(g) for g in self.groups))
        assert self.data_size > 0, "LatentEpisodeSampler: no cached-latent windows found"
        self.batch_size = batch_size
        self.total_size = int(math.ceil(self.data_size / batch_size)) * batch_size if batch_size else self.data_size

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.total_size

    def __iter__(self):
        while True:
            rng = np.random.default_rng(self.seed + self.epoch)
            self.epoch += 1
            order = np.arange(len(self.groups))
            if self.shuffle:
                rng.shuffle(order)
            idx = []
            for gi in order:
                g = self.groups[gi]
                if self.shuffle and len(g) > 1:
                    g = g[rng.permutation(len(g))]
                idx.extend(g.tolist())
            if len(idx) < self.total_size:  # pad to multiple of batch_size
                idx.extend(idx[: self.total_size - len(idx)])
            idx = idx[: self.total_size]
            yield from idx
            if not self.infinite:
                break
