"""Canonical latent -> image decoder for LMWM: RETRIEVAL (nearest real frame).

LMWM predicts real-frame DINOv3-H latents, so the correct way to turn a predicted
latent into an image is to retrieve the nearest real frame -- sharp, real, and
faithful to the prediction. The pooled synthesis decoder
(`scripts/train_dinov3h_decoder.py`) is inherently blurry (pooled 1280-D discards
spatial layout; L1/L2 loss predicts the mean) and is DEPRECATED for visualization
and VLA subgoal rendering. Use this class everywhere instead.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


class LatentRetrievalDecoder:
    """Map a (batch of) DINOv3-H latent(s) to the nearest real frame image(s)."""

    def __init__(
        self,
        feature_dir: str | Path = "temp/crave_full_dinov3h",
        dataset_root: str | Path = "kai0/data/Task_A/kai0_base",
        camera: str = "observation.images.top_head",
        device: str = "cpu",
        res: int | None = None,
    ) -> None:
        import torch

        self.feature_dir = Path(feature_dir)
        self.dataset_root = Path(dataset_root)
        self.camera = camera
        self.res = res
        self.device = torch.device(device if (device != "cpu" and torch.cuda.is_available()) else "cpu")
        self.chunks_size = int(json.loads((self.dataset_root / "meta/info.json").read_text())["chunks_size"])

        idx = np.load(self.feature_dir / "index.npz")
        self.E = idx["E"].astype(np.int64)
        self.FR = idx["FR"].astype(np.int64)
        n = int(idx["n"])
        feat = np.zeros((n, 1280), dtype=np.float16)
        valid = np.zeros(n, dtype=bool)
        for shard in sorted(self.feature_dir.glob("shard_*.npz")):
            z = np.load(shard)
            gi = z["gidx"].astype(np.int64)
            feat[gi] = z["feat"]
            valid[gi] = z["valid"].astype(bool)
        self.E = self.E[valid]
        self.FR = self.FR[valid]
        self.feat = _l2(feat[valid].astype(np.float32))
        self._feat_t = torch.from_numpy(self.feat).to(self.device)
        self._caps: dict[int, object] = {}

    def retrieve(self, latents: np.ndarray, topk: int = 1, exclude_episode: int | None = None):
        """Return (gidx[B,k], cos[B,k]) of the nearest real frames to each latent."""
        import torch

        q = _l2(np.atleast_2d(latents).astype(np.float32))
        qt = torch.from_numpy(q).to(self.device)
        sim = qt @ self._feat_t.T  # (B, N)
        if exclude_episode is not None:
            mask = torch.from_numpy(self.E == exclude_episode).to(self.device)
            sim = sim.masked_fill(mask[None, :], -2.0)
        cos, idx = sim.topk(topk, dim=1)
        return idx.cpu().numpy(), cos.cpu().numpy()

    def frame(self, gidx: int) -> np.ndarray:
        """Read the real RGB frame for a global feature index."""
        import cv2

        ep = int(self.E[gidx])
        if ep not in self._caps:
            mp4 = self.dataset_root / f"videos/chunk-{ep // self.chunks_size:03d}/{self.camera}/episode_{ep:06d}.mp4"
            self._caps[ep] = cv2.VideoCapture(str(mp4))
        cap = self._caps[ep]
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(self.FR[gidx]))
        ok, im = cap.read()
        if not ok:
            h = self.res or 128
            return np.zeros((h, h, 3), np.uint8)
        rgb = im[:, :, ::-1]
        if self.res is not None:
            import cv2 as _cv2
            rgb = _cv2.resize(rgb, (self.res, self.res))
        return np.ascontiguousarray(rgb)

    def decode(self, latents: np.ndarray, exclude_episode: int | None = None) -> np.ndarray:
        """Canonical decode: latent -> nearest real frame image(s). Returns (B,H,W,3) uint8."""
        idx, _ = self.retrieve(latents, topk=1, exclude_episode=exclude_episode)
        return np.stack([self.frame(int(i[0])) for i in idx])

    def close(self) -> None:
        for c in self._caps.values():
            c.release()
        self._caps.clear()
