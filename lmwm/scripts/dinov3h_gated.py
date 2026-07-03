#!/usr/bin/env python
"""Self-contained DINOv3 ViT-H/16+ (gated SwiGLU MLP) encoder — pure torch, loads the
HF lvd1689m safetensors directly. Needed on gf3 where kai0's transformers (4.53.2)
lacks dinov3. Reuses the verified building blocks from openpi's dinov3_vit_standalone
(embeddings/RoPE/attention/layerscale) and adds the gated MLP (use_gated_mlp=True).

Output matches crave's HFDinoEncoder.encode_grid: (N, 256, 1280) patch tokens.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file

_STD = Path(__file__).resolve().parents[2] / "kai0/src/openpi/models_pytorch/dinov3_vit_standalone.py"
_spec = importlib.util.spec_from_file_location("_d3s", _STD)
_d3s = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_d3s)
_Embeddings, _Rope, _Attn, _LayerScale = _d3s._Embeddings, _d3s._Rope, _d3s._Attn, _d3s._LayerScale


class _GatedMLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        b = c.get("mlp_bias", True)
        self.gate_proj = nn.Linear(c["hidden_size"], c["intermediate_size"], bias=b)
        self.up_proj = nn.Linear(c["hidden_size"], c["intermediate_size"], bias=b)
        self.down_proj = nn.Linear(c["intermediate_size"], c["hidden_size"], bias=b)
        self.act = nn.SiLU()  # hidden_act=silu

    def forward(self, x):
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class _GatedLayer(nn.Module):
    def __init__(self, c):
        super().__init__()
        eps, d = c["layer_norm_eps"], c["hidden_size"]
        self.norm1 = nn.LayerNorm(d, eps=eps); self.attention = _Attn(c); self.layer_scale1 = _LayerScale(c)
        self.norm2 = nn.LayerNorm(d, eps=eps); self.mlp = _GatedMLP(c); self.layer_scale2 = _LayerScale(c)

    def forward(self, x, pos):
        x = x + self.layer_scale1(self.attention(self.norm1(x), pos))
        x = x + self.layer_scale2(self.mlp(self.norm2(x)))
        return x


class DINOv3HGated(nn.Module):
    def __init__(self, model_dir: str, device: str = "cuda:0", dtype=torch.bfloat16):
        super().__init__()
        c = json.loads((Path(model_dir) / "config.json").read_text())
        self.n_prefix = 1 + c["num_register_tokens"]
        self.embeddings = _Embeddings(c)
        self.rope_embeddings = _Rope(c)
        self.layer = nn.ModuleList([_GatedLayer(c) for _ in range(c["num_hidden_layers"])])
        self.norm = nn.LayerNorm(c["hidden_size"], eps=c["layer_norm_eps"])
        sd = load_file(str(Path(model_dir) / "model.safetensors"))
        missing, unexpected = self.load_state_dict(sd, strict=False)
        missing = [m for m in missing if "inv_freq" not in m]
        if missing or unexpected:
            raise RuntimeError(f"DINOv3-H load mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
        p = json.loads((Path(model_dir) / "preprocessor_config.json").read_text())
        self.register_buffer("mean", torch.tensor(p["image_mean"]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(p["image_std"]).view(1, 3, 1, 1))
        self.to(device=device, dtype=dtype).eval()
        for pp in self.parameters():
            pp.requires_grad_(False)
        self.device, self.dtype = device, dtype

    def _forward(self, px):
        x = self.embeddings(px)
        pos = self.rope_embeddings(px)
        for L in self.layer:
            x = L(x, pos)
        return self.norm(x)

    @torch.no_grad()
    def forward(self, px):
        return self._forward(px)

    def encode_pooled_tensor(self, x01):
        """Differentiable pooled encode. x01: (B,3,H,W) in [0,1]. Returns (B,1280) pooled."""
        import torch.nn.functional as _F
        if x01.shape[-1] != 256:
            x01 = _F.interpolate(x01, size=(256, 256), mode="bilinear", align_corners=False)
        x = ((x01.to(self.dtype) - self.mean.to(self.dtype)) / self.std.to(self.dtype))
        return self._forward(x)[:, self.n_prefix:].mean(1).float()

    @torch.no_grad()
    def encode_grid(self, imgs, bs: int = 64) -> np.ndarray:
        """imgs: (N,H,W,3) uint8 (H=W=256 expected) -> (N,256,1280) fp32 patch tokens."""
        mean = self.mean.to(dtype=self.dtype); std = self.std.to(dtype=self.dtype)
        out = []
        for s in range(0, len(imgs), bs):
            b = torch.from_numpy(imgs[s:s + bs].astype(np.float32) / 255.0).permute(0, 3, 1, 2)
            b = ((b.to(self.device, self.dtype) - mean) / std)
            y = self.forward(b)[:, self.n_prefix:]        # (b, P, 1280)
            out.append(y.float().cpu().numpy())
        return np.concatenate(out)


if __name__ == "__main__":  # local verification vs crave HFDinoEncoder
    import argparse
    import sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--dino_path", default="/vePFS/xiezhicong/.cache/huggingface/hub/dinov3-vith16plus-pretrain-lvd1689m")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "crave/src"))
    from crave.encoders import load_encoder
    rng = np.random.default_rng(0)
    imgs = rng.integers(0, 256, (12, 256, 256, 3), dtype=np.uint8)
    mine = DINOv3HGated(args.dino_path, args.device).encode_grid(imgs)      # (12,256,1280)
    ref = load_encoder("dinov3-h", device=args.device).encode_grid(imgs)    # (12,1280,16,16)
    ref = ref.reshape(12, 1280, 256).transpose(0, 2, 1)                     # (12,256,1280)
    a = mine.reshape(-1, 1280); b = ref.reshape(-1, 1280)
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8)
    print(f"gated-H vs crave: per-token cos mean={cos.mean():.5f} min={cos.min():.5f} | L1={np.abs(mine-ref).mean():.4f}")
