"""Standalone pure-PyTorch DINOv3-ViT (config-driven, supports gated SwiGLU MLP) — NO transformers.

Port of transformers DINOv3ViTModel forward with param names matching the HF safetensors so
load_state_dict maps 1:1. Handles both ViT-L/16 (non-gated GELU MLP) and ViT-H+/16 lvd1689m
(use_gated_mlp=True, silu SwiGLU). Needed because kai0 pins transformers==4.53.2 which lacks dinov3,
and gf3's env can't load it. Weights: the exact HF lvd1689m safetensors the LMWM was trained on.

forward(pixels [B,3,H,W], ImageNet-normalized) -> last_hidden_state (B, 1+R+P, hidden).
Token order = [CLS, R register tokens, P patch tokens]; patches = [:, 1+R:].
"""
from __future__ import annotations
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


class _Embeddings(nn.Module):
    def __init__(self, c):
        super().__init__()
        h = c["hidden_size"]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, h))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, h))
        self.register_tokens = nn.Parameter(torch.zeros(1, c["num_register_tokens"], h))
        self.patch_embeddings = nn.Conv2d(c["num_channels"], h, c["patch_size"], c["patch_size"])

    def forward(self, px):
        pe = self.patch_embeddings(px).flatten(2).transpose(1, 2)
        b = px.shape[0]
        return torch.cat([self.cls_token.expand(b, -1, -1),
                          self.register_tokens.expand(b, -1, -1), pe], dim=1)


def _rope_coords(nph, npw, device):
    ch = torch.arange(0.5, nph, dtype=torch.float32, device=device) / nph
    cw = torch.arange(0.5, npw, dtype=torch.float32, device=device) / npw
    coords = torch.stack(torch.meshgrid(ch, cw, indexing="ij"), dim=-1).flatten(0, 1)
    return 2.0 * coords - 1.0


class _Rope(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.patch = c["patch_size"]
        head_dim = c["hidden_size"] // c["num_attention_heads"]
        inv = 1.0 / (c["rope_theta"] ** torch.arange(0, 1, 4 / head_dim, dtype=torch.float32))
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, px):
        _, _, H, W = px.shape
        coords = _rope_coords(H // self.patch, W // self.patch, px.device)
        angles = 2 * math.pi * coords[:, :, None] * self.inv_freq[None, None, :]
        angles = angles.flatten(1, 2).tile(2)
        return torch.cos(angles).to(px.dtype), torch.sin(angles).to(px.dtype)


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(q, k, cos, sin):
    np_ = sin.shape[-2]
    npre = q.shape[-2] - np_
    qp, qP = q.split((npre, np_), dim=-2)
    kp, kP = k.split((npre, np_), dim=-2)
    qP = qP * cos + _rotate_half(qP) * sin
    kP = kP * cos + _rotate_half(kP) * sin
    return torch.cat((qp, qP), dim=-2), torch.cat((kp, kP), dim=-2)


class _Attn(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.nh = c["num_attention_heads"]; self.hd = c["hidden_size"] // self.nh; d = c["hidden_size"]
        self.q_proj = nn.Linear(d, d, bias=c["query_bias"])
        self.k_proj = nn.Linear(d, d, bias=c["key_bias"])
        self.v_proj = nn.Linear(d, d, bias=c["value_bias"])
        self.o_proj = nn.Linear(d, d, bias=c["proj_bias"])

    def forward(self, x, pos):
        b, n, _ = x.shape
        q = self.q_proj(x).view(b, n, self.nh, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(b, n, self.nh, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(b, n, self.nh, self.hd).transpose(1, 2)
        q, k = _apply_rope(q, k, *pos)
        o = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, -1)
        return self.o_proj(o)


class _LayerScale(nn.Module):
    def __init__(self, c):
        super().__init__(); self.lambda1 = nn.Parameter(torch.ones(c["hidden_size"]) * c["layerscale_value"])

    def forward(self, x):
        return x * self.lambda1


class _MLP(nn.Module):
    """GELU up/down (ViT-L)."""
    def __init__(self, c):
        super().__init__()
        self.up_proj = nn.Linear(c["hidden_size"], c["intermediate_size"], bias=c["mlp_bias"])
        self.down_proj = nn.Linear(c["intermediate_size"], c["hidden_size"], bias=c["mlp_bias"])
        self.act = nn.GELU()

    def forward(self, x):
        return self.down_proj(self.act(self.up_proj(x)))


class _MLPGated(nn.Module):
    """SwiGLU gated MLP (ViT-H+): down(act(gate(x)) * up(x)). act = silu per config hidden_act."""
    def __init__(self, c):
        super().__init__()
        self.gate_proj = nn.Linear(c["hidden_size"], c["intermediate_size"], bias=c["mlp_bias"])
        self.up_proj = nn.Linear(c["hidden_size"], c["intermediate_size"], bias=c["mlp_bias"])
        self.down_proj = nn.Linear(c["intermediate_size"], c["hidden_size"], bias=c["mlp_bias"])
        self.act = nn.SiLU() if c.get("hidden_act", "silu") == "silu" else nn.GELU()

    def forward(self, x):
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class _Layer(nn.Module):
    def __init__(self, c):
        super().__init__()
        eps, d = c["layer_norm_eps"], c["hidden_size"]
        self.norm1 = nn.LayerNorm(d, eps=eps); self.attention = _Attn(c); self.layer_scale1 = _LayerScale(c)
        self.norm2 = nn.LayerNorm(d, eps=eps)
        self.mlp = _MLPGated(c) if c.get("use_gated_mlp", False) else _MLP(c)
        self.layer_scale2 = _LayerScale(c)

    def forward(self, x, pos):
        x = x + self.layer_scale1(self.attention(self.norm1(x), pos))
        x = x + self.layer_scale2(self.mlp(self.norm2(x)))
        return x


class DINOv3ViTStandalone(nn.Module):
    def __init__(self, model_dir: str):
        super().__init__()
        c = json.loads((Path(model_dir) / "config.json").read_text())
        c.setdefault("num_channels", 3)
        for k in ("query_bias", "key_bias", "value_bias", "proj_bias"):
            c.setdefault(k, True)
        self.cfg = c; self.hidden = c["hidden_size"]; self.n_prefix = 1 + c["num_register_tokens"]
        self.embeddings = _Embeddings(c); self.rope_embeddings = _Rope(c)
        self.layer = nn.ModuleList([_Layer(c) for _ in range(c["num_hidden_layers"])])
        self.norm = nn.LayerNorm(self.hidden, eps=c["layer_norm_eps"])
        sd = load_file(str(Path(model_dir) / "model.safetensors"))
        missing, unexpected = self.load_state_dict(sd, strict=False)
        missing = [m for m in missing if "inv_freq" not in m]
        if missing or unexpected:
            raise RuntimeError(f"DINOv3 standalone weight mismatch: missing={missing[:6]}, unexpected={unexpected[:6]}")
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, px, return_all_hidden=False):
        x = self.embeddings(px); pos = self.rope_embeddings(px)
        hs = [x]                                       # HF-style: hidden_states[0] = embeddings
        for L in self.layer:
            x = L(x, pos); hs.append(x)                # hidden_states[i] = block i-1 output (pre-final-norm)
        last = self.norm(x)
        return (last, hs) if return_all_hidden else last
