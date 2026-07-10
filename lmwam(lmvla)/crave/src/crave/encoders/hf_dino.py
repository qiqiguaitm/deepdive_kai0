"""DINOv2 / DINOv3 encoder via HuggingFace transformers.

Handles all three dtypes (fp16 for dinov2, bf16 for dinov3, int8 via bitsandbytes for
the 7B), the prefix-token skip (CLS + register tokens), and the per-encoder resolution
that pins the patch grid to grid×grid. Lazy-loads the model on first use.
"""
from __future__ import annotations

import os

import numpy as np

from crave.config.encoders import EncoderSpec
from crave.encoders.base import Encoder


class HFDinoEncoder(Encoder):
    def __init__(self, spec: EncoderSpec, device: str = "cuda"):
        super().__init__(spec, device)
        self._proc = None
        self._model = None
        self._standalone = None   # set if transformers can't load this dinov3 (e.g. gf3 tf==4.53.2)

    # -- lazy load --------------------------------------------------------
    def _ensure(self):
        if self._model is not None or self._standalone is not None:
            return
        import torch
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        self._torch = torch
        try:
            from transformers import AutoImageProcessor, AutoModel
            self._proc = AutoImageProcessor.from_pretrained(self.spec.path)
            dt = self.spec.dtype
            if dt == "int8":
                from transformers import BitsAndBytesConfig
                self._model = AutoModel.from_pretrained(
                    self.spec.path, quantization_config=BitsAndBytesConfig(load_in_8bit=True)).eval()
                self._in_dtype = torch.bfloat16
            elif dt == "bf16":
                self._model = AutoModel.from_pretrained(self.spec.path, dtype=torch.bfloat16).to(self.device).eval()
                self._in_dtype = torch.bfloat16
            elif dt == "fp32":
                self._model = AutoModel.from_pretrained(self.spec.path).to(self.device).eval()
                self._in_dtype = None
            else:  # fp16
                self._model = AutoModel.from_pretrained(self.spec.path).half().to(self.device).eval()
                self._in_dtype = torch.float16
        except Exception as e:  # transformers lacks dinov3 (kai0 pins 4.53.2) -> pure-torch standalone port
            from crave.encoders._dino_vit_standalone import DINOv3ViTStandalone
            self._in_dtype = torch.float16 if self.spec.dtype == "fp16" else (None if self.spec.dtype == "fp32" else torch.bfloat16)
            m = DINOv3ViTStandalone(self.spec.path).to(self.device)
            if self._in_dtype is not None:
                m = m.to(self._in_dtype)
            self._standalone = m
            print(f"[crave.hf_dino] transformers can't load {self.spec.name} ({type(e).__name__}); "
                  f"using pure-torch DINOv3ViTStandalone (dtype={self.spec.dtype})", flush=True)

    def _prep(self, batch):
        if self._standalone is None:
            sz = {"height": self.spec.res, "width": self.spec.res}
            px = self._proc(images=batch, return_tensors="pt", size=sz).to(self.device)
            if self._in_dtype is not None:
                px = {k: (v.to(self._in_dtype) if self._torch.is_floating_point(v) else v) for k, v in px.items()}
            return px
        # standalone: uint8 (N,H,W,3) RGB -> ImageNet-normalized (N,3,res,res) tensor (matches HF processor)
        import torch.nn.functional as F
        torch = self._torch
        x = torch.from_numpy(np.asarray(batch)).to(self.device).float().permute(0, 3, 1, 2) / 255.0
        if x.shape[-1] != self.spec.res or x.shape[-2] != self.spec.res:
            x = F.interpolate(x, size=(self.spec.res, self.spec.res), mode="bicubic", align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        return x.to(self._in_dtype) if self._in_dtype is not None else x

    def _hidden(self, batch):
        """last_hidden_state (B, 1+R+P, dim) for both transformers and standalone paths."""
        px = self._prep(batch)
        if self._standalone is not None:
            return self._standalone(px)
        return self._model(**px).last_hidden_state

    # -- public API -------------------------------------------------------
    def encode_pooled(self, imgs, bs: int = 128) -> np.ndarray:
        self._ensure()
        torch = self._torch
        out = np.zeros((len(imgs), self.spec.dim), np.float32)
        npf = self.spec.nprefix
        for b in range(0, len(imgs), bs):
            with torch.no_grad():
                hs = self._hidden(imgs[b:b + bs])[:, npf:].mean(1)
                out[b:b + bs] = hs.float().cpu().numpy()
        return out

    def encode_grid(self, imgs, bs: int = 64) -> np.ndarray:
        self._ensure()
        torch = self._torch
        P, dim, npf = self.spec.grid, self.spec.dim, self.spec.nprefix
        grids = np.zeros((len(imgs), dim, P, P), np.float16)
        for b in range(0, len(imgs), bs):
            with torch.no_grad():
                toks = self._hidden(imgs[b:b + bs])[:, npf:]
                side = int(round(toks.shape[1] ** 0.5))
                assert side == P, f"grid {side}!=P({P}) for {self.spec.name}; res={self.spec.res} patch mismatch"
                g = toks.reshape(toks.shape[0], P, P, dim).permute(0, 3, 1, 2).float().contiguous().cpu().numpy()
            grids[b:b + bs] = g.astype(np.float16)
        return grids

    def unload(self):
        if self._model is not None:
            del self._model
            self._model = None
            import torch
            torch.cuda.empty_cache()
