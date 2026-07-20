from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from latent_action_model.core.lam_model import load_latent_action_model
from starVLA.model.framework.latent_world.processor_utils import (
    build_latent_world_processor_spec,
    configure_latent_world_processor,
)
from starVLA.model.framework.latent_world.types import LatentWorldPolicyInferBatch, LatentWorldPolicyTrainBatch
from .vlm_auto import (
    load_vlm_auto,
    remove_lm_head,
)
from starVLA.model.tools import sync_managed_modules_training_mode

from .flowmatching_expert import ConditionalFlowMatchingConfig, ConditionalFlowMatchingHead


# ============================================================================
# Config & Constants
# ============================================================================

DEFAULT_LAM_ROOT = Path("latent_action_model")


@dataclass
class LatentWorldPolicyConfig:
    """Independent LatentWorldPolicyBackend config."""

    # Flow head config
    flow_cfg: ConditionalFlowMatchingConfig = field(default_factory=ConditionalFlowMatchingConfig)

    # Action chunk (Qwen-compatible schema)
    future_action_window_size: int = 7
    past_action_window_size: int = 0
    action_horizon: int = 8

    # Base checkpoints
    hf_cache_dir: Optional[Union[str, Path]] = None
    lam_ckpt_path: str = str(DEFAULT_LAM_ROOT / "logs/dino_base_ae_bridge/version_0/checkpoints/epoch=39.ckpt")
    lam_yaml_path: str = str(DEFAULT_LAM_ROOT / "logs/dino_base_ae_bridge/version_0/dino_base_ae.yaml")

    # VLM dtype
    vlm_dtype: torch.dtype = torch.bfloat16
    remove_lm_head: bool = True

    # Placeholder token
    latent_action_placeholder_token: str = "<ACT_PH>"

    # World / distill losses
    perceptual_weight: float = 0.1
    enable_loss_distill: bool = True
    latent_loss_type: str = "mse"  # "mse" | "cosine"
    lam_encoder_distill_weight: float = 1.0

    # Flow conditioning variants
    future_prediction: bool = False
    repeated_diffusion_steps: int = 4
    flow_only_mode: bool = False
    enable_flow_h_t1_scheduled_sampling: bool = False
    flow_h_t1_pred_prob_start: float = 0.0
    flow_h_t1_pred_prob_end: float = 1.0
    flow_h_t1_pred_ramp_steps: int = 20000
    detach_future_feature: bool = False

    # Independent additions
    num_action_queries: int = 8
    flow_action_num_queries: int = 8


# ============================================================================
# Lightweight Blocks
# ============================================================================


class VLMToLAMQFormer(nn.Module):
    """Refine VLM query hidden states into one latent action embedding."""

    def __init__(
        self,
        *,
        vlm_hidden_dim: int,
        lam_code_dim: int,
        num_layers: int = 1,
        num_heads: int = 8,
        ffn_expansion_factor: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.query = nn.Parameter(torch.randn(1, 1, int(lam_code_dim)) * 0.02)
        self.cross_attns = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=int(lam_code_dim),
                    kdim=int(vlm_hidden_dim),
                    vdim=int(vlm_hidden_dim),
                    num_heads=int(num_heads),
                    dropout=float(dropout),
                    batch_first=True,
                )
                for _ in range(int(num_layers))
            ]
        )
        self.norm_qs = nn.ModuleList([nn.LayerNorm(int(lam_code_dim)) for _ in range(int(num_layers))])
        self.norm_kvs = nn.ModuleList([nn.LayerNorm(int(vlm_hidden_dim)) for _ in range(int(num_layers))])
        hidden_dim = int(int(lam_code_dim) * float(ffn_expansion_factor))
        self.ffns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(int(lam_code_dim)),
                    nn.Linear(int(lam_code_dim), hidden_dim),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                    nn.Linear(hidden_dim, int(lam_code_dim)),
                )
                for _ in range(int(num_layers))
            ]
        )
        self.final_norm = nn.LayerNorm(int(lam_code_dim))

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        bsz = int(context.shape[0])
        queries = self.query.expand(bsz, -1, -1).to(device=context.device, dtype=context.dtype)
        for norm_q, norm_kv, xattn, ffn in zip(self.norm_qs, self.norm_kvs, self.cross_attns, self.ffns):
            q = norm_q(queries)
            kv = norm_kv(context)
            attn_out, _ = xattn(q, kv, kv)
            queries = queries + attn_out
            queries = queries + ffn(queries)
        return self.final_norm(queries)


# ============================================================================
# VLM / LAM Load Helpers
# ============================================================================


def _load_vlm_and_processor(
    cfg: LatentWorldPolicyConfig,
    vlm_model_id: str,
) -> Tuple[nn.Module, Any, Any, int]:
    processor_spec = build_latent_world_processor_spec(policy_cfg=cfg, vlm_model_id=vlm_model_id)
    vlm, processor = load_vlm_auto(processor_spec.model_id, processor_spec.cache_dir, dtype=cfg.vlm_dtype)
    processor, tokenizer, placeholder_token_id = configure_latent_world_processor(
        processor,
        placeholder_token=processor_spec.placeholder_token,
    )

    # Ensure tokenizer size and embedding rows stay aligned after adding placeholder tokens.
    target_vocab_size = max(int(len(tokenizer)), int(placeholder_token_id + 1))
    embed = vlm.get_input_embeddings()
    if embed is None:
        raise RuntimeError("[LatentWorldPolicyBackend] VLM does not expose input embeddings.")
    embed_rows = int(embed.weight.shape[0])
    if target_vocab_size > embed_rows:
        if not hasattr(vlm, "resize_token_embeddings"):
            raise RuntimeError(
                "[LatentWorldPolicyBackend] tokenizer vocab exceeds embedding size, "
                "but model does not support `resize_token_embeddings`."
            )
        vlm.resize_token_embeddings(target_vocab_size)

    if hasattr(vlm, "generation_config") and vlm.generation_config is not None:
        vlm.generation_config.max_new_tokens = 4
    if hasattr(vlm, "config") and vlm.config is not None:
        try:
            vlm.config.loss_type = "ForCausalLMLoss"
            vlm.config.use_cache = False
        except Exception:
            pass

    if bool(getattr(cfg, "remove_lm_head", True)):
        remove_lm_head(vlm)

    return vlm, processor, tokenizer, placeholder_token_id


def _apply_flow_only_grad_to_h_vlm(
    *,
    h_vlm: torch.Tensor,
    act_placeholder_mask: torch.Tensor,
    enable_flow_only: bool,
) -> torch.Tensor:
    """
    Keep full VLM context values for flow, but split direct-flow gradients by sequence boundary:
    only tokens after the last act placeholder keep direct flow gradients.
    """
    if not bool(enable_flow_only):
        return h_vlm
    if h_vlm.dim() != 3:
        raise ValueError(f"[LatentWorldPolicyBackend] expected h_vlm [B, L, D], got {tuple(h_vlm.shape)}")
    if act_placeholder_mask is None:
        raise ValueError("[LatentWorldPolicyBackend] flow_only_mode=True requires act_placeholder_mask.")
    if act_placeholder_mask.dim() != 2:
        raise ValueError(
            f"[LatentWorldPolicyBackend] expected act_placeholder_mask [B, L], got {tuple(act_placeholder_mask.shape)}"
        )
    if act_placeholder_mask.shape[0] != h_vlm.shape[0] or act_placeholder_mask.shape[1] != h_vlm.shape[1]:
        raise ValueError(
            "[LatentWorldPolicyBackend] act_placeholder_mask shape mismatch with h_vlm: "
            f"mask={tuple(act_placeholder_mask.shape)} vs h_vlm={tuple(h_vlm.shape)}"
        )
    act_mask = act_placeholder_mask.to(device=h_vlm.device, dtype=torch.bool)
    if not torch.all(act_mask.any(dim=1)):
        raise ValueError("[LatentWorldPolicyBackend] flow_only_mode=True requires at least one act placeholder per sample.")

    seq_positions = torch.arange(h_vlm.shape[1], device=h_vlm.device).unsqueeze(0)
    last_act_idx = torch.where(act_mask, seq_positions, -1).max(dim=1).values
    post_act_mask = seq_positions > last_act_idx.unsqueeze(1)
    post_act_mask_f = post_act_mask.unsqueeze(-1).to(dtype=h_vlm.dtype)
    return h_vlm * post_act_mask_f + h_vlm.detach() * (1.0 - post_act_mask_f)


def _module_param_dtype(module: Optional[nn.Module], default: torch.dtype) -> torch.dtype:
    if module is None:
        return default
    try:
        p = next(module.parameters())
        return p.dtype
    except Exception:
        return default


def _cuda_autocast(dtype: torch.dtype):
    if torch.cuda.is_available():
        return torch.autocast("cuda", dtype=dtype)
    return nullcontext()


@dataclass
class PolicyEncodingState:
    h_vlm: torch.Tensor
    pred_action_emb: torch.Tensor
    h_t: torch.Tensor
    h_t1_pred: torch.Tensor
    h_t1_gt: torch.Tensor
    h_t_original: torch.Tensor
    # [V8 dual] 全局(milestone)通道; 非 dual 时为 None
    pred_action_emb_ms: Optional[torch.Tensor] = None
    h_ms_pred: Optional[torch.Tensor] = None
    h_ms_gt: Optional[torch.Tensor] = None


# ============================================================================
# LatentWorldPolicyBackend
# ============================================================================


class LatentWorldPolicyBackend(nn.Module):
    """Independent world-model VLA without LatentVLAModel dependency."""

    def __init__(self, model_cfg: LatentWorldPolicyConfig, *, vlm_model_id: str) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.vlm_model_id = str(vlm_model_id)

        # 1) Load VLM + processor/tokenizer and register placeholder token.
        self.vlm, self.processor, self.tokenizer, self.placeholder_token_id = _load_vlm_and_processor(
            self.model_cfg,
            self.vlm_model_id,
        )

        # 2) Load LAM.
        self.lam = load_latent_action_model(self.model_cfg.lam_ckpt_path, self.model_cfg.lam_yaml_path)
        # [LMWM swap] env-gated: 用我们的 LMWM 生成器替 LaWM decoder(对比实验, 唯一变量=世界模型)
        # [V8 LMWM_DUAL] 双尺度并联: 不 swap, 局部通道保 LaWM(t+7), 另挂 LMWM 全局通道(milestone)。
        import os as _os
        # [V8 Plan B] LMWM_DUAL_2Q=1: 双 query 变体(局部/全局各一组 act placeholder, 解单 query 双头容量瓶颈)。蕴含 dual。
        self._lmwm_dual_2q = _os.environ.get("LMWM_DUAL_2Q") == "1"
        self._lmwm_dual = _os.environ.get("LMWM_DUAL") == "1" or self._lmwm_dual_2q
        self._lmwm_dec = None      # V8 全局通道 decoder(生成器); dual 时挂
        self._lmwm_inv = None      # V8 全局通道 teacher(InverseEnc, 冻); dual 时挂
        if _os.environ.get("LMWM_CKPT"):
            import sys as _sys
            # repo root = 4 dirs up from this file (<repo>/starVLA/model/framework/vlas/lawam.py)
            _repo = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))))
            _sys.path.insert(0, _os.environ.get("LMWM_ADAPTER_DIR", _repo))
            if self._lmwm_dual:
                # 双通道: LaWM decoder 原封不动(守 t8/t7 精度), LMWM 生成器另挂作全局通道(守 t6/t9 指引)
                from lmwm_adapter import load_lmwm_parts
                self._lmwm_dec, self._lmwm_inv = load_lmwm_parts(_os.environ["LMWM_CKPT"])
                print(f"[LMWM][DUAL] loaded LMWM parts (gen+inv) alongside LaWM from {_os.environ['LMWM_CKPT']}; NO decoder swap", flush=True)
            else:
                from lmwm_adapter import make_lmwm_lam
                # swap_teacher 默认 True: 否则 vlm_to_lam 蒸馏到 LaWM code 空间, 与 LMWM generator 期望的 code 失配(BUG_AUDIT MAJOR-2)
                self.lam = make_lmwm_lam(self.lam, _os.environ["LMWM_CKPT"],
                                         swap_teacher=_os.environ.get("LMWM_SWAP_TEACHER", "1") == "1")
                print(f"[LMWM] swapped LAM decoder with LMWM generator from {_os.environ['LMWM_CKPT']}", flush=True)

        # [LMWM Path A] 世界模型目标 h_t1_gt: t+7 近未来帧 -> milestone+1 帧特征(BUG_AUDIT CRITICAL-1)。
        # 全逻辑在 lmwm_milestone_target 模块, 这里只装 provider; forward 里用它覆盖 h_t1_gt。
        self._lmwm_target_provider = None
        if _os.environ.get("LMWM_MILESTONE_TARGET"):
            import sys as _sys2
            _repo2 = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))))
            _sys2.path.insert(0, _os.environ.get("LMWM_ADAPTER_DIR", _repo2))
            from lmwm_milestone_target import get_provider as _lmwm_get_prov
            self._lmwm_target_provider = _lmwm_get_prov()

        # 3) Create trainable query and mapping head.
        self.num_action_queries = int(self.model_cfg.num_action_queries)
        # [V8 Plan B] 双 query: act placeholder 翻倍(前半=局部 query, 后半=全局 ms query)。
        # 注意 dataloader/__init__.py 与 runtime/components.py 也须按 LMWM_DUAL_2Q 传相同的 act_queries 数(prompt 占位符数一致)。
        self._base_num_action_queries = self.num_action_queries
        if self._lmwm_dual_2q:
            self.num_action_queries = self._base_num_action_queries * 2

        vlm_cfg = getattr(self.vlm, "config", None)
        text_cfg = getattr(vlm_cfg, "text_config", None)
        # NOTE: don't nest `getattr` in default value; default expression is evaluated eagerly.
        vlm_hidden_size = getattr(text_cfg, "hidden_size", None)
        if vlm_hidden_size is None:
            vlm_hidden_size = getattr(vlm_cfg, "hidden_size", None)
        if vlm_hidden_size is None:
            raise AttributeError("[LatentWorldPolicyBackend] cannot resolve VLM hidden_size from config.")
        vlm_hidden_dim = int(vlm_hidden_size)
        lam_code_dim = int(self.lam.code_dim)
        self.act_query = nn.Parameter(torch.randn(self.num_action_queries, vlm_hidden_dim) * 0.02)
        self.flow_action_num_queries = int(self.model_cfg.flow_action_num_queries)
        self.flow_action_query = nn.Parameter(torch.randn(self.flow_action_num_queries, vlm_hidden_dim) * 0.02)
        self.vlm_to_lam = VLMToLAMQFormer(
            vlm_hidden_dim=vlm_hidden_dim,
            lam_code_dim=lam_code_dim,
            num_layers=1,
            num_heads=8,
            ffn_expansion_factor=4.0,
            dropout=0.0,
        )
        # [V8 dual] 全局(ms)通道的第二个投影头: 共享 h_act, 蒸馏到 LMWM InverseEnc code 空间。
        # 注册为子模块 → 随 model.to(device)/DDP 自动搬运; 生成器/teacher 同理挂为子模块。
        if self._lmwm_dual:
            if self._lmwm_dec is None or self._lmwm_inv is None:
                raise RuntimeError("[LatentWorldPolicyBackend] LMWM_DUAL=1 需同时设 LMWM_CKPT 以加载 LMWM 生成器/teacher。")
            self.vlm_to_lam_ms = VLMToLAMQFormer(
                vlm_hidden_dim=vlm_hidden_dim,
                lam_code_dim=lam_code_dim,   # LaWM code_dim == LMWM code_dim == 32(见 make_lmwm_lam 断言)
                num_layers=1,
                num_heads=8,
                ffn_expansion_factor=4.0,
                dropout=0.0,
            )
            self.lmwm_dec = self._lmwm_dec        # 子模块名(可训生成器)
            self.lmwm_teacher = self._lmwm_inv    # 子模块名(冻结 teacher)

        # 4) Align flow config to LAM output dimensions.
        lam_vision_dim = int(self.lam.input_dim)
        lam_grid_h = int(getattr(self.lam.encoder, "grid_height", 0) or 0)
        lam_grid_w = int(getattr(self.lam.encoder, "grid_width", 0) or 0)
        lam_num_tokens = int(lam_grid_h * lam_grid_w) if lam_grid_h > 0 and lam_grid_w > 0 else int(
            self.model_cfg.flow_cfg.num_vision_tokens
        )
        if int(self.model_cfg.flow_cfg.vision_dim) != lam_vision_dim:
            print(
                f"[LatentWorldPolicyBackend] flow_cfg.vision_dim={self.model_cfg.flow_cfg.vision_dim} "
                f"!= LAM vision dim={lam_vision_dim}; auto-aligned."
            )
        self.model_cfg.flow_cfg.vision_dim = lam_vision_dim

        if int(self.model_cfg.flow_cfg.num_vision_tokens) != lam_num_tokens:
            print(
                f"[LatentWorldPolicyBackend] flow_cfg.num_vision_tokens={self.model_cfg.flow_cfg.num_vision_tokens} "
                f"!= LAM token count={lam_num_tokens}; auto-aligned."
            )
            self.model_cfg.flow_cfg.num_vision_tokens = lam_num_tokens

        # 5) Flow head.
        self.flow = ConditionalFlowMatchingHead(config=self.model_cfg.flow_cfg)
        self.flow.action_horizon = int(self.model_cfg.action_horizon)
        self._flow_train_step: int = 0
        if bool(self.model_cfg.enable_flow_h_t1_scheduled_sampling) and not bool(self.model_cfg.detach_future_feature):
            print(
                "[LatentWorldPolicyBackend][warn] enable_flow_h_t1_scheduled_sampling=true and "
                "detach_future_feature=false: flow gradients through GT-conditioned branch can increase loss_perceptual."
            )
        self.sync_training_modes(mode=self.training)

    def _inject_queries(
        self,
        *,
        inputs_embeds: torch.Tensor,
        placeholder_mask: torch.BoolTensor,
        queries: torch.Tensor,
        num_queries: int,
        name: str,
    ) -> None:
        device = inputs_embeds.device
        placeholder_mask = placeholder_mask.to(device=device, dtype=torch.bool)

        bsz, seq_len = int(inputs_embeds.shape[0]), int(inputs_embeds.shape[1])
        expected = int(bsz * num_queries)
        got = int(placeholder_mask.sum().item())
        if got != expected:
            raise ValueError(
                f"[LatentWorldPolicyBackend] {name} placeholder count mismatch: got={got}, expected={expected}"
            )

        per_sample = placeholder_mask.sum(dim=1)
        if not torch.all(per_sample == int(num_queries)):
            bad = torch.nonzero(per_sample != int(num_queries), as_tuple=False).flatten()
            b = int(bad[0].item()) if bad.numel() > 0 else -1
            got_b = int(per_sample[b].item()) if b >= 0 else -1
            raise ValueError(
                f"[LatentWorldPolicyBackend] {name} placeholder count mismatch for sample {b}: got={got_b}, expected={num_queries}"
            )

        idx = placeholder_mask.nonzero(as_tuple=False)
        flat = idx[:, 0] * seq_len + idx[:, 1]
        idx = idx[flat.argsort()]
        b_idx, p_idx = idx[:, 0], idx[:, 1]
        q_idx = torch.arange(int(num_queries), device=device).repeat(bsz)
        inputs_embeds[b_idx, p_idx, :] = queries[q_idx]

    def _run_vlm_stage(
        self,
        *,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        image_grid_thw: Optional[torch.LongTensor],
        act_placeholder_mask: torch.BoolTensor,
        flow_placeholder_mask: torch.BoolTensor,
        act_query: torch.Tensor,
        flow_query: torch.Tensor,
    ) -> Dict[str, Any]:
        embed = self.vlm.get_input_embeddings()
        if embed is None:
            raise RuntimeError("[LatentWorldPolicyBackend] VLM does not expose input embeddings.")
        inputs_embeds = embed(input_ids)

        self._inject_queries(
            inputs_embeds=inputs_embeds,
            placeholder_mask=act_placeholder_mask,
            queries=act_query,
            num_queries=int(self.num_action_queries),
            name="act_query",
        )
        self._inject_queries(
            inputs_embeds=inputs_embeds,
            placeholder_mask=flow_placeholder_mask,
            queries=flow_query,
            num_queries=int(flow_query.shape[0]),
            name="flow_query",
        )

        # HF transformers 4.57 Qwen3VL causal wrapper drops `hidden_states`
        # from the returned output; use the underlying multimodal model output.
        vlm_out = self.vlm.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        if vlm_out is None:
            raise RuntimeError(
                "[LatentWorldPolicyBackend] Qwen3VL base model forward returned None; "
                "expected `last_hidden_state`."
            )
        hidden = getattr(vlm_out, "last_hidden_state", None)
        if hidden is None:
            output_keys = list(vlm_out.keys()) if hasattr(vlm_out, "keys") else None
            raise RuntimeError(
                "[LatentWorldPolicyBackend] Qwen3VL base model output does not expose "
                "`last_hidden_state`. "
                f"output_type={type(vlm_out).__name__}, output_keys={output_keys}."
            )
        bsz = int(hidden.shape[0])
        q = int(self.num_action_queries)
        h_act = hidden[act_placeholder_mask]
        if h_act.numel() == 0:
            raise ValueError("[LatentWorldPolicyBackend] empty action hidden selection; check act_placeholder_mask.")
        if h_act.shape[0] != bsz * q:
            raise ValueError(
                f"[LatentWorldPolicyBackend] action hidden count mismatch: got={h_act.shape[0]}, expected={bsz * q}"
            )

        h_act_q = h_act.view(bsz, q, -1)
        if getattr(self, "_lmwm_dual_2q", False):
            # [V8 Plan B] 双 query: 前半 = 局部 query 隐藏态, 后半 = 全局 ms query 隐藏态, 各喂独立投影头
            Q = self._base_num_action_queries
            pred_latent = self.vlm_to_lam(h_act_q[:, :Q, :])
            out = {"h_vlm": hidden, "pred_latent": pred_latent, "vlm_out": vlm_out}
            out["pred_latent_ms"] = self.vlm_to_lam_ms(h_act_q[:, Q:, :])
        else:
            pred_latent = self.vlm_to_lam(h_act_q)
            out = {"h_vlm": hidden, "pred_latent": pred_latent, "vlm_out": vlm_out}
            if getattr(self, "_lmwm_dual", False):
                # [V8 E1] 单 query 双头: 同 h_act 经第二投影头得全局(ms)通道 code
                out["pred_latent_ms"] = self.vlm_to_lam_ms(h_act_q)
        return out

    def _build_lam_teacher_inputs_for_distill(self, primary_video: torch.Tensor) -> torch.Tensor:
        expected_t = int(getattr(getattr(self.lam, "encoder", None), "num_frames", 0) or 0)
        if expected_t <= 0:
            return primary_video

        cur_t = int(primary_video.shape[1])
        if cur_t == expected_t:
            return primary_video
        if cur_t < expected_t:
            raise ValueError(
                f"[LatentWorldPolicyBackend] distill teacher temporal mismatch: got T={cur_t}, "
                f"but LAM encoder expects num_frames={expected_t}."
            )
        idx = torch.linspace(0, cur_t - 1, steps=expected_t, device=primary_video.device).round().long()
        return primary_video.index_select(1, idx)

    def _run_lam_teacher(self, *, primary_video: torch.Tensor, embodiment_id: torch.Tensor) -> torch.Tensor:
        primary_video_t = self._build_lam_teacher_inputs_for_distill(primary_video)
        with torch.no_grad():
            lam_out = self.lam.get_latent_action(
                videos=primary_video_t,
                states=None,
                dec_videos=primary_video_t,
                predict_future_frame=False,
                embodiment_ids=embodiment_id,
            )
        # Some LAM implementations return inference tensors here. Materialize a normal
        # tensor before using it in training losses, otherwise autograd cannot save it.
        return lam_out["quantized"].detach().clone()

    def _compute_latent_loss(
        self,
        *,
        pred_latent: torch.Tensor,
        teacher_latent: torch.Tensor,
        latent_loss_type: str,
    ) -> torch.Tensor:
        if pred_latent.shape != teacher_latent.shape:
            raise ValueError(
                f"[LatentWorldPolicyBackend] latent shape mismatch: pred={tuple(pred_latent.shape)}, "
                f"teacher={tuple(teacher_latent.shape)}"
            )
        if str(latent_loss_type).lower() == "mse":
            return F.mse_loss(pred_latent, teacher_latent)
        return 1 - F.cosine_similarity(pred_latent, teacher_latent, dim=-1).mean()

    def _compute_distill_loss(
        self,
        *,
        pred_latent: torch.Tensor,
        primary_video: torch.Tensor,
        embodiment_id: torch.Tensor,
    ) -> torch.Tensor:
        teacher_latent = self._run_lam_teacher(primary_video=primary_video, embodiment_id=embodiment_id)
        return self._compute_latent_loss(
            pred_latent=pred_latent,
            teacher_latent=teacher_latent,
            latent_loss_type=self.model_cfg.latent_loss_type,
        )

    def _decode_future_tokens_strict_single_query(
        self,
        *,
        h_t: torch.Tensor,
        pred_action_emb: torch.Tensor,
        source: str,
    ) -> torch.Tensor:
        if pred_action_emb.shape[1] != 1:
            raise ValueError(
                f"[{source}] future_prediction requires single-query latent action, "
                f"got query_dim={pred_action_emb.shape[1]}."
            )
        decoded = self.lam.decoder(h_t, pred_action_emb)
        if isinstance(decoded, tuple):
            decoded = decoded[0]
        if decoded.dim() == 4:
            decoded = decoded[:, 0, :, :] if decoded.shape[1] == 1 else decoded[:, -1, :, :]
        return decoded

    def _decode_ms_future(self, *, h_t: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
        """[V8 dual] 全局通道: 用 LMWM 生成器从 h_t + ms code 生成 milestone 特征。
        code: [B,1,code_dim]; 回 [B,K,D]。"""
        decoded = self.lmwm_dec(h_t, code)   # LMWMDecoder -> [B,1,K,D]
        if isinstance(decoded, tuple):
            decoded = decoded[0]
        if decoded.dim() == 4:
            decoded = decoded[:, 0, :, :] if decoded.shape[1] == 1 else decoded[:, -1, :, :]
        return decoded

    def _compute_distill_loss_ms(self, *, pred_ms_emb: torch.Tensor, g_t: torch.Tensor, g_f: torch.Tensor) -> torch.Tensor:
        """[V8 dual] 全局通道蒸馏: InverseEnc(g_t,g_f)->code 作 teacher(冻), 蒸 pred_ms_emb。
        与单通道 swap_teacher 的 _gla 闭包一致(g_t=features[:,0], g_f=features[:,-1])。"""
        B, K, D = g_t.shape
        P = int(K ** 0.5)
        gt = g_t.transpose(1, 2).reshape(B, D, P, P)
        gf = g_f.transpose(1, 2).reshape(B, D, P, P)
        with torch.no_grad():
            code = self.lmwm_teacher(gt, gf).unsqueeze(1).detach()   # [B,1,code_dim]
        return self._compute_latent_loss(
            pred_latent=pred_ms_emb,
            teacher_latent=code,
            latent_loss_type=self.model_cfg.latent_loss_type,
        )

    @classmethod
    def build(
        cls,
        cfg: LatentWorldPolicyConfig,
        *,
        vlm_model_id: str,
    ) -> "LatentWorldPolicyBackend":
        model = cls(cfg, vlm_model_id=vlm_model_id)
        return model

    def train(self, mode: bool = True):
        super().train(mode)
        self.sync_training_modes(mode=mode)
        return self

    def sync_training_modes(self, mode: bool = True) -> None:
        sync_managed_modules_training_mode(
            self.vlm,
            self.lam,
            self.flow,
            self.vlm_to_lam,
            mode=mode,
        )
        if getattr(self, "_lmwm_dual", False):
            # 生成器随主训练模式; teacher(InverseEnc)恒 eval(冻结, 只作蒸馏目标)
            self.lmwm_dec.train(mode)
            self.vlm_to_lam_ms.train(mode)
            self.lmwm_teacher.eval()

    def set_flow_train_step(self, step: int) -> None:
        self._flow_train_step = max(0, int(step))

    def _flow_h_t1_pred_prob(self) -> float:
        if not bool(self.model_cfg.enable_flow_h_t1_scheduled_sampling):
            return 1.0
        start = float(self.model_cfg.flow_h_t1_pred_prob_start)
        end = float(self.model_cfg.flow_h_t1_pred_prob_end)
        ramp_steps = max(1, int(self.model_cfg.flow_h_t1_pred_ramp_steps))
        ratio = min(1.0, float(self._flow_train_step) / float(ramp_steps))
        prob = start + (end - start) * ratio
        return float(max(0.0, min(1.0, prob)))

    def _build_flow_future_condition(
        self,
        *,
        h_t1_pred: torch.Tensor,
        h_t1_gt: torch.Tensor,
    ) -> torch.Tensor:
        if h_t1_pred.shape != h_t1_gt.shape:
            raise ValueError(
                f"[LatentWorldPolicyBackend] h_t1 shape mismatch: pred={tuple(h_t1_pred.shape)}, gt={tuple(h_t1_gt.shape)}"
            )

        if not self.training:
            return h_t1_pred

        prob_pred = self._flow_h_t1_pred_prob()
        bsz = int(h_t1_pred.shape[0])
        pred_mask = (torch.rand(bsz, 1, 1, device=h_t1_pred.device) < prob_pred)
        cond_future = torch.where(pred_mask, h_t1_pred, h_t1_gt)

        if not bool(self.model_cfg.detach_future_feature):
            gt_mask = (~pred_mask).to(dtype=h_t1_pred.dtype)
            # Straight-through bridge:
            cond_future = cond_future + gt_mask * (h_t1_pred - h_t1_pred.detach())
        else:
            cond_future = cond_future.detach()

        return cond_future

    # ------------------
    # Checkpoint IO
    # ------------------
    def save_pretrained(self, save_directory: Union[str, Path], **kwargs):
        save_dir = Path(str(save_directory))
        save_dir.mkdir(parents=True, exist_ok=True)

        out = self.vlm.save_pretrained(str(save_dir), **kwargs)
        ckpt_path = save_dir / "pytorch_model.pt"
        state_dict = self.state_dict()
        torch.save(state_dict, ckpt_path)
        num_params = sum(p.numel() for p in self.parameters())
        print(f"[LatentWorldPolicyBackend] Full checkpoint saved to: {ckpt_path} (num_params={num_params})")
        return out

    # ------------------
    # Forward
    # ------------------
    def _prepare_queries(self, *, device: torch.device, vlm_stage_dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        flow_query = getattr(self, "flow_action_query", None)
        if flow_query is None:
            raise ValueError("[LatentWorldPolicyBackend] flow_action_query is None; check policy backend initialization.")
        vlm_embed_dtype = _module_param_dtype(self.vlm.get_input_embeddings(), default=vlm_stage_dtype)
        act_query = self.act_query.to(device=device, dtype=vlm_embed_dtype)
        flow_query = flow_query.to(device=device, dtype=vlm_embed_dtype)
        return act_query, flow_query

    def _prepare_train_batch(self, *, batch: LatentWorldPolicyTrainBatch) -> LatentWorldPolicyTrainBatch:
        return batch

    def _prepare_infer_batch(self, *, batch: LatentWorldPolicyInferBatch) -> LatentWorldPolicyInferBatch:
        return batch

    def _run_shared_encoding_core(
        self,
        *,
        prepared_batch: Union[LatentWorldPolicyInferBatch, LatentWorldPolicyTrainBatch],
        primary_visual_input: torch.Tensor,
        source: str,
        lam_features_with_no_grad: bool,
    ) -> PolicyEncodingState:
        vlm_stage_dtype = self.model_cfg.vlm_dtype
        lam_stage_dtype = torch.bfloat16

        device = prepared_batch["input_ids"].device
        act_query, flow_query = self._prepare_queries(device=device, vlm_stage_dtype=vlm_stage_dtype)

        with _cuda_autocast(vlm_stage_dtype):
            vlm_out_dict = self._run_vlm_stage(
                input_ids=prepared_batch["input_ids"],
                attention_mask=prepared_batch["attention_mask"],
                pixel_values=prepared_batch["pixel_values"],
                image_grid_thw=prepared_batch["image_grid_thw"],
                act_placeholder_mask=prepared_batch["act_placeholder_mask"],
                flow_placeholder_mask=prepared_batch["flow_placeholder_mask"],
                act_query=act_query,
                flow_query=flow_query,
            )
        h_vlm = vlm_out_dict["h_vlm"]
        pred_action_emb = vlm_out_dict["pred_latent"]

        with _cuda_autocast(lam_stage_dtype):
            if lam_features_with_no_grad:
                with torch.no_grad():
                    features = self.lam.extract_vision_features(primary_visual_input)
            else:
                features = self.lam.extract_vision_features(primary_visual_input)

            if features is None:
                raise ValueError(f"[{source}] lam visual feature extraction returned None; check LAM config.")
            h_t_original = features[:, 0, :, :]
            h_t = h_t_original
            h_t7_gt = features[:, -1, :, :]           # 局部 t+7 目标(原样, 从不被覆盖)
            _dual = getattr(self, "_lmwm_dual", False)

            # [LMWM] milestone+1 目标(provider): 无 milestone 的帧退回 t+7。
            _prov = getattr(self, "_lmwm_target_provider", None)
            _ms_target = h_t7_gt
            if _prov is not None and prepared_batch.get("episode_index") is not None:
                _tgt, _valid = _prov.get_target(
                    prepared_batch["episode_index"], prepared_batch["frame_index"],
                    out_shape=h_t7_gt.shape[1:], device=h_t7_gt.device, dtype=h_t7_gt.dtype)
                _ms_target = torch.where(_valid[:, None, None], _tgt, h_t7_gt)

            pred_action_emb_ms = vlm_out_dict.get("pred_latent_ms") if _dual else None
            h_ms_pred = None
            h_ms_gt = None
            if _dual:
                # 双尺度: 局部通道 GT=t+7(不覆盖), 全局通道 GT=milestone
                h_t1_gt = h_t7_gt
                h_ms_gt = _ms_target
            else:
                # 单通道(旧 Path A): provider 就绪时 h_t1_gt 被 milestone 覆盖
                h_t1_gt = _ms_target

            if self.model_cfg.future_prediction:
                # 局部通道(LaWM decoder): dual 时保 LaWM 原 decoder, 非 dual 时可能是 swap 后的 LMWM decoder
                h_t1_pred = self._decode_future_tokens_strict_single_query(
                    h_t=h_t,
                    pred_action_emb=pred_action_emb,
                    source=source,
                )
                if _dual:
                    # 全局通道(LMWM 生成器): 从 ms code 生成 milestone 特征
                    h_ms_pred = self._decode_ms_future(h_t=h_t, code=pred_action_emb_ms)
            else:
                h_t1_pred = h_t
                if _dual:
                    h_ms_pred = h_t

        return PolicyEncodingState(
            h_vlm=h_vlm,
            pred_action_emb=pred_action_emb,
            h_t=h_t,
            h_t1_pred=h_t1_pred,
            h_t1_gt=h_t1_gt,
            h_t_original=h_t_original,
            pred_action_emb_ms=pred_action_emb_ms,
            h_ms_pred=h_ms_pred,
            h_ms_gt=h_ms_gt,
        )

    def _run_shared_encoding_train(
        self,
        *,
        prepared_batch: LatentWorldPolicyTrainBatch,
        source: str,
        lam_features_with_no_grad: bool,
    ) -> PolicyEncodingState:
        return self._run_shared_encoding_core(
            prepared_batch=prepared_batch,
            primary_visual_input=prepared_batch["primary_video"],
            source=source,
            lam_features_with_no_grad=lam_features_with_no_grad,
        )

    def _run_shared_encoding_infer(
        self,
        *,
        prepared_batch: LatentWorldPolicyInferBatch,
        source: str,
        lam_features_with_no_grad: bool,
    ) -> PolicyEncodingState:
        return self._run_shared_encoding_core(
            prepared_batch=prepared_batch,
            primary_visual_input=prepared_batch["primary_image"],
            source=source,
            lam_features_with_no_grad=lam_features_with_no_grad,
        )

    def forward(
        self,
        *,
        batch: LatentWorldPolicyTrainBatch,
    ) -> Dict[str, torch.Tensor]:
        # Precision contract (QwenGR00T-style):
        # - VLM stage: bf16 autocast
        # - LAM stage: bf16 autocast
        # - Flow/loss stage: float32 autocast
        lam_stage_dtype = torch.bfloat16
        flow_stage_dtype = torch.float32
        prepared_batch = cast(
            LatentWorldPolicyTrainBatch,
            self._prepare_train_batch(batch=batch),
        )
        device = prepared_batch["input_ids"].device

        shared = self._run_shared_encoding_train(
            prepared_batch=prepared_batch,
            source="LatentWorldPolicyBackend.forward",
            lam_features_with_no_grad=True,
        )

        _dual = getattr(self, "_lmwm_dual", False)
        with _cuda_autocast(lam_stage_dtype):
            loss_distill = torch.tensor(0.0, device=device, dtype=lam_stage_dtype)
            loss_distill_local = torch.tensor(0.0, device=device, dtype=lam_stage_dtype)
            loss_distill_ms = torch.tensor(0.0, device=device, dtype=lam_stage_dtype)
            if bool(self.model_cfg.enable_loss_distill):
                # 局部通道(LaWM teacher): dual 时 self.lam 是纯 LaWM, 蒸馏目标=LaWM code
                loss_distill_local = self._compute_distill_loss(
                    pred_latent=shared.pred_action_emb,
                    primary_video=prepared_batch["primary_video"],
                    embodiment_id=prepared_batch["embodiment_id"],
                )
                loss_distill = loss_distill_local
                if _dual:
                    # 全局通道(InverseEnc teacher): g_f = shared.h_t1_gt(dual 时=features[:,-1]=t+7)
                    loss_distill_ms = self._compute_distill_loss_ms(
                        pred_ms_emb=shared.pred_action_emb_ms,
                        g_t=shared.h_t,
                        g_f=shared.h_t1_gt,
                    )
                    loss_distill = loss_distill_local + loss_distill_ms

            loss_perceptual_local = torch.tensor(0.0, device=device, dtype=lam_stage_dtype)
            loss_perceptual_ms = torch.tensor(0.0, device=device, dtype=lam_stage_dtype)
            if self.model_cfg.future_prediction:
                loss_perceptual_local = F.mse_loss(shared.h_t1_pred, shared.h_t1_gt)
                loss_perceptual = loss_perceptual_local
                if _dual:
                    loss_perceptual_ms = F.mse_loss(shared.h_ms_pred, shared.h_ms_gt)
                    loss_perceptual = loss_perceptual_local + loss_perceptual_ms
            else:
                loss_perceptual = torch.tensor(0.0, device=device, dtype=lam_stage_dtype)

        h_vlm_for_flow = _apply_flow_only_grad_to_h_vlm(
            h_vlm=shared.h_vlm,
            act_placeholder_mask=prepared_batch["act_placeholder_mask"],
            enable_flow_only=bool(self.model_cfg.flow_only_mode),
        )
        attn_flow = prepared_batch["attention_mask"] == 1

        with _cuda_autocast(flow_stage_dtype):
            repeat_steps = int(self.model_cfg.repeated_diffusion_steps)
            h_t_rep = shared.h_t.repeat(repeat_steps, *([1] * (shared.h_t.ndim - 1)))
            h_t1_pred_rep = shared.h_t1_pred.repeat(repeat_steps, *([1] * (shared.h_t1_pred.ndim - 1)))
            h_t1_gt_rep = shared.h_t1_gt.repeat(repeat_steps, *([1] * (shared.h_t1_gt.ndim - 1)))
            h_t1_cond_rep = self._build_flow_future_condition(
                h_t1_pred=h_t1_pred_rep,
                h_t1_gt=h_t1_gt_rep,
            )
            # [V8 dual] 全局通道条件(scheduled sampling 逐通道复用同一函数)
            h_ms_cond_rep = None
            if _dual:
                h_ms_pred_rep = shared.h_ms_pred.repeat(repeat_steps, *([1] * (shared.h_ms_pred.ndim - 1)))
                h_ms_gt_rep = shared.h_ms_gt.repeat(repeat_steps, *([1] * (shared.h_ms_gt.ndim - 1)))
                h_ms_cond_rep = self._build_flow_future_condition(
                    h_t1_pred=h_ms_pred_rep,
                    h_t1_gt=h_ms_gt_rep,
                )
            h_vlm_rep = h_vlm_for_flow.repeat(repeat_steps, *([1] * (h_vlm_for_flow.ndim - 1)))
            state_rep = prepared_batch["state"].repeat(repeat_steps, *([1] * (prepared_batch["state"].ndim - 1)))
            state_mask_rep = prepared_batch["state_mask"].repeat(
                repeat_steps, *([1] * (prepared_batch["state_mask"].ndim - 1))
            )
            actions_rep = prepared_batch["actions"].repeat(
                repeat_steps, *([1] * (prepared_batch["actions"].ndim - 1))
            )
            actions_mask_rep = prepared_batch["actions_mask"].repeat(
                repeat_steps, *([1] * (prepared_batch["actions_mask"].ndim - 1))
            )
            action_hz_rep = prepared_batch["action_hz"].repeat(repeat_steps)
            embodiment_rep = prepared_batch["embodiment_id"].repeat(repeat_steps)
            attn_rep = attn_flow.repeat(repeat_steps, *([1] * (attn_flow.ndim - 1)))

            loss_flow = self.flow(
                h_t=h_t_rep,
                h_t1_star=h_t1_cond_rep,
                h_vlm=h_vlm_rep,
                state=state_rep,
                actions=actions_rep,
                action_hz=action_hz_rep,
                embodiment_id=embodiment_rep,
                state_mask=state_mask_rep,
                actions_mask=actions_mask_rep,
                attention_mask=attn_rep,
                h_ms_star=h_ms_cond_rep,   # [V8 dual] None 时 flow 走单通道老路
            )

            loss_total = (
                loss_flow
                + self.model_cfg.perceptual_weight * loss_perceptual
                + self.model_cfg.lam_encoder_distill_weight * loss_distill
            )

        zero = torch.tensor(0.0, device=device, dtype=loss_total.dtype)
        out = {
            "loss_flow": loss_flow,
            "loss_perceptual": loss_perceptual,
            "loss_distill": loss_distill,
            "loss_vlm": zero,
            "loss_total": loss_total,
        }
        if _dual:
            # 逐通道监控(plan §3: loss_perceptual_ms→~0.011, local 更低)
            out["loss_perceptual_local"] = loss_perceptual_local.detach()
            out["loss_perceptual_ms"] = loss_perceptual_ms.detach()
            out["loss_distill_local"] = loss_distill_local.detach()
            out["loss_distill_ms"] = loss_distill_ms.detach()
        return out

    @torch.inference_mode()
    def predict_action(
        self,
        *,
        batch: LatentWorldPolicyInferBatch,
        guidance_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        return_intermediates: bool = False,
        return_padded: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        flow_stage_dtype = torch.float32
        prepared_batch = cast(
            LatentWorldPolicyInferBatch,
            self._prepare_infer_batch(batch=batch),
        )

        if guidance_scale is None:
            import os as _os_g   # V7-CFG扫: env 覆盖 guidance(w<1降hint依赖, w=0纯base局部精度)
            guidance_scale = float(_os_g.environ.get("LMWM_CFG_GUIDANCE", self.flow.config.cfg_guidance_scale))
        if num_inference_steps is None:
            num_inference_steps = int(self.flow.config.num_inference_steps)

        shared = self._run_shared_encoding_infer(
            prepared_batch=prepared_batch,
            source="LatentWorldPolicyBackend.predict_action",
            lam_features_with_no_grad=False,
        )
        attn_flow = prepared_batch["attention_mask"] == 1

        _dual = getattr(self, "_lmwm_dual", False)
        with _cuda_autocast(flow_stage_dtype):
            actions = self.flow.sample_actions_cfg(
                h_t=shared.h_t,
                h_t1_star=shared.h_t1_pred,
                h_vlm=shared.h_vlm,
                state=prepared_batch["state"],
                state_mask=prepared_batch["state_mask"],
                action_hz=prepared_batch["action_hz"],
                embodiment_id=prepared_batch["embodiment_id"],
                cfg_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                attention_mask=attn_flow,
                return_padded=bool(return_padded),
                h_ms_star=shared.h_ms_pred if _dual else None,   # [V8 dual] 全局通道; CFG 只调此段
            )

        if not return_intermediates:
            return actions

        num_tokens = shared.h_t_original.shape[1]
        hw = int(num_tokens ** 0.5)
        if hw * hw != num_tokens:
            hw = 16
        intermediates = {
            "h_t": shared.h_t_original.detach().cpu(),
            "h_t1_pred": shared.h_t1_pred.detach().cpu(),
            "vision_tokens_hw": (hw, hw),
        }
        return actions, intermediates
