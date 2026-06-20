"""ACWMTrainer —— 把 GigaWorld-Policy(WAM)退化/改造为【动作条件世界模型 AC-WM】(Ctrl-World 范式),
并吸收 Ctrl-World 的训练 trick。

继承 CasualWATrainer,只覆写 get_models(读新旋钮) + forward_step(注入 AC-WM 与 trick)。

吸收的 Ctrl-World trick:
  ✅ AC-WM:动作以 **clean** 注入视频去噪(acwm_clean_action=True 时强制 t_action=0),且不监督动作
     (配合 config lambda_action=0)→ 模型只学 P(视频|首帧,给定动作,文本);外部 π0.5 在环提供动作。
  ✅ 条件帧噪声增强(cond_noise_aug):对 ref(首帧)条件 latent 注入小噪声(flow 下向噪声插值),
     缩小自回归 rollout 的 train/test gap(推理时 ref 是模型自生成的不完美 latent),抗漂移。
  ⏳ 带噪多帧 history(num_history>1):需改 transformer。CasualWorldActionTransformer.forward 里
     `hidden_states=cat([ref_latents,noisy_latents],dim=2)` 且 `num_ref_tokens=pw*ph`(硬编码 ref=单帧)。
     扩 K 帧需:num_ref_tokens=K*pw*ph、相应 RoPE 切片(line ~743-750)、因果 mask。本 trainer 对
     num_history>1 抛 NotImplementedError 并指明改点;M1 先用 num_history=1。

注:本 trainer 的 forward_step 由 CasualWATrainer.forward_step 改写而来(仅注入 [ACWM] 标记处),
    首次跑 M1 需做一次前向 smoke 验证。
"""
import torch

from .wa_casual_trainer import CasualWATrainer


class ACWMTrainer(CasualWATrainer):
    def get_models(self, model_config):
        model = super().get_models(model_config)
        # 新旋钮
        self.acwm_clean_action = bool(model_config.get("acwm_clean_action", True))
        self.cond_noise_aug = float(model_config.get("cond_noise_aug", 0.0))
        self.num_history = int(model_config.get("num_history", 1))
        # 多帧 history 已由 transformer_wa_casual 的 num_ref_frames=ref_latents.shape[2] 泛化支持(K>=1)。
        assert self.num_history >= 1
        # AC-WM 下动作必须能 clean 注入:确保走 independent_action_sigma 分支(便于把 t_action 置 0)
        if self.acwm_clean_action and not (self.independent_action_sigma or self.async_noise):
            self.independent_action_sigma = True
        return model

    def _augment_ref(self, cond_latents):
        """条件/历史帧噪声增强:flow-matching 下把每个条件帧独立向噪声插值一个随机小 σ(∈[0,cond_noise_aug))。
        cond_latents: (bs, C, K, h, w);每帧独立 σ(Ctrl-World 风格)。"""
        if self.cond_noise_aug <= 0:
            return cond_latents
        bs, _, K = cond_latents.shape[0], cond_latents.shape[1], cond_latents.shape[2]
        sigma_r = torch.rand(bs, 1, K, 1, 1, device=cond_latents.device, dtype=cond_latents.dtype) * self.cond_noise_aug
        noise = torch.randn_like(cond_latents)
        return cond_latents * (1 - sigma_r) + noise * sigma_r

    def forward_step(self, batch_dict):
        # 仅支持 latent-cache + expand_timesteps 路径(acwm_clothfold 配置即此路径);其余回退基类。
        use_latent_cache = "visual_latents" in batch_dict
        if not (use_latent_cache and self.expand_timesteps):
            return super().forward_step(batch_dict)

        import functools
        transformer = functools.partial(self.model, "transformer")

        visual_latents = batch_dict["visual_latents"].to(self.dtype)
        _bs, _ndim = visual_latents.shape[0], visual_latents.ndim
        bs = _bs
        prompt_embeds = batch_dict["prompt_embeds"].to(self.dtype)

        if self.async_noise:
            sigma, ans_action_sigma, timestep, ans_action_ts, ans_clean = self.get_ans_timesteps(_bs, _ndim)
        else:
            timestep, sigma = self.get_timestep_and_sigma(_bs, _ndim)
            ans_action_sigma = ans_action_ts = ans_clean = None
        _ts_per_sample = timestep.float()
        if self.independent_action_sigma and not self.async_noise:
            ans_action_ts, ans_action_sigma = self.get_timestep_and_sigma(_bs, ndim=3)

        action = batch_dict["action"]
        state = batch_dict["state"]
        if self.state_repeats > 1:
            state = state.repeat(1, self.state_repeats, 1)
        if self.action_repeats > 1:
            action = action.repeat(1, self.action_repeats, 1)

        # --- video flow-matching ---
        visual_noise = torch.randn_like(visual_latents)
        visual_target = visual_noise - visual_latents
        noisy_latents = visual_noise * sigma + visual_latents * (1 - sigma)

        # --- action：[ACWM] 干净注入 ---
        action_sigma = ans_action_sigma if (self.async_noise or self.independent_action_sigma) else sigma.squeeze(-1).squeeze(-1)
        action_noise = torch.randn_like(action)
        action_target = action_noise - action
        noisy_action = action_noise * action_sigma + action * (1 - action_sigma)
        if self.acwm_clean_action:
            # [ACWM] 视频以"给定动作"为条件 → 动作 clean(t_action=0);动作不被监督(config λ_action=0)
            noisy_action = action.clone()

        # --- [ACWM] 带噪多帧 history 条件(K=num_history;前 K 帧作条件,各帧独立加噪)---
        num_latent_frames = visual_latents.shape[2]
        latent_height = visual_latents.shape[-2]
        latent_width = visual_latents.shape[-1]
        K = self.num_history
        assert 1 <= K < num_latent_frames, (
            f"num_history={K} 必须满足 1<=K<latent 窗口帧数 T={num_latent_frames}"
            f"(当前 latent 窗 T 太短;需重抽更长窗或减小 num_history)"
        )
        # 历史条件 = clip 前 K 帧,逐帧噪声增强(抗自回归漂移)
        history_cond = self._augment_ref(visual_latents[:, :, :K])          # (bs,C,K,h,w)
        cond_full = torch.zeros_like(visual_latents)
        cond_full[:, :, :K] = history_cond

        first_frame_mask = torch.ones(
            bs, 1, num_latent_frames, latent_height, latent_width,
            dtype=visual_latents.dtype, device=visual_latents.device,
        )
        first_frame_mask[:, :, :K] = 0                                       # 前 K 帧为条件(不去噪/不计损失)
        insert_noisy_latents = (1 - first_frame_mask) * cond_full + first_frame_mask * noisy_latents
        temp_ts = (first_frame_mask[:, :, :, ::2, ::2] * timestep[:, None, None, None, None]).reshape(bs, -1)
        timestep = temp_ts

        insert_noisy_latents = insert_noisy_latents.to(self.dtype)
        num_state_tokens = state.shape[1]
        num_action_tokens = action.shape[1]
        noise_t = timestep[:, -2:-1]
        noisy_action = noisy_action.to(self.dtype)
        state = state.to(self.dtype)
        ref_latents = insert_noisy_latents[:, :, :K]                        # K 帧 history → transformer ref
        noisy_latents = insert_noisy_latents[:, :, K:]
        frame_per_tokens = first_frame_mask.shape[-1] * first_frame_mask.shape[-2] // 4
        num_latent_tokens = frame_per_tokens * first_frame_mask.shape[2]
        timestep = torch.zeros(
            bs, num_state_tokens + num_action_tokens + num_latent_tokens,
            device=noisy_latents.device, dtype=noisy_latents.dtype,
        )
        num_clean_latent_tokens = frame_per_tokens * K                      # 前 K 帧 token 为 clean 条件
        timestep[:, num_state_tokens + num_clean_latent_tokens:] = noise_t
        if self.async_noise or self.independent_action_sigma:
            a0 = num_state_tokens + num_clean_latent_tokens
            timestep[:, a0:a0 + num_action_tokens] = ans_action_ts.to(timestep.dtype)[:, None]

        visual_pred, action_pred = transformer(
            ref_latents=ref_latents,
            noisy_latents=noisy_latents,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
            action=noisy_action,
            state=state,
        )

        # --- loss:[ACWM] 只回传视频损失(动作不监督)---
        visual_loss = ((visual_pred.float() - visual_target.float()) * first_frame_mask) ** 2
        if self.training_weight_enabled:
            _vl_per = visual_loss.mean(dim=list(range(1, visual_loss.ndim)))
            visual_loss = (_vl_per * self._training_weight(_ts_per_sample)).mean()
        else:
            visual_loss = visual_loss.mean()

        # 返回 dict;backward 的 total=sum(dict)。AC-WM 只含视频损失。
        return {"visual_loss": self.lambda_video * visual_loss}
