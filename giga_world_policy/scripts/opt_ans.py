"""优化推理引擎:把 prefix-KV/compile/FP8 加速阶梯接到【真实 ckpt + 真实 serving 输入】上,
同时支持两条路径:
  - gwp_ori(切断):action_only 快路径(48 active tokens)—— 复用 PrefixCachedRunner;
  - gwp_ans(异步耦合):全量路径(action 48 + noisy-video 144 active tokens,T_a<T_O)
    —— 本文件新增 AnsPrefixRunner:prefix(state+ref 145 tok)因掩码不 attend action/noisy,
    跨步恒定 → 同样可一次编码缓存 K/V;每步只算活跃 192 tok,且 action/video 各用自己的
    per-token timestep(t_a / t_O),与 wa_pipeline 的 ANS 推理逐位等价。

用法(bench+parity,真实权重):
  python -m scripts.opt_ans --transformer_dir <dir> --model_id <Wan-Diffusers> \
      [--tier exact|fp8] [--steps_video 10 --steps_act 5] [--parity] [--bench N]
episode_report 集成:--engine opt(见 episode_report 补丁),infer 路径换成 opt_call()。
"""
import argparse
import time

import torch

from scripts.prefix_cache import PrefixCachedRunner
from world_action_model.models.transformer_wa_casual import CasualWorldActionTransformer


class AnsPrefixRunner(PrefixCachedRunner):
    """全量路径(action+noisy-video 活跃)的前缀缓存 runner。

    与父类共享:prefix 编码/缓存(prepare)、CachedSelfAttnProcessor("read" 模式把
    K/V 拼成 [prefix; active])。差异:active 集多了 video tokens,且 timestep 按
    [t_a×48 ; t_O×144] per-token 注入;输出端 action_decoder + proj_out(unpatchify 仅 noisy 帧)。
    """

    def prepare_ans(self, ref_latents, noisy_latents, encoder_hidden_states, state):
        self.prepare(ref_latents, noisy_latents, encoder_hidden_states, state)
        m = self.m
        # noisy 位置的 video rope(常量,跨步不变):整段 video rope 去掉 prefix 的 ref 部分
        hidden = torch.cat([ref_latents, noisy_latents], dim=2)
        vr = m.rope(hidden)
        self.rope_noisy = (vr[0][:, self.num_ref:].contiguous(), vr[1][:, self.num_ref:].contiguous())
        # unpatchify 几何
        p_t, p_h, p_w = m.config.patch_size
        self._geo = (noisy_latents.shape[2] // p_t, noisy_latents.shape[3] // p_h, noisy_latents.shape[4] // p_w,
                     p_t, p_h, p_w)

    def compile_step_ans(self, mode="reduce-overhead"):
        self._step_ans_compiled = torch.compile(self._step_ans_impl, mode=mode, fullgraph=False)

    def step_ans(self, action, noisy_latents, t_act, t_vid):
        if getattr(self, "_step_ans_compiled", None) is not None:
            torch.compiler.cudagraph_mark_step_begin()
            a, v = self._step_ans_compiled(action, noisy_latents, t_act, t_vid)
            return a.clone(), v.clone()
        return self._step_ans_impl(action, noisy_latents, t_act, t_vid)

    @torch.no_grad()
    def _step_ans_impl(self, action, noisy_latents, t_act, t_vid):
        m = self.m
        bs, seq_a, _ = action.shape
        a_states = m.action_encoder(action)
        v_tokens = m.patch_embedding(noisy_latents.to(a_states.dtype)).flatten(2).transpose(1, 2)
        seq_v = v_tokens.shape[1]
        ha = torch.cat([a_states, v_tokens], dim=1)

        # per-token timestep:[t_a×seq_a ; t_O×seq_v](与 pipeline 的 timestep_full 活跃段一致)
        ts = torch.cat([t_act.reshape(1).expand(seq_a), t_vid.reshape(1).expand(seq_v)]).to(a_states.dtype)
        ts = ts.unsqueeze(0).expand(bs, -1).reshape(-1)
        temb, tproj, _, _ = m.condition_embedder(ts, self.enc_proj_raw, None, timestep_seq_len=seq_a + seq_v)
        tproj = tproj.unflatten(2, (6, -1))

        rope = (torch.cat([self.action_rope[0], self.rope_noisy[0]], dim=1),
                torch.cat([self.action_rope[1], self.rope_noisy[1]], dim=1))
        for blk in m.blocks:
            ha = blk(ha, self.enc_proj, tproj, rope, None)

        shift, scale = (m.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
        shift, scale = shift.squeeze(2), scale.squeeze(2)
        ha = (m.norm_out(ha.float()) * (1 + scale) + shift).type_as(ha)

        action_pred = m.action_decoder(ha[:, :seq_a])
        vh = m.proj_out(ha[:, seq_a:])
        nf, gh, gw, p_t, p_h, p_w = self._geo
        vh = vh.reshape(bs, nf, gh, gw, p_t, p_h, p_w, -1).permute(0, 7, 1, 4, 2, 5, 3, 6)
        noise_pred = vh.flatten(6, 7).flatten(4, 5).flatten(2, 3)
        return action_pred, noise_pred

    # ---------- BAC for ANS path(192 active tokens 的逐 block 残差缓存)----------
    def compile_bac_ans(self, refresh_mode="max-autotune-no-cudagraphs", cached_mode="reduce-overhead"):
        self._ans_refresh_compiled = torch.compile(self._step_ans_refresh_impl, mode=refresh_mode, fullgraph=False)
        self._ans_cached_compiled = torch.compile(self._step_ans_cached_impl, mode=cached_mode, fullgraph=False)

    def _ans_pre(self, action, noisy_latents, t_act, t_vid):
        m = self.m
        bs, seq_a, _ = action.shape
        a_states = m.action_encoder(action)
        v_tokens = m.patch_embedding(noisy_latents.to(a_states.dtype)).flatten(2).transpose(1, 2)
        seq_v = v_tokens.shape[1]
        ha = torch.cat([a_states, v_tokens], dim=1)
        ts = torch.cat([t_act.reshape(1).expand(seq_a), t_vid.reshape(1).expand(seq_v)]).to(a_states.dtype)
        ts = ts.unsqueeze(0).expand(bs, -1).reshape(-1)
        temb, tproj, _, _ = m.condition_embedder(ts, self.enc_proj_raw, None, timestep_seq_len=seq_a + seq_v)
        rope = (torch.cat([self.action_rope[0], self.rope_noisy[0]], dim=1),
                torch.cat([self.action_rope[1], self.rope_noisy[1]], dim=1))
        return ha, temb, tproj.unflatten(2, (6, -1)), rope, seq_a

    def _ans_post(self, ha, temb, seq_a, bs):
        m = self.m
        shift, scale = (m.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
        ha = (m.norm_out(ha.float()) * (1 + scale.squeeze(2)) + shift.squeeze(2)).type_as(ha)
        action_pred = m.action_decoder(ha[:, :seq_a])
        vh = m.proj_out(ha[:, seq_a:])
        nf, gh, gw, p_t, p_h, p_w = self._geo
        vh = vh.reshape(bs, nf, gh, gw, p_t, p_h, p_w, -1).permute(0, 7, 1, 4, 2, 5, 3, 6)
        return action_pred, vh.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    @torch.no_grad()
    def _step_ans_refresh_impl(self, action, noisy_latents, t_act, t_vid):
        ha, temb, tproj, rope, seq_a = self._ans_pre(action, noisy_latents, t_act, t_vid)
        for i, blk in enumerate(self.m.blocks):
            out = blk(ha, self.enc_proj, tproj, rope, None)
            d = out - ha
            if self.delta_buf[i] is None or self.delta_buf[i].shape != d.shape:
                self.delta_buf[i] = torch.empty_like(d)
            self.delta_buf[i].copy_(d)
            ha = out
        return self._ans_post(ha, temb, seq_a, action.shape[0])

    @torch.no_grad()
    def _step_ans_cached_impl(self, action, noisy_latents, t_act, t_vid):
        ha, temb, tproj, rope, seq_a = self._ans_pre(action, noisy_latents, t_act, t_vid)
        for i, blk in enumerate(self.m.blocks):
            if self._bac_mask[i]:
                ha = blk(ha, self.enc_proj, tproj, rope, None)
            else:
                ha = ha + self.delta_buf[i]
        return self._ans_post(ha, temb, seq_a, action.shape[0])

    def step_ans_refresh(self, action, noisy, ta, tv):
        f = getattr(self, "_ans_refresh_compiled", None) or self._step_ans_refresh_impl
        a, v = f(action, noisy, ta, tv)
        return a, v

    def step_ans_cached(self, action, noisy, ta, tv, mask):
        self._bac_mask = tuple(mask)
        f = getattr(self, "_ans_cached_compiled", None)
        if f is not None:
            torch.compiler.cudagraph_mark_step_begin()
            a, v = f(action, noisy, ta, tv)
            return a.clone(), v.clone()
        return self._step_ans_cached_impl(action, noisy, ta, tv)


# ---------------- 端到端 opt_call:替换 WAPipeline.__call__ 的去噪循环 ----------------

@torch.no_grad()
def opt_call(pipe, runner, *, image, state, prompt_embeds, height, width, num_frames,
             action_chunk, num_inference_steps, action_num_inference_steps=None,
             generator=None, is_ans=False, bac_skip=0, _prepared=[False]):
    """复用 pipe 的 VAE/预处理/调度器,去噪循环换成 runner。返回 action(归一化空间)。
    数学上与 stock 路径等价(exact 档),输入输出契约对齐 episode_report 的用法。"""
    device = pipe._execution_device
    dt = pipe.transformer.dtype
    t_a_steps = action_num_inference_steps or num_inference_steps

    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    pipe.action_scheduler.set_timesteps(t_a_steps, device=device)
    action_timesteps = pipe.action_scheduler.timesteps

    img = pipe.video_processor.preprocess(image, height=height, width=width).to(device, dtype=torch.float32)
    st = state.unsqueeze(0).to(device=device, dtype=dt)
    latents, condition, first_frame_mask, action = pipe.prepare_latents(
        img, 1, pipe.vae.config.z_dim, height, width, num_frames, torch.float32, device, generator, None, None,
        action_chunk)
    action = action.to(dtype=dt)

    ref_const = ((1 - first_frame_mask) * condition + first_frame_mask * latents)[:, :, :1].to(dt)
    enc = prompt_embeds.to(dt)

    # prefix 编码(每次推理一次)
    if is_ans:
        runner.prepare_ans(ref_const, latents[:, :, 1:].to(dt), enc, st)
    else:
        runner.prepare(ref_const, latents[:, :, 1:].to(dt), enc, st)
    runner.set_action_rope(action_chunk)

    for i, t in enumerate(timesteps):
        if i >= t_a_steps and not is_ans:
            break
        if is_ans:
            noisy = latents[:, :, 1:].to(dt)
            ta_i, tv_i = action_timesteps[i].to(dt), t.to(dt)
            if bac_skip and i == 0:
                a_pred, n_pred = runner.step_ans_refresh(action, noisy, ta_i, tv_i)
            elif bac_skip:
                nb = len(runner.m.blocks); st0 = (nb - bac_skip) // 2
                mask = [not (st0 <= j < st0 + bac_skip) for j in range(nb)]
                a_pred, n_pred = runner.step_ans_cached(action, noisy, ta_i, tv_i, mask)
            else:
                a_pred, n_pred = runner.step_ans(action, noisy, ta_i, tv_i)
            lat_noisy = pipe.scheduler.step(n_pred.float(), t, latents[:, :, 1:], return_dict=False)[0]
            latents = torch.cat([latents[:, :, :1], lat_noisy], dim=2)
        else:
            ta_i = t.to(dt)
            if bac_skip and i == 0:
                a_pred = runner.step_refresh(action, ta_i)
            elif bac_skip:
                nb = len(runner.m.blocks); st0 = (nb - bac_skip) // 2
                mask = [not (st0 <= j < st0 + bac_skip) for j in range(nb)]
                a_pred = runner.step_cached(action, ta_i, mask)
            else:
                a_pred = runner.step(action, ta_i)
        action = pipe.action_scheduler.step(a_pred.float(), action_timesteps[i], action.float(),
                                            return_dict=False)[0].to(dt)
        if i + 1 >= t_a_steps:
            break
    return action


# ---------------- bench + parity(真实权重) ----------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--transformer_dir", required=True)
    p.add_argument("--tier", default="exact", choices=["eager", "exact", "fp8"])
    p.add_argument("--compile_mode", default="max-autotune-no-cudagraphs")
    p.add_argument("--steps_video", type=int, default=10)
    p.add_argument("--steps_act", type=int, default=0, help="0=自动(ans→5, ori→steps_video)")
    p.add_argument("--height", type=int, default=192); p.add_argument("--width", type=int, default=768)
    p.add_argument("--action_chunk", type=int, default=48)
    p.add_argument("--t5_len", type=int, default=64)
    p.add_argument("--bench", type=int, default=20)
    p.add_argument("--parity", action="store_true")
    p.add_argument("--bac", type=int, default=0, help="跳过中段 block 数(0=off)")
    a = p.parse_args()
    dev, dt = torch.device("cuda"), torch.bfloat16
    torch.manual_seed(0)

    m = CasualWorldActionTransformer.from_pretrained(a.transformer_dir).to(device=dev, dtype=dt).eval()
    is_ans = bool(getattr(m.config, "action_attends_video", False))
    steps_act = a.steps_act or (5 if is_ans else a.steps_video)
    print(f"[opt] real ckpt loaded | is_ans={is_ans} | tier={a.tier} | T_a={steps_act}/T_O={a.steps_video}")

    if a.tier == "fp8":
        from scripts.fp8_linear import swap_linears_to_fp8
        n = swap_linears_to_fp8(m.blocks)
        print(f"[opt] fp8 swapped {n} linears")
    if a.tier in ("exact", "fp8"):
        for mod in m.modules():
            if hasattr(mod, "fuse_projections") and hasattr(mod, "set_processor"):
                try: mod.fuse_projections()
                except Exception: pass

    # 合成 serving 形状输入(latent 域;几何同 768×192/5帧;权重真实 → parity 有意义)
    z = m.config.in_channels
    lh, lw = a.height // 16, a.width // 16
    ref = torch.randn(1, z, 1, lh, lw, device=dev, dtype=dt)
    noisy0 = torch.randn(1, z, 1, lh, lw, device=dev, dtype=dt)
    state = torch.randn(1, 1, 14, device=dev, dtype=dt)
    act0 = torch.randn(1, a.action_chunk, 14, device=dev, dtype=dt)
    enc = torch.randn(1, a.t5_len, m.config.text_dim, device=dev, dtype=dt)

    runner = AnsPrefixRunner(m) if is_ans else PrefixCachedRunner(m)
    if a.bac:
        runner.init_bac(len(m.blocks))
        if a.tier in ("exact", "fp8"):
            (runner.compile_bac_ans if is_ans else runner.compile_bac)()
    if a.tier in ("exact", "fp8"):
        runner.compile_prepare("reduce-overhead")
        (runner.compile_step_ans if is_ans else runner.compile_step)("reduce-overhead")

    def prep():
        (runner.prepare_ans if is_ans else runner.prepare)(ref, noisy0, enc, state)
        runner.set_action_rope(a.action_chunk)

    _step_idx = [0]
    def one_step(act, noisy, ta, tv):
        i = _step_idx[0]; _step_idx[0] += 1
        if a.bac:
            nb = len(m.blocks); st0 = (nb - a.bac) // 2
            mask = [not (st0 <= j < st0 + a.bac) for j in range(nb)]
            if is_ans:
                return runner.step_ans_refresh(act, noisy, ta, tv) if i == 0 else runner.step_ans_cached(act, noisy, ta, tv, mask)
            return (runner.step_refresh(act, ta) if i == 0 else runner.step_cached(act, ta, mask)), None
        if is_ans:
            return runner.step_ans(act, noisy, ta, tv)
        return runner.step(act, ta), None  # ori:action token 的 t 才是 step 的 t

    # ---- parity vs stock forward(同输入逐位对比)----
    if a.parity:
        prep()
        ta = torch.tensor(420.0, device=dev); tv = torch.tensor(700.0, device=dev)
        ap, npred = one_step(act0, noisy0, ta, tv)
        fpt = (lh // 2) * (lw // 2); total = 1 + a.action_chunk + 2 * fpt
        ts = torch.zeros(1, total, device=dev, dtype=dt)
        ts[:, 1 + fpt:1 + fpt + a.action_chunk] = ta
        ts[:, 1 + fpt + a.action_chunk:] = tv
        runner._set_mode("off")  # stock 参考前向不能带 prefix-KV 注入
        with torch.no_grad():
            if is_ans:
                out_full, ap_ref = m._forward_inference(ref_latents=ref, noisy_latents=noisy0, timestep=ts,
                                                        encoder_hidden_states=enc, action=act0, state=state,
                                                        return_dict=False)
                np_ref = out_full[:, :, 1:]
                print(f"[parity] action max|d|={(ap-ap_ref).abs().max().item():.2e} "
                      f"video max|d|={(npred-np_ref).abs().max().item():.2e}")
            else:
                ap_ref = m(ref_latents=ref, noisy_latents=noisy0, timestep=ts, encoder_hidden_states=enc,
                           action=act0, state=state, action_only=True, return_dict=False)
                print(f"[parity] action max|d|={(ap-ap_ref).abs().max().item():.2e}")
        runner._set_mode("read")

    # ---- bench:prepare 1次 + steps_act 步 ----
    if a.bench:
        sched_t = torch.linspace(1000, 100, a.steps_video, device=dev)
        act_t = torch.linspace(1000, 200, steps_act, device=dev)
        for _ in range(3):  # warmup
            prep(); _step_idx[0] = 0
            act = act0.clone(); noisy = noisy0.clone()
            for i in range(steps_act):
                ap, npred = one_step(act, noisy, act_t[i], sched_t[i])
        torch.cuda.synchronize(); times = []
        for _ in range(a.bench):
            t0 = time.perf_counter()
            prep(); _step_idx[0] = 0
            act = act0.clone(); noisy = noisy0.clone()
            for i in range(steps_act):
                ap, npred = one_step(act, noisy, act_t[i], sched_t[i])
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1e3)
        ts_ = torch.tensor(times)
        print(f"[bench] rollout({steps_act} step): mean {ts_.mean():.1f}ms ± {ts_.std():.1f} | min {ts_.min():.1f}ms "
              f"| mem {torch.cuda.max_memory_allocated()/2**30:.1f}G")


if __name__ == "__main__":
    main()
