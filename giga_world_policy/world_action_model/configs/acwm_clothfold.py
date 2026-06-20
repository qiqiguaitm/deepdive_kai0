"""acwm_clothfold —— 基于 Wan2.2-TI2V-5B 的【动作条件世界模型 AC-WM】(Ctrl-World 范式)。

与 GigaWorld-Policy 默认的 WAM(动作为主、视频可选)相反:这里 **视频为主、动作为给定条件**
  P(未来视频 | 首帧/历史, 给定动作, 文本)；外部策略(π0.5)提供动作，模型只 rollout 视频。

实现要点：
  - AC-WM：lambda_action=0(不监督动作) + 动作以 clean 方式注入(见 ACWMTrainer，强制 t_action=0)。
  - 吸收 Ctrl-World tricks：带噪多帧 history 条件(num_history>1) + 条件帧噪声增强(cond_noise_aug)。
    → 这两条需要 ACWMTrainer(本配置 runners 已指向它)。
  - runners 指向 world_action_model.trainer.acwm_trainer.ACWMTrainer。

数据(M1 先单库，保证可跑通)：visrobot01_v3_train + vae_latent_v3fix(相机顺序修正版，top_head 全分辨率主图)。
  ⚠️ 不要直接混 kairobot01_v3：其 vae_latent 布局(12×48, visual/ref schema, 主图不同)与 visrobot
     的 vae_latent_v3fix(24×20)【不一致】，混训会视角错位。混库需先用 scripts/wam_pipeline/compute_latents.py
     以【统一布局】重抽两库后再开启(见文末 MIX 块)。
"""
import copy

from world_action_model.configs.visrobot01_gwp_abs_v5 import config as _base

config = copy.deepcopy(_base)

PROJ = "runs/acwm_clothfold"
config["project_dir"] = PROJ
config["models"]["view_dir"] = PROJ

# ---- runner: 用 AC-WM trainer(吸收 Ctrl-World tricks)----
config["runners"] = ["world_action_model.trainer.acwm_trainer.ACWMTrainer"]

# ---- AC-WM：视频为主、动作为给定条件 ----
config["models"]["lambda_video"] = 1.0
config["models"]["lambda_action"] = 0.0          # 不监督动作（动作仅作条件）
config["models"]["independent_action_sigma"] = True
config["models"]["acwm_clean_action"] = True     # ACWMTrainer: 强制 t_action=0(动作 clean 注入)

# ---- 吸收 Ctrl-World tricks（由 ACWMTrainer 读取）----
# cond_noise_aug:条件(首帧/历史)帧噪声增强 —— Ctrl-World 抗自回归漂移的核心,低风险,M1 即开。
config["models"]["cond_noise_aug"] = 0.3         # flow 下把条件帧向噪声插值的 σ 上界(0=关闭)
# num_history:带噪多帧 history。transformer_wa_casual 已泛化支持(num_ref_frames=ref_latents.shape[2])。
#   约束:K < latent 窗口帧数 T。visrobot vae_latent_v3fix 的 T=4 → K≤3;
#   ⚠️ 混 kairobot(T=2)时 K 只能=1;混训前请设 num_history=1(或用 compute_latents 重抽更长窗)。
config["models"]["num_history"] = 2
# 随机时间步长(skip=randint(1,2)):受预计算固定 stride latent 限制，暂不启用;
# 如需，应在 compute_latents 重抽时生成多 stride latent，或改走 raw-video 解码路径。

# ---- 内联 fold 评测(沿用 v5；visrobot01_v3_val)----
config["models"].setdefault("eval_fold", {})
config["models"]["eval_fold"].update(dict(enabled=True, every=1000, n_eps=100))

# =====================================================================================
# MIX 块(待 kairobot 与 visrobot 用统一布局重抽 latent 后启用)：
#   for _ent in extra_kairobot_entries: config["dataloaders"]["train"]["data_or_config"].append(_ent)
#   并把 latent_dir 指向统一目录(如 vae_latent_uni)，view_keys 统一相机序。
# =====================================================================================
