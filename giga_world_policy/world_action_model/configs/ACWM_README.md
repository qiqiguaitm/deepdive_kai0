# AC-WM (Wan2.2) on cloth-fold — 运行说明

把 GigaWorld-Policy 改造为 **动作条件世界模型 (AC-WM, Ctrl-World 范式)**：视频为主、动作为给定条件，
外部 π0.5 在环 rollout。新增两文件（不改动原有文件）：

- `world_action_model/configs/acwm_clothfold.py` — AC-WM 配置（继承 `visrobot01_gwp_abs_v5`）。
- `world_action_model/trainer/acwm_trainer.py` — `ACWMTrainer`（`CasualWATrainer` 子类）。

## 吸收的 Ctrl-World trick
| trick | 状态 | 实现 |
|---|---|---|
| AC-WM：动作 clean 注入 + 不监督动作 | ✅ | `acwm_clean_action=True`(强制 t_action=0) + `lambda_action=0` |
| 条件帧噪声增强(抗自回归漂移) | ✅ | `cond_noise_aug=0.3`，`ACWMTrainer._augment_ref` |
| 文本/动作 CFG、future-only loss、多相机 | ✅ | giga 基类已有 |
| 带噪多帧 history | ⏳ 需改 transformer | `num_history>1` 当前抛错；改点见下 |
| 随机时间步长 skip | ❌ 受限 | 预计算 latent 固定 stride=4；需重抽多 stride 或走 raw-video |

### 多帧 history 的 transformer 改点（后续）
`world_action_model/models/transformer_wa_casual.py::forward`：
- L713 `hidden_states = cat([ref_latents, noisy_latents], dim=2)` — ref 现为 1 帧。
- L726 `num_ref_tokens = post_patch_width * post_patch_height` → 改 `K * pw * ph`。
- L743-750 RoPE 切片、因果 mask 按 K 帧 ref 调整。
改完后把 `ACWMTrainer.get_models` 的 `num_history!=1` 拦截放开，并在 forward_step 把前 K 帧作 noised 条件
（mask[:, :, :K]=0，条件内容 = `_augment_ref(visual_latents[:, :, :K])`）。

## ⚠️ 数据一致性（关键）
- **M1 只用 `visrobot01_v3_train` + `vae_latent_v3fix`**（相机顺序修正：top_head 全分辨率主图，latent 24×20）。
- **不要直接混 `kairobot01_v3`**：其 `vae_latent` 是另一布局（12×48, `visual/ref` schema, 主图不同），
  与 visrobot-v3fix 不一致，混训会视角错位。
- 混库前用 `scripts/wam_pipeline/compute_latents.py` 以**统一布局/相机序/分辨率/stride** 重抽两库，
  再启用 `acwm_clothfold.py` 文末的 MIX 块。

## 运行 M1
```bash
# 1. 环境（giga 独立 env）
conda create -n gigaworld-policy python==3.11 && conda activate gigaworld-policy
pip install ./third_party/giga-train ./third_party/giga-models ./third_party/giga-datasets
# 模型：本地 checkpoints/Wan2.2-TI2V-5B-Diffusers + Wan2.2-T5（配置 redirect_common_files=false 指向本地）

# 2. 先做一次前向 smoke（验证 ACWMTrainer.forward_step 改写无误）—— 1 step：
#    将 acwm_clothfold 的 train.max_steps 临时设小、batch=1、gpu_ids=[0]
python -m scripts.train --config world_action_model.configs.acwm_clothfold

# 3. 正式：8 卡 deepspeed zero2；内联 eval_fold(every=1000, n_eps=100) 出 GT-vs-预测 + MAE@{1,10,24,48}
```

> 首次跑务必先 1-step smoke：`ACWMTrainer.forward_step` 由基类 `forward_step` 改写而来（注入 [ACWM] 处），
> 需验证张量形状/返回 dict 与基类一致后再开长训。
