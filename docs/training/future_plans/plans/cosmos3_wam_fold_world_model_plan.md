# Cosmos3 × wam_fold 柔性衣物世界模型方案(Forward Dynamics WM)

> 目标:用 **Cosmos3-Nano (16B omni MoT)** + 本机 **wam_fold_v1 叠衣服数据集(去重 8,610 episodes / 82.3 h)**,
> 后训练一个 **动作可控、衣物形变真实** 的视频世界模型(action-conditioned forward dynamics),
> 服务两个下游闭环:① 策略离线评测器(WorldEval 式,替代部分真机评测);② 数据引擎(DreamGen 式合成轨迹)。
>
> 定位与 `wam_fold_policy`(policy 模式,输出动作)互补:本方案是 **FD 模式,输入动作、输出未来视频**。
> 呼应 abs-lookahead 实验结论"TF 上限 0.0598 → 杠杆在视频侧":视频侧的预测/仿真能力正是本方案主体。

---

## 1. 调研结论摘要(2026-06 深度调研,完整引用见文末)

### 1.1 Cosmos3 侧(决定"怎么训")

- **动作是 Cosmos3 的一等模态**,不是外挂 conditioning branch。MoT 扩散子序列里 video/audio/action
  token 并列,同一 checkpoint 支持三种生成模式:**forward dynamics(给定干净动作去噪视频)**、
  inverse dynamics(给定视频去噪动作)、policy(联合去噪)。后训练后模型**特化到单一模式与频率**。
  (技术报告 arXiv:2606.02800 §2.2)
- 动作中训数据 8.4M episodes / 61.3K h,其中机器人 5.4K h(AgiBot/DROID/Bridge/UMI 等),
  **保留失败与 idle 片段**——我们的数据处理应一致(保留失败 episode,idle 走 idle-count 元数据)。
- 信封:10–30 FPS、5–400 帧、256p/480p/720p;多视角 = **单画布拼接(concat_view)**,无原生多流。
- 官方动作后训练配方已开源(cosmos-framework `action_policy_droid_posttrain.md`):
  LeRobot 格式、TOML recipe、全量微调、bf16、lr 2e-4、动作头 fresh-init(`keys_to_skip_loading`)、
  动作参数 5× lr multiplier、8 卡单机起步、HSDP 多机。**不需要 Blackwell/FP8**,A100 bf16 可行
  (本地 wam_fold_policy 已在本集群实际跑通,见 §6)。
- 评测上 NVIDIA **不用 FVD**:机器人 FD 用 **PSNR**(短时序对齐代理,DROID 16帧:Nano 25.52 dB,
  Ctrl-World 22.99);动作跟随用"从生成视频反推轨迹 vs 条件轨迹";policy 用闭环成功率。
- 已知短板:长时序 action-state drift、形变物体只有定性宣称("fabric 比 Ctrl-World 真实")、
  **没有任何形变定量指标**——这正是本方案要补的评测空白。

### 1.2 相关工作侧(决定"怎么做可控与评测")

- **帧级动作注入 > 轨迹级**:IRASim ablation、ACT-Bench(44.1% vs 30.7% 指令一致性)。
  Cosmos3 原生帧级(action token = 相邻帧转移),天然满足。
- **delta/absolute 混合是事实最佳实践**(GE-Sim、EVAC):空间锚定用绝对、时间增量用 delta。
  本地数据集已实现:臂关节 delta(锚=窗口首帧 proprio)、夹爪绝对(`_DELTA_MASK`),保持不变。
- **可控性手段**:训练期 action CFG dropout(本地 packer 已有 `cfg_dropout_rate=0.1`)→ 推理期
  action-branch CFG(guidance 2–5 扫描);进阶可上 **FreeAction**(training-free:无条件分支用
  **取负动作 −a**,引导权重 ∝‖a‖₂,只作用早期去噪步)修复"零动作 ≠ 无条件"的混淆。
- **可控性度量**(领域共识,缺一不可):
  1. **Δaction 扰动**:同一首帧,真动作 vs 换成别的 episode 的动作,生成结果差异
     (ΔPSNR / Δ-LPIPS,Genie/AdaWorld/dWorldEval 协议);
  2. **IDM 探针**:在真数据上训的逆动力学模型去生成视频上反推动作,看误差比(AVID
     Action Error Ratio;DreamGen Bench 与下游策略成功率相关 >90%);
  3. **轨迹重检测**:生成视频里跟踪 EE/物体关键点 vs GT 轨迹(EWMBench:Hausdorff/DTW)。
     我们有 `config/calibration.yml` 手眼标定 + Piper FK,可把 GT EE 像素轨迹直接投影出来——
     比 EWMBench 的 SAM2 跟踪更强。
- **衣物难点有直接证据**:ACWM-Phys(2605.08567)实测动作条件 WM 在 **deformable contact 上
  OOD 掉点最大**(外观学习而非物理学习);视频模型物理守恒仅 ~20–33%(VideoPhy)。
  对策:单任务大数据强 in-distribution(82.3 h 单任务在该领域是超大规模)、评测门禁、
  不宣称 OOD 物理泛化。
- **截至 2026-06 没有任何以"叠衣服视频世界模型"为核心贡献的发表工作**(验证过的空白),
  最接近的是 GE-Sim 双 Franka 叠衣(250 demos 适配)与 WorldEval(**1,400 条 AgileX 双臂
  14 维轨迹** LoRA WAN-14B,策略评测 Pearson r=0.942——与我们硬件/动作维度同类,
  说明 8.6k episodes 全量微调 16B 模型在数据上绰绰有余)。
- 数据量分界:**LoRA 适用于几十~几百 episodes;≳万级轨迹用全量微调**。我们去重 8,610 episodes
  (官方 DROID 配方 76k,WorldEval LoRA 仅 1.4k,居中偏全量微调侧)→ 全量微调。
- 长时序:Ctrl-World 经验——**腕部相机比外部相机退化快得多**(11s 时 SSIM 0.62 vs 0.83);
  pose-conditioned 稀疏记忆可撑 >20 s。Cosmos3 单次生成 ≤400 帧,长时序走 AR 分块。

### 1.3 本地资产(决定"从哪起步")

| 资产 | 状态 |
|---|---|
| 数据 `kai0/data/wam_fold_v1` | visrobot01 2,098 eps / 28.8 h(切分为 train 1,898 + val 200,**与全量目录同一份数据,勿重复计数**)+ kairobot01 6,512 eps / 53.5 h;**去重合计 8,610 eps / 82.3 h**(均按 info.json total_frames÷30fps 计);3 相机 480×640@30fps H264;14 维绝对关节动作;T5 embedding 已预存;磁盘 ~1.5 TB(含切分目录副本) |
| 框架 `packages/cosmos3`(cosmos-framework v1.2.2 本地改造版) | `WamFoldLeRobotDataset` **已支持 mode=forward_dynamics/inverse_dynamics/policy/joint**;`ActionDataPacker`(CFG dropout、channel mask、idle 元数据);VAE latent 预计算缓存(省 ~34% 步时);跨 rig 域混合(domain 16/17 + 每 rig 分位数归一化) |
| 已跑通的训练基建 | `wam_fold_policy/train/`:单机 8 卡 / 2 节点 / 4n8g 多机 FSDP 全量微调均有产出(`wam_fold_policy_runs/train_out_*`);PFS L2 cache OOM 已修(LRU VideoDecoderCache + pod mem);实测 ~45k token/步、7–8.6 s/步 |
| 评测基建 | `wam_fold_policy/eval/`(MAE/视频报告、16 卡分片);`eval_i2v/`:**Nano/Super/Super-I2V 三模型零样本 I2V 报告已产出**(3 相机 × 1s/3s/7s)→ 即 Phase 0 真实感基线 |
| 检查点 | Policy-DROID DCP 已就绪;**FD 需 `nvidia/Cosmos3-Nano` 基座(MT 中训版)→ 需确认下载 + `convert_model_to_dcp`** |
| 算力 | 本机 8×A100-80GB(驱动 535 → CUDA 12.x → `cu128-train` 组);AIHC 多机 wrapper 现成 |

---

## 2. 总体设计

```
              ┌────────────── 条件 ──────────────┐
首帧/上下文帧(concat_view 画布 ~720×640:cam_high 全幅 + 双腕 ½ 幅)
+ 14 维动作 chunk(臂关节 delta + 夹爪绝对,每 rig 分位数归一化,domain id 16/17)
+ 文本("Flatten and fold the cloth.",T5 已预存)
              │
   Cosmos3-Nano 16B MoT,FD 模式(动作 token 干净,去噪视频 token)
              │
        未来视频 chunk(480p 画布,30 FPS)──AR 分块拼接──→ 长时序 rollout
```

关键设计决策(均有调研依据):

| 决策 | 选择 | 理由 |
|---|---|---|
| 基座 | `nvidia/Cosmos3-Nano`(MT 中训),**不是** Policy-DROID | 报告 LIBERO 实验:MT-init 适配速度/上限全面优于 PT-init;Policy-DROID 已特化 policy 模式,不适合 FD |
| 模式 | 纯 `forward_dynamics`(不用 joint 混合) | 报告明确后训练特化单模式;ID 探针单独训小型号(§4.2) |
| 动作表征 | 沿用现有 delta-arm + abs-gripper + 每 rig 分位数归一化 | 与 GWP/Policy-DROID 约定一致,abs→delta 曾带来 ~55× mae@1 改善;混合表征是领域最佳实践 |
| 视角 | 沿用 concat_view 单画布(3 相机) | Cosmos3 唯一原生多视角方式;DROID 同款;跨视角一致性免费获得 |
| chunk | **32 帧 @ 30 FPS(≈1.07 s)** 起步;Phase 3 试 64 帧 @ 15 FPS(≈4.3 s) | 32 帧 ≈3.7k token/样本,45k token 预算下 ~12 样本/pack,步时可控;衣物形变在 1 s 内已可观察;更长视野交给 AR rollout |
| 微调方式 | 全量微调(动作输入侧投影 fresh-init,5× lr multiplier) | 8.6k episodes 处于全量微调 regime;官方配方同款 |
| CFG | 训练 `cfg_dropout_rate=0.1`(已有);推理 guidance 扫 1–5;FreeAction 作为免训练升级备选 | §1.2 可控性手段 |
| 数据混比 | visrobot01_train ×3 + kairobot01 ×1(≈3:1 平衡,沿用 GWP/policy 配方);**保留失败 episodes** | 与已验证的 policy 配方一致;Cosmos3 中训保留失败数据 |

---

## 3. 训练方案(四阶段)

### Phase 0 — 基线锚定(基本已完成,补齐量化)
- `eval_i2v/` 三模型零样本报告已有 → 抽出 **PSNR/SSIM/LPIPS@{1s,3s,7s}** 数值表作为
  "未后训练真实感基线";补跑 base Nano 的 **零样本 FD**(给动作)对照,验证"基座动作先验
  在 wam 域多差"(预期:动作几乎被忽略,作为可控性基线)。
- 产物:`baseline.json`(后续所有 Phase 与之对比)。

### Phase 1 — FD 后训练 v1(核心交付)
1. 下载/转换基座:`nvidia/Cosmos3-Nano` → `convert_model_to_dcp` → `wam_fold_wm_runs/checkpoints/`。
2. 新建实验配置 `wam_fold_wm_nano.py`(复制 `wam_fold_nano.py`,改动極小):
   - `build_cross_rig_data_source(..., mode="forward_dynamics", chunk_length=32)`;
   - `keys_to_skip_loading` 保留动作侧模块 fresh-init(action2llm/action_modality_embed/
     action_pos_embed;**llm2action 输出头 FD 不更新,可一并 reset 不影响**);
   - lr 对齐官方动作配方 **2e-4** + 动作参数 5× multiplier(policy 版用 2e-5 是因 warm-start
     Policy-DROID;FD 从 MT 基座 + fresh 动作头,按官方配方走),warmup 30,cosine 到 0;
   - `max_tokens=45056`、`max_batch_size=12`(32 帧 token 翻倍,按 §1.3 实测步时再调);
   - VAE latent 缓存:对 32 帧窗口**重新预计算**(cache_key 含 `L32`,自动区分,不与 16 帧冲突)。
3. Smoke:单机 8×A100 300 步(`smoke_validate.sh` 改 experiment 名),断言 video-loss 下降。
4. 正式:多机(沿用 4n8g wrapper)**10k 步**,save_iter 1000,每 1000 步跑 §4 快评(val 200 eps
   抽 32 条),早停看 val PSNR 平台 + 可控性指标(防过拟合:视频扩散比图像更易复读训练片段,
   **不看 train loss 看 held-out 首帧生成**)。
5. 预算:32 帧步时按 16 帧 ×(1.5~2) 估 12–17 s/步,10k 步 ≈ 33–47 h(4n8g)。

### Phase 2 — 可控性强化与推理调参
- guidance 扫描 {1, 2, 3, 5} × 去噪步数 {10, 20, 50} → 可控性/真实感/速度三维 trade-off 表;
- 若 action-following 不达标(§4 门禁):
  a) `cfg_dropout_rate` 0.1→0.15 + 重训尾段 2k 步;
  b) FreeAction 负动作引导(纯推理改动,先试);
  c) 检查 idle 片段占比——idle 多会教模型"忽略动作"(Cosmos3 用 idle-count 元数据规避,确认
     `append_idle_frames=True` 生效)。

### Phase 3 — 长时序 AR rollout
- 用 vLLM-Omni AR 分块协议:每段 32 帧,末 K 帧作下段上下文,滚 10–30 s;
- 测每步 PSNR/SSIM/LPIPS 退化曲线(对照 Ctrl-World:10s PSNR 23.56);**分相机统计**——
  预期腕部视角先崩,若严重则评估"仅 cam_high 单视角变体"或上下文帧数加倍;
- 可选实验:64 帧 @ 15 FPS 变体(动作下采样 2×,时间视野 4.3 s)对比 32@30,看衣物慢动态
  (铺平后回弹、对折下落)谁更真;
- 远期(挂账不在本期):self-generated history 训练 / RL 后训练抗漂移(Persistent Robot WM,
  LPIPS 0.091→0.070 的先例)。

### Phase 4 — 两个下游闭环(价值变现)
- **策略评测器(优先)**:取 wam 实验已有真机/离线评测过的策略族(gwp_ori、gwp_ans、
  pi0.5、delta 家族…),各自在 WM 里 rollout(策略出动作 → WM 出视频 → 策略闭环),
  成功判定用 **kai0 stage classifier(`stage_advantage`)+ VLM judge** 双裁判;
  与真实评测算 **Pearson r / MMRV**。门禁:r ≥ 0.8(WorldEval 同硬件类做到 0.942)。
  打通后:新 checkpoint 先过 WM 评测再上真机,省真机时间。
- **数据引擎**:对失败/稀缺状态(衣服皱团、半折)首帧生成多样 rollout → 现有 ID/policy 模型
  反推伪动作 → 注入 AWBC 负/正样本池(DreamGen 协议:合成轨迹量与增益 log-linear)。

---

## 4. 评测方案(每 1000 步快评 + 里程碑全评)

| 维度 | 指标 | 协议 | 门禁(v1) |
|---|---|---|---|
| 短时真实感 | PSNR / SSIM / LPIPS @32帧 | visrobot01_val 200 eps,GT 动作条件生成 vs 真视频 | PSNR > 24 dB(DROID 参照 25.5)且 > I2V 零样本基线 +2 dB |
| 可控性-扰动 | ΔPSNR / Δ-LPIPS | 同首帧,GT 动作 vs 随机换 episode 动作 / 取负动作 / 零动作 三组 | GT 显著优于扰动组(ΔPSNR > 2 dB),且零动作组生成静止 |
| 可控性-探针 | IDM Action Error Ratio | 单独后训练一个 ID 模式 Nano(或复用 GWP world_action_model IDM)在生成视频上反推动作 | 误差比 < 1.5×(AVID 全量微调参照 1.297) |
| 可控性-轨迹 | EE 像素轨迹 DTW/Hausdorff | FK + `calibration.yml` 投影 GT EE 轨迹,生成视频上光流/点跟踪重检测 | 报告值,趋势向好 |
| 衣物语义 | stage 进度一致性 | kai0 stage classifier 在生成 vs 真视频上的 stage 序列编辑距离 | 报告值;铺平→对折阶段顺序不乱 |
| 长时序 | 逐步退化曲线、round-trip LPIPS | Phase 3 协议,分相机 | 10 s PSNR > 22(Ctrl-World 23.56 参照) |
| 终极 | 策略评测相关性 Pearson r | Phase 4 协议 | r ≥ 0.8 |
| (候选)3D 形变保真 | RGBench real-to-sim CD/HD | 生成视频 3D 提升(深度/3DGS)后对真实点云;**RGBench 的真机 GT 就采自 Agilex Piper(同款臂)** | 报告值;3D 提升噪声待标定 |

> 形变保真与 3D 评测的深度调研(22 条对抗验证论断:PBD 不可用作保真后端、FEM SOTA、PGND/Cloth-Splatting 3D 估计器、real2sim 仅准静态有效等)见
> [`cosmos/wam_fold_wm/docs/deformable_optimization_strategies.md`](../../../../cosmos/wam_fold_wm/docs/deformable_optimization_strategies.md);
> 开源柔性数据集盘点(协同训练/评测/AF 混合三类)见 [`cosmos/wam_fold_wm/docs/open_deformable_datasets.md`](../../../../cosmos/wam_fold_wm/docs/open_deformable_datasets.md)。

工程上:复用 `wam_fold_policy/eval/` 分片框架,新增 `wam_fold_wm/eval/` 目录放 FD 评测脚本;
所有指标进一个 `report.html`(沿用 eval_i2v 的报告生成器)。

---

## 5. 风险与对策

| 风险 | 依据 | 对策 |
|---|---|---|
| 动作被忽略(prior 压过 conditioning) | WorldEval 明确报告该失败模式 | CFG dropout 已埋好;guidance 扫描;FreeAction;idle 元数据;Δaction 扰动评测尽早做(smoke 后立即测) |
| 形变接触段失真(抓-提-放) | ACWM-Phys:deformable contact 掉点最大 | 82.3 h 单任务 in-distribution;评测里专门切接触段统计;不承诺 OOD |
| 腕部视角长时序崩坏 | Ctrl-World 实测 | 分相机评测;必要时单视角变体或画布权重 |
| 过拟合/复读训练片段 | 视频扩散已知问题 | 只看 held-out;val 首帧 + 新颖衣物摆放抽查;10k 步内多 ckpt 留档 |
| A100 上 32 帧 OOM | H200 139G 上 64 样本/卡 480p 即 OOM | token-pack 预算制(45k)天然防;先 smoke 实测,`max_batch_size` 12→8 退让 |
| 基座 Nano 未下载/网络 | gf 机器走 29290 反代 | 提前用 ModelScope 镜像 `nv-community/Cosmos3-Nano` 兜底 |
| PFS L2 cache 节点 OOM 复发 | 历史事故(已修) | 沿用 LRU VideoDecoderCache + pod mem 957G 配置,`cpu_mem_watch` 监控 |

---

## 6. 资源与排期(粗估)

| 阶段 | 算力 | 时长 |
|---|---|---|
| Phase 0 补齐 + 基座转换 + latent 预计算(32 帧) | 单机 8×A100 | 1–2 天 |
| Phase 1 smoke + 10k 步全量微调 | 4n8g(AIHC) | 2–3 天 |
| Phase 2 推理调参 + 可控性迭代 | 单机 | 2–3 天 |
| Phase 3 长时序 | 单机/2 机 | 2–3 天 |
| Phase 4 策略评测相关性研究 | 16 卡分片评测 | 3–5 天 |

合计 ~2–3 周到"可用的策略评测器";数据引擎闭环另计。

## 7. 执行详解与状态(2026-06-12 启动,本机先行)

**执行原则**:所有 Phase 的准备与验证先在本机 8×A100 完成,验证通过后才扩展到 AIHC 多机大规模训练。
排序逻辑是"先证伪、再投入"——最大风险(动作被忽略)在 smoke 后立即用 Δaction 扰动测试暴露。

```
Phase 0 (1-2天)            Phase 1 (2-3天)         Phase 2 (2-3天)      Phase 3 (2-3天)    Phase 4 (3-5天)
baseline.json ─────────→ smoke 300步 ──Δaction──→ guidance扫描 ──→ AR rollout ──→ 策略相关性 r≥0.8
基座→DCP ──────────────→   ↑提前证伪点              ↑FreeAction退路    ↑腕部视角风险      ↑= 可用评测器
latent缓存(L32) ───────→ 10k步多机训练
```

### 代码与产物位置

- 工作目录:`cosmos/wam_fold_wm/`(train/recipe_wm_nano.toml、train/smoke_validate.sh、eval/baseline_from_i2v.py)
- 实验配置:`packages/cosmos3/cosmos_framework/configs/base/experiment/action/posttrain_config/wam_fold_wm_nano.py`
  (FD 模式、chunk 32、lr 2e-4 + 动作模块 5×、复用 policy 的 ActionDataPacker 与跨 rig 混合)
- 运行产物:`cosmos/wam_fold_wm_runs/`(checkpoints/Cosmos3-Nano-dcp、reports/、smoke_out/)

### 执行状态

| 项 | 状态 | 结果/备注 |
|---|---|---|
| 基座 Cosmos3-Nano 在本地 | ✅ | `models/modelscope/Cosmos3-Nano`(33G,ModelScope 下载,完整 HF 布局) |
| Phase 0 真实感基线 baseline.json | ✅ 2026-06-12 | 零样本 I2V(cam 均值):**Nano 1s/3s/7s = 12.88/12.61/12.22 dB PSNR**;Super 11.79/11.01/10.77;Super-I2V 11.36/10.36/10.34。Nano 反而最好;地板极低 → 后训练空间大;§4 的"基线+2dB"门禁实际以绝对值 24 dB 为准 |
| FD 实验配置注册 + 加载验证 | ✅ | `experiment=wam_fold_wm_nano` 注册成功,lr/multipliers/mode/chunk 字段核验通过 |
| FD 数据通路单样本验证 | ✅ | video [3,33,720,640](33 帧观测窗,720×640 画布)、action [32,14]、domain 16、cache_key 带 L32;共 13,820,352 窗口(vis×3+kai) |
| 本机 venv 解释器修复 | ✅ | 训练 venv 指向 uv 托管 CPython 3.13(仅 AIHC 容器有);本机以 `ln -s /mnt/pfs/p46h4f/cosmos/uvpy/cpython-3.13.0-* → /root/.local/share/uv/python/` 修复,torch 2.10+cu128 8 卡可用 |
| 基座转 DCP + 30 步 smoke | ✅ 2026-06-12 18:58 | **PASS**:DCP 转换成功(`Cosmos3-Nano-dcp`);30 步 loss 0.1532→min 0.1028;token 统计确认 FD 模式(vision gen 19,440 tok / action 192 tok 干净条件 / 无 causal) |
| 300 步本机验证训练 | 🔄 运行中 | SMOKE_ITERS=300,实测稳定步时/显存 → 校准 10k 步多机预算;产出 ckpt@100/200/300 供 Δaction 测试 |
| FD 推理 harness 搭建 | ✅ 2026-06-12 | `wam_fold_wm/eval/{export_iter300.sh, fd_infer.py, run_fd_infer.sh}`。export iter300 DCP→HF(29G/7分片)✅;复用 `eval_report.CosmosFoldPolicy` 加载;**delta+quantile 归一化 self-check 通过**(dataset delta action `[0.106,0.016,-0.069]`→norm`[0.673,0.05,-0.212]`,全在[-1,1],与训练一致) |
| Δaction 扰动测试(提前证伪) | ✅ 管线验证通过 2026-06-12 | iter_300 上跑通 FORWARD_DYNAMICS 三组对照:**GT psnr=11.82 / other=11.79 / zero=12.13;ΔPSNR(gt−other)=0.03、(gt−zero)=−0.31**。即 300 步 smoke 时**动作尚未被遵循**(ΔPSNR≈0,符合预期判据)——**管线/证伪框架已就位,真实可控性数值等 10k 步正式训练后换 `--export-dir` 复跑**。注:此时 PSNR 11.82 < I2V 零样本基线 12.88,说明 fresh-init 动作头+300 步远未收敛,正常 |
| L32 latent 预计算 | ⏳ | 复用 `wam_fold_policy/train/precompute_latents.py`,改 chunk 32;验证训练用在线编码即可,正式 10k 步前完成 |
| 10k 步正式训练 | ⏳ 不在本机 | 待上述全部通过后上 AIHC 4n8g(recipe 的 replicate_degree 改 4) |
| 外部布料数据第一批下载 | 🔄 2026-06-12 启动,三路并行 | → `kai0/data/external_cloth/`:① RoboCOIN fold_clothes 29.7G(**LeRobot v2.1 同版本**,AgileX 双臂 584 ep,三相机布局同我们);② AgiBot task_570 叠T恤 188.5G(amap LeRobot v2 转换,per-task tar);③ RoboCOIN 毛巾系列 5 仓 ~36G。脚本 `wam_fold_wm/train/ms_download.sh`,日志 `wam_fold_wm_runs/reports/downloads/`。ModelScope 可用性排查见 `wam_fold_wm/docs/open_deformable_datasets.md` |

### 各 Phase 详解要点(对 §3 的补充)

- **Phase 0**:基线已锚定(上表)。要点:零样本可控性基线(基座 FD 给动作)留待 DCP 就绪后补——预期动作被忽略,作为可控性零点。
- **Phase 1**:smoke(30 步)只验证管道与 loss 趋势;**smoke 通过后立刻做 Δaction 扰动**,几小时内证伪"动作通路是否有效",再投入 10k 步。正式训练早停看 held-out 首帧生成指标,不看 train loss(视频扩散易复读训练片段)。
- **Phase 2**:不动权重,先拧推理旋钮:guidance {1,2,3,5} × 去噪步数 {10,20,50} 三维 trade-off 表。不达标的退路按代价排序:FreeAction(纯推理,当天可验)→ 查 idle 占比 → cfg_dropout 0.1→0.15 重训尾段 2k 步。
- **Phase 3**:AR 分块滚 10–30 s,逐步退化曲线**分相机**统计(预期双腕先崩);对照实验 64帧@15fps vs 32帧@30fps 看衣物慢动态。门禁 10s PSNR>22。
- **Phase 4**:用已有真机/离线成绩的策略族(gwp_ori/gwp_ans/pi0.5/delta 家族)在 WM 内闭环 rollout,stage classifier + VLM 双裁判判成功,与真实成绩算 Pearson r/MMRV。**r≥0.8 即"可用评测器"验收**:新策略 checkpoint 先过 WM 评测(16 卡分片,几十分钟)再决定是否上真机。数据引擎(稀缺状态合成轨迹→伪动作→AWBC 样本池)在评测器建立信任后另行排期。

## 8. 引用(调研来源)

Cosmos3 技术报告 arXiv:2606.02800;cosmos-framework(training.md / action_policy_droid_posttrain.md /
custom_dataset.md);HF nvidia/Cosmos3-Nano(-Policy-DROID);IRASim 2406.14540;Ctrl-World 2510.10125;
GE/GE-Sim 2508.05635;DreamGen 2505.12705;EnerVerse-AC 2505.09723;Vid2World 2505.14357;
FreeAction 2509.24241;GAIA-2 2503.20523;WorldEval 2505.19017;EWMBench 2505.09694;
dWorldEval 2604.22152;ACWM-Phys 2605.08567;ACT-Bench 2412.05337;VideoPhy-2 2503.06800;
World-in-World 2510.18135(画质≠具身效用,可控性更重要);Persistent Robot WM 2603.25685;
DreamZero 2602.15922;V-JEPA 2 2506.09985;RoboLab(NVlabs);RoboArena 2506.18123。
