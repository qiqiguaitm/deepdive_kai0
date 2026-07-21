# DESIGN · pi05 P0 建栈实现细节（2026-07-21）

> 配套 `PLAN_pi05_lmwm_sameencoder_2026-07-21.md`。本文 = P0 落地的**精确实现细节**(勘查已确认的 key/路径/骨架),
> 让 P0a(LIBERO)/P2a(RoboTwin)/hint 建栈 turnkey,不重复调研。

---

## 0. P0b 注入路径 —— ✅ 已落地(本轮)

改了 4 文件(纯加法, config 门控, `lmwm_hint_dim=0` 逐位等价上游):
- `kai0/src/openpi/models/pi0_config.py`: +`lmwm_hint_dim:int=0` / `lmwm_hint_len:int=1` / `lmwm_hint_target:str="prefix"`
- `kai0/src/openpi/models/model.py`: +`Observation.lmwm_hint: Float["*b hl hd"] | None`(from_dict + preprocess_observation 透传)。
  **注意**:轴名必须 `hl hd`,不能用 `h d`——`h` 与图像 `images:"*b h w c"` 的高度轴撞名会 jaxtyping 报错(本轮踩过)。
- `kai0/src/openpi/models/pi0.py`:
  - `__init__`: `self.lmwm_hint_proj = nnx.Linear(hint_dim, W)`;`W` = prefix→`paligemma_config.width` / suffix→`action_expert_config.width`。
  - `embed_prefix`(image token 后、language 前):投影→append token,`ar_mask += [False]*hint_len`(双向可见,仿 soft_prompt)。
  - `embed_suffix`(action_head_cond 后):投影→prepend token,`ar_mask += [True]*hint_len`(仅 action expert 见,仿 domain token)。
- 冒烟 `scratchpad/smoke_hint.py`:dummy gemma 变体,测 dim0 / prefix-1152 / suffix-768 / best-of-K=4 前向 + dim0-guard 等价。
  **CPU 上因 gf0 满载(load~100)极慢,须用 GPU**(`CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.15`)。

### 0.1 ⭐关键 gotcha:随机初始化下注入"看似死了"是预期,非 bug
冒烟发现三路 loss 逐位相同(prefix/suffix/dim0 全 2.5484…)。诊断 `scratchpad/diag4.py` 证:harness 灵敏
(改 actions Δ=0.096 / 改 noise Δ=0.396 都 CHANGES),但**改 image 和改 hint 都精确 Δ=0.000e+00**。
**根因**:`gemma.py:121/128` adaRMS 的 modulation Dense 是 `kernel_init=nn.initializers.zeros`(零初始化)
→ `gate=0` → `_gated_residual(x,y,gate)=x+0·y=x` → 随机初始化时**每个 block 恒等映射,prefix(含 image)零影响**。
这是 DiT/flow-transformer 标准"零门控从恒等起步"。**⇒ image 和 hint 同行为 = 我代码正确;零影响只是随机初始化产物。**
**warm-start pi05_base 时门控是学到的非零值 → hint 从 fine-tune 第 0 步就生效。** 真 liveness = 训练 SR,不是初始化 loss。
(若要正向铁证:载 pi05_base 真权重跑 diag,image+hint 都会动 loss——分析已充分,未做。)

---

## 1. LIBERO(P0a)—— 数据现成,无需下载/转换

### 1.1 数据(勘查确认)
- **本地 v2.1 lerobot 四套件全齐**,在 submodule 树外 `/vePFS/tim/workspace/LIBERO_fastwam/`:
  | 套件目录 | ep | frames |
  |---|---|---|
  | `libero_10_no_noops_lerobot` | 388 | 104280 |
  | `libero_goal_no_noops_lerobot` | 433 | 52895 |
  | `libero_object_no_noops_lerobot` | 457 | 67309 |
  | `libero_spatial_no_noops_lerobot` | 434 | 53229 |
  - 均 LeRobot **v2.1** / franka / **fps 20** / `data/chunk-000/episode_*.parquet`。openpi 钉的 lerobot 直读。
- ⛔ **不要用** lawam 侧 `lawam_local/dataset/libero_merged_no_noops_20hz`(那是 v3.0/GR00T,openpi 读不了)。

### 1.2 实际数据 key(parquet 列 + video 目录, 勘查确认)
```
observation.images.image          (video, 512x512x3)   ← prefix 相机
observation.images.wrist_image    (video, 512x512x3)   ← 腕部
observation.state                 (list<float>, 8)
action                            (list<float>, 7)
task_index                        (int64)  → prompt 走 prompt_from_task→tasks.jsonl
frame_index / episode_index / index / timestamp
```
LeRobot 交付**扁平点号 key**(非嵌套);openpi `flatten_dict(sep="/")` 不会把点号变斜杠 → **repack 必须写点号 key**。
默认 `LeRobotLiberoDataConfig`(config.py:455)的 repack 用 `observation/image`(斜杠)**不匹配本数据**,需自定义。

### 1.3 需写的 config(骨架)
```python
# 新 DataConfigFactory(或给 LeRobotLiberoDataConfig 加一个点号-key 变体):
repack = RepackTransform({
    "image":       "observation.images.image",
    "wrist_image": "observation.images.wrist_image",
    "state":       "observation.state",
    "actions":     "action",
    "prompt":      "prompt",           # prompt_from_task=True 时自动注入
})
# data_transforms = LiberoInputs(model_type) / LiberoOutputs()  ← 现成, state8/action7 已对齐
# base_config = DataConfig(prompt_from_task=True)
# 四套件多路: datasets_yaml 列 4 个本地路径 → ConcatDataset(见 data_loader _load_repos)

TrainConfig(
  name="pi05_libero_a0",
  model=Pi0Config(pi05=True),          # A0=纯基线, 无 hint
  data=<上面的 LiberoLocalDataConfig>(datasets_yaml="<4 suites>.yaml", ...),
  weight_loader=CheckpointWeightLoader("<repo>/kai0/checkpoints/pi05_base/params"),
  lr_schedule=CosineDecaySchedule(warmup_steps=1000, peak_lr=1.5e-5, decay_steps=30000, decay_lr=1.5e-6),
  ema_decay=0.9999, num_train_steps=30000, batch_size=128, fsdp_devices=8, num_workers=8,
)
# A1/A2: 同上 + model=Pi0Config(pi05=True, lmwm_hint_dim=768/1152, lmwm_hint_target="prefix"|"suffix")
#        + data 侧喂 lmwm_hint 字段(LiberoInputs 需加 inputs["lmwm_hint"]=data["lmwm_hint"])
```

### 1.4 P0a 剩余待验证(需 GPU,盒子空了再跑)
1. **实测 batch key**:构 `LiberoLocalDataConfig(repo_id=本地路径).create(...)`,拉一个 batch,print keys,确认 repack 命中。
2. **norm_stats**:`scripts/compute_norm_states_fast.py --config-name pi05_libero_a0`(四套件合并统计)。
3. **eval 客户端**:kai0 无 `examples/libero`。lawam `examples/LIBERO/eval_files/` 的 libero rollout client 可复用,
   但需 **openpi-websocket ↔ libero 胶水层**(pi05 用 `serve_policy.py` 起 websocket server;client 侧复用 libero env)。
   环境变量抄 lawam `run_libero_suite_benchmark.sh`:`STAR_VLA_PYTHON / LIBERO_PYTHON / LIBERO_HOME / PYTHONPATH(注入libero) / MUJOCO_GL=egl`。

---

## 2. RoboTwin(P2a)—— 意外地轻(~1-1.5 天)

### 2.1 数据(勘查确认)
- 全量 v2.1:`lmvla/lawam/dataset/robotwin2.0/`(27500ep / 6.08M frames / **50fps** / aloha)。北京同源 `/vePFS-North-E/vis_robot/...`。
- 训练子集 v3.0:`lmvla/lawam_local/dataset/robotwin2_lmwm_v30/`(1315ep / 504k frames / 117 tasks)。
- **state=14 / action=14** aloha layout:`left_joints[0:6]+left_gripper[6]+right_joints[7:13]+right_gripper[13]`。
- 3 相机:`cam_high / cam_left_wrist / cam_right_wrist`(480×640)。

### 2.2 复用 vs 新建
- **transform 复用**:pi05 `aloha_policy.py` 的 `AlohaInputs/AlohaOutputs`(14维+3相机+`make_bool_mask(6,-1,6,-1)`)原生认这 3 相机名。
- **config**:改 `LeRobotAlohaDataConfig`(config.py:402)的 repack 指 robotwin v3.0 键;`Pi0Config(pi05=True, action_dim=32, action_horizon=50)`。
  - ⚠️ `adapt_to_pi=False`(sim 无真机 Interbotix 夹爪 linear↔radian 换算)。
- **horizon 天然对齐**:50fps × sec_chunk 1.0 = **action_horizon 50**(pi05 已是 50)。
- **eval 协议复用**:pi05 `websocket_policy_server` 与 lawam client **同源 openpi msgpack**,wire 直通。
  整套编排 `batched_eval_runner.py`+`robotwin_batch_bridge.py`+`auto_eval_robotwin.sh` 照用,只换 server 命令为 pi05 `serve_policy.py`。
  - 需薄 obs 适配:bridge 打 `{primary_image, wrist_image, lang, state, action_hz}` → pi05 aloha obs 键(~50 行 `RobotwinInputs`)。
  - server metadata 不塞 `ckpt_path` 即可绕过 client 的 `_validate_server_metadata`。
- **norm_stats**:pi05 侧无 robotwin norm,须对 `robotwin2_lmwm_v30` 重算。

### 2.3 ⚠️ 坑(运行前必补)
- `_cluster_env.sh` 引用的 `$REPO/lmvla/lawam/robotwin_python_wrapper{,_northe}.sh` **不存在**,实存于 `lmvla/lmwam/scripts/`。
  eval yaml 有 `[ -x ]` 自检,不补则 fail-fast exit 13。**建栈第一步补这 2 个 wrapper**(软链或拷贝)。

---

## 3. hint 离线抽取（两侧唯一接口）
- `lmwm/scripts/export_pi05_hint.py`(新):载 LMWM ckpt → 对数据集逐帧算 ĝ_next → `lmwm/data/pi05_hint/{libero,robotwin}_{dino,so400m}/hint.npz`(+`_env.json`,带 episode/frame 索引 + K 候选轴)。
- A1=`lmwm_libero_rvalley`(DINOv3 768D);A2-lite=`lmwm_libero_so400m`(§4.20 生成器 1152D)。
- 消费端:pi05 `LiberoInputs/AlohaInputs` 加 `inputs["lmwm_hint"]=data["lmwm_hint"]`;data 侧按 episode/frame 索引对齐挂列。

---

## 4. 修订 P0 执行序（both 无数据阻断, 并行）
1. **P0b** ✅ 完成。
2. **LIBERO**:datasets_yaml(4 套件)→ LiberoLocalDataConfig(点号 repack)→ pi05_libero_a0 → batch-key 实测 → norm_stats → A0 smoke → eval 胶水。
3. **RoboTwin**:补 2 wrapper → LeRobotAlohaDataConfig(robotwin repack)→ norm_stats → A0 → eval bridge 换 pi05 server。
4. **hint**:export_pi05_hint.py 与 2/3 并行。
5. **A1/A2 训练**:hint 产好 + LiberoInputs/AlohaInputs 消费 lmwm_hint → 全矩阵并行铺北京/上海。

> 判据纪律见 PLAN §1 脚注(变 seed / 聚合差<1.5pt 不声称 / t6 判据 t8 不可用 / ckpt 带 config.yaml+dataset_statistics.json)。
