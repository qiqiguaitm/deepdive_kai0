# PLAN · pi05 × LMWM 同编码器空间三臂实验（2026-07-21）

> **核心问题（P3）**：把 LMWM 的子目标 hint 放进 **VLA(pi05)自身视觉编码器空间(So400m)** 训练/注入,
> 相比放在**外挂 DINOv3 空间**,是否带来**进一步的下游 SR 提升**?
> 判据 = 下游 SR(LIBERO + RoboTwin),不是内在 recon_cos(§4.7 教训:内在 gain 2.1× 未换 SR)。
>
> 详源:roadmap §4.19(r 场 So400m 空间成立,边界一致性 +0.25 最优)、§4.20(生成器 So400m 空间 recon_cos 0.95)、
> `ARCH_predictor_vs_single`(两模型价值=多模态 best-of-K)。本文 = 执行规划 + 代码环境设计。

---

## 1. 三臂实验设计（每臂 × {LIBERO, RoboTwin}，共 6 训练）

| 臂 | hint 来源空间 | 隔离了什么 | 用现成资产 |
|---|---|---|---|
| **A0 · pi05 基线** | 无 hint | 下限 | pi05_base |
| **A1 · LMWM@外挂 DINOv3** | 冻结 DINOv3-base pooled(768D)的 LMWM(`lmwm_libero_rvalley`) | 「注入 LMWM 有没有用」 | ✅ ckpt 已在 |
| **A2 · LMWM@pi05 So400m** | So400m patch-mean(1152D)空间的 LMWM(`lmwm_libero_so400m`) | 「换到 VLA 自身空间有没有**进一步**增益」 | ✅ §4.20 已训生成器 |

- **A1 vs A0** = LMWM 注入的净收益;**A2 vs A1** = 同编码器空间的**增量**(本 plan 的核心问句)。
- **So400m "同空间"两档**(A2 的精度阶梯):
  - **A2-lite**（先做）：hint 来自 lmwm HF-So400m 特征(§4.20 管线,已建)。是"So400m 语义族空间",非 pi05 精确权重。
  - **A2-true**（定论追加）：hint 从 **pi05 自身 PaliGemma-SigLIP** 抽 patch token。彻底消除对齐,才是"真·同编码器"。
  - 由于注入端有**可学习 Linear 适配层**能吸收分布差,A2-lite vs A1 已能回答"So400m 语义 vs DINOv3 几何"的核心问题;A2-true 是把权重/预处理差也扣掉的定论版。

> **判据纪律(硬约束,违反结论无效,承 §4.18)**:① 重复评测必须变 seed;② 聚合差 <1.5pt 不可声称;
> ③ per-task 只用低方差任务(t6 std 1.7~3.0),**t8 不可作判据**;④ ckpt 跨集群带 `config.yaml`+`dataset_statistics.json`。

---

## 2. 代码环境设计：高内聚 · 低耦合 · 可并行

### 2.1 一句话架构:**hint 离线预计算 → 作为数据字段喂 pi05,LMWM 与 pi05 训练完全解耦**

```
[LMWM 侧, PyTorch, 离线]                    [pi05 侧, JAX, 训练]
 g_t(帧特征) → LMWM 预测器+生成器 → ĝ_next    obs.lmwm_hint(新字段, 逐帧向量)
   → 逐帧 hint 向量, 存成数据集旁挂列/npz    → Linear(D→width) → prepend token
   (A1=DINOv3空间 / A2=So400m空间, 只是不同文件)   → 与 dataset_id 走同一条路
```

**为什么解耦**:pi05 训练**不 import LMWM**,只读一个 `lmwm_hint` 数据字段。A1/A2 的差别**仅是喂哪个 hint 文件**——
pi05 训练代码**一份不变**跑三臂。这满足:
- **高内聚**:pi05 侧改动集中在"一条加法注入路径"(下 2.2);LMWM 侧改动集中在"离线出 hint"(下 2.3)。
- **低耦合**:两侧唯一接口 = hint 数据文件格式(`[N_frame, D] fp16 + episode/frame 索引`)。换空间=换文件,零代码改动。
- **可并行**:hint 一旦离线产好,6 个训练互相独立,可同时铺满两队列。

### 2.2 pi05 侧改动（一条 config 门控的加法路径,默认关=上游行为）

| 层 | 改动 | 位置(勘查确认) |
|---|---|---|
| Observation | 加字段 `lmwm_hint: Float[*b, D] \| None` | `model.py:124` 附近 + `from_dict:161` + `preprocess_observation:272` 透传(与 `dataset_id`/`progress` 同构) |
| 数据 Inputs | `inputs["lmwm_hint"] = data["lmwm_hint"]` | 各 policy 的 `*Inputs.__call__`(LIBERO: `libero_policy.py:56`) |
| 模型消费 | `hint_proj = nnx.Linear(D, width)` → prepend token | **默认 prefix**:`pi0.py:205`(`tokens.append(image_tokens)` 后)追加 hint token,`ar_mask += [False]*n`。语义与 So400m patch token 同域,最自然 |
| config 开关 | `lmwm_hint_dim / lmwm_hint_len / lmwm_hint_target(prefix\|suffix)`,默认 0/关 | 仿 `pi0_config.py:59-69` soft_prompt/action_head_cond |
| curriculum | 可复用 `freeze_mode` 先只训 hint_proj | `pi0_config.py:140-158` |

- **注入点二选一(需拍板,见 §6)**:prefix(A,全 token 可见,vision 域自然)/ suffix(B,仅 action expert 可见,不扰 VLM 语言对齐,更保守)。默认 prefix,B 留作对照 flag。
- 所有改动 config-gated,`lmwm_hint_dim=0` 时**与上游 pi05 逐位一致**,老 config/ckpt 不受影响 → 低耦合。

### 2.3 LMWM 侧改动（离线出 hint,复用现成脚本）

- A1 hint:`lmwm_libero_rvalley`(DINOv3)预测器 → 逐帧 ĝ_next(768D)。
- A2 hint:`lmwm_libero_so400m`(§4.20 生成器)→ 逐帧 ĝ_next(1152D)。
- 新增一个薄脚本 `lmwm/scripts/export_pi05_hint.py`:载 LMWM ckpt → 对数据集逐帧算 hint → 存 `lmwm/data/pi05_hint/{libero,robotwin}_{dino,so400m}/hint.npz`(+`_env.json`)。**这是两侧唯一接口产物。**
- **多模态(可选增强)**:§ ARCH_predictor_vs_single 结论——两模型价值在 best-of-K。若 pi05 端能消费多候选(如 prefix 放 K 个 hint token),可出 `[N, K, D]`;先做单发 mode(K=1),best-of-K 作 A2 的增强对照。

### 2.4 代码物理位置（高内聚)

```
kai0/src/openpi/models/pi0.py            + hint 投影/prepend(门控)
kai0/src/openpi/models/model.py          + Observation.lmwm_hint 字段透传
kai0/src/openpi/models/pi0_config.py     + lmwm_hint_* 开关
kai0/src/openpi/policies/libero_policy.py + inputs["lmwm_hint"]
kai0/src/openpi/policies/<robotwin>_policy.py(新)
kai0/src/openpi/training/config.py       + pi05_libero_{a0,a1,a2} / pi05_robotwin_{a0,a1,a2} TrainConfig
lmvla/lmwm/scripts/export_pi05_hint.py   (新, 两侧唯一接口)
train_scripts/kai/volc/pi05_{libero,robotwin}_{a0,a1,a2}_*.yaml(新, mkyaml.py 生成)
```

---

## 3. 前置缺口（P0,必须先补;勘查已确认这些在 pi05 侧不存在）

| 缺口 | 现状 | 补法 |
|---|---|---|
| pi05 **LIBERO** TrainConfig | 只有未接入的 `LeRobotLiberoDataConfig`(config.py:455) | 注册 `pi05_libero_a0`,repo_id 指 LeRobot 版 LIBERO |
| pi05 **LIBERO 数据**(LeRobot v2.1) | lawam 侧有 `libero_merged_no_noops`,openpi 格式未知 | 核对/转换成 openpi LeRobot;或用上游 openpi LIBERO repo |
| pi05 **LIBERO eval 客户端** | kai0 无 `examples/libero` | 移植上游 openpi libero 客户端 或 参考 `lmwm/scripts/eval_lawm_libero.py`+`serve_policy.py` |
| pi05 **RoboTwin** 全链路 | pi05 侧 0 支持(仅 lawam/fastwam 有 RoboTwin) | **最大新工作**:policy wrapper + eval 桥接 + 数据(robotwin2.0 LeRobot);50fps horizon 坑(sec_chunk×fps==action_horizon) |
| pi05 base ckpt | `pi05_base/params` 存在(config.py:1318) | 直接用 |

> ⚠️ **RoboTwin-for-pi05 是最大成本**,但也是**最有价值的场**(LIBERO 已饱和无分辨力,roadmap §4 反复强调转 RoboTwin)。
> 建议 **LIBERO 先行**(脚手架轻、验证注入路径通),**RoboTwin 紧随**(拿真分辨力)。

---

## 4. 分阶段计划（P0 建栈 → P1 LIBERO 三臂 → P2 RoboTwin 三臂）

| 阶段 | 内容 | 产出/判据 | 可并行度 |
|---|---|---|---|
| **P0a** | pi05 LIBERO 脚手架(config+data+eval 客户端),跑通 A0 基线 smoke | pi05 在 LIBERO 出非乱码 SR | 串行(基建) |
| **P0b** | pi05 hint 注入路径(§2.2)+ `export_pi05_hint.py`;`lmwm_hint_dim=0` 回归 = A0 | 门控关时逐位等价 A0;开时能前向 | 与 P0a 部分并行 |
| **P1** | **LIBERO 三臂 A0/A1/A2-lite**,各 ×4 seed | A1−A0、A2−A1 的 SR Δ(变 seed) | ✅ 3臂×seed 全并行 |
| **P1+** | A2-true(pi05 自身 SigLIP 抽 hint)+ best-of-K 增强 | 定论版 So400m | 视 P1 结果 |
| **P2a** | pi05 RoboTwin 脚手架(policy+eval+data+50fps horizon) | pi05 在 RoboTwin 出 SR | 串行(基建) |
| **P2** | **RoboTwin 三臂 A0/A1/A2**,各 ×seed | 未饱和场的 A1−A0、A2−A1 | ✅ 全并行 |

**关键路径**:P0a→P1(LIBERO 先出信号)‖ P2a 同时起(RoboTwin 建栈不阻塞 LIBERO)。hint 离线产好后 P1/P2 训练全并行。

---

## 5. 队列策略（北京为主 · 上海为辅）

- **北京 Robot-North-H20(主)**:承载 P1/P2 的训练主力。8 卡整节点 `ml.hpcpni3ln.45xlarge`;**无 4 卡规格**(用 1 卡 `ml.pni3ln.5xlarge` 或整节点)。
- **上海 robot-task A100(辅)**:承载 hint 离线抽取(GPU 抽特征)、A2-true 的 pi05-SigLIP 抽取、以及溢出的并行训练臂。
- **提交**:一份可移植 body → `mkyaml.py --cluster both` 生成两集群 yaml;entrypoint `source _cluster_env.sh`,只用 `$REPO/$PYTHON`,不写集群字面量。
- **6 训练臂分配建议**:LIBERO 三臂(轻)优先北京并排;RoboTwin 三臂(重)北京为主、上海分流;hint 抽取放上海 A100。
- 环境:pi05 训练用 `kai0/.venv`(JAX);hint 抽取用 `kai0/.venv`(So400m)/`srpo`(DINOv3 r 场);LMWM 训练已有环境。见 `ENV_SELECTION_RULES.md`。

---

## 6. 决策（2026-07-21 已定）

1. **注入点 = prefix + suffix 双 flag 对照**。`lmwm_hint_target ∈ {prefix, suffix}` 都实现,做 A/B。
2. **两环境齐头并进**：P0 同时建 LIBERO + RoboTwin 脚手架(RoboTwin 建栈重,并行推进不阻塞)。
3. **A2 = lite 先行**(HF So400m,复用 §4.20);A2-true(pi05 自身 SigLIP)作定论追加。

### 6.1 实验矩阵（决策 1 展开）

每环境:`A0`(无 hint)· `A1-prefix` · `A1-suffix` · `A2lite-prefix` · `A2lite-suffix` = **5 臂**。
- **注入点 A/B 先在 LIBERO 定胜负**(轻场,快),胜出的注入点带去 RoboTwin,避免 RoboTwin 跑满 5 臂的重成本。
- ⇒ LIBERO 5 臂全跑;RoboTwin 跑 `A0 · A1-{winner} · A2lite-{winner}` = 3 臂(+ 视情 A2-true)。
- 全部 ×4 seed 变 seed 重复评测。**同一份 pi05 代码 + config flag + hint 文件**驱动全矩阵,零重复代码。

---

## 8. P0 可行性核查结果（2026-07-21，两 Explore 子代理并行勘查）

> **关键反转**:原 plan 假设「LIBERO 先行(脚手架轻)」。核查后发现 **RoboTwin 对 pi05 反而比 LIBERO 更省**,
> 因为 LIBERO 侧有数据格式硬阻断,而 RoboTwin 侧数据/transform/协议几乎全现成。

### 8.1 pi05 注入路径(P0b)—— ✅ 已落地并冒烟
`pi0_config.py`(3 flag)+`model.py`(Observation.lmwm_hint 透传)+`pi0.py`(hint_proj + prefix/suffix 两注入)
+ 冒烟脚本 `scratchpad/smoke_hint.py`。`lmwm_hint_dim=0` 守卫保证与上游逐位一致。

### 8.2 LIBERO(P0a)—— ⚠️ 数据是硬阻断
| 环节 | 状态 | 结论 |
|---|---|---|
| policy transform `libero_policy.py` | ✅ 现成 | state8/action7 维度已对齐,不用写 |
| `LeRobotLiberoDataConfig`(config.py:455) | stub | 定义好但从未实例化,无 repo_id |
| pi05 LIBERO TrainConfig | 缺失 | 需新写 + norm_stats |
| **LIBERO 数据** | ⛔**硬阻断** | lawam 侧 `libero_merged_no_noops_20hz` 是 LeRobot **v3.0/GR00T**(`file-*.parquet` 聚合布局 + key 命名不匹配 repack);openpi 钉的 lerobot 只认 **v2.1**。→ **走下载官方 `physical-intelligence/libero`(v2.1)** 直接匹配 repack,绕开双重转换 |
| eval 客户端 | kai0 缺失 | lawam `examples/LIBERO/eval_files/` 可复用,但需写 **openpi-websocket↔libero 胶水层** |

### 8.3 RoboTwin(P2a)—— ✅ 意外地轻(~1-1.5 天)
| 环节 | 状态 | 结论 |
|---|---|---|
| **数据** | ✅ 现成 | `lmvla/lawam/dataset/robotwin2.0/` LeRobot **v2.1**(27500ep/50fps);v3.0 子集 `robotwin2_lmwm_v30`(1315ep) |
| state/action | ✅ 对齐 | **14 维 aloha layout**,直接用 pi05 `LeRobotAlohaDataConfig`(config.py:402);`AlohaInputs` 原生认 3 相机 |
| horizon | ✅ 天然对齐 | 50fps×1.0s=**action_horizon 50**,pi05 config 已是 50 |
| eval 协议 | ✅ 互通 | pi05 `websocket_policy_server` 与 lawam client **同源 openpi msgpack 协议**,wire 层直接通 |
| eval 编排 | ✅ 整套复用 | `batched_eval_runner.py`+`robotwin_batch_bridge.py`+`auto_eval_robotwin.sh` 照用,只换 server 命令为 pi05 serve |
| 需新建 | 薄 | TrainConfig 一段 + obs 适配 transform(~50 行,`{primary_image,wrist_image,lang}`→aloha 键) + norm_stats + pi05 版 `deploy_policy.yml` |
| ⚠️ 坑 | | (a) `_cluster_env.sh` 引用的 `lmvla/lawam/robotwin_python_wrapper{,_northe}.sh` **不存在**(实存于 `lmvla/lmwam/scripts/`),运行前须补;(b) `adapt_to_pi=False`(sim 无真机夹爪几何换算);(c) norm_stats 须对 pi05 重算 |

### 8.4 修订后的 P0 执行序（并行）
1. **P0b 注入路径** ✅ 已完成(本轮)。
2. **RoboTwin 先行**(数据现成、真判据场):补 2 wrapper → pi05 RoboTwin TrainConfig(改 `LeRobotAlohaDataConfig` repack)→ norm_stats → A0 基线 smoke → eval bridge 换 server。
3. **LIBERO 并行起**:下载 `physical-intelligence/libero` v2.1(di-* 绕代理)→ 注册 TrainConfig → eval 胶水层。**下载耗时可与 RoboTwin 建栈重叠**。
4. hint 离线抽取(`export_pi05_hint.py`)与上面并行:LMWM ckpt 已在,产 `[N,K,D] hint.npz`。

---

## 7. 风险与失败模式

- **So400m 对齐**(§E):A2-lite 的 hint 空间 ≠ pi05 精确权重;靠 Linear 适配层吸收,但若 A2 无增益,需 A2-true 排除"是对齐问题还是空间本身无用"。
- **内在≠SR**(§4.7):§4.20 的 recon_cos 0.95 不保证 SR;必须下游收口。
- **LIBERO 饱和封顶**(§4 零和张力 ~94.8):LIBERO 上 A2−A1 可能被封顶淹没 → RoboTwin 才是真判据。
- **注入扰动 VLM**:prefix 注入若破坏语言对齐→ 退 suffix;先跑 `lmwm_hint_dim=0` 回归确保无害。
- **50fps horizon 坑**(RoboTwin):sec_chunk×fps 必须==action_horizon(历史踩过)。
```
