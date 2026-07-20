# LMVLA 环境与依赖关系图(2026-07-20 勘查)

> 起因:环境混乱已直接造成实验事故 —— 见 §4。本文 = 「哪段代码该用哪个解释器」的单一事实源。

---

## 1. 先理清命名(五个名字,常被混用)

| 名字 | 是什么 | 位置 |
|---|---|---|
| **LaWAM** | **外部参考工作**(RLinf,arXiv 2606.15768)。我们的上游孪生 | vendored: `lmvla/lmwm/vendor/LaWAM/` |
| **LMVLA** | **我们的伞形项目名**(= LAWAM 别名,易与上面混淆) | `lmvla/` |
| **lawam/** | 我们改造的 **starVLA 训练/评测框架**(下游 VLA) | `lmvla/lawam/` |
| **CRAVE** | **信号层**:跨-episode 复现密度场 r(o),零训练 | `lmvla/crave/` |
| **LMWM** | **世界模型**:r-谷分段 + r-脊目标 + (生成器+MDN) | `lmvla/lmwm/` |
| **LMWAM** | LMWM **注入** lawam 后的集成体(报告页用名) | 无独立目录 |

> ⚠️ `lmvla/lmvla/` 是嵌套同名目录,`lmvla/temp/`、`lmvla/xvla/` 未在当前主线使用。

---

## 2. 环境清单(gf0 本机)

| 环境 | Python | torch | transformers | sklearn | numpy | 状态 |
|---|---|---|---|---|---|---|
| **`conda:srpo`** | 3.10.20 | 2.10.0+cu128 | **4.57.6** | 1.7.2 | 2.2.6 | ✅ **CRAVE/LMWM 正统环境** |
| **`kai0/.venv`** | 3.12.13 | 2.6.0+cu124 | 5.13.1 | **❌无** | 2.2.6 | ✅ VLA 训练/评测 |
| `kai0/.venv_py311_bak` | 3.11.15 | 2.7.1+cu126 | **4.53.2** | 1.8.0 | 1.26.4 | ⚠️ 见 §4 |
| `.venv_dinov3` | 3.11.15 | — | — | — | — | 🗑️ **空壳(3个包),已死** |
| `.venv_wanvae` | 3.11.15 | — | — | — | — | 🗑️ **Wan-VAE 编码器已被架构定版淘汰** |

**North-E**(`/vePFS-North-E/vis_robot/`):

| 环境 | Python | 备注 |
|---|---|---|
| `workspace/deepdive_kai0/kai0/.venv` | 3.12.13 | 与 gf0 的 kai0/.venv 版本一致(torch 2.6.0/tf 5.13.1);**无 sklearn** |
| `huanqian/conda_envs/RoboTwin` | 3.10.20 | sapien 3.0.0b1 + mplib + curobo。**RoboTwin 仿真专用** |
| `kai0/.venv_py311_bak` | — | **不存在**(仅 gf0 有) |

---

## 3. 组件 → 环境 映射(必须遵守)

| 环节 | 代码 | 必须用 | 硬性理由 |
|---|---|---|---|
| **DINOv3 特征提取** | `p1_*_dinov3base_extract.py`、`crave/encoders` | **`srpo`** | **DINOv3 需 transformers ≥ 4.56**(`dinov3_vit` model_type);`py311_bak` 的 4.53.2 **加载不了** |
| **CRAVE r 场 / 分段** | `p1_*_rvalley_pairs.py`、`rvalley_segmenter.py` | **`srpo`** | 脚本注释即写 `srpo python`;需 scipy+sklearn |
| **LMWM 模型训练** | `p1_train_lmwm_{libero,robotwin}.py` | **`srpo`** | 同上 |
| **Qwen3-VL 特征提取** | `p1_libero_qwen3vl_extract.py` | `kai0/.venv` | 需 transformers 5.x 的 `Qwen3VLForConditionalGeneration` |
| **VLA 训练/评测** | `lawam/starVLA/*`、volc yaml | `kai0/.venv` | `STAR_VLA_PYTHON` |
| **RoboTwin 仿真侧** | `envs/*`、`robotwin_batch_bridge` | **RoboTwin conda** | sapien/mplib/curobo 不在任何 venv 里 |

### RoboTwin 解释器包装(两集群路径不同)

```
cnsh    : lmvla/lawam/robotwin_python_wrapper.sh          → /vePFS/HuanQian/conda_envs/RoboTwin
North-E : lmvla/lawam/robotwin_python_wrapper_northe.sh   → /vePFS-North-E/vis_robot/huanqian/conda_envs/RoboTwin
```
两者都需设 `VK_ICD_FILENAMES`(vulkan)与 `PYTHONPATH=<...>/robotwin_client_deps`
(json_numpy/omegaconf/rich,conda env 本身缺这些)。评测时经 `ROBOTWIN_PYTHON` 传入。

---

## 4. 已造成事故的坑(全部实际发生过)

1. **`kai0/.venv` 无 sklearn** → CRAVE 分段脚本 `ModuleNotFoundError: sklearn`。
   我当时改用 `py311_bak` 绕过 —— **这是错的**,正解是 `srpo`。
2. **`py311_bak` 的 transformers 4.53.2 < 4.56** → `AutoModel` 加载 DINOv3 失败,
   `hf_dino.py` 的 try/except 会**静默回退**到纯 torch 移植版 `DINOv3ViTStandalone`。

   > ✅ **[2026-07-20 已实证:两条路径等价,风险解除]**
   > 在**同一环境、同一批帧**上强制走两条路径(monkeypatch `AutoModel` 使其抛异常,
   > 从而进入真实的 except 分支,连同其自带预处理),对比 3 个 episode / 60 帧:
   > **逐元素完全相同,`max|A−B| = 0.000e+00`**,逐帧余弦 = 1.000000。
   > ⇒ standalone 是**忠实移植**,历史上经该路径抽取的特征**可与 HF 路径产物混用**。
   >
   > ⚠️ 但仍**不建议**用 `py311_bak` 抽特征:回退是静默的,一旦移植版将来与上游脱节
   > 就会无声地产生差异。保留"必须走 HF 路径"的硬断言作为防线。
   >
   > 📌 方法学教训:首轮我用"gist 帧间相似度矩阵的相关系数"作下游判据,得到 **−0.50**,
   > 与"特征逐元素相同"直接矛盾。真因是**该指标在此处病态** —— 同一 episode 内帧极度相似,
   > 帧间相似度离散度趋近 0,对近乎常数的两组求相关无意义。
   > **指标本身需要先验条件数,否则会得出与事实相反的结论。**
3. ~~**同一脚本在不同环境结果不同**(原判断:sklearn/numpy 版本差异改变了 BGMM/PCA 输出)~~
   **❌ [2026-07-20 已证伪]**:用 `srpo`(sklearn 1.7.2 / numpy 2.2.6)复跑同一 `finalarch` 脚本,
   得 **3.44 段/ep —— 与 `py311_bak`(sklearn 1.8.0 / numpy 1.26.4)完全相同**。
   ⇒ 3.44 vs 生产 5.87 的差异**不是环境造成的**,而是**脚本版本变更**
   (该脚本 07-19 被改过、生产产物 07-15 生成,且**未纳入 git 无法 diff 回原版**)。

   真正的教训不变但换了原因:**产物必须同时记录「生成环境 + 脚本 git hash」**;
   本例中缺的恰恰是后者 —— 环境查得再清也归因不了。**未纳入 git 的脚本不应用于生产产物。**
4. **North-E 没有 `py311_bak`** → 本机能跑的脚本搬过去可能直接找不到解释器。
5. **RoboTwin 仿真依赖不在任何 venv** → 不设 `ROBOTWIN_PYTHON` 就 `No module named 'sapien'`。

---

## 6. 跨集群一致性(cnsh ↔ North-E)

实测指纹逐项对比(2026-07-20):

| 环节 | cnsh(gf0 /vePFS) | North-E | 一致性 |
|---|---|---|---|
| **VLA 训练/评测** `kai0/.venv` | py3.12.13 · torch **2.6.0+cu124** · tf **5.13.1** · acc 1.13.0 · numpy 2.2.6 · scipy 1.15.3 · pandas 2.2.3 · pyarrow 20.0.0 · av 13.1.0 | **逐项相同** | ✅ **完全一致** |
| **RoboTwin 仿真** conda | py3.10.20 · torch **2.4.1+cu121** · sklearn 1.7.2 · numpy 1.26.4 · scipy 1.15.3 · pandas 2.3.3 · av 17.0.1 | **逐项相同** | ✅ **完全一致** |
| **CRAVE/LMWM** `srpo` | ✅ py3.10.20 · torch 2.10.0+cu128 · tf 4.57.6 · sklearn 1.7.2 | ❌ **不存在** | ⚠️ **单点依赖** |
| repo git | `494c17e` (07-18) | `e3edce5` (07-08) | ⚠️ 差 12 天,**但 starVLA 55 文件 md5 逐一相同**(差异在 docs/kai0) |
| GPU | 8×A100-80G | 8×H20 | ⚠️ 硬件不同 |

### 由此得出的三条结论

1. **✅ 跨集群 VLA 实验是可比的。** 2026-07-19 的 RoboTwin 双臂(LMWM 训在 cnsh、baseline 训在 North-E)
   环境逐项一致,该对照**成立**;唯一残留变量是 **A100 vs H20**(同步数/同 batch/同 seed,bf16 数值差异可忽略,
   但若最终差距 <2pt 则不能排除)。
2. **✅ RoboTwin 仿真行为跨集群一致**,评测可在任一集群跑。
   注意两边 wrapper 路径不同(见 §3),`ROBOTWIN_PYTHON` 必须按集群设。
3. **⚠️ CRAVE/LMWM 链路是 gf0 单点。** North-E 上**没有** srpo 等价环境
   (现有 conda 仅 RoboTwin/abot-physworld/fastwam/gigaworld-policy/lerobot/uniVP,均非本链路)。
   ⇒ **所有特征提取、r 场计算、分段、LMWM 训练只能在 gf0 跑**,产物(pairs.npz / target_compact.npz /
   lmwm.pt)靠传输同步到 North-E。这也是一个可用性风险:gf0 挂了则整条信号链停摆。

### 与 §4 的关联

`kai0/.venv` 在两个集群完全一致,**但两边都缺 sklearn** —— 所以"用 kai0/.venv 跑 CRAVE 脚本"
在任何集群都会失败,不是某台机器的问题,是**该环境本就不为此链路设计**。

---

## 5. 建议(按收益排序)

1. **删除死环境**:`.venv_dinov3`(空壳)、`.venv_wanvae`(编码器已淘汰)。减少误选。
2. **统一 CRAVE/LMWM 全链路到 `srpo`**,并在每个脚本头部写死断言:
   ```python
   import transformers, sys
   assert tuple(map(int, transformers.__version__.split(".")[:2])) >= (4, 56), \
       f"DINOv3 需 transformers>=4.56, 当前 {transformers.__version__} (应使用 srpo 环境)"
   ```
3. **产物旁写 `_env.json`**(python/torch/transformers/sklearn/numpy 版本 + 脚本 git hash),
   解决 §4.3 那类「同脚本不同结果、无法归因」的问题。
4. **`py311_bak` 明确定位**为"只读 npz 的下游分析",在其名字或 README 中标注**禁止用于特征提取**。
5. North-E 若需跑 CRAVE 链路,应同步一个 `srpo` 等价环境(当前只有 `kai0/.venv`,缺 sklearn)。
