# Idle(静止/投放)数据裁剪对训练的影响 — 汇总 + 分步规划

> **核心目的(本系列的真正主线,之前文档未点明)**: 验证**裁掉 episode 里的 idle(静止)帧能否让模型真机表现更好**。idle 帧分两类:① **前端**"投放等待"长静止段(机械臂不动、操作员往台上放衣服);② **中段**操作里的停顿/犹豫/反复。假设:idle 帧被 BC 忠实模仿 → 真机走停 / 犹豫 / cloth loop / 拉取松手。
> **分步走**: **Step 1 前端投放裁剪**(= 之前的 v2→v3 / no_release,已做)→ **Step 2 中段 idle 裁剪**(未来)→ Step 3 节奏归一(可选)。
> **状态**: Step 1 ✅ 真机初步成立(裁前端真机明显改善);Step 2 📋 规划。
> **建立**: 2026-06-07(**合并自** [`v2v3_data_window_scaling_experiments.md`](v2v3_data_window_scaling_experiments.md) + [`data_root_cause_probe_experiments.md`](data_root_cause_probe_experiments.md) H1,围绕 idle 主题重组)。
> ⚠️ **方法学铁律**: **真机为终判,offline MAE 系统性反指** —— idle 多的慢/停顿轨迹逐帧 teacher-forcing MAE 反而低,真机却灾难。MAE 仅用于确认训练健康 + 选 ckpt。

---

## 0. 为什么怀疑 idle 数据(已坐实的签名)

后期数据 offline MAE 更低却真机 fail;早期 smooth_800 offline 略差却真机 work。两者最大的数据侧差异之一就是 **idle/静止帧占比 + episode 长度**:

| 段 | 真机 | ep 中位长 | 静止帧 %(\|Δ\|<2e-3) | 投放 onset 中位 |
|---|---|---:|---:|---:|
| smooth 4-25~5-09(work) | ✅ | 1091 | **32.7%** | 短 |
| 后期 5-18~5-27(fail) | ❌ | 1600+ | **37~40%** | ~127 帧 |

→ 后期 ep **长 50%、静止帧多 5~7pp、开头投放等待长**。假设这些 idle 被学进策略 → 真机犹豫/走停。**本系列就是逐步裁掉 idle 验证之。**

---

## 1. idle 数据 + 裁剪机制(代码)

- **检测 motion-onset**(`build_no_release.py`):12D `|Δaction|` 均值持续 > `thr=3e-3`(rad/帧)达 `win=10` 帧的首帧 = 真运动起点;`margin=15` 帧。
- **前端裁**:`cut = max(0, onset - margin)`,删 parquet 行 `[0:cut]` + 同步裁 3 路 mp4(`assert video_frames == parquet_rows`)。
  - `--mode no_release`:对指定 2 天做前裁(单实验对照)。
  - `--per-date`(v3):对 `vis_base/v2/<date>` 每个 ep 前裁 → 输出 `vis_base/v3/<date>-v3`。
- **版本含义**: **v2 = 未裁原始;v3 = 前端投放已裁**。(只裁前端,中段 idle 仍在 → Step 2。)

---

## 1.5 ⭐ 互联网深度调研结论:是否需要删除"所有"idle 数据(2026-06-07)

> deep-research(101 agents / 19 源 / 23 条 3-票核验 claim)。**Bottom line:删掉"所有"idle 帧 —— 既非必要、也不是领域默认、且有风险;证据支持"靶向中间路线"而非一刀切全删。**

1. **idling/卡顿的真因是 single-step / history-conditioned BC 的 copycat 过拟合**,不是 idle 帧本身有毒 —— 单步策略学会"重复上一动作"(训练 near-zero 误差却真机不动)。[Diffusion Policy `2303.04137`:BC-RNN/IBC "get stuck when idle actions not removed";Causal Confusion `1905.11979`;Copycat `2010.14876`;ACT RSS19]
2. **chunked 策略(ACT / Diffusion Policy / pi0 / pi0.5 / X-VLA)天生吸收 chunk 内 pause** → 对保留的 idle 鲁棒(DP:horizon>1 "compensate for idle portions",h=8 最优)。**我们 pi05/XVLA 都是 chunked → 删中段 pause 的边际收益很可能很小。** 但**超过 chunk 长度的"前端长 idle"chunking 盖不住 → 仍需裁**(正好解释前端裁有效)。
3. **主流数据集不做"全删低速帧"**:DROID 只删 operator-gated idle(投放/setup 等待),非速度阈值全删;openpi pi0-DROID 只过滤"整块都 idle 的 chunk"。"删 idle 是常见标准做法"被**证伪(0-3 vote)**。
4. **全删的风险**:任务需要的 pause(叠衣 settle / regrasp 稳定)被删 → 策略学不会"该等时等";train/deploy 分布漂移;丢失恢复力。[DP 明确"因任务需要故意不删 idle"]
5. **领域推荐的是"删的替代方案"**:action chunking(已有)+ **upweight changepoint 关键帧**(Keyframe-Focused IL `2106.06452`)+ **语义/熵感知选择性下采样**(ESPADA `2512.07371` ~2×、DemoSpeedup `2506.05064` ~3×,压缩 casual 段、保留高精度段;**ESPADA 含真机叠衣**)。
6. **关键空白 = 本实验的价值**:目前**无任何"在 chunked VLA 上直接 ablate 删中段 pause vs 靠 chunking 吸收"的工作** → 本系列 v3.1 实验正好填空白。研究**预测:全删 ≈ 或略差于 前端裁(v3)**。

> **据此修正 Step 2 设计**:不盲目全删。**前端裁(已验证)+ 中段长 pause 设上限/下采样、保留短功能性 settle** 是证据支持的最优。但要实证"全删是否必要",就把 **v3.1(全删)当一个 arm**,并**必须和 v3(前端裁)baseline 同数据对比**(+ 可选"中间路线"arm)。详见 §3。
>
> **Sources**: DP `2303.04137` · ACT `roboticsproceedings.org/rss19/p016` · PIP `2508.15669` · Causal Confusion `1905.11979` · Copycat `2010.14876` · Keyframe-Focused `2106.06452` · ESPADA `2512.07371` · DemoSpeedup `2506.05064` · DROID `droid_policy_learning` · openpi droid README。

---

## 2. Step 1 — 前端投放裁剪:已完成实验汇总

| 实验 | 数据 | 裁剪 | offline | 真机 | 结论 |
|---|---|---|---|---|---|
| **no_release probe**(data_root_cause Exp-1)| `A_0522_0526`(后期 fail 2 天 200ep)| no_release(前裁)vs raw(未裁)| best step20k MAE@1 **0.0160**(与 raw 持平,offline 看不出)| ✅ **no-release 明显改善**(用户 2026-06-02)| 🟢 **H1 投放污染初步成立**:前裁真机更流畅 |
| v2/v3 window 系列 | v2/5-18(Exp-A 未裁)· v3 窗口/全量/去脏(Exp-B/C/D 前裁)| 混合(见下注)| 各 horizon MAE | ⏳ 真机待做 | 量数据量/时窗/去脏,**顺带都在 v3 前裁基础上** |

> ⚠️ **诚实标注(之前的混淆)**: v2/v3 window 系列(Exp-A~D)其实**混了多个变量**(trim v2/v3 × 窗口 1日/多日/全量 × 去脏 Exp-D),**没把"idle 裁剪"作为单变量隔离**。真正干净的"裁 vs 不裁"单变量对照是 **no_release probe(Exp-1 no_release vs raw,同 2 天同量)** → 这才是 Step 1 的关键证据。Exp-A~D 的窗口/去脏结论与 idle 主题正交(Exp-D `t-20260607104053-jgbgw` 仍在跑,细节见原 v2v3 文档)。

**Step 1 结论**: **前端投放裁剪(no_release / v2→v3)真机明显改善**(H1 初步成立)。offline MAE 看不出(反指)。→ **idle 数据确实伤真机,裁了有用;下一步裁中段。**

---

## 3. ⭐ Step 2 — v3.1(删除全部 idle)实验 + 调研建议的对照设计

> 用户计划:**通过代码把数据集里所有 idle(前端+中段+尾部)删掉 → 生成 v3.1**,然后训两个模型,验证"是否需要删所有 idle"。
> ⚠️ **调研提示(§1.5)**:全删非必要、有风险,且 pi05 是 chunked → 删中段边际收益预计很小。**所以实验必须带 v3(前端裁)baseline 对比 + 一个"中间路线"arm**,否则只跑两个 v3.1 无法回答"是否需要全删"。

### 3.1 v3.1 数据构建(待实现)
- **定义**: v3.1 = v3(前端投放已裁)**再删掉中段 + 尾部所有静止段** = "全删 idle"。
- 检测:逐帧 `|Δaction|` 持续 `< thr`(静止)的连续段 → 删除(不分前/中/尾)。
- ⚠️ **关键风险:轨迹连续性**(调研也强调)。删中段后 action chunk 会**跨被删段跳变** → 引入 jump。必须:(a) 删后重排 `frame_index`/`index` + 重建 chunk;(b) 保留段边界过渡帧;(c) 严阈值,只删"完全静止",不碰慢速有效动作。
- 实现:扩 `build_no_release.py` 加 `--mode all_idle`(或新脚本),输出 `vis_base/v3.1/<date>`。**先 2a 小规模验证裁剪正确再批量。**

### 3.2 实验矩阵(单变量 = idle 删除程度;同 init/参数/pipeline/部署)
| arm | 数据 | idle 处理 | 用途 |
|---|---|---|---|
| **B-early(baseline)** | ≤5-10 | **v3(仅前端裁)** | 早期 work 锚(≈ smooth800)对照 |
| **Exp-2a** ⭐(用户)| ≤5-10 | **v3.1(全删)** | 早期数据上"全删 vs 前端裁" |
| **B-all(baseline)** | 全量(排 5-16) | **v3(仅前端裁)** | = Exp-C/Exp-D 系列 |
| **Exp-2b** ⭐(用户)| 全量 | **v3.1(全删)** | 全量上"全删 vs 前端裁" |
| (可选)**Exp-2c 中间路线** | ≤5-10 或全量 | **前端裁 + 中段长 pause 设上限/下采样、留短 settle** | 调研推荐解,大概率最优 |

> 注:用户原计划是 Exp-2a + Exp-2b(两个 v3.1)。**B-early / B-all 是必须补的对照**(否则无法判定"全删是否必要");它们多半已有现成 v3 ckpt 可复用(smooth800 / Exp-C / Exp-D)。

### 3.3 判据(真机为终判)
| 结果 | 结论 |
|---|---|
| v3.1 真机 **明显优于** v3 | 全删有用 → 值得(推翻调研预测,填补 chunked-VLA 空白) |
| v3.1 **≈** v3 | **全删非必要**(调研预测) → 省事用 v3;或转 Exp-2c 中间路线找增量 |
| v3.1 **差于** v3 | 全删**有害**(删了功能性 settle / 分布漂移)→ 明确不要全删 |
> offline MAE 仅看训练健康(idle 多的轨迹 MAE 反指,§1.5/铁律)。

### 3.4 分步执行
- **2a** 扩 build 加全删 idle 模式;1~2 ep 验证"只删静止、轨迹不断裂、chunk 不跳变"(可视化)。
- **2b** build `≤5-10 v3.1` + `全量 v3.1`;注册 config(克隆 Exp-C/smooth800 同参,仅换数据);提交训练。
- **2c** 对齐 B-early/B-all 的 v3 baseline(复用现成 ckpt 或补训);**真机三方对比** → 落 §3.3 判据。
- **2d**(条件)若全删 ≈/差 → 跑 Exp-2c 中间路线(前端裁 + 中段下采样)验证证据推荐解。

---

## 4. 关联文档
- **合并来源(已并入本文档)**: `v2v3_data_window_scaling_experiments.md`(v2/v3 + 窗口 Exp-A~D,细节与 Exp-D 任务跟踪留存其中)。
- **前端裁根因线**: `data_root_cause_probe_experiments.md`(H1=投放污染=本系列 Step 1;H2 慢节奏 / H3 gripper 漂移 / H4 wrist OOD 是**其它**真机失败根因,不属 idle 主题,各自单查)。
- **裁剪脚本**: `train_scripts/kai/data/build_no_release.py`(`--mode no_release` 前裁 / `--per-date` v3 / 待加中段裁模式)。
- **work 锚点**: `task_a_new_smooth_800_new_norm_results.md`。
