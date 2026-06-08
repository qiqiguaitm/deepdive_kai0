# X3.C (我们的 X-VLA) vs 官方 X-VLA + vs pi05 — 真机差的根因分析

> **问题**: X3.C 用 smooth_800 数据训练, **offline action MAE 健康** (单调收敛, 见 [`../future_plans/plans/xvla_track_x_curriculum.md`](../future_plans/plans/xvla_track_x_curriculum.md) §0.NEW.2.5), 但**真机机械臂完成任务不尽人意**; **同一份数据训练的 pi05 真机 work**。
>
> **建立**: 2026-06-01
> **方法**: 逐行对比 ① 官方 X-VLA repo (`xvla/X-VLA/`, THU-AIR) ② lerobot port (`lerobot/policies/xvla/`, 实际训练用) ③ 我们的训练/推理代码 (`train_scripts/xvla/`, `kai0/scripts/serve_policy_xvla.py`)。
> **真机实证**: [`x3c_realrobot_trace_20260601.md`](x3c_realrobot_trace_20260601.md) — 一次真机执行 trace 证实任务失败 = EE 预测高频震荡 (lag1 自相关 −0.34) + IK 放大 (EE↔joint 相关仅 0.47) + 折返比 9~13× + 夹爪 16 次开合, 与 R1/R2 吻合。
> **数据集审计**: [`xvla_dataset_vs_official.md`](xvla_dataset_vs_official.md) — 我们 vis EE6D vs 官方 Agilex 逐项对齐: action 表示/rot6d/gripper/loss-scale/标签语义**全对齐无致命错**; 唯一实质差异 **D5 = action chunk 时间窗口 1s vs 官方 2s** (qdur), 列为 P0 后下一候选。
> **关联**: [`xvla_vs_official_gap_rootcause`] · pi05 同标尺对比 (xvla_track_x_curriculum §0.NEW.2.5b: pi05 EE6D MAE ≈ X3.C 的 1/5)

---

## 0. TL;DR — 三层根因 (按置信度)

| # | 根因 | 证据 | 影响 | 置信 |
|---|---|---|---|---|
| **R1** | **训练全程缺 ImageNet 图像归一化** | `xvla_train.py:330` 直接 `model.forward(batch)`, 绕过 lerobot 的 `XVLAImageNetNormalizeProcessorStep`; dataset 只 `/255.0` | base ckpt 在 ImageNet 域预训练, 我们用 [0,1] 域 finetune → 视觉特征**输入分布错位**, 预训练白费 + 泛化差 | 🔴 高 (代码坐实) |
| **R2** | **EE6D→IK 真机执行链 vs pi05 直出 joint** | X3.C 出 20D EE6D → server `_ee6d_to_world8` → ROS node IK→joint; pi05 直接出 14D joint | IK 在奇异点/容差 (pos 0.04m/rot 0.12rad) 处放大误差; offline 不经此链所以测不出 | 🔴 高 (架构性) |
| **R3** | **欠训 + 弱配方**: 30k step/warmup500 vs 官方 50k/warmup2000 | 官方 README finetune 示例 `--iters 50000 --warmup_steps 2000`; 我们 30k/500; 训练曲线末端仍降 | action fidelity 未到位 (§0.NEW.2.5 显示 30k 仅 2.1 epoch, 末段还在降) | 🟡 中 |
| **R4** | **架构/容量**: X-VLA-base 0.9B vs pi05 2.2B + 成熟 flow-matching | §0.NEW.2.5b 同标尺 pi05 EE6D MAE 各 horizon 4~6× 优于 X3.C | action 保真度天花板更低 | 🟡 中 |

> **为什么 pi05 同数据 work 而 X3.C 不**: pi05 (a) 图像归一化正确 (openpi AgilexInputs 标准管线), (b) 直接输出 joint **不经 IK**, (c) 模型更大更成熟, (d) 训练配方匹配 (50k step EMA)。X3.C 在 R1/R2/R3/R4 四处同时吃亏。
>
> ⚠️ **2026-06-07 重定位(以 §7 为准)**: P0 修了 R1 后真机仍"抓不到衣角",实测复盘 → **R4 容量排除**(官方 0.9B 能 fold)、D5 = 动作 chunk 表示(我们 1s 稠密 vs 官方 2s intention-abstraction anchor)一度升为主因。上表 R1/R4 权重已过时,详见 §7。
> ⚠️⚠️ **2026-06-08 再修正(以 §7.5 为准)**: D5 对照 `X3C_smooth800_d5anchor` 训完,offline 欠到位探针实测 → **D5 没治好接近/抓取欠到位(0.71→0.70),只治好持握乱飘+震荡 → D5 从主因降为"部分改善"**;欠到位与动作表示无关(dense/anchor 都 ~0.70)→ **头号嫌疑转向"配方/数据"**(单域 finetune-from-base vs 官方 290K 多域 co-train),待 Exp-O vs Exp-S 数据隔离实验定论。

---

## 1. 关键差异表 — 官方 X-VLA / lerobot port / 我们的实现

| 维度 | 官方 X-VLA repo | lerobot port (训练用) | 我们的代码 | 一致? |
|---|---|---|---|---|
| **图像归一化** | `transforms.Normalize(ImageNet mean/std)` (dataset.py:81) | `XVLAImageNetNormalizeProcessorStep` (processor_xvla.py:73) — **在 processor, 不在 forward** | ❌ **dataset 只 /255.0** (multi_domain_dataset.py:157), `forward` 不归一化, 训练脚本不调 processor | 🔴 **不一致 (R1)** |
| 图像增强 | ColorJitter(0.2) 训练时 (dataset.py:78) | (processor 可选) | ❌ 无 ColorJitter | 🟡 缺增强 |
| 图像 resize | Resize(224,224) bicubic | `resize_with_pad` in forward (modeling_xvla:318) | resize_pad → 256/256/224 + forward 内再 resize_with_pad | ✅ 近似 |
| 动作表示 | EE6D 20D, gripper BCE | 同 | 同 (joint_to_ee6d.py) | ✅ |
| 动作归一化 | 无 mean/std, gripper mask | 无 | 无 | ✅ |
| Rot6D 排布 | interleaved `[r00,r01,r10,r11,r20,r21]` | 同 | interleaved (fixed, joint_to_ee6d:42) | ✅ (已修 bug) |
| **总步数** | 默认 1M iters; README finetune **50k** | — | **30k** | 🟡 **偏少 (R3)** |
| **warmup** | **2000** | — | **500** | 🟡 偏少 |
| freeze_steps | 1000 | — | 1000 | ✅ |
| lr | 1e-4 | — | 5e-5 | 🟡 (更小, 配 30k 合理) |
| optimizer | AdamW β(0.9,0.95) wd0 | — | AdamW β(0.9,0.95) wd **1e-4** | 🟡 wd 不同 |
| batch | 16 | — | 8/gpu × 8 = **64 eff** | 🟡 更大 |
| 推理 denoise | 10 步 | — | 10 步 | ✅ |
| **推理平滑** | **无** temporal ensembling (官方) | — | ✅ ROS node `StreamActionBuffer` min_jerk 混合 (policy_inference_node) | ✅ 我们反而更好 |
| **EE→joint** | 官方 deploy 出 world EE pose, 下游 IK | — | server `_ee6d_to_world8` → ROS IK (firmware/host) | 🔴 **额外 IK 链 (R2)** |

---

## 2. R1 (主因) — 训练全程缺 ImageNet 归一化 [代码坐实]

### 证据链

1. **lerobot XVLAPolicy 的 forward 不做归一化**: `modeling_xvla.py:_prepare_images` (line 313-320) 只 `resize_with_pad`, **无 mean/std**。
2. **归一化是 processor 层职责**: `processor_xvla.py:73` `XVLAImageNetNormalizeProcessorStep()` — 文档明确 "validates [0,1] range before normalizing, formula (image-mean)/std" (line 350-355)。**必须在调 forward 前手动跑 processor**。
3. **我们的训练脚本绕过 processor**: `xvla_train.py:330` `model.forward(batch)` 直接调, **全文件无 `make_xvla_pre_post_processors` / `Normalize` / processor import**。
4. **dataset 只 /255.0**: `multi_domain_dataset.py:157` `torch.from_numpy(frame).permute(2,0,1).float()/255.0` → 图像停在 [0,1], 再没归一化。
5. **官方 dataset 明确归一化**: `xvla/X-VLA/datasets/dataset.py:81` `transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))`。

### 为什么这是真机杀手 (而 offline 看不出)

- **base ckpt `lerobot/xvla-base` 是在 ImageNet 归一化域预训练的** (Florence2 视觉塔标准做法)。我们用 [0,1] 域 (均值 ~0.5, 非 0 中心) finetune → **预训练视觉特征的输入统计被破坏**, 模型得从头重学视觉前端 → 等效"半随机初始化 + 欠训"。
- **训练/推理自洽**: 因为训练和 eval 都缺归一化, 模型在 [0,1] 域内自洽学习 → **offline MAE 正常** (这就是 §0.NEW.2.5 单调收敛但绝对值差的原因)。
- **真机差**: 视觉特征质量低 → 对光照/视角/布料外观的泛化差 → 真机新场景动作犹豫/不准。pi05 用 openpi 标准图像管线 (归一化正确), 视觉前端泛化好。

### 验证方法 (推荐先做, 成本最低)

```bash
# A/B: 同 smooth_800, 唯一改 = 加 ImageNet 归一化, 重训 X3.C → 对比 offline MAE + 真机
# 实现: multi_domain_dataset.py 图像 /255 后追加 (img-mean)/std (ImageNet),
#       serve_policy_xvla.py 推理侧同步加 (训练/推理必须一致!)
```

---

## 3. R2 — EE6D→IK 真机执行链 (pi05 无此链)

| | pi05 | X3.C XVLA |
|---|---|---|
| 模型输出 | **14D joint** (直接关节角) | 20D EE6D (末端位姿 + 二值 gripper) |
| 真机执行 | joint 直接下发 | EE6D → world pose (`_ee6d_to_world8`) → **IK → joint** |
| 误差传导 | 无额外环节 | IK 在奇异点 / 容差边界放大 (serve 配置 pos 容差 0.04m, rot 0.12rad, max_jump 0.15rad) |
| offline 是否覆盖 | 是 (MAE 直接在 joint) | **否** (eval 在 EE6D 空间, 不经 IK) → 真机才暴露 |

**机理**: 即使 EE6D 预测精确, IK 求解可能 (a) 多解跳变 (b) 接近奇异点时关节剧烈变化 (c) 容差内"够用即停"导致末端漂移。这些**只在真机闭环出现, offline EE6D MAE 完全测不到**。pi05 直出 joint 绕开整条链。

> 注: 我们的 ROS `StreamActionBuffer` (min_jerk 混合) 已经比官方更努力地平滑, 但平滑的是 IK 之后的 joint 轨迹, 救不了 IK 本身的多解/奇异问题。

---

## 4. R3 — 欠训 + 弱配方

- **步数**: 官方 README finetune 示例 `--iters 50000`, 我们 30k。§0.NEW.2.5 训练曲线显示 **30k 末段 MAE 仍在降** (24k→30k @1 -4%), 30k 仅 ≈2.1 epoch → 确实欠训。
- **warmup**: 官方 2000, 我们 500 — 短 warmup + frozen backbone 1000 步, 解冻后 lr 冲击更大。
- **wd**: 官方 0, 我们 1e-4 — 轻微但方向性差异。
- → 已规划 100k 延长训练 (§0.NEW.5) 验证欠训分量。**但注意: 若 R1 (归一化) 不修, 单纯加步数只是在错误输入域上练更久, 收益有限。R1 应优先。**

---

## 5. R4 — 架构/容量天花板

§0.NEW.2.5b 同标尺 (同 val 窗口 + 同 FK) 对比: **pi05 EE6D MAE 各 horizon 4~6× 优于 X3.C**, xyz/rot6d/gripper 三组全胜。即使修了 R1/R2/R3, X-VLA-base (0.9B, lerobot port) 的 action 保真度天花板大概率仍低于 pi05 (2.2B + 成熟 flow-matching + EMA)。**这是 Track X 的结构性劣势**, 决定了它的价值只能来自跨域/跨本体迁移 (soft prompt 多域共享), 而非单任务 action fidelity。

---

## 6. 修复优先级 + 行动

| 优先级 | 行动 | 成本 | 预期 |
|---|---|---|---|
| ⭐⭐⭐ **P0** 🔄 **执行中** | **修 R1 + 对齐官方配方**, 重训 X3.C (`X3C_smooth800_p0`, uc01, 2026-06-02 起, ETA ~11h) | 1 训练 | 若真机显著改善 → R1 坐实 |
| ⭐⭐ P1 | **R2 诊断**: 已有真机 trace ([`x3c_realrobot_trace_20260601.md`](x3c_realrobot_trace_20260601.md)) 证 EE↔joint 相关仅 0.47; P0 后复测看 IK 是否独立残余 | — | 定位 IK 残余 |
| ⭐ P2 | P0 真机若仍不足, 再评估 R3 (更多步) / R4 (架构) | — | — |

### 6.1 P0 重训配置 (`X3C_smooth800_p0`, 2026-06-02) — 修 R1 + 对齐官方

| 参数 | 官方 X-VLA | 30k 旧版 | **P0 版** | 说明 |
|---|---|---|---|---|
| **图像归一化** | ImageNet | ❌ 只 /255 | ✅ **ImageNet (img-mean)/std** | **R1 修复, 主改动** |
| ColorJitter | 0.2 | ❌ | ✅ 0.2 (brightness/contrast/saturation) | 对齐官方 |
| steps | 50k (finetune 示例) | 30k | **60k** | 适配 eff batch 64 (官方 4×) + R1 后视觉重适应余量 (≈4.3 epoch) |
| lr | 1e-4 | 5e-5 | **1e-4** | 对齐官方 (VLM lr = 1e-5 via scale 0.1) |
| warmup | 2000 | 500 | **2000** | 对齐官方 |
| freeze | 1000 | 1000 | 1000 | 已一致 |
| weight_decay | 0.0 | 1e-4 | **0.0** | 对齐官方 |
| lr schedule | constant (默认) | cosine | cosine (适配) | 定长训练 cosine 更稳, 非照搬 |
| batch | 16 (单卡) | 64 | 64 (适配) | 单机 8 A800 |
| betas / grad clip | (0.9,0.95) / 1.0 | 同 | 同 | ✅ |

- ckpt: `uc01:/data/shared/ubuntu/local_ckpts/xvla_x3c_smooth800_p0/` (每 2k step), 不覆盖 30k 版。
- **train/serve parity 铁律**: `multi_domain_dataset.imagenet_normalize_chw` 与 `serve_policy_xvla` 的 `_IMAGENET_MEAN/STD` 数值完全一致; 推理须加 `--imagenet_norm` (旧 30k ckpt 用 `--no-imagenet_norm`)。ColorJitter 仅训练 (eval/serve image_aug=False)。
- ⚠️ eval: `eval_xvla_ee6d.py` 经 `multi_domain_dataset` 现也归一化 → **P0 ckpt eval 自洽; 旧 30k ckpt 勿再用此 eval** (mismatch)。

### 6.2 P0 验收 — 真机客观判据 (复测 trace 指标)

P0 ckpt 真机录 trace, 对比 [`x3c_realrobot_trace_20260601.md`](x3c_realrobot_trace_20260601.md) 的 30k 基线:

| 指标 | 30k 基线 (差) | P0 目标 |
|---|---|---|
| EE-L y/z 速度 lag1 自相关 | −0.34 / −0.35 (震荡) | → 接近 0 |
| EE 折返比 (路程/净位移) | 9.1× / 13.1× | → ~2× (smooth 数据级) |
| 关节方向反转率 | 0.45~0.69 | → ~0.1 |
| 夹爪开合切换 (200帧) | 16 | → 个位数 |
| 任务完成 | ❌ 需人干预 | → 自主折叠 |
| ⭐ P2 | R1 修复后重跑 §0.NEW.2.5b pi05 对比, 看 gap 缩小多少 → 估 R4 残余 | 2 eval | 量化架构天花板 |

> **关键判断**: pi05 同数据 work 证明**数据没问题、任务可学**。X3.C 的差是**实现/配方**问题 (R1 主导) + **架构**劣势 (R4)。**先修 R1 (零数据成本, 纯代码) 是性价比最高的一步。**

---

## 7. P0 后实测复盘 + 根因重定位 (2026-06-07) — 容量排除,动作表示(D5)是主因

> P0 版 `xvla_x3c_smooth800_p0/step58000` 训完:offline xyz MAE@1 0.0052(P0 前 0.0122→减半,R1 见效),**但真机仍"连衣角都抓不到"**(用户实测)。两个针对性探针(Claude 离线 A100,bart 真 token + 在线 EE6D 转换 + held-out vis;脚本在 gitignored `_xvla_gripper_debug/`)定位真因。

### 7.1 实测诊断

**(a) 运动剖面 — MAE@1 误导,模型没学会"运动幅度"**(按 GT chunk 位移分高/低动量):

| 窗口 | xyz MAE@1 | GT chunk 位移 | pred chunk 位移 | pred/GT |
|---|---:|---:|---:|---:|
| 接近/抓取(高动量) | ~0.005 | ~1.2m | ~0.7–0.9m | **0.60–0.80(欠到位 → 够不到衣角)** |
| 持握(低动量) | ~0.004 | ~0.012m | ~0.31m | **~26×(该静止却乱飘)** |

→ MAE@1 看着好(≈5mm)但只测**第一步 + teacher-forcing**;整条 chunk 的**运动幅度是错的**:接近只走 60-80%(够不到衣角=真机抓不到),持握乱飘 26×。**又一次应验"offline MAE 系统性反指"。**

**(b) denoise 步数无效(零成本旋钮已排除)**:10→20→50 步,欠到位仅 0.74→0.77→0.80(补 6pp,仍够不到),持握更糟 0.33→0.38m,MAE@1 略升。→ 非推理问题。

### 7.2 ⭐ 根因重定位:R4 容量**排除**,D5 动作表示**升为主因**

- **官方 X-VLA-0.9B 能学会 Agilex 叠衣**(Soft-Fold ~100% 成功 / 33 folds·h⁻¹,paper)→ **同架构同 0.9B 容量足够、任务可学 → R4 容量不成立,DEMOTE**。差异必在我们的实现。
- 代码坐实 **D5(动作 chunk 表示)= 真凶**:

| | 官方 X-VLA (Agilex real-world) | 我们 |
|---|---|---|
| 动作 chunk 构造 | `freq=30, qdur=2.0` → `np.linspace(cur, cur+2s, 31)` = **30 anchor 均匀铺在 2 秒**(intention abstraction 插值下采样) | `stack(action[f_idx:f_idx+30])` = **30 连续帧 = 1 秒稠密** |
| 代码 | `base.py:152` + `real_world.py:40`(`qdur=2.0`) | `multi_domain_dataset.py:162-163` |
| 时间窗口 | **2s** | **1s(仅一半)** |
| 采样 | linspace 时间插值 anchor | 连续帧,无下采样 |

- **机理**:`lerobot/xvla-base` 预训练用的是"30 anchor over 2s"的 **intention-abstraction** 动作表示;我们用"30 连续帧 1s 稠密"finetune → **动作的时间窗 + 尺度与预训练先验冲突** → 模型 hedge → (a) 欠到位(走 60-80% → 够不到衣角)、(b) 持握乱飘。与 7.1 实测完全吻合,**且不依赖容量**(官方 0.9B 用对表示就 fold)。
- 这正是数据审计 [`xvla_dataset_vs_official.md`](xvla_dataset_vs_official.md) 早标的 **D5(唯一实质差异:chunk 1s vs 官方 2s)**;P0 后经实测 + 代码坐实,**从"下一候选"升为头号主因**。

### 7.3 根因权重修正(P0 后)

| 根因 | P0 前 | P0 后 | D5 对照训完后(2026-06-08,见 §7.5) |
|---|---|---|---|
| **D5 动作表示(1s 稠密 vs 官方 2s anchor)** | 未列 | 🟠 候选主因(代码坐实) | 🟡 **降级:部分改善,非根治**——治持握乱飘(8.75→1.61)+ 治震荡,但**接近/抓取欠到位 0.71→0.70 没动** |
| R1 ImageNet 归一 | 主因 | P0 已修,offline 减半但真机仍废 → 非真机主因 | 同 |
| R3 欠训 | 中 | 次(58k 近 plateau) | 同 |
| **R4 容量 0.9B** | 中 | ❌ **排除**(官方 0.9B 能 fold) | 维持排除 |
| R2 EE6D→IK 链 | 高 | 放大器,非源头 | 同 |
| **配方/数据(单域 finetune-from-base vs 官方 290K 多域 co-train)** | — | 次要嫌疑 | 🔴 **升为头号嫌疑**——欠到位与动作表示无关(dense/anchor 都 ~0.70)、R4 已排除 → Exp-O vs Exp-S 数据隔离实验定论 |
| gripper 部署映射(SoftFold -0.0055) | — | 叠加项 | 同 |

### 7.4 验证 / 修复 — D5 对照实验已提交 (2026-06-07)

- **✅ 已落地 (读取时重采样, 无需重建数据)**:`LeRobotEE6DDataset` 加 `action_qdur` 参数(`multi_domain_dataset.py`);设 2.0 时,action chunk = `linspace(f_idx, f_idx+qdur·fps, N+1)[1:]` 取 30 anchor(对齐官方 `base.py:152` + `real_world.py qdur=2.0`),per-frame EE6D 按帧重采样,**默认 None=legacy 不影响其它实验**。smoke 实测:anchor chunk 位移 **1.59m vs legacy 0.117m(13.6×)**,样本数不变。
- **对照实验** `X3C_smooth800_d5anchor`(单变量 vs `X3C_smooth800_p0`,仅 `action_qdur=2.0`):
  - YAML `xvla_x3c_d5anchor_cnsh_8gpu.yaml`,**task `t-20260607152340-4j7q5`**(cn-shanghai / robot-task 8 A100,60k)。
  - **验证判据(offline)**:训完用运动剖面探针测"接近/抓取 pred chunk 位移/GT"——若从 60-80% 回到接近 100%(欠到位消失)→ **D5 坐实为真因**;否则查次要嫌疑。
  - ⚠️ **真机验证还需对齐执行时序**:30 anchor 现表示 2s 运动,部署执行须按 2s 时序(非 30Hz 稠密),否则 2× 过快。offline 先验证表示假说,真机时序为下一步。
- 次要嫌疑:官方 Soft-Fold 可能 **co-train 在 290K 多域语料**(享跨本体先验),我们是**单域 finetune-from-base on 811ep** → 配方差异。D5 修复后若仍不足再查。
- ⭐ **数据隔离实验(已规划)**: 官方 Soft-Fold(已迁 gf0,§5 证数据/处理正确)vs 我们数据,**同 X-VLA-base + 同参数 + 同 D5-pipeline + 同部署** → 若官方能 fold、我们不能,则**我们的 demonstration 数据本身有问题**;若都不能,则 pipeline 有 bug。详细 plan 见 [`../future_plans/plans/xvla_track_x_curriculum.md`](../future_plans/plans/xvla_track_x_curriculum.md) §0.NEW.7。

---

## 7.5 ⭐ D5 对照训完 — offline 实测复盘 (2026-06-08):D5 ≠ 根治,降级

> `X3C_smooth800_d5anchor`(单变量 vs `X3C_smooth800_p0`,仅 `action_qdur=2.0`)训完 60k。跑两个 offline:matched-protocol MAE + §7.4 预注册的欠到位探针。**脚本**:`eval_xvla_ee6d.py --action-qdur`(新增) + `_xvla_gripper_debug/probe6_d5anchor_underreach.py`。**完整表见** curriculum §0.NEW.7。

### 7.5.1 证伪"错配放大 MAE"(旧 §0.NEW.7 解释)
旧版称 d5anchor 的高 MAE 是"anchor-pred vs dense-GT 失配的假象"。实测:同 val 同窗口把 GT 也换成 anchor 2s(`--action-qdur 2.0`)→ **MAE 几乎不变**(@1 dense 0.0387 vs anchor 0.0386;@30 0.1006 vs 0.1007)。→ 高 MAE **是真实的,非 GT 标尺假象**。(d5anchor 仍不宜与 p0 比绝对值,但真因是"2s horizon 物理跨度 2×、长程更难",非标尺错。)

### 7.5.2 ⭐ 欠到位探针 — D5 没治好"够不到衣角"
每模型按各自原生 GT,高/低动量窗口由 dense-GT 位移分桶、同 f_idx:

| 模型 / 原生 GT | 接近/抓取 pred/GT | 持握 pred/GT | pred lag1(GT) |
|---|---|---|---|
| X3C_p0 (dense 1s) | **0.71** 欠到位 | 8.75 乱飘 | 0.66 (0.77) |
| d5anchor (anchor 2s) | **0.70** 欠到位 | 1.61 | 0.85 (0.84) |

- **§7.4 预注册判据(欠到位回 ~100%)未达成**:接近/抓取 0.71→0.70,**几乎没动 → D5 不是抓取欠到位的根治**。
- D5 **确实改善**:持握乱飘 8.75→1.61、时序震荡 pred lag1 现合 GT(0.85 vs 0.84,p0 偏抖)。绝对指令位移 0.81m→1.22m(+51%,2s 时序回放下绝对到达更远,真机或仍有正作用)。

### 7.5.3 根因重定位(再修)
- **D5 降级**:主因 → "部分改善"(治乱飘/震荡,不治欠到位)。
- **接近欠到位在 dense 与 anchor 两种动作表示下都 ~0.70 → 与动作表示无关**。叠加 R4 容量已排除(官方同 0.9B 能 fold)→ **头号嫌疑 = 配方/数据**:我们单域 finetune-from-base on 811ep vs 官方 **290K 多域 co-train**。
- **→ Exp-O(官方 Soft-Fold)vs Exp-S(我们数据)数据隔离实验从"佐证"升为头号下一步**(§0.NEW.7);真机 fold 须 2s anchor 时序回放 + gripper 各自实测行程。

---

## 8. ⭐⭐ 深入:数据差异 vs 训练 regime 差异 — 为什么训练不对 (2026-06-08)

> 目标:欠到位(接近/抓取 pred/GT ~0.70)+ 持握过冲 = **动态范围压缩 / 回归均值** 的特征(模型对"该走多远"条件化不足,朝动作幅度的边际均值压)。在 R4 容量排除、D5 排除后,逐项查"配方"与"数据"两条。**结论:配方结构不是区分点,矛头落在数据(质量/覆盖)或 base ckpt/pipeline,Exp-O 为决定性检验。**

### 8.1 soft-prompt / domain 槽 freshness(直接 introspect base ckpt)

`lerobot/xvla-base`(= X-VLA-Pt 基座)的**全部 30-行 domain 嵌入**(`soft_prompt_hub` + `action_encoder.fc/bias` + `action_decoder.fc/bias`,`models/transformer.py:245,395`)一致显示:**只有 slot 10–17 被训过**(norm 显著高;bias≠0),**slot 5 / 18–29 全是 fresh**(near-init;`action_encoder/decoder.bias` 这些行**精确=0.000**,即从未接收梯度)。

| domain_id | 含义(`xvla/X-VLA/datasets/domain_config.py:41`) | base 中状态 |
|---|---|---|
| 5 | **AIR-AGILEX-HQ**(eef_6d 20D,= 我们数据格式) | **FRESH**(bias=0) |
| 10 | AIR-AGILEX(eef_quaternion 16D) | ✅ 训练过 |
| 16 | robomind-agilex | ✅ 训练过 |
| 20 | 我们的 vis(自定义) | **FRESH**(bias=0,与 5 等价) |

### 8.2 ⭐ 关键反转:fresh-slot **不是**我们独有的问题
官方 **X-VLA-SoftFold(cloth fold 100% 成功)部署用 `domain_id=5`**(`evaluation/SoftFold-Agilex/deploy/client_eef6d_xvla.py:116`,且是 `client_eef6d_xvla` = eef_6d 20D,**和我们同格式**)。而 slot 5 在 X-VLA-Pt 里**和我们的 slot 20 一样是全新空槽(bias 精确=0)**。
→ **官方也是"从 fresh 槽 + 单域 finetune-from-Pt"**,与我们(slot 20)**结构完全相同** → **"cold-start soft-prompt / 单域 finetune"不可能是我们失败的原因**(否则官方也会失败)。先前 Explore agent 把此列为头号嫌疑 → **经 ckpt 实测证伪,撤销**。
> 注:slot 10(AIR-AGILEX,已训)用 quaternion 表示;官方 SoftFold 故意改用 fresh 的 HQ 槽(5)+eef_6d → 说明**fresh 槽对官方 work 没障碍**。domain_id 5 与 20 在此 base 中等价(都 fresh),Exp-O 用 21 也无碍,只要 train/deploy 一致即可。

### 8.3 数据分布实测对比(官方 Soft-Fold vs 我们 smooth_800,各采 50 ep)
脚本 `/tmp/data_cmp.py`(eef_6d / EE6D action 的 xyz):

| 指标 | 官方 Soft-Fold | 我们 smooth_800 | 解读 |
|---|---|---|---|
| ep 时长 | **58.2s**(1747帧) | 32.7s(980帧) | 我们 ~**½ 长**(更快完成/更少帧) |
| 中位 xyz 速度 | 21.0 cm/s | 14.7 cm/s | 我们更慢(中位) |
| p90 / p99 速度 | 56.2 cm/s / 0.040 | **60.6** cm/s / **0.045** | 我们峰值更快 |
| 2s chunk 位移 中位/p90 | 0.43 / 1.02 m | 0.45 / **1.20** m | **我们大动作不少反更多** |
| idle 占比(<1.5cm/s) | 7.1% | **12.6%** | 我们 idle ~**2×** |

→ **"我们数据缺大动作"被排除**:2s 位移分布相当、p90 我们更大。**欠到位 = 模型学不出数据里已有的大 reach**(GT 高动量位移 1.1–1.7m,pred 只 70%),非数据没有。数据侧真实差异 = **ep 更短 + idle ~2×**,以及**未测的 demonstration 一致性/多模态/覆盖**(本探针测不到)。

### 8.4 base ckpt 溯源 — 确认正版 X-VLA-Pt(排除"base 错"嫌疑)
查 `xvla/xvla_ckpts`:model card = **`lerobot/xvla-base`**(README 明写"Reference impl: github.com/2toinf/X-VLA;LeRobot impl follows original";下载 cache HF commit `cdb7964`)。三重证据它就是**官方 X-VLA-Pt 基座**(lerobot port):
1. **参数量一致**:HF `lerobot/xvla-base` 与 `2toINF/X-VLA-Pt` 都 **879.7M**,同 arXiv 2510.10274,同 Florence-2-large;Pt 自称 "Foundation Edition"。
2. **权重指纹(决定性)**:base 训练过的 domain 槽 **正好 = 10–17**,精确对应 `domain_config.py` 标注的 **"pretraining" 域(robomind×4 / Droid×2 / AGIBOT = 11–17)+ AIR-AGILEX=10**;所有 **"# ft" 域(Bridge=0…HQ=5…AIRBOT=18)全 fresh**。这只可能是**未经任何下游 finetune 的纯 Pt 基座**(若是某 ft 版,对应 ft 槽会被训过)。
3. → **"base ckpt 不对/残缺"嫌疑排除**。`xvla-base` = 正版 X-VLA-Pt,与官方 SoftFold 同起点。

### 8.5 根因落点(本次净结论)
排除链:R4 容量(官方 0.9B 能 fold)❌ → D5 动作表示(dense/anchor 都欠到位 0.70)❌ → fresh-slot/单域配方(官方同样 fresh 槽 5 单域 finetune)❌ → base ckpt(确认正版 X-VLA-Pt,§8.4)❌。**剩下两条**:
1. **数据质量/覆盖/一致性**(非幅度;leading):我们 811ep vs 官方 1532ep;更短、idle ~2×;state→action 条件可预测性可能更低 → 弱条件化 → 范围压缩。
2. **pipeline 偏差**(非 base 权重):训练/serve 链路某处未发现的 bug(图像/anchor/归一/IK 时序等),非基座问题。

→ **Exp-O 升为决定性检验**,等价于"**用我们(已确认正版)的 base+pipeline+配方能否复现官方 100% SoftFold**":
- Exp-O(官方数据)**fold** + Exp-S(我们数据)**不 fold** → **我们的 vis 数据质量/覆盖是问题**(配方/base/pipeline 没问题)。
- Exp-O **也不 fold** → **pipeline 有 bug**(base 已确认正版 §8.4 + 数据+配方都对齐官方仍失败)→ 逐行核训练/serve 链路。
- 两者都 fold → 之前真机失败 = 部署侧(gripper 映射 / 2s 时序回放)。

> 下一步(按性价比):① 等 Exp-O 训完,同样跑欠到位探针 + 真机 fold(domain/gripper/2s 时序对齐);② base 溯源**已完成**(§8.4 确认正版 X-VLA-Pt);③ 数据侧补测 demonstration 一致性(同状态下动作的多模态/方差)。

---

## 附录 — 关键文件:行

| 项 | 位置 |
|---|---|
| **官方 SoftFold 部署 domain_id=5 (eef_6d)** | `xvla/X-VLA/evaluation/SoftFold-Agilex/deploy/client_eef6d_xvla.py:116` |
| **domain_id 映射 (AIR-AGILEX=10, HQ=5)** | `xvla/X-VLA/datasets/domain_config.py:41-63` |
| **base 溯源 = 正版 X-VLA-Pt** | `xvla/xvla_ckpts/README.md`(lerobot/xvla-base)+ HF 879.7M 同 `2toINF/X-VLA-Pt`;权重指纹槽 10-17=Pt 预训练域 |
| **base ckpt domain 槽 freshness 检查** | `_xvla_gripper_debug/check_softprompt.py` + `check_alldomain.py`(只 10-17 训练) |
| **数据分布对比脚本** | `_xvla_gripper_debug/data_cmp.py`(官方 58s/idle7% vs 我们 33s/idle13%) |
| **官方动作 anchor(intention abstraction)** | `xvla/X-VLA/datasets/domain_handler/base.py:152` (`linspace(cur,cur+qdur,N+1)`) + `real_world.py:40` (`qdur=2.0`) |
| **我们动作 chunk(30 连续帧 1s)** | `train_scripts/xvla/data/multi_domain_dataset.py:162-163` |
| 我们训练 forward (绕过 processor) | `train_scripts/xvla/launch/xvla_train.py:330` |
| 我们 dataset 图像 /255 无归一化 | `train_scripts/xvla/data/multi_domain_dataset.py:157` |
| lerobot 归一化在 processor | `lerobot/policies/xvla/processor_xvla.py:73,349` |
| lerobot forward 只 resize | `lerobot/policies/xvla/modeling_xvla.py:313-320` |
| 官方 dataset 归一化 | `xvla/X-VLA/datasets/dataset.py:81` |
| 官方 finetune 50k/warmup2000 | `xvla/X-VLA/README.md:273-275` |
| EE6D→world→IK 真机链 | `kai0/scripts/serve_policy_xvla.py` (`_ee6d_to_world8`) + `ros2_ws/.../policy_inference_node.py` (IK) |
| pi05 同标尺对比 | `xvla_track_x_curriculum.md` §0.NEW.2.5b |
