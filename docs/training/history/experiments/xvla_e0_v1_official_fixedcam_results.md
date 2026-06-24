# E0_v1_official_FIXEDCAM 训练结果 — vision-blind 根因修复**验证**实验 (✅ 成功)

> 时间：2026-06-23 (训练 50k, volc 8×A100 dev 队列) ~ 2026-06-24 (离线视觉消融 + MAE 判定)
> 硬件：volc 8×A100-80GB (训练) · gf0 1×A100-80GB (离线消融)
> 训练任务：dev 队列 `t-20260623145309-8clks`，OUT `xvla/ckpts/xvla_e0_v1_official_fixedcam`
> 唯一变量 (相对 [E0_v1_official](xvla_e0_v1_official_results.md))：**修好 dataloader 视频路径 bug** (commit `3b46252`，分支 `main-pre-crave-backup`)。配方、数据、`use_proprio=True`、domain_id=20 完全不变。
> 相关：根因复盘 [`../xvla_blackimage_dataloader_lesson.md`](../xvla_blackimage_dataloader_lesson.md)

---

## 0. 一句话结论

**修复成功。** 把 `LeRobotEE6DDataset._video_path` 的全键/短名目录回退补上 (原 bug：用完整 feature key `observation.images.top_head` 拼视频目录，而 v1 数据集按短名 `top_head/` 存 → `av.open` FileNotFound → `__getitem__` 静默返回 `torch.zeros` 黑图)，**完全相同的官方配方 + v1 数据 + `use_proprio=True` 重训 50k → 模型第一次真正读视觉**：

- **视觉/本体影响比 `d_img/d_state` = 5.17 (xyz) / 10.56 (grip)** — 换图改动作 >> 换 state；旧 E0 (黑图训练) 是 **0.000**。
- 离线 MAE `@1=0.0305 / @30=0.0522` — vs 官方 soft_fold `@30≈0.0473` 仅 **1.1×**，vs 黑图 E0 `@30=0.1353` **好 2.6×**。

→ **推翻了旧 E0 复盘 (`xvla_e0_v1_official_results.md` §0) 把失明归因于 "proprio 早融合捷径 / 数据链架构链需同时切断" 的结论。** 真正的根因是**数据 loader 静默喂黑图**，与 proprio、与 `use_proprio` 开关无关。把图修对，`use_proprio=True` 照样学出强视觉闭环。

---

## 1. 实验设定

与 [E0_v1_official](xvla_e0_v1_official_results.md) §1 **逐项相同** (config `E0_v1_official`：v1 `A_v1_noRelabel_ee6d` 6 日期 / static-skip 后 ~581k sample / domain_id=20 / action≠state / use_proprio=True / 4group_official param-groups / VLM 10×LR bug 已修 / freeze 1000 / 50k step / eff bs 128 / lr 1e-4 constant / wd 0 / action_qdur=2.0 / ImageNet norm + ColorJitter / bf16 / static_skip)。

**唯一差异**：`train_scripts/xvla/data/multi_domain_dataset.py` 的 `_video_path` 现先试完整 key 路径、不存在再回退短名目录；`__getitem__` 的 except 由静默 `torch.zeros` 改为**响亮告警 + 计数** (`_decode_fail_count`)。修复后该 loader 喂的是**真实图**。

> smoke 验证 (训练前)：2k step smoke 全程 black-fallback 计数 = **1 / 581k** (单条孤立坏 mp4，非系统性)，确认 bug 已除。

---

## 2. 训练曲线

无 inline-eval MAE (该 trainer 只记 flow-matching loss + gnorm；且 MAE/loss 本就测不出 vision-blind)。flow loss 正常震荡收敛，gnorm 未发散，step 49,950 正常落 `step_final` (3.52 GB)。50k 吞吐 ~1.33 it/s → ~10.4h。

---

## 3. ⭐ 离线视觉消融 (决定性判定门禁)

方法同旧 E0 §3：[`eval_xvla_vision_ablation_dataset.py`](../../../../train_scripts/kai/eval/eval_xvla_vision_ablation_dataset.py)，数据 `A_v1_noRelabel_ee6d/2026-04-23`，gf0 A100，固定 flow-matching seed → 每次推理是 `(images, state)` 的确定函数。**关键：本次跑在修好的 loader 上 → 喂真实图。**

### 3.1 final (step 50000)，n=24

| 扰动 | xyz (mm) | gripper |
|---|---|---|
| 换图 hold state (`d_img`) | **327.77** | 0.4470 |
| 整图置黑 hold state (`d_blank`) | 272.23 | 0.5397 |
| 换 state hold image (`d_state`) | 63.43 | 0.0424 |
| **视觉/本体影响比 `d_img/d_state`** | **5.167** | **10.555** |

### 3.2 训练过程趋势 (n=12)

| step | `d_img` xyz (mm) | `d_state` xyz (mm) | 比值 xyz | 比值 grip |
|---|---|---|---|---|
| 4,000 | 225.9 | 122.0 | 1.85 | 3.26 |
| 24,000 | 319.6 | 141.8 | 2.26 | 3.23 |
| 50,000 (final) | 327.8 | 63.4 | **5.17** | **10.56** |

→ 视觉依赖**从训练极早期 (4k) 就已健康** (1.85 ≫ 0)，并随训练增强；`d_state` 反而下降 (122→63mm)，即模型越训越**偏向视觉、越少 proprio 开环复述**。这是健康闭环动力学。对照黑图 E0 全程 `d_img≈0.00 / 比值 0.000`。

---

## 4. 离线 MAE 对照

| 模型 | 数据 / loader | @1 | @30 | 备注 |
|---|---|---|---|---|
| 官方 soft_fold (Exp-O) | 官方 hdf5 真实图 | 0.0160 | 0.0473 | 上界基准 |
| **E0 fixedcam (本实验)** | v1 + **修好 loader (真实图)** | **0.0305** | **0.0522** | **vs 官方 @30 仅 1.10×** |
| E0 (黑图, 旧) | v1 + 坏 loader (全黑) | 0.0654 | 0.1353 | 失明，@30 差 2.6× |

→ 修好 loader 后 @30 从 0.1353 砍到 0.0522，逼近官方。残余 1.1× gap 属数据规模/质量层面 (官方 hdf5 vs 自建 v1 6 日期)，**已非失明级**。

---

## 5. Checkpoint 路径

```
gf0:/vePFS/tim/workspace/deepdive_kai0/xvla/ckpts/xvla_e0_v1_official_fixedcam/
  ├── config.json
  ├── step_002000 .. step_048000/ (每 2000 一存)
  └── step_final/state_dict.pt          (3,519,354,117 B, step≈50000, lerobot XVLAPolicy)
本地 (已拉取 + sidecar, 字节一致 3519354117):
  /data1/DATA_IMP/checkpoints/ckpt_xvla/xvla_e0_v1_official_fixedcam_step_final/
  ├── state_dict.pt   ├── config.json   └── sidecar.json
    (image_norm=imagenet, deploy_domain_id=20, use_proprio=true, vision_blind=false,
     vision_ablation 比值 5.167, offline_mae @1=0.0305/@30=0.0522)
```

部署：标准 XVLA server (`kai0/scripts/serve_policy_xvla.py`，已支持读 ckpt config.json 覆盖 use_proprio)，强制 domain_id=20，prompt "Flatten and fold the cloth."。

---

## 6. 结论 / 与对照实验

| 组 | 数据 action | use_proprio | loader | `d_img` xyz | `d_img/d_state` | 结论 |
|---|---|---|---|---|---|---|
| p0 / d5anchor | ≡state | True | 坏(黑图) | 0.03–0.08mm | 0.000 | 失明 |
| E1 (旧) | ≡state | False | 坏(黑图) | 0.00mm | 0.000 | 失明 |
| E0 (旧) | ≠state (v1) | True | 坏(黑图) | 0.00mm | 0.000 | 失明 |
| **E0 fixedcam (本)** | ≠state (v1) | **True** | **修好(真实图)** | **327.8mm** | **5.167** | **✅ 健康视觉闭环** |

**关键推论**：之前 p0/E0/E1 三组失明**全部是同一个 loader 黑图 bug** 造成的假象 (它们共享同一坏 loader)。`action≡state` 数据约定、proprio 早融合、`use_proprio` 开关**都不是真因**——把同样的配方喂真实图，`use_proprio=True` 第一次就学出强视觉依赖。这正是 [lesson 文档](../xvla_blackimage_dataloader_lesson.md) "共享同一上游 bug ≠ 独立验证" "loss 收敛 ≠ 输入健康" 的实证。

- ⭐ **下一步**：可上真机 A/B (这是第一个非失明的自建数据 X-VLA)。先过夹爪 SNR 门禁；@30 vs 官方 1.1× 说明动作质量接近，值得真机验证折叠闭环。
- ❌ 旧 `xvla_e0_v1_official_results.md` / `xvla_proprio_shortcut_openloop_fix.md` 的 "proprio 捷径" 结论已作废 (见各文 §0 banner)。
