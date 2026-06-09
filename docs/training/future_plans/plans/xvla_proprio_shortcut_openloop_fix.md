# X-VLA proprioception 捷径 / vision-blind 开环 — 根因认证与修复训练规划

> **目的**: 认证"X-VLA 真机不抓衣服、固定动作"的根因, 给出 probe 证据 + 可执行的修复训练方案 + 离线门禁。
> **建立**: 2026-06-09 · **方法**: 离线 vision-ablation (复用 serve infer 路径重放真机 trace, 固定 seed + 关 proprio-feedback), 不训练。
> **关联**: [`xvla_track_x_curriculum.md`](xvla_track_x_curriculum.md) (Track X, p0/d5anchor 来源) · [`xvla_camera_robust_grasp_final.md`](xvla_camera_robust_grasp_final.md) (相机 gap, 假定模型读视觉的前提被本文推翻) · memory `reference_xvla_vision_blind_openloop` / `reference_vision_ablation_openloop`。
> **诊断工具**: [`train_scripts/kai/eval/eval_xvla_vision_ablation_offline.py`](../../../../train_scripts/kai/eval/eval_xvla_vision_ablation_offline.py)

---

## 0. 结论 (一句话)

**整条 X-VLA smooth800 管线 (p0 + d5anchor) 训练就是纯开环 (vision-blind): 动作是 proprioception 的纯函数, 三路相机像素对输出的影响 = 0.000。不是数据问题、不是部署问题、不是 qdur/归一化问题。在修好 `use_proprio` 捷径前, 换任何数据/qdur/norm 训出的 X-VLA ckpt 都会瞎。**

这也推翻了 Track X 之前的诊断链 ("X3.C 真机失败 = R1 缺 ImageNet 归一化 → 重训 p0 修复"): p0 已修 R1 + ImageNet 归一化, 真机**依旧失败**, 因为真根因是 proprio 捷径, R1 只是表层。

---

## 1. 认证证据 (离线 vision-ablation)

方法: 用 `eval_xvla_vision_ablation_offline.py` 复用 **serve 的 `XVLAServerPolicy.infer` 路径** (预处理与真机逐字节一致), **固定 seed + `proprio_feedback=OFF`** → 每次推理是 `(image, state)` 的独立确定函数。从真机 `--trace` dump 重放真实 obs, 做三种扰动比对 action chunk:

| 扰动 | d5anchor (trace 11:55) | p0 (trace 06-07) | 含义 |
|---|---|---|---|
| **换一张完全不同的图** (state 不变) | xyz **0.03mm** / grip 0.0000 | xyz **0.03mm** / grip 0.0000 | 图像内容对动作无影响 |
| **整张图置黑** (state 不变) | xyz **0.07mm** / grip 0.0000 | xyz **0.08mm** / grip 0.0000 | 删掉视觉对动作无影响 |
| **换 proprio state** (图不变) | xyz **311mm** / grip 0.041 | xyz **248mm** / grip 0.012 | 本体一动, 动作大变 |
| **视觉/本体影响比 (d_img/d_state)** | **0.000** | **0.000** | →0 = 纯开环 vision-blind |

> 对照健康基线: 同一批 smooth800 数据上, **pi0** 之前测出 blank/real MAE = **13.6×** (视觉健康, 见 `reference_vision_ablation_openloop`)。X-VLA 在同数据上是 **0.000**。

---

## 2. 已排除的假设 (为何不是数据/部署/时序)

| 假设 | 排除证据 |
|---|---|
| **部署 bug (图没喂进模型)** | `config.image_features` = serve 喂的三键 `observation.images.image/image2/image3`; `num_image_views=3`, `empty_cameras=0` → 三路全消费、无 zero-pad; `resize_imgs_with_padding=(224,224)` 对齐。`_prepare_images` 不报错 = 键命中。**全部三路置黑仍 0 变化** → 被消费的视图确实无影响。 |
| **数据质量 (5-19~5-27 漂移)** | p0/d5anchor 用的是 **smooth800 好数据 (04-23~05-09)**, 非漂移期。同数据 pi0 视觉健康。 |
| **qdur / publish_rate 时序** | 已修正: d5anchor `publish_rate:=15` 后速度回到 1.00× (34/35mm/s), 行为**不变**, 仍不读视觉。时序与视觉盲是两件独立事。 |
| **ImageNet 归一化 (R1)** | p0 已修 R1, sidecar `image_norm=imagenet`, ablation 用 `imagenet_norm=True`, 仍 0.000。 |
| **采样随机性** | seed 固定, 三扰动同 seed; state 扰动能产生 248~311mm 变化 = 推理管线本身正常响应输入。 |

---

## 3. 根因

折叠任务**高度程式化** (从 home 位姿走固定折叠序列) + **单 domain(20)** + `configuration_xvla.py:85 use_proprio=True` 把 `state20` (EE6D) 当强条件输入 → 模型用 proprioception 就能把整条轨迹拟合到低 loss, 训练梯度没有动力流向 Florence2 视觉编码器。这是 **causal confusion / 开环捷径**的教科书案例。

同时解释了长期困惑"MAE 没问题但真机不可用": **开环复述本体轨迹本就能达到低 MAE / 低 val loss**, MAE 完全测不出视觉盲 (见 `feedback_offline_eval_protocol` / `feedback_real_machine_oscillation_data_tail`)。

为何 X-VLA 瞎而 pi0 不瞎 (同数据): 待证, 假设是 X-VLA 的 proprio 条件强度 / action 表征 (绝对 EE6D, 首步≈当前 proprio) 让捷径更易赢; pi0 的 state 条件较弱或表征不同。

---

## 4. 修复训练规划

`lerobot` XVLAConfig **只有 `use_proprio` 总开关, 无 proprio-dropout 部分遮蔽**。按代价/收益:

### 4.1 实验 E1 — 确诊性 A/B: `use_proprio=False` (先做, 最快)
- **做法**: 复制 p0 训练 config, 仅置 `use_proprio: false` (关 state 输入), 其余 (数据 smooth800 / 60k / lr 1e-4 / ImageNet norm) 完全不变。
- **目的**: 强制模型只能用视觉。是**确诊**而非最终模型 — 验证"只要拿掉 proprio 捷径, 模型就会读视觉"。
- **判据**: 训完跑 `eval_xvla_vision_ablation_offline.py`, **视觉/本体影响比从 0.000 抬到 ≳0.5** = 根因坐实。再上真机看是否会"找衣服"。
- **风险**: 丢掉 proprio 平滑信号, 连续控制可能更抖 (可叠 publish-time EMA 缓解)。

### 4.2 实验 E2 — proprio-dropout (更好的最终模型, 需 patch)
- **做法**: patch lerobot XVLA 训练, 对每个样本以概率 `p_drop` (起步 0.5) 把 `observation.state` 置零/替换占位 token; 推理仍可给 proprio。
- **目的**: 保留 proprio 收益 (平滑/精度) 又强制模型读视觉。causal confusion 的标准解。
- **依赖**: E1 确诊为 proprio 捷径后再投入实现成本。
- **判据**: 同 E1 门禁 (比值 ≳0.5) + 真机抓取成功率 ≥ E1。

### 4.3 实验 E3 — 数据多样性 (治本, 最慢, 次选)
- **做法**: 采集/合成"同 proprio (同 home 起点) 下衣服位置不同 → 需不同抓取目标"的样本, 打破 proprio→action 确定性。
- **目的**: 即使有 proprio, 也强迫视觉成为消歧的唯一信息源。
- **判据**: 同门禁 + 跨衣服位置泛化。

### 4.4 对照矩阵

| 组 | use_proprio | proprio-dropout | 数据 | 终判 |
|---|---|---|---|---|
| **baseline (现状)** | True | — | smooth800 | 视觉比 **0.000** ❌ |
| **E1** | **False** | — | smooth800 | 视觉比 ≳0.5? + 真机找衣服? |
| **E2** | True | **0.5** | smooth800 | 视觉比 ≳0.5 + 真机成功率 |
| **E3** | True/dropout | 0.5 | + 位置多样 | 跨位置泛化 |

---

## 5. 离线门禁 (新增, 强制)

**以后任何 X-VLA ckpt 上真机前, 必须先跑离线 vision-ablation, 视觉/本体影响比 ≳ 0.5 才放行** (类比 pi0 的夹爪 SNR≳15× 门禁)。MAE / val loss **不作为视觉依赖判据** (测不出开环)。

```bash
CUDA_VISIBLE_DEVICES=<free> kai0/.venv_xvla/bin/python \
  train_scripts/kai/eval/eval_xvla_vision_ablation_offline.py \
  --trace /tmp/xvla_stack/trace_<ts> \
  --ckpt /data1/DATA_IMP/checkpoints/ckpt_xvla/<ckpt> --n 12
```
读 "视觉/本体影响比": →0 = vision-blind 禁止上机; ~1 = 健康闭环。

---

## 6. 行动顺序

1. **停止盲调 X-VLA ckpt** (换数据/qdur/norm 都不会改变 vision-blind)。
2. 跑 **E1 (`use_proprio=False`)** 做确诊 → ablation 比值是否抬起。
3. 比值抬起 + 真机会找衣服 → 实现 **E2 (proprio-dropout)** 做最终模型。
4. 把 §5 门禁纳入 X-VLA 上机流程。
