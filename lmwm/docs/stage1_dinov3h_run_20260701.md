# LMWM Stage-1 DINOv3-H 运行 — 2026-07-01

## 目的

在真实 CRAVE DINOv3-H milestone prototype 上验证第一个 LaWM 形状的 LMWM 训练路径,而非早期的 one-hot smoke 表示。

这仍然是一个小步 prototype-to-prototype 实验。它验证数据导出、episode 划分、双 GPU 执行、checkpoint 保存和转移模型训练路径在 1280D CRAVE 特征上是否工作。

## 源特征

- 特征缓存:`temp/crave_full_dinov3h`
- 编码器:DINOv3-H
- 特征维度:1280
- 有效帧:334875
- Episode:3055
- Milestone 文件:`temp/crave_full_dinov3h/milestones_uniform_dinov3h.npz`
- Milestone prototype:37 个 DINOv3-H 簇中心

本轮未找到匹配的 DINOv3-H `kai0_dagger` 缓存。本次运行使用已有的全量 `kai0_base` DINOv3-H 缓存。

## 导出的数据集

命令:

```bash
python lmwm/scripts/export_dinov3h_milestone_pairs.py \
  --config lmwm/configs/datasets/kai0base_dinov3h_fixed.yaml

python lmwm/scripts/export_dinov3h_milestone_pairs.py \
  --config lmwm/configs/datasets/kai0base_dinov3h_next_unique.yaml
```

输出:

- `lmwm/data/crave_sequences/kai0base_dinov3h/pairs_fixed_h3.npz`
- `lmwm/data/crave_sequences/kai0base_dinov3h/pairs_next_unique.npz`

两个导出各包含 200000 对。每对为 LaWM 形状:

```text
current DINOv3-H milestone prototype r_t
future DINOv3-H milestone prototype r_{t+h}
current/future milestone id
episode_id 用于 episode 级 train/val 划分
progress_t / progress_future 作为 CRAVE progress 元数据
```

## 模型形状

Stage-1 模型有意遵循 LAWM 第一阶段模式:

```text
r_t, r_future -> inverse transition code u_t
r_t, u_t      -> predicted future latent r_hat_future
r_hat_future  -> future milestone classifier
```

这使得第一版保持接近 LAWM,避免直接跳到最终的循环图/规划模型。

## 训练命令

GPU0,固定 horizon Stage-1A/B:

```bash
CUDA_VISIBLE_DEVICES=0 python lmwm/scripts/train_state_world_model.py \
  --config lmwm/configs/training/kai0base_dinov3h_stage1ab_fixed.yaml
```

GPU1,next-unique Stage-1C:

```bash
CUDA_VISIBLE_DEVICES=1 python lmwm/scripts/train_state_world_model.py \
  --config lmwm/configs/training/kai0base_dinov3h_stage1c_next_unique.yaml
```

两个配置皆使用 `device: cuda:0`,因为 `CUDA_VISIBLE_DEVICES` 分配后每个进程仅看到其分配的物理 GPU。

## 结果

Fixed-horizon 运行:

- 运行目录:`lmwm/logs/stage1ab/20260701_140401+kai0base_dinov3h_stage1ab_fixed`
- Checkpoint:`lmwm/checkpoints/stage1ab/20260701_140401+kai0base_dinov3h_stage1ab_fixed`
- 训练对:159428
- 验证对:40572
- 最终步:1200
- Val loss:0.0082213071
- Val MSE:0.0070061434
- Val CE:0.0024303274
- Val top1:1.0
- Val top3:1.0

Next-unique 运行:

- 运行目录:`lmwm/logs/stage1c/20260701_140401+kai0base_dinov3h_stage1c_next_unique`
- Checkpoint:`lmwm/checkpoints/stage1c/20260701_140401+kai0base_dinov3h_stage1c_next_unique`
- 训练对:159428
- 验证对:40572
- 最终步:1200
- Val loss:0.0038437987
- Val MSE:0.0021513989
- Val CE:0.0033847997
- Val top1:1.0
- Val top3:1.0

## 解读

该步的高准确率是预期的,因为输入和目标都取自 37 个 milestone prototype 的有限表。此运行应被视为管线和架构验证,而非最终隐变量里程碑世界模型已解决的证据。

下一个有用的步骤是使预测问题不那么像查表:

- 在帧级 DINOv3-H 特征上训练,同时监督 milestone-prototype 目标;
- 添加多候选下一 milestone 似然而非仅一个目标;
- 按 held-out episode 和转移类型(尤其是低 support 转移)评估;
- 只有在此之后才引入循环图规划和 Viterbi / max-product 路径监督。

## 第二个划分管线检查:kai0bd

`kai0bd_feature_stage1` 现在在较小的 base+dagger 缓存特征划分上运行了相同的 LMWM 管线:501 episode(251 base-like / 250 dagger-like),45k 帧,796D 特征状态,64 milestone。它已导出 fixed-horizon 和 next-unique pair、pair 级循环图、Stage-1/2/3 checkpoint 和运行时摘要。这仍是管线验证,因为图标签是确定性表目标,但它证明了代码路径不限于原始的 kai0_base DINOv3-H 缓存。
