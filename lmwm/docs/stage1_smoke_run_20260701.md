# LMWM Stage-1 Smoke 运行 — 2026-07-01

## 资源划分

本机有 2 x NVIDIA A100-SXM4-80GB。

- GPU0:固定 horizon Stage-1A/B smoke 运行。
- GPU1:next-unique-milestone Stage-1C smoke 运行。

对于这个小模型,两个独立的单 GPU 运行比 DDP 更高效,因为模型仅有约 69k 参数,通信开销将占主导。

## 数据导出

源产物:

```text
temp/crave_interp_ep2302_30hz_decoded/_cache.npz
```

命令:

```bash
python lmwm/scripts/export_crave_sequences.py \
  --config lmwm/configs/datasets/ep2302_smoke.yaml

python lmwm/scripts/export_crave_sequences.py \
  --config lmwm/configs/datasets/ep2302_next_unique_smoke.yaml
```

输出:

- `lmwm/data/crave_sequences/ep2302_smoke/pairs_fixed_h30.npz`
  - 2930 对
  - 39 个 milestone
- `lmwm/data/crave_sequences/ep2302_smoke/pairs_next_unique.npz`
  - 2953 对
  - 39 个 milestone

重要局限:

```text
prototype_source = one_hot_progress_smoke
```

此 smoke 运行使用 `[one_hot(milestone_id), progress]` 作为 prototype 向量。它验证了 LaWM 形状的训练循环和文件管理,但尚不是完整的 CRAVE prototype-latent 实验。

## 训练运行

### Fixed Horizon h=30

命令:

```bash
CUDA_VISIBLE_DEVICES=0 python lmwm/scripts/train_state_world_model.py \
  --config lmwm/configs/training/ep2302_stage1ab_smoke.yaml
```

运行目录:

```text
lmwm/logs/stage1ab/20260701_135327+ep2302_stage1ab_smoke
```

Checkpoint 目录:

```text
lmwm/checkpoints/stage1ab/20260701_135327+ep2302_stage1ab_smoke
```

最终指标:

```json
{
  "step": 600,
  "train_loss": 0.04764937236905098,
  "val_loss": 0.040797843957008355,
  "val_mse": 0.014962628426557921,
  "val_ce": 0.05167043106090087,
  "val_top1": 0.9965870307167235,
  "val_top3": 0.9982935153583617
}
```

### Next Unique Milestone

命令:

```bash
CUDA_VISIBLE_DEVICES=1 python lmwm/scripts/train_state_world_model.py \
  --config lmwm/configs/training/ep2302_stage1c_next_unique_smoke.yaml
```

注意:当使用 `CUDA_VISIBLE_DEVICES=1` 时,进程将物理 GPU1 视为 `cuda:0`,因此配置使用 `device: cuda:0`。

运行目录:

```text
lmwm/logs/stage1c/20260701_135353+ep2302_stage1c_next_unique_smoke
```

Checkpoint 目录:

```text
lmwm/checkpoints/stage1c/20260701_135353+ep2302_stage1c_next_unique_smoke
```

最终指标:

```json
{
  "step": 600,
  "train_loss": 0.034386344254016876,
  "val_loss": 0.04841142873927421,
  "val_mse": 0.023584733324758897,
  "val_ce": 0.049653390331064584,
  "val_top1": 0.9983079526226735,
  "val_top3": 0.9983079526226735
}
```

## 解读

本轮已验证:

- 项目结构;
- 配置驱动的运行管理;
- CRAVE `_cache.npz` 到 pair 数据集导出;
- LaWM 形状的逆向转移编码器和正向解码器;
- milestone 分类头;
- 日志和 checkpoint 输出。

本轮尚未验证:

- 在真实 CRAVE prototype 隐变量上的学习;
- 跨 episode 泛化;
- 动作条件动态;
- 图规划或最大完成路径推理。

## 下一步

将 smoke prototype 向量替换为真实 CRAVE milestone prototype 隐变量。下一导出目标应包括:

```text
prototype_latent[milestone_id] -> r_t 原始输入
milestone_id / progress         -> 仅辅助元数据
episode 级 train/val 划分       -> 无随机帧划分泄露
```
