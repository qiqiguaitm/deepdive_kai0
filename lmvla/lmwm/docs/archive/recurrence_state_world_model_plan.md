# LMWM 规划:循环状态世界模型

## 1. 目标

在 CRAVE 循环 milestone 状态之上训练第一个 CRAVE-native 世界模型。

目标不是像素预测,也不是完整 VLA 策略。第一个目标是训练一个紧凑模型,消费当前任务感知的 CRAVE 状态并预测:

- Greedy next(贪心下一 milestone):单步局部最大,`argmax P(stage_{t+1} | stage_t)`;
- Max-product next(最大积下一 milestone):向终点/完成 milestone 的有限 horizon 最高积路径上的下一步;
- 候选未来 milestone 的分布;
- 路径级概率或置信度分数,后续可成为 VLA 上下文。

该模型是以下目标的第一步:

```text
CRAVE recurrence states -> Latent Milestone World Model -> latent subgoal / planning prior -> VLA action model
```

### 1.1 术语锁定

两个下一 milestone 输出固定如下:

- `Greedy`:单步局部预测,`argmax P(stage_{t+1} | stage_t)`。它回答:"从当前阶段,最可能的直接下一阶段是什么?"
- `Max-product`:向终点 milestone 的有限 horizon 动态规划/最大积搜索。它回答:"哪一直接下一阶段位于到完成的最高积路径上?"

这两个名称不应互换。避免在用户可见文本中使用 `Max-Probability Milestone`,因其听起来像单步 argmax 造成歧义。

## 2. LaWAM 参考点

LaWAM 有用是因为它避免了直接像素未来预测。它在冻结的视觉特征空间中预测未来观测特征,并将其注入为动作生成的隐变量视觉子目标。

从 LaWAM 项目页面,相关的设计主张是:

- 不为控制重建未来视频;
- 在冻结编码器空间(如 DINOv3)中预测一个紧凑的隐变量视觉子目标;
- 保持推理非迭代、低延迟;
- 先学习世界模型,再让策略预测/使用其隐变量转移码。

本地复制参考代码:

```text
lmwm/vendor/LaWAM/
```

概念上值得复用的部分:

- `latent_action_model/`:在冻结视觉特征上的隐变量未来表示学习。
- `starVLA/dataloader/latent_world_train_collator.py`:隐世界输出如何与 VLA 训练样本打包。
- `starVLA/config/training/*.yaml`:运行配置风格。
- `train_lawam.sh` 和 `train_lawam_distributed.sh`:环境设置、运行目录、配置快照、日志和分布式启动模式。

不应直接复制到第一个 LMWM 原型中的部分:

- Qwen/VLA 训练栈。
- LIBERO/RoboTwin 仿真器特定适配器。
- LAM 图像 token 重建目标。

LMWM 应借鉴接口思路,而非精确的目标函数。我们的隐变量是 CRAVE 的任务感知 milestone 状态,而非通用视觉未来 token。

## 3. 本地项目结构

`lmwm` 目录遵循本地 `kai0` 项目风格:

```text
lmwm/
  README.md
  pyproject.toml
  configs/
    datasets/
    models/
    training/
  data/
  checkpoints/
  logs/
  docs/
  scripts/
  src/lmwm/
  vendor/LaWAM/
```

规则:

- 保持上游 LaWAM 不修改于 `vendor/LaWAM`。
- 仅将一手代码放在 `src/lmwm` 和 `scripts`。
- 大数据集和模型 checkpoint 不纳入源代码控制。
- 每次训练运行应将其 YAML 配置快照到运行日志目录。

## 4. CRAVE 数据接口

第一个训练数据集应从已有 CRAVE 产物构建。

所需逐帧字段:

- `episode_id`
- `frame_index` 或特征时间步
- `milestone_id`
- `milestone_progress`
- `milestone_latent_prototype`
- 可选的解码 prototype 图像路径
- 可选的 DINO/state 特征
- 可选的 action chunk 或机器人状态增量
- 如果可用,完成标志 / 端点质量标志

初始候选源:

- `crave` 实验输出,如 `_cache.npz`、`Pord`、`nm30`、milestone 中心、解码中心、转移矩阵。
- 已计算 milestone 序列和 Viterbi 读出的 CRAVE 脚本。

导出格式先保持简单:

```text
data/crave_sequences/<dataset_name>/
  manifest.jsonl
  sequences.npz
  prototypes.npz
  split.json
```

## 5. 模型范围:第一版

第一版应尽可能贴近 LaWAM Stage-1 LaWM。不要直接跳到最终的图/规划器模型。首要目标是复用相同的学习形状:

```text
LaWM:  z_t, z_{t+h} -> inverse transition code u_t
       z_t, u_t     -> predicted future visual latent z_hat_{t+h}

LMWM:  r_t, r_{t+h} -> inverse transition code u_t
       r_t, u_t     -> predicted future milestone latent r_hat_{t+h}
```

其中 `r_t` 是 CRAVE 循环状态隐变量,而非原始图像特征。

这保留了输入/输出维度和训练逻辑接近 LaWM,同时仍使用 CRAVE 的 milestone 抽象。最终的转移图和 Viterbi / max-product 规划器应在首个 LaWM 形状模型工作后再添加。

### 5.1 Stage-1A:LaWM 形状的 LMWM

使用以固定 horizon `h` 分隔的 CRAVE 状态对:

```text
current state:  r_t
future state:   r_{t+h}
```

推荐表示:

```text
r_t = concat(
  milestone_prototype_latent_t,   # 投影到 code_dim,如 32 或 64
  milestone_id_embedding_t,       # 可选,相同 code_dim 尺度
  progress_scalar_projection_t,   # 可选小投影
  robot_state_projection_t        # 可选,仅当稳定时
)
```

首次运行保持最小:

```text
r_t = projected milestone_prototype_latent_t
r_{t+h} = projected milestone_prototype_latent_{t+h}
code_dim = 32 or 64
num_queries = 1
```

这有意镜像 LaWAM 的 LAM 配置,其中 `code_dim=32` 且 `num_queries=1`。如果 CRAVE prototype 隐变量是高维的,在世界模型前加一个小线性投影器,使可见模型接口保持 LaWM 风格。

架构:

```text
r_t ------------------------------+
                                   v
r_t, r_{t+h} -> Inverse Encoder -> u_t
                                   |
                                   v
                      Forward Decoder(r_t, u_t)
                                   |
                                   v
                            r_hat_{t+h}

loss: distance(r_hat_{t+h}, r_{t+h})
      + optional CE(classifier(r_hat_{t+h}), milestone_{t+h})
      + optional state/progress auxiliary loss
```

核心模块:

- `StateProjector`:将 CRAVE prototype / id / progress 映射到固定的 `code_dim`。
- `InverseTransitionEncoder`:从 `(r_t, r_{t+h})` 预测紧凑转移码 `u_t`。
- `ForwardStateDecoder`:从 `(r_t, u_t)` 预测 `r_hat_{t+h}`。
- `MilestoneClassifier`:将 `r_hat` 映射到 milestone logits 的可选头。

为什么先做这个:

- 匹配 LaWM 的逆向+正向训练配方。
- 可复用 LaWAM 紧凑转移码的思路。
- 避免同时解决图规划、历史建模和 VLA 集成。
- 产出一个未来隐变量子目标,后续可类似 LaWAM 注入到策略中。

### 5.2 Stage-1B:分类对齐输出

因为 CRAVE milestone 是离散的,仅隐变量回归不够。添加 milestone id 分类器:

```text
r_hat_{t+h} -> logits over M milestones
```

损失:

```text
L = L_latent(r_hat_{t+h}, r_{t+h})
  + alpha * CE(logits, milestone_{t+h})
```

这自然地处理了多对一性质。多个当前状态可以产生分类为同一未来 milestone 的隐变量。模型仍是 LaWM 形状,但评估可以用 milestone 准确率和图行为。

### 5.3 Stage-1C:从固定 horizon 到下一 milestone

在固定 horizon 预测工作后,用 `h = next unique milestone` 替代固定帧偏移训练相同架构:

```text
r_t -> r_{next_unique_milestone}
```

这是 LMWM 开始从 LaWM 分叉的第一个点:目标不再只是未来时间偏移,而是下一个循环任务阶段。

### 5.4 Stage-2:转移分布与图

仅在 Stage-1A/B/C 稳定后,添加图风格输出:

```text
P(m_{t+1} | r_t, optional history/action)
P(done | r_t)
```

该阶段支持:

- 通过单步 `argmax P(m_{t+1}|m_t)` 的贪心下一 milestone;
- 通过学习的图上向终点 milestone 的有限 horizon Viterbi/最大积动态规划的最大积下一 milestone;
- 通过路径似然的异常评分。

Stage-2 是通向最终循环状态世界模型的路径,但不应是第一个实现。

## 6. Stage-1 输入/输出契约

### 6.1 训练样本

从 CRAVE episode 导出 pair 样本:

```text
episode_id
timestep_t
timestep_future
milestone_t
milestone_future
prototype_latent_t
prototype_latent_future
progress_t
progress_future
robot_state_t optional
action_chunk_t optional
```

从固定 horizon pairs 开始:

```text
future = t + h
```

然后添加 next-unique-milestone pairs:

```text
future = first tau > t where milestone_tau != milestone_t
```

### 6.2 模型输入

最小首次运行:

```text
x_current = prototype_latent_t
x_future  = prototype_latent_future
```

投影后:

```text
r_t       : [code_dim]
r_future  : [code_dim]
```

最小运行工作后可选的添加:

```text
milestone_id_embedding_t
progress_projection_t
robot_state_projection_t
action_chunk_projection_t
```

### 6.3 模型输出

Stage-1A 输出:

```text
u_t              : transition code [code_dim]
r_hat_future     : predicted future CRAVE latent [code_dim]
```

Stage-1B 输出添加:

```text
logits_future_milestone : [num_milestones]
```

输出解释:

- `r_hat_future` 是 LaWAM 兼容的隐变量子目标。
- `logits_future_milestone` 是 CRAVE 特定的离散落地。
- `r_hat_future` 的最近 prototype 给出用于调试的解码中心图像。

### 6.4 为什么维度保持接近 LaWM

保持第一个模型接近 LaWM 有用,因为:

- LaWAM 的 LAM 使用紧凑码(`code_dim=32`,`num_queries=1`),因此我们可以测试 CRAVE milestone 动态是否也适合通过小转移瓶颈。
- 固定的 `code_dim` 接口使后续 VLA 条件更容易。
- 我们可以在引入 CRAVE 特定图规划前借鉴 LaWAM 的逆向/正向训练模式。
- 如果小瓶颈失败,失败也是有信息的:CRAVE 循环动态可能需要历史、动作条件或多 query 码。

## 7. 评估

使用 held-out CRAVE episode。

指标:

- 单步 top-1/top-3 下一 milestone 准确率;
- 真实下一 milestone 的负对数似然;
- 完整 held-out milestone 序列的路径似然;
- 完整路径准确率:最大完成路径是否到达观测到的后续 milestone;
- 异常排序:合成回退/重排破坏的低概率;
- 子目标有用性:解码的下一 prototype 在视觉上合理并遵循任务顺序。

重要 sanity checks:

- 与经验转移图对比。
- 与无神经训练的原始 Viterbi milestone 序列对比。
- 报告 CRAVE milestone 歧义或循环的情况。
- 保持 Viterbi 平滑/显示平滑与训练标签分开。

## 8. 训练阶段

### Phase 0:仓库与数据审计

- 将 LaWAM 代码保留在 `vendor/LaWAM` 下。
- 映射序列导出所需的 CRAVE 产物。
- 确定规范标签名称:`progress`、`milestone_id`、`prototype`。
- 编写导出脚本骨架。

### Phase 1:序列导出

- 为 kai0_base 和 kai0_dagger 导出 milestone 序列。
- 按 episode 而非按帧保存 train/val/test 划分。
- 保存 prototype 隐变量和解码 prototype 元数据。
- 从导出序列重新计算经验转移图。

### Phase 2:非神经图基线

- 实现转移图估计器。
- 实现贪心和最大完成规划。
- 在 held-out episode 和合成破坏上评估。
- 产出可视化报告:当前 milestone → 预测下一 milestone。

### Phase 3:神经循环状态世界模型

- 训练 MLP/GRU 基线。
- 如果可用添加动作/状态条件。
- 与经验图对比。
- 导出模型 checkpoint 和推理 API。

### Phase 4:VLA 接口

- 将下一 milestone 隐变量 prototype 转换为策略条件对象。
- 导出 prompt/上下文字段:
  - 当前 milestone;
  - 贪心下一 milestone;
  - 最大完成下一 milestone;
  - 路径概率;
  - 用于调试的解码 prototype 图像路径(如果需要)。
- 保持此接口与后续 RPVLA/LaWAM 风格的动作模型兼容。

## 9. 风险与决策

- 根据定义,CRAVE progress 不是奖励。它可以转换为 progress 衍生的过程奖励,但模型应输出 `progress` 和 milestone 概率。
- Viterbi 是一个解码器/规划器,不是学习的世界模型。它对伪标签、图搜索和最大积规划有用。
- 显示平滑不得混入训练标签。在 milestone id 和图转移上训练;仅将平滑 progress 用于可视化。
- 循环 milestone 会使单步预测歧义。评估应包括 top-k 和路径级指标,而非仅 top-1。

## 10. 即时后续命令

脚本就绪后的计划命令:

```bash
uv run python scripts/export_crave_sequences.py \
  --config configs/datasets/kai0_crave.yaml

uv run python scripts/train_state_world_model.py \
  --config configs/training/kai0_lmwm_gru.yaml

uv run python scripts/eval_state_world_model.py \
  --checkpoint checkpoints/kai0_lmwm_gru/latest.pt
```
