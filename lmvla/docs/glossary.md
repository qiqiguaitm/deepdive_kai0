# LMVLA 术语锁

> 全项目（docs / figures / configs / 运行时 summary）统一命名。歧义命名一律不用。

## 核心对象

| 术语 | 定义 | 别名/禁用 |
|---|---|---|
| **milestone** | 跨 episode 重复出现的技能相位（KMeans 簇 + 顺序化后的离散状态）；比时间多解释 2× 动作方差 | 不叫 "phase"/"stage" 混用；stage 仅指 milestone 的序号 |
| **milestone 图** | milestone 之间的转移/先后结构（precedence / transition matrix） | — |
| **prototype（latent）** | 某 milestone 的代表 latent（簇心或 episode-local medoid）；在 SigLIP/DINOv3 空间，可作视觉子目标 | — |
| **value / progress** | CRAVE 从 milestone 序列 readout 的每帧标量进度（离散 V2.4 / 连续 TCC） | — |
| **LMWAM** | 最终系统 = **LMWM × kai0 π0.5**，action expert 被 latent milestone 引导 | 不是子项目目录名（那是历史命名）；目录用 `lmvla/` |

## 预测规则（**严格区分，勿互换**）

| 术语 | 定义 |
|---|---|
| **Greedy** | 一步局部预测：`argmax P(stage_{t+1} \| stage_t)` |
| **Max-product** | 有限视野 DP / max-product 搜向终点 milestone，报告路径上的**下一步** |

> ⚠️ 一步规则**不叫** "Max-Probability Milestone"——该措辞歧义、已弃用（会把 Greedy 与 Max-product 两个含义对调）。

## 注入 / 训练

| 术语 | 定义 |
|---|---|
| **KI（Knowledge Insulation）** | 保护预训练：连续 action expert → backbone 的 K/V 套 `stop-grad`，**非冻 backbone** |
| **prefix 虚拟图像 token** | milestone prototype 经原图像投影，作"虚拟未来图像 token"进 VLM prefix（主注入方案 P） |
| **GT-first** | 先用 GT milestone 验证注入机制本身，再换真预测器（隔离预测误差污染） |
| **hybrid fallback** | 神经预测 + 图规划先验的兜底（`*_fallback_mask`）；fallback 是**规划先验**非独立学习修正 |

## 频率 / 处理约定

| 术语 | 约定 |
|---|---|
| **30Hz 原生处理** | 所有新 episode 推理直接在 30Hz 特征上跑；不再用 3Hz stride=10 → upsample（会引入 aliasing）。KMeans 模型仍可 3Hz 训 |
| **DC** | distance correction（α=2.0），在线 readout 的一环 |
| **sym adaptive vote** | 对称自适应投票（wd=10, t=0.3），削尖峰 |

## 编码器

| 名 | 说明 |
|---|---|
| **DINOv3-H** | 默认编码器；bf16（fp16 溢出），res 256 → 16×16 grid；跳过 1 CLS + 4 register token 再 pool |
| 可选 | `dinov2-{small,base,large}` / `dinov3-7b{,-int8}` / `wan-vae`（`crave.config.encoders.ENCODERS` 加一行即可扩） |
