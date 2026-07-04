# LMWM 最优解码器交付 — flow-matching 像素解码器 (2026-07-03)

## 结论：`dec_best.pt` = 条件 flow-matching 像素解码器（原 `flow_b160`）

把模型预测的 pooled DINOv3-H 隐向量（1280-D, L2 归一化）解码成**锐利且保真**的图像。
它在 gf3 八卡上对 4 个配置（base 96/128/160 + 80k 数据）扫参后胜出。

## 客观排名（gf3 公共 held-out，`flow_eval.json`）

| 配置 | step | reencode_cos↑ | sharpness | pixel_L1 |
|---|---|---|---|---|
| **flow_b160 (交付)** | 24000 | **0.667** | 749 | 0.110 |
| flow_b128_80k | 30000 | 0.657 | 700 | 0.109 |
| flow_b128 | 24000 | 0.655 | 719 | 0.109 |
| flow_b96 | 24000 | 0.593 | 788 | 0.110 |
| *real* | — | — | *923* | — |

## 三方最终对比（本地公共 held-out，`final_decoder_compare.json` / `.png`）

| 解码器 | reencode_cos↑（语义保真） | sharpness | pixel_L1 | 视觉 |
|---|---|---|---|---|
| dec_v2 (L1) | 0.348 | 180（糊） | **0.070** | 条件均值模糊，无褶皱 |
| dec_gan_v2 (GAN) | 0.413 | 879 | 0.089 | 锐但**幻觉**（凭空造夹爪/纹理）|
| **flow_b160 (交付)** | **0.681** | 500 | 0.111 | **锐且保真**：衣物形状/颜色/褶皱、夹爪位置、桌沿都对 |

**为什么 flow 的 pixel_L1 反而更高**：L1 解码器专门最小化像素 L1 → 输出模糊的条件均值（L1 低但语义差 0.35）。
生成式解码器从图像分布采样，不做像素平均，所以 pixel_L1 高，但**语义保真度（reencode_cos）几乎翻倍**、且锐利。
对"展示模型自己的预测长什么样"这个目的，reencode_cos + 视觉锐度才是对的指标，pixel_L1 会误导。

**为什么 flow 是对的路线**：它在真实图像分布内采样，结构上无法产生对抗式噪声（这正是"再编码一致性损失"失败的原因——
直接优化冻结编码器输出会得到 cos 0.88 但视觉是高频垃圾）。flow 同时拿到 GAN 的锐度 + 超过 GAN/L1 的保真度，且无幻觉。

## ⚠️ 逐帧视频 / milestone 提示场景：为什么改用确定性解码器（2026-07-04 修正）

上面的排名是**单帧保真**的结论——flow 确实最锐利最保真。但**做 milestone 提示（把预测 milestone 解码成图给策略/人看）时最终没有用 flow**，原因是**时序稳定性**：

- 我预测的 milestone 是**分段常值信号**：连续多帧对应同一个 milestone（同一个目标隐向量）。
- flow 是**生成式**解码器，每次调用从新采样的噪声出发跑 ODE → 即使输入隐向量**完全相同**，逐帧解码也会得到**不同**的图 → 提示在同一个 milestone 内部**来回跳变 / 闪烁**。这对"稳定提示下一步该长什么样"是致命的。
- 确定性解码器（`dec_v2` L1，单次前向、无采样）：**同隐向量 → 同像素** → 同一 milestone 内提示**纹丝不动**，只在 milestone 切换时变化。这正是想要的行为。

**交付选择**：milestone 提示视频用 `dec_v2`（L1，确定性、保真、无幻觉，代价是糊）。`dec_best`（flow）仍是**单帧/静态展示**的最佳解码器。见 `make_prod_video_vis_fwd.py --decoder`（`load_any_decoder()` 自动识别 flow vs pooled）。

**关键认知：稳定性 与「生成式与否」是两个正交轴**。闪烁的根因是**每帧重采样噪声**，不是"生成式"本身。若 (a) 每段只解码一次并保持、或 (b) 固定噪声种子，flow 也能变成时序稳定——即不必为了稳定牺牲 flow 的锐度。更优方案见下节。

## 📋 更优解码方案调研（sharp + faithful + 时序稳定，2026-07-04）

目标重构：milestone 提示的正确目标是**「只在 milestone 切换时变化、段内绝对稳定、且锐利保真」**。按契合度排序：

1. **检索式解码（最近真实帧）— 首选**。把预测隐向量在真实帧 bank 里取最近邻，直接显示那张**真实图**。确定性、锐利（真像素）、**零幻觉**、段内稳定（常值隐向量→常值最近邻）。已是本项目既定方向（见 memory `feedback_lmwm_retrieval_decoder`：LMWM latent→图统一用检索解码器）。唯一闪烁风险在切换点附近的 Voronoi 边界抖动 → 加**滞回 / top-k 平均 / 要求最近邻持续 N 帧**即可消除。**最贴合本场景**。
2. **每段解码一次 + 保持（配 flow）**。对每个 unique milestone 用 flow **只解码一次**，整段沿用这张图 → 拿到 flow 锐度 + 结构上零闪烁，还便宜（每段一次而非每帧）。前提：隐向量分段常值（absolute 表示成立；forward-from-current 因输入含变化的当前观测则不成立）。
3. **固定噪声 / 共享种子的 flow**。冻结 ODE 初始噪声跨帧 → flow 退化为隐向量的确定性函数 → 常值隐向量得到完全相同的输出。文献支撑：latent reuse / consistent DDIM init / FastInit（temporally consistent video 的噪声初始化）。常值隐向量下与方案 2 等价；隐向量有微抖时仍需要它。
4. **时序相干噪声 / 光流 warp（Go-with-the-Flow）**。针对 forward-from-current 这类隐向量**真的逐帧漂移**的情形：让噪声沿光流传播，使生成式视频平滑而非闪烁。最重，一般用不到（提示是分段常值，方案 1/2 更简单）。
5. **`dec_v2`（当前交付，L1）**。确定性 → 本就无闪烁，只是糊。已交付的安全基线。
6. **`dec_gan_v2`（GAN）**。也是确定性 → 同样无闪烁、且锐利，但会**幻觉**（凭空造夹爪/纹理）；被否是因为保真而非稳定。若要"更锐但仍稳定"，它是确定性选项。

**建议**：milestone 提示切到**方案 1（检索式解码）**——确定性、锐利、无幻觉、段内稳定，且与项目既定检索解码器方向一致；`dec_v2` 作为糊但保真的合成式兜底。若要合成式锐度又要稳定，用**方案 2/3**（decode-once-hold 或 fixed-noise flow）而非逐帧重采样 flow。

参考文献：
- Go-with-the-Flow: Motion-Controllable Video Diffusion Using Real-Time Warped Noise — https://arxiv.org/abs/2501.08331
- FastInit: Fast Noise Initialization for Temporally Consistent Video Generation — https://arxiv.org/abs/2506.16119
- TokenFlow: Consistent Diffusion Features for Consistent Video Editing — https://arxiv.org/abs/2307.10373
- Warped Diffusion: Solving Video Inverse Problems with Image Diffusion Models (NeurIPS 2024) — https://arxiv.org/abs/2410.16152

## 用法

```python
from decode_best import load_best_decoder          # lmwm/scripts/decode_best.py
dec = load_best_decoder("lmwm/checkpoints/dinov3h_decoder/dec_best.pt", "cuda:0")
imgs = dec(latents)      # (N,1280) L2-normed pooled DINOv3-H -> (N,128,128,3) uint8 RGB
```

- 输入必须是**统一 gated DINOv3-H pooled 空间**的隐向量（`crave.encoders.encode_pooled` / `DINOv3HGated`，
  训练即部署同一空间——见 pitfalls B8 编码空间统一）。喂 bank-space 隐向量已交叉验证可用。
- ODE 25 步 Euler；`dec(latents, ode_steps=50)` 可换更慢更稳。res=128，base=160 UNet，80MB。
- 生视频：把 `make_prod_video_bankspace.py` 的 pooled 解码换成 `decode_best`（逐帧 ODE 采样，比 L1 慢但锐利保真）。

## 训练复现（gf3 八卡）

```
kai0/.venv/bin/python lmwm/scripts/flow_decoder_gf3.py \
  --base 160 --n 50000 --res 128 --steps 24000 --bs 64 \
  --out temp/lmwm_p0/flow_b160.pt --device cuda:2
```
rectified flow：`xt=(1-t)·noise + t·image`，UNet 回归速度场 `image-noise`，pooled 隐向量经 FiLM + 时间嵌入做条件。

## 产物

- `lmwm/checkpoints/dinov3h_decoder/dec_best.pt` — 交付解码器（flow_b160, 80MB）
- `lmwm/scripts/decode_best.py` — 干净的解码 API（`load_best_decoder`）
- `lmwm/scripts/final_decoder_compare.py` — 三方对比复现脚本
- `lmwm/docs/assets/final_decoder_compare.png` — real | L1 | GAN | flow 对比图
- `lmwm/outputs/{flow_eval,final_decoder_compare}.json` — 客观指标
