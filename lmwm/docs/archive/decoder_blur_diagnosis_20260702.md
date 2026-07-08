# 解码模糊归因诊断(2026-07-02)

> 问题:预测 latent 解码后糊,是 decode 的问题、还是 LMWM 预测精度不足、还是解码器训练不足?
>
> 注:**解码器方法本身属 CRAVE**,pooled/patch-grid 保真对比的方法学结论沉淀在
> [`crave/docs/milestone_centroid_decoding.md`](../../crave/docs/milestone_centroid_decoding.md) §2b;
> 本文聚焦 **LMWM 侧的应用**(解码 LMWM 预测的 latent、归因、路线选择)。

## 决定性实验

用**锐度 = Laplacian variance**(越高越清晰)+ 用**真实帧自己的 latent**解码(完美输入,隔离预测误差)。held-out 300 帧:

| 对象 | 锐度 | L1 误差 |
|---|---|---|
| 真实帧 | **992** | — |
| decode(帧**自己的真** latent)[L1 解码器] | **152** | 15.9/255 |
| decode(帧自己的真 latent)[L1+GDL 梯度锐化] | 114 | 16.0 |

## 三个问题的答案

**Q1:确定是 decode 的问题吗?→ 是。** 用**完美 latent**(帧自己的)解码,锐度仍只有 152 vs 真实 992(糊 6.5×)。糊在解码器,与预测无关。

**Q2:是 LMWM 输出精度不足吗?→ 不是(对"糊"而言)。** 完美 latent 都糊,所以糊不是预测误差造成的。预测的 cos0.86 不精确只会在"糊"之上再加一点点偏差,不是糊的来源。

**Q3:训练不足、能训得更好吗?→ 简单损失救不动。** 加梯度锐化损失(GDL)非但没锐,反而 152→114。根因是 **L1/L2 损失逐像素求均值 → 天生糊**,加边缘损失治标不治本;要真锐需**生成式**损失(GAN/扩散,采样一个清晰模态而非平均)。

## 用户的关键逻辑(对的)

> "如果解码器能把数据集帧的 DINOv3-H latent 解码回和原图几乎一样,那预测 latent 解码糊就是 LMWM 的问题。"

**完全正确的实验设计**:先让解码器做到**自重建近乎无损**(把它变成"测量仪"),再解码预测 latent —— 那时若还糊/错,才是 LMWM 的锅。

**但当前前提不成立**:pooled DINOv3-H 自重建就糊(152 vs 992),所以现在**还不能**把糊归给 LMWM。要先造出自重建保真的解码器。

**pooled 能做到自重建近乎无损吗?证据倾向"不能(逐像素)"**:
- pooled(CLS/mean)是全局摘要,DINOv3 的**空间细节在 16×16 patch token 里**,不在 pooled 向量里(CRAVE 早验证:pooled 解码糊,patch-grid 解码锐)。
- pooled 特征有歧义:检索里不同帧的 pooled cos 可达 0.9+(自检索 top2/3 就 0.96),不同布料/构型映射到相近 pooled → 解码器无法区分 → 逐像素精确复原不可能。
- L1、L1+GDL 都糊,进一步佐证。

## 结论与路线

1. **糊是解码器(pooled + L1 损失),不是 LMWM 预测精度。**(Q1 是、Q2 否、Q3 简单法救不动)
2. **要"清晰的合成图"**:用生成式解码器(GAN/扩散,plausible-sharp)——语义/构型清晰但非逐像素精确;或用检索(真实帧,已是规范)。
3. **要"解码后能当 LMWM 质量判据"(用户真正目的)**:必须先有自重建保真的解码器 —— 这要求 LMWM 预测 **patch-grid 空间 latent**(而非 pooled)+ 空间解码器(CRAVE 证明其锐利)。只有那时,预测解码的糊/错才能归因于 LMWM。
4. **在此之前,LMWM 预测质量用 latent 空间的 cosine(0.86)衡量**,它与解码器无关,是干净的预测质量指标。

## 更新:GAN 解码器把糊救回来了(根因确认是损失,不是 pooling)

L1/L2 天生糊(逐像素求均值)。换成 **pix2pix 式对抗训练**(PooledDecoder 作生成器 + PatchGAN 判别器 + hinge 对抗 + L1),self-recon 锐度:

| 解码器 | self-recon 锐度(真实=992) | L1 误差 |
|---|---|---|
| L1(旧) | 152 | 15.9 |
| L1+GDL | 114 | 16.0 |
| **GAN(新)** | **852**(接近真实 992) | 18.1 |

**结论更新**:decode(帧自己的真 latent)锐度 152→**852**(5.6×,接近真实)。**说明 pooled 1280-D 其实保留了足够信息合成清晰单帧图 —— 糊是 L1 损失(求均值),不是 pooling 的信息损失。** 我之前"pooled 根本性有损"的说法要收回:对**单帧级**目标,pooled+GAN 就能锐;CRAVE 的 ill-posed 只针对**平均/簇中心**目标。

**用户逻辑现在成立了**:GAN 解码器自重建接近无损(852≈992),已是合格"测量仪"。此时 decode(LMWM 预测)—— 见 `ep793_subgoal_decode_GAN.png` —— **图变锐了**,剩下的颜色/褶皱偏差(预测偏 tan、真值偏粉)**归因于 LMWM 预测误差(cos 0.86)**,不是解码器。

## CRAVE 的 DINOv3-H 解码器(用户问:能否自重建恢复原样)

CRAVE 确实实现了 DINOv3-H+ **patch-grid**(16×16 空间)解码器,并做过详尽消融(`milestone_centroid_decoding.md`):
- **单帧 encode→decode 自重建**:recon L1 **0.029**(~3% 误差),结构分 **234**,能还原布料形态/颜色/手位 —— 单帧可恢复。
- **但平均/簇中心目标始终软**:锐度 gridavg 82 / medoid 62 vs 真实帧 369;结构分簇中心 80-112 vs 单帧 234。
- **规模消融 A-I(编码器 22M→300M、解码器 0.76M→12.8M、数据 9k→24k)证明:加规模救不了簇中心** —— "平均输入 ill-posed",与规模无关。pix2pix-GAN:单帧/medoid 更锐,簇平均仍软。
- CRAVE 硬结论:**清晰合成只能来自单帧;代表图最清晰用 medoid(最近真实帧)/ Wan2.2-VAE 单帧重建(L1 0.003 照片级)。**

与我们 GAN 结论一致:**单帧级目标可锐(GAN 852 / CRAVE 单帧 234 / Wan L1 0.003);平均目标 ill-posed。** LMWM 预测的是近似单帧 latent(非精确),GAN 解码后清晰但颜色/褶皱略偏 = 预测误差。

## 最终规范(两个解码器各司其职)
- **要清晰合成图(可视化/subgoal 渲染)**:用 **GAN 解码器**(`dinov3h_decoder/dec_gan.pt`),清晰、可当 LMWM 质量测量仪。
- **要逐像素真实/精确**:用**检索**(最近真实帧,`LatentRetrievalDecoder`)——真实照片级。
- L1/GDL pooled 解码器弃用。

## 保真才是目标:patch-grid 解码器实测(2026-07-02)

用户澄清:目标是**解码保留原图相同信息(保真)**,不是清晰。据此:
- **GAN 是错方向**:锐是幻觉,颜色漂移(粉→tan),保真更差。
- **L1 pooled 更忠实但天花板 ~6.2%**(pooled 平均掉空间细节)。

提取 **DINOv3-H patch-grid 特征**(16×16×1280,`enc.encode_grid`)+ 训 CRAVE 空间解码器,held-out 自重建:

| 解码器 | 自重建 L1(越低越忠实) | 锐度 |
|---|---|---|
| pooled L1 | 6.2% | 152 |
| pooled GAN | ~7%(+掉色) | 852 |
| **DINOv3-H patch-grid** | **2.7%** | 214 |

**结论**:patch-grid 解码 **2.7% L1,比 pooled(6.2%)忠实约 2.3×** —— 颜色(teal/navy/pink 全对)、形状、臂位都保住(见 `outputs/patch_decoder/patch_recon_compare.png`)。**这才满足"保留相同信息"。** patch-grid 保住了 pooling 平均掉的空间信息,这是保真的真正杠杆(换损失救不了,换表示才行)。

**对 LMWM 的含义**:LMWM 现在预测 **pooled**(1280),无法喂 patch 解码器。要让 LMWM 预测能被忠实解码,需 LMWM 预测 **patch-grid latent**(16×16×1280,输出大 256×)—— 或学一个 pooled→patch-grid 的 un-pool 映射。真实帧/medoid 的忠实可视化现在即可用 patch 解码器。

## 两条路线并行实测:直接 patch-grid vs pooled→un-pool(2026-07-02)

要让"LMWM 预测能被忠实解码",两条路,held-out 单帧自重建对比(`scripts/unpool_vs_patch.py`):

| 路线 | 方法 | 自重建 L1 | 说明 |
|---|---|---|---|
| **Path 1(治本)** | 直接预测 patch-grid latent(1280×16×16)→ CRAVE 解码 | **2.7%** | 忠实(颜色/形状/臂位保住) |
| **Path 2(轻量)** | 保持预测 pooled → 学 un-pool → patch-grid → 解码 | **6.9%** | **失败,甚至不如 pooled 基线 6.2%** |
| pooled 基线 | pooled 直接解码 | 6.2% | — |

**关键**:un-pool 出的 grid 对真 grid 只有 **cos 0.77** —— pooled 1280-D **真的把空间信息平均丢了,无法还原 patch-grid**。所以 Path 2(6.9%)≈ 甚至略差于 pooled 基线,**un-pool 是死路**。见 `assets/unpool_vs_patch.png`(行1真实 / 行2 path1 忠实 / 行3 path2 退化)。

**结论**:**只有 Path 1(LMWM 直接预测 patch-grid latent)能拿到 2.7% 忠实解码。** 代价是 proto 头从 pooled(1280)换成空间头(1280×16×16,输出大 256×),但这是唯一有效路线;轻量 un-pool 捷径走不通(信息已在 pooling 时丢失)。

## 产物
- `scripts/train_patch_decoder.py`(patch-grid 保真 2.7%)、`scripts/unpool_vs_patch.py`(路线对比)→ `outputs/patch_decoder/`,图 `assets/patch_recon_compare.png`、`assets/unpool_vs_patch.png`
- `scripts/train_dinov3h_decoder.py` 加 `--gdl_weight`(不足以救糊)
- `scripts/train_dinov3h_decoder_gan.py`(GAN 解码器,锐但幻觉,保真差)
- `scripts/train_patch_decoder.py`(DINOv3-H patch-grid 提取 + CRAVE 解码器,**2.7% 保真**)→ `outputs/patch_decoder/{summary.json,patch_recon_compare.png}`
- 图:`ep793_subgoal_decode_{L1,GAN}.png`
