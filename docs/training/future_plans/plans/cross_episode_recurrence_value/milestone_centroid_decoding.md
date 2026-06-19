# CRAVE 簇中心解码 与 解码器对比试验

> 2026-06-18/19。把 milestone 簇中心从"取最近真实帧"扩展为"**解码出合成质心图**",并系统对比各编码器/解码器。
> **图集统一放** `docs/visualization/cross_episode_recurrence_value/centroid_decoder/`(本文件所有图引用均在该子目录)。
> 选型结论(标准配置)单列见 [`centroid_representation_config.md`](centroid_representation_config.md);完整实验留痕见 [`center_representative_research.md`](../../../../visualization/cross_episode_recurrence_value/center_representative_research.md)。

---

## 1. 背景与目标
milestone 代表图原来用"离簇心最近的真实帧(medoid)"。本线探索:**能否训一个解码器,把簇中心(latent)直接解码成一张合成质心图**,作为去具体化、不绑单 episode 的"可读原型"。核心要回答两件事:① 用什么编码器/解码器最好;② 合成质心能否替代/补充 medoid。

## 2. 解码器对比试验(实验线)

| 阶段 | 方法 | 结论 | 图(在 `centroid_decoder/`) |
|---|---|---|---|
| 2.1 | **池化向量解码**(mean-pooled latent → 图) | 鬼影/糊(池化丢空间)→ 基线否定 | `crave_center_decoder_compare.png` / `_recon.png` |
| 2.2 | **patch-token 空间解码**(16×16 网格 → 图) | **重建变清晰**(保空间);但簇内 grid 平均仍软 | `crave_patch_decoder_compare.png` / `_recon.png` |
| 2.3a | 刚性对齐再平均(掩码质心+主轴) | 比朴素平均好,仍软原型(可形变布料平均 ill-posed) | `crave_aligned_centroid.png` |
| 2.3b | pix2pix-GAN(含低频 L1 修复掉色) | 单帧/medoid 更锐;簇平均仍软;对抗会掉色(已修) | `crave_patch_gan_compare.png` / `_recon.png` |
| 2.4 | **规模消融 A–I**(编码器 22M→300M、解码器 0.76M→12.8M、数据 9k→24k 全顶满) | **加规模救不了簇中心**:单帧 medoid 冲到 234,簇中心始终 80–112 → 瓶颈是"平均输入 ill-posed",与规模无关 | `crave_scale_ablation_AtoI.png` |
| 2.5a | **解码器倒 U 选型**(固定 large+9k:tiny→small→medium→big→xl) | 簇中心结构 52 / **112(small 峰)** / 107 / 79 / 88 → **small(0.92M)最优**(太小糊、太大噪) | `crave_large_decoder_ladder.png` |
| 2.5b | 编码器选型 + 可读候选 | small→base→**large** 结构 92→89→112,large 最优;可读候选 B/D/E | `crave_readable_centroid_candidates.png` |

**两条硬结论**:① "清晰的平均质心"对可形变布料**数学上 ill-posed**,加规模/换损失都救不动;② 清晰合成只能来自**单帧**(medoid 渲染),本质是美化 exemplar。→ 代表图最清晰仍用 **medoid(最近真实帧)**;合成质心定位为"平滑可读原型"。

> 附:挖矿 episode 数对 value 质量的影响(~100–200 ep 饱和)`crave_mine_episode_sweep.png`。

## 3. ep2302 30Hz 端到端演示(本标准配置)
脚本 `train_scripts/kai/data/crave_ep2302_30hz_decoded.py`。全程 DINOv2-large 编码 + small 解码器:

- **自适应 milestone**(KMeans-96→覆盖率→Otsu)+ **自适应 value bins**(bins=milestone 位置,value↔milestone 精确对应)+ **报告式平滑**(`smooth_monotone`,fps 标定移动平均)。
- 产物:`crave_ep2302_30hz_decoded.png`(value 曲线 + milestone 变化 + 每 milestone 簇中心解码图)、`crave_ep2302_30hz_decoded.mp4`(左相机 / 右上 value 游标 / 右下解码质心,**并行 compose + libx264** 编码,**逐帧 2960 全验证 ALL_PASS**)、`crave_ep2302_transition_3row.png`(跳变帧四行:原始 / encode-decode 重建 / 簇中心解码 / 最近真实 medoid)。

## 4. 标准配置(定稿)
**DINOv2-large 编码器 + small(0.92M)空间解码器 + patch-grid 输入 + L1 + 9k + 自适应 milestone + 自适应 value bins**。详见 [`centroid_representation_config.md`](centroid_representation_config.md)。

## 5. 图集清单(`centroid_decoder/`)
- 解码器对比:`crave_center_decoder_*` / `crave_patch_decoder_*` / `crave_aligned_centroid` / `crave_patch_gan_*`
- 规模/选型:`crave_scale_ablation_AtoI`(A–I 总图)+ 各 per-config tag / `crave_large_decoder_ladder` / `crave_readable_centroid_candidates`
- 演示:`crave_ep2302_30hz_decoded.{png,mp4}` / `crave_ep2302_transition_3row.png`
- value/数据:`crave_mine_episode_sweep`

---

## 6. 未来规划

> 以下两节由 deep research(2026-06-19,100 agents / 18 源 / 对抗校验)定调。结论先行,后续按"便宜首验"推进。

### 6.1 V-JEPA 2 编解码器测试 —— **结论:聚类/解码不用换;只在做时序预测时评估 V-JEPA 2-AC**
- **V-JEPA 2 / 2.1 是 latent-only,无原生像素解码器**(masked latent 预测,encoder + predictor)。要出簇中心**图**仍得**另训一个像素解码器**(Meta 自己也是单独训了个 ViT-L decoder 仅作可视化)→ 在"解码"这步 V-JEPA 2 **不省事**,相对 DINOv2-large + 训练解码器无优势。
- **dense 语义聚类 DINO 系仍更强**(ADE20K seg 47.9 vs 55.9 mIoU);V-JEPA 2.1 仅在 depth 上追平 DINOv3;"视频编码器聚类胜 DINOv3"的说法被**否决(0-3)**。→ **milestone 聚类 + 簇中心解码:保持 DINOv2-large + small 解码器(现标准配置)。**
- **V-JEPA 2 唯一差异化价值 = V-JEPA 2-AC**(动作条件世界模型,潜空间预测/规划)→ **只在 6.2 要做"时序 milestone 预测 / 世界模型规划"时才值得评估**,不是为聚类/解码。
- 工程:V-JEPA 2 ckpt 在 HF(`facebook/vjepa2-*`),但本机 HF/镜像被墙(同 dinov2-large,需另找权重)。
- **便宜首验(若评估)**:用 V-JEPA 2-AC 在 latent 空间预测 ep 的下一段 → 比对 CRAVE milestone 序列,看时序预测是否比 DINOv2 逐帧聚类更早/更准锚定相位;**不替换现有聚类**,只作时序预测候选。

### 6.2 milestone 预测赋能 VLA —— **结论:可行,但作"补充"而非替代;Design 1 优先 + 强制子目标过滤**
范式成熟(子目标/目标图条件化:SuSIE / UniPi / GR-2 / V-JEPA 2-AC),所以"预测 milestone+1 子目标条件化 VLA"**可行**。两设计实测建议:

- **Design 1 · goal-image(解码 milestone+1 簇中心 → 目标图喂 VLA)** —— **首选、范式最成熟、最低风险**。
  - ⚠️ **关键风险**:我们的解码质心是**平滑软原型**(非写实),生成子目标带伪影/非真进度会**拖垮**低层策略(已证 3-0)。→ **必须配 GHIL-Glue 式子目标过滤器**(过滤"无进度/有害"子目标),否则适得其反。
- **Design 2 · latent(milestone+1 簇中心 latent 作子目标 embedding 注入 VLA)** —— 更省(无需解码),但**前提是 milestone latent 在 VLA 编码器空间**。我们 milestone 用 DINOv2-large,pi0/pi05 用别的视觉编码器 → **不共享** → 需让 **milestone 聚类改用 VLA 同款编码器** 或 **训一个对齐投影**(额外工程)。
- **诚实红线**:"子目标图条件 > 语言/纯 BC"被**否决(0-3)** → milestone 条件化**不一定**优于我们现有的**稠密 value(优势加权 BC)** → 定位为**补充**(尤其长程/多模态 cloth fold),先验证有无增量,别当替代。

**便宜首验(各设计)**:
1. **Design 1**:离线用 **GT milestone+1 的解码图**当目标图(先用 GT、不用预测,排除预测误差),训 goal-conditioned BC vs 无目标 BC,sim01 比成功率 + 套 GHIL-Glue 过滤。增量为正再上"预测 milestone+1"。
2. **Design 2**:把 milestone **聚类改用 VLA 的编码器**(SigLIP/DINO,与 pi0 对齐)→ 子目标 latent 天然在 VLA 空间 → 注入一个子目标 token,同样先用 GT milestone+1。
3. 两者都先回答"**子目标条件化相对现有稠密 value 有没有正增量**";有,再投入预测器 + 过滤器工程。

> 详细引用见 deep research 输出(task `wzge796p5`)。

### 6.3 仿真验证方案 —— **验"方法有效性",与叠衣域解耦**
**原则**:milestone/子目标条件是**跨任务通用**方法,先在标准 VLA 基准上验"有没有效",不必迁就 cloth;验通再回真机 kai0(叠衣)验域迁移。

**选定环境:LIBERO-Long(openpi/pi0 原生闭环,本仓已接好)**——长程多阶段(milestone 天然有意义)+ pi0 闭环零集成。
- 现成资产:`kai0/src/openpi/policies/libero_policy.py`(pi0 LIBERO 策略)、`fastwam/experiments/libero/`(`run_libero_manager.py` / `eval_libero_single.py` / 并行评测 + `libero_{uncond,idm,joint}_2cam224` configs)、`xvla/X-VLA/evaluation/libero/`。
- 为何不选 cloth sim 首验:DexGarmentLab/SoftGym 等重/慢/域特化,**方法有效性**不需要布料;布料 sim(DexGarmentLab)留作最终域确认。

**对照臂(干净隔离"子目标条件是否有正增量")**:
- **A0 baseline**:原版 pi0(无子目标)。
- **A1 = Design-1**:pi0 + **GT milestone+1 解码图**作目标图(先用 GT、不预测 → 上界,排除预测误差)。
- **A2 = Design-2**:pi0 + **GT milestone+1 latent**子目标(用 pi0 同款编码器抽 milestone 特征,保证 latent 同空间)。
- **A3(若 A1/A2 有正增量再做)**:学一个 milestone+1 预测器 + **GHIL-Glue 式子目标过滤**,换掉 GT。

**流程**:CRAVE 套到 LIBERO demo(逐帧特征→聚类→milestone→每帧取 milestone+1 子目标)→ 训 A0/A1/A2(集群,长训不走本地 2 卡)→ LIBERO-Long 闭环比**成功率 + 子阶段通过率**。
**判据**:A1/A2 > A0 → 方法有头部空间 → 投预测器(A3);A1/A2 ≈ A0 → 子目标条件在此不优于纯 BC,回退/只做稠密 value。**先答"GT 子目标有没有用",再谈预测**。
