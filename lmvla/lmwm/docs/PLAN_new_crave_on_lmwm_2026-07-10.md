# 计划：用最新 CRAVE 架构在 LMWM 上跑一版结果（2026-07-10）

> 触发：CRAVE 方法架构更新（[`crave/docs/final_architecture.md`](../../crave/docs/final_architecture.md)，2026-07-09 收口）。
> 目标：把**新 CRAVE teacher**接到 LMWM intrinsic 预测器上，重训并出指标，对照旧 DINOv3-H 37-milestone 基线。
> 状态：**规划完成，等用户重生成 DINO-base 数据后执行**。

---

## 0. 接口理解（已核实代码）

- **CRAVE = 训练期 teacher**：只负责 ① milestone 定义（哪帧→哪 milestone）② 排序/进度值。部署期零 CRAVE/零 DINO。
- **LMWM intrinsic 预测器**：学 `frame 特征 → 下一 milestone(码/prototype)`。
  - 桥接脚本 `lmwm/scripts/export_dinov3h_milestone_pairs.py`：读 `feature_dir`(逐帧特征) + `milestone_file{C,Pord}` → 最近质心分配 → `next_unique`/`fixed_horizon` 配对 → `pairs_*.npz`。
  - 指标面板训练器 `lmwm/scripts/train_ablation.py`：产出文档里的 `deploy grid-cos / identity top-N` 数字（`--teacher medoid|center --anchor ce|progress --code_dim --K`）。
- **旧基线**：`temp/crave_full_dinov3h/milestones_uniform_dinov3h.npz` = **C(37,1280) + Pord(37)**（DINOv3-H）。kai0 单任务 deploy≈0.703 / id_top3≈0.473；3-task mean deploy 0.753 / id_top3 0.710（见 `ARCHITECTURE_AND_BASELINE.md` §4）。

## 1. 新 CRAVE 相对旧的变化（只动 teacher，部署 SigLIP 不变）

| 环节 | 旧(基线) | 新(final_architecture) |
|---|---|---|
| 发现编码器 | DINOv3-H 1280D | **DINOv3-base 768→PCA128 ⊕ proprio位置14（各自 L2，能量1:1）** |
| 聚类 | KMeans/uniform → **37** milestone | **BayesianGMM(自适应K)+per-mode coverage≥0.5 → 12** milestone |
| 进度值 | — | **中位数(median)**（保序，0 倒挂） |
| 读出 | — | **双锚 Viterbi(无smooth·无norm01)**，corr vs 监督GT **0.943** |

生成脚本 `crave/experiments/gen_final_v3.py`：吃 `temp/crave_d3b_pca128/{feats/ep*.npy, milestones.npz(取pca_mean/components)}` + `temp/crave_full_dinov3h/index.npz(E,FR)` + kai0_base parquet(state) → 出 `temp/crave_final_v3b.npz{Ctgt(M,142), vals, cluster_idx, SMU/SSD,...}`。

## 2. 执行步骤（数据就绪后）

1. **生成新 CRAVE spec** — `PYTHONPATH=crave/src:lmwm/src python crave/experiments/gen_final_v3.py`
   验收：milestone 数≈12、末值>0.9≈100%、单调≈0.98、milestone 值升序无倒挂。[CPU，分钟级]
2. **桥接 spec → LMWM** — 写小适配器：
   - `milestone_file`：`C = Ctgt(142D)`、`Pord = rank(vals)`；
   - `feature_dir`：把 gen_final_v3 的逐帧 `jointF(img128⊕pos14)` 落成 `index.npz{E,FR,T,n}`+`shard_*.npz{gidx,feat,valid}`（复用 export 的 loader）；
   - 逐帧 milestone id 优先用 **Viterbi 路径**（与 CRAVE 一致），否则退最近质心。
3. **导出配对** — `export_dinov3h_milestone_pairs.py --config <新dataset yaml>`（`next_unique`，`frame_feature→prototype`，max_pairs 200k）→ `pairs_next_unique.npz`。
4. **训练预测器** — 两选一（见 §3 决策）：
   - `train_ablation.py --feature_dir <新joint> --teacher medoid --anchor ce --code_dim 128 --K 4 --tag newcrave12`（出 deploy/id 面板，直接对标基线）；或
   - stage1d frame2proto 轻量 MDN（只出 next-ms top1）。[GPU，分钟级]
5. **评估+对照表** — 新12 vs 旧37：`deploy grid-cos / id_topN / next-ms top1 / value mono·corr`。落 `lmwm/outputs/newcrave12/` + 回填本 doc。

## 3. 待用户确认的设计决策

**决策①：intrinsic 预测器的输入特征空间**
- **A（推荐，最忠实"新架构端到端"）**：预测器直接在**新 img⊕pos 142D** 空间；teacher=12 BayesianGMM milestone。完整体现新 CRAVE。注意：输入空间与旧基线(1280D)不同 → 与旧 0.753 非严格同口径，需注明。
- **B（受控对照）**：输入空间**保持不变**(DINOv3-H 或 SigLIP)，只把帧**重标注**到新 12 milestone + 重算该空间下的中心。隔离"milestone 定义本身是否更好"，与基线严格同口径。
- **C**：A+B 都跑，出两行对照。

**决策②：训练器**
- `train_ablation.py`（推荐，产出与文档同名的 deploy/id_top3 指标，直接可比）；或 stage1d 轻量（只 next-ms top1，更快）。

> 我的建议：**A + train_ablation.py** 作为主"版本结果"，若时间够加 **B** 做受控对照（都很便宜）。

## 4. 计算量 & 风险

- **计算**：BayesianGMM(~335k×142D,n=40) 几分钟 CPU；Viterbi×3055ep 几分钟；预测器 1.2k–9k step 分钟级 GPU。**整链分钟级，可快速迭代。**
- **风险/坑**：
  - Ctgt 是 `l2(jointF)` 均值未再归一化；export 会对 `C` 再 l2 → 分配口径需对齐（用 Viterbi 路径分配可规避）。
  - `gen_final_v3.py` 存的是 `crave_final_v3b.npz` 且其**自带 label 输出仍做 per-ep norm01**（125-126 行），与"双锚无norm01"文档不一致；LMWM 只用它的 **spec(Ctgt/vals)**，不用它的 label，故不受影响；若要双锚 label 另跑 `gen_anchored_labels.py`。
  - 12 milestone 比 37 粗 → next_unique 配对数下降、id_topN 基数变（对照时注明簇数）。

## 5. 等待项

用户正在重生成 **DINO-base 数据**（`temp/crave_d3b_pca128/feats` + `milestones.npz`；当前已有 7月9 版 3055ep）。**用户通知就绪后**从 §2 步骤1 开始。
