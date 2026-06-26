# FastWAM-v6 / GWP_ABS_v5 真机执行不正确 —— 排查记录(2026-06-26)

## 1. 现象
- **fastwam-v6**(独立 ActionDiT,WAM)和 **gwp_abs_v5**(共享 transformer,WAM)真机叠衣服时**机械臂执行不正确**,无法正常叠衣;**kai0 π₀.₅(A_smooth800_dagger_full)真机正常**。
- 三者**离线 MAE 都很好**(visrobot01_v3_val 100ep:kai0 cum@48=0.0483 / fastwam 0.0540 / gwp 0.0874;详见 [[../../../cosmos/wam_fold_wm/eval/three_way_compare_v3val.md]] 与 memory `jpsz-3way-eval-reproduction`)。
- 用户明确约束:**差异不在夹爪开合、也不在 RTC 平滑稳定性**上 —— 是**关节轨迹本身错**。

## 2. 方法论
关键认识:**离线 eval 路径 == 部署推理路径**(都走 `infer_action`/serve 的同一套 prep_image→VAE→action)。所以凡是能在离线复现的因素,都应同时影响离线 MAE。又因 **pi05 真机能用**,把 pi05 当**对照组**:任何"fastwam 特有缺陷"必须在 pi05 上不成立才能解释差异。

## 3. 逐一证伪(全部有实测/对照)

| # | 假设 | 验证 | 结论 |
|---|---|---|---|
| 1 | norm stats v1/v3 错配 | 对比训练 config 与部署脚本 | ❌ 同一文件 `visrobot01_v3_fold/dataset_stats.json`,z-score |
| 2 | proprio_cmd_feedback(部署喂"上次指令"当 proprio) | 闭环 rollout measured vs commanded;**pi05 对照** | ❌ pi05 同样发散 ~5×(且该离线测试本身有缺陷:commanded-proprio 配 GT 图像=人为不一致,真机两者一致) |
| 3 | fastwam 视觉过敏 | 换图/噪声敏感度;**pi05 对照** | ❌ pi05 同样/更敏感(噪声 ×1.32 vs ×1.35;换图 ×2.13 vs ×1.73) |
| 4 | action==state(action 列=state 拷贝,无控制超前) | 跨数据集 | ❌ **全项目通用**(Task_A/base/dagger、A_smooth800 全是 action==state),pi05 也是,能用 |
| 5 | v3 数据内容异常 | 对比 A_smooth800(stats/图像/结构/同步) | ❌ 关节 stats Δ<0.08rad、图像像素级同源、帧数==parquet 行数 |
| 6 | **WAM latent 消费(训练13帧视频 vs 部署单帧)** | 实测两种 VAE 编码的首帧 latent | ❌ **逐位相同,cosine=1.0000, MAE=0.0000**(Wan VAE 因果,首帧独立) |
| 7 | **部署 NFE=4 太少** | NFE 扫描 {2,4,6,10,20} | ❌ nfe4(0.115)≈nfe10(0.114),fastwam-v6 收敛快 |
| 8 | **opt 引擎(部署用)产出错** | opt-exact vs stock 同窗口 | ❌ 逐位相同,\|差\|=0.0013 |
| 9 | **fastwam-v6 过拟合 v3 采集** | 跑 6.10 后新采集数据 + A_new_pure(训练外) | ❌ **泛化良好**(见 §4) |

`merge_and_split.py` 经核查:state/action **原样保留**(只重写 episode_index/index),视频 symlink,无值变换。

## 4. 跨采集泛化对照(fastwam-v6,8ep/全窗口/nfe10,cum MAE@48)
| 数据集 | fastwam-v6 | gwp_abs_v5 | kai0 π₀.₅(真机能用) |
|---|---|---|---|
| v3_val(训练同分布) | 0.0842 | 0.1258 | 0.0458 |
| NEW 2026-06-16(61ep,训练外) | **0.0748** | **0.0991** | **0.0799** |
| NEW 2026-06-17(88ep,训练外) | **0.0694** | **0.1013** | **0.0729** |
| NEW 2026-06-23(13ep,训练外) | **0.0768** | **0.0969** | **0.0758** |
| A_new_pure_200(另一采集) | 0.0282 | — | — |

> **三方在 6.10 后新采集数据上都不退化**(fastwam/gwp 甚至更好)→ **都非过拟合**。关键:**新数据上 fastwam-v6(0.069–0.077)≈ pi05(0.073–0.080)**,泛化能力相当 —— 所以"泛化/过拟合"**不是** fastwam(真机崩)与 pi05(真机好)的区分点。新数据由用户提供(gf0 `/vePFS/.../Task_A/vis_dagger/v3` 6.10 后子目录),经 **gf0→BOS→本机/jpsz(跳代理)** 中转,落 `kai0/data/newval_v3_after0610/`(8ep/全窗口/同协议)。

## 5. 结论
**fastwam-v6 / gwp_abs_v5 在数据、模型、latent、推理引擎、泛化每一项都正确,且与能用的 pi05 一致;部署推理路径(nfe4+opt-exact)产出的动作 = 离线动作。离线无论如何复现不出真机失败。**

→ 真因只能在**真机运行时**,不在任何离线可复现的因素里。**唯一未被覆盖的差异**:所有离线测试都用**解码 mp4(压缩视频)的帧**,而真机喂**原始相机帧**(raw,无 AV1 压缩 / 不同 ISP)。结合 WAM 对噪声/高频敏感(×1.35)、且走 Wan-VAE-latent(对像素高频敏感),**raw-vs-压缩的视觉分布差异**是唯一无法离线测的因素(手里没有 raw 帧)。pi05 走 SigLIP 语义特征,可能对此更鲁棒。

## 6. 下一步(下周真机排查)
1. **出真机 dump**:`./start_scripts/kai/start_autonomy_fastwam_v6.sh --execute --debug-dump /tmp/fwdump`(serve 落盘每次推理的拼接图 PNG + io npz(state/action))。
2. 对比 dump 的**拼接图 vs v3 训练图**(raw-vs-压缩、相机分配、形变),**state/action 数值**是否合理、`motion` 是否异常。
3. 若确认 raw-vs-压缩:训练侧加图像增广(噪声/模糊/压缩),或部署侧对齐到训练的压缩/编码链。
4. 部署侧已保留 `proprio_cmd_feedback:=false`(对 WAM 更对齐训练,虽非真因)。

## 附:复现脚本(本次排查产物,jpsz `/tmp/`)
`vis_ablate3.py`(视觉敏感度)、`proprio_test.py`、`closedloop_test.py`(measured vs commanded proprio)、`nfe_sweep.py`、`opt_vs_stock.py`、`latent_cmp.py`(13帧vs单帧latent)、`crosseval.py`/`newval_eval.py`(跨采集泛化)、`cl_pi05.py`/`vis_pi05.py`(pi05 对照)。
