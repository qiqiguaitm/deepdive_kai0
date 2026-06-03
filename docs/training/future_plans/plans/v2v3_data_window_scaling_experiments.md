# v2/v3 数据量×时窗 扩展实验 (Data Window Scaling)

> **目的**: 用 3 个 pi05 cloth-fold 训练实验,考察**数据时窗/数量**对真机表现的影响 —— 单日 vs 多日窗口 vs 全量。与 [`data_root_cause_probe_experiments.md`](data_root_cause_probe_experiments.md) 互补(后者查"数据质量/裁剪",本系列查"数据数量/时窗")。
> **状态**: 📝 规划中 (待数据上 vePFS + config 注册)
> **建立**: 2026-06-03
>
> ⚠️ **方法学铁律**(沿用本项目一贯结论): **真机为终判, offline MAE 系统性反指**。下面每个实验出 ckpt 后**必须真机测**,MAE 仅用于确认收敛 + 选 best ckpt。

---

## 0. 研究问题

后期数据(5-18 起)offline SOTA 但真机犯病(走停/犹豫/松手)。本系列换一个轴:**喂多少天、喂哪个时窗,真机怎么变**?
- **Exp-A 单日 5-18**: 最小数据(201 ep),看单日能否 work / 过拟。
- **Exp-B 5-18~5-28 窗口**: 后期 8 天(955 ep),"近期多日"。
- **Exp-C 全 v3(排 5-16)**: 全量 1940 ep,"全历史"。
- **Exp-B→C 顺序**: 先窗口后全量,对比"加早期数据(4-23~5-10)是帮忙还是稀释"。

---

## 1. 数据清单(2026-06-03 实测,源在 uc `vis_base/`)

### Exp-A 源: `vis_base/v2/2026-05-18-v2`
- **201 ep, 25G**。⚠️ **v2 版本**(与 B/C 的 v3 是不同处理 pipeline,见 §5 注意)。

### Exp-B / Exp-C 源: `vis_base/v3/<date>`
| date | ep | | date | ep |
|---|--:|---|---|--:|
| 2026-04-23-v3 | 21 | | 2026-05-10-v3 | 95 |
| 2026-04-24-v3 | 187 | | 2026-05-16-v3 | 16 ⛔排除(残缺3.3M) |
| 2026-04-25-v3 | 96 | | **2026-05-18-v3** | **201** |
| 2026-04-28-v3 | 152 | | **2026-05-19-v3** | **100** |
| 2026-04-29-v3 | 100 | | **2026-05-20-v3** | **100** |
| 2026-04-30-v3 | 83 | | **2026-05-21-v3** | **100** |
| 2026-05-06-v3 | 100 | | **2026-05-22-v3** | **100** |
| 2026-05-07-v3 | 20 | | **2026-05-26-v3** | **100** |
| 2026-05-08-v3 | 101 | | **2026-05-27-v3** | **105** |
| 2026-05-09-v3 | 30 | | **2026-05-28-v3** | **149** |

- **Exp-B (5-18~5-28, 加粗 8 日)** = **955 ep** (注:5-23/24/25 无采集)。
- **Exp-C (全 v3 排 5-16)** = **1940 ep** (4-23~5-10 共 985 + 5-18~5-28 共 955)。

---

## 2. 三个实验规格

| | **Exp-A** | **Exp-B** | **Exp-C** |
|---|---|---|---|
| 数据 | v2/2026-05-18-v2 单日 | v3 5-18~5-28 窗口 | v3 全量(排 5-16) |
| ep 数 | 201 | 955 | 1940 |
| 建议 config 名 | `pi05_flatten_fold_v2_0518_201` | `pi05_flatten_fold_v3_0518_0528` | `pi05_flatten_fold_v3_all_no0516` |
| 集群 | **cnsh** (robot-task, A100) | **cnbj** (Robot-North-H20) | **cnbj** (排队) |
| 卡数 | 16 (2节点) | 16 (2节点) | 16 (2节点) |
| 顺序 | 独立 | 先跑 | **Exp-B 完成后顺序跑** |

### 统一训练配置(单变量:只改数据)
对齐 work 锚点 smooth_800 / A_0522_0526 系列:
| 项 | 值 |
|---|---|
| Model | pi05 (`Pi0Config(pi05=True)`) |
| 框架 | JAX/Flax NNX (`scripts/train.py`) |
| Init | `mixed_1_clean`(与既往 work 锚点一致) |
| Prompt | `"Flatten and fold the cloth."` |
| use_delta_joint_actions | False (absolute) |
| LR | Cosine, warmup=1k, peak=1.5e-5, decay→1.5e-6 |
| EMA | 0.9999 |
| batch_size / fsdp_devices | 128 / 16 |
| **Steps** | A 单日小集 **30k 够**;B/C **50k**(待 inline-eval plateau 微调) |
| norm_stats | **各自重算**(`compute_norm_states_fast.py`),不复用 |
| inline_eval_val_root | `vis_v2_merged_val`(与既往 cross-val 一致,便于横比) |

---

## 3. 每实验执行链(逐个)

1. **build 数据集**(合并选定日期 → lerobot v2.1 单集, episode_index 重排):
   - 参考 `train_scripts/kai/data/build_no_release.py`(`--mode raw` 不裁)/ `build_task_a_new_100.py`。
   - Exp-A: 仅 `v2/2026-05-18-v2`。Exp-B: 合并 v3 的 8 个日期。Exp-C: 合并 v3 除 5-16 外全部。
   - 产物落各集群 vePFS 的 `self_built/<config数据名>/`。
2. **compute_norm_states_fast.py --config-name <config>**(数据所在机)。
3. **注册 config** 到 `kai0/src/openpi/training/config.py` + `git commit && push`(gf3/cnbj、gf0/cnsh 由 cron/pull 同步)。
4. **init ckpt** `mixed_1_clean/params` 在目标集群 vePFS 就位(size 校验 22G)。
5. **提交 16 卡 YAML**(cnsh / cnbj 对应 queue + image + SubPath,详见 [`../../deployment/training_ops/submission/`](../../deployment/training_ops/submission/) + 共性坑 `training_pitfalls_common.md`)。
6. **验证**: log 出 `Generating train split` + `Step N: loss` + 熬过第一次 ckpt save。

> **Exp-C 顺序触发**: Exp-B 在 cnbj 完成(50k + final save)后,再提交 Exp-C(同 cnbj 16 卡)。可挂 watcher 监 Exp-B 终态自动提交,或手动。

---

## 4. ⚠️ 关键前置 — 数据上 vePFS(uc 即将回退!)

raw `vis_base/v2,v3` 当前**只确认在 uc**(323G)。uc **正在回退**,且 volc 训练需数据在 **cnsh/cnbj vePFS**。**必须先确认/搬运**:

| 待确认 | 动作 |
|---|---|
| vis_base v2/v3 在 gf0(cnsh)vePFS 有无副本? | 有 → 直接 build;无 → **uc 回退前** rsync/TOS 搬 5-18-v2(25G)+ v3(61G)到 cnsh/cnbj vePFS |
| cnbj vePFS 有无 v3? | 同上,Exp-B/C 需 v3 在 cnbj |
| build 在哪做 | 建议在数据落地的 vePFS 机器(gf0/gf3)本地 build,避免跨集群读 |

> 🔴 **最高优先级**: 在 uc 回退前把这两块 raw(共 ~86G:5-18-v2 25G + v3 61G)确保有 vePFS/本地副本,否则实验无源。

---

## 5. 注意事项 / 待决

1. **v2 vs v3 版本混用**: Exp-A 用 v2、B/C 用 v3。若想干净对比"单日 vs 窗口",理想应同版本。**待确认**: 是否把 Exp-A 也换 v3 的 5-18(v3 也有 201ep),以消除版本混淆?(当前按你给的 v2 路径写。)
2. **早期数据增益方向**: Exp-C vs Exp-B 的差(加 4-23~5-10 的 985ep)是本系列核心对比 → 真机判定"早期数据帮忙/稀释"。
3. **5-16 排除**: 仅 16ep/3.3M,残缺,排除无争议。
4. **steps**: A 单日 201ep 用 50k 会重过拟,建议 30k;B/C 50k。最终看 inline-eval plateau。

---

## 6. 关联 XVLA 8 卡 volc 任务(并行轨,另行细化)

另需"通过 volc 提交 8 卡 XVLA 训练任务队列"——与本 pi05 系列独立。X-VLA(torch DDP, 非 JAX)在 volc 提交要点:
- Framework 需给 torchrun env;8 卡 = 单节点 8 GPU(`ml.hpcpni3ln.45xlarge` 类);
- 数据 = `A_new_smooth_800_xvla`(已改名)等 EE6D 集,需上目标 vePFS;
- init = `xvla_ckpts`(3.3G)上 vePFS;
- 详细 YAML + 提交在本 plan 批准后单独出(见 待办)。

---

## 7. 待你拍板
- [ ] **§4 数据搬运**: 确认 vis_base v2/v3 在 vePFS 有无副本;无则授权我 uc 回退前搬运。
- [ ] **§5.1 v2/v3**: Exp-A 保持 v2 单日,还是改 v3 的 5-18 以同版本?
- [ ] config 名 / steps(A=30k? B/C=50k?)是否按建议。
- [ ] Exp-C 顺序触发要自动(watcher)还是手动。
- [ ] XVLA 8 卡 volc 是否本轮一起细化提交。
