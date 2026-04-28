# `kai0/checkpoints/` 目录规范

> 真机 autonomy 推理用的 checkpoint 落盘约定。**新 ckpt 上架前对照本文逐项过 checklist；任何一项缺失或路径不对 → autonomy 启动会在不同位置炸 (missing config / missing params / missing norm_stats / 关节乱跑)。**

---

## 1. 物理布局 (统一单点, 2026-04-27 起)

唯一权威位置: **`/data1/DATA_IMP/checkpoints/`** (同盘 `/dev/nvme0n1p1`, 7 TB).

工程内访问通过顶层 symlink:

```
/home/tim/workspace/deepdive_kai0/kai0/checkpoints
  → /data1/DATA_IMP/checkpoints                      # 单层 symlink, 项目脚本/openpi 都通过这访问
```

(`/home/tim/workspace` 自身已经是 → `/data1/tim/workspace` 的 symlink, 所以工程内的 `kai0/...` 路径全部最终落到 `/data1` 上, 不占系统盘.)

每条 ckpt 直接在 `/data1/DATA_IMP/checkpoints/` 下作为**真实目录**存在, 不要再走 `KAI0/ckpt_downloads/` 这种二级中转 — 历史上有过, 已 2026-04-27 合并掉.

---

## 2. 两种 ckpt 拓扑 (二选一)

### A. 扁平 (lightweight, inference-only ckpt)
适用于"params + assets + meta, 无 train_state"的轻量 ckpt (例如外部下发或 `from_tos_file.py` 拉来的 best ckpt, 12 GB 量级).

```
/data1/DATA_IMP/checkpoints/<run_label>_<milestone>/
├── _CHECKPOINT_METADATA
├── assets/<asset_id>/         # 见 §3.3 (a) 路径; 用 (b) 默认 asset_id 时此目录可空
│   └── norm_stats.json
└── params/                    # orbax 权重
```

例:
```
/data1/DATA_IMP/checkpoints/mixed_gf0_best_at_4k/
/data1/DATA_IMP/checkpoints/mixed_gf0_step12999_final/
/data1/DATA_IMP/checkpoints/visrobot01_only_best_step6000/
```

启动命令里 `checkpoint_dir:=` **直接指向这一层**, 例如 `checkpoint_dir:=kai0/checkpoints/mixed_gf0_best_at_4k`.

### B. run-style (训练全量 ckpt, 多 step 共存)
适用于"params + assets + train_state"的完整训练快照, 或同一 run 下要保留多个 step 的对比:

```
/data1/DATA_IMP/checkpoints/<config_name>/<run_id>/<step>/
├── _CHECKPOINT_METADATA
├── assets/
├── params/
└── train_state/               # 续训才需要; 推理可剥掉
```

例:
```
/data1/DATA_IMP/checkpoints/pi05_flatten_fold_mix_vis600/mix_vis600_v1/38000/
/data1/DATA_IMP/checkpoints/pi05_pick_place_box_kai0_unfreeze_20k/p_unfreeze_20k_v1/4000/
```

`checkpoint_dir:=` 指向**到 step 那一层**.

> 同一个 ckpt 选 A 或 B 中一种, 取决于它来自训练全量保存还是裁剪过的推理-only 包. **不要混用**.

---

## 3. 必备四件套 (autonomy 推理读取顺序)

每次新 ckpt 上线前, **必须**确认以下 4 件全部就位:

### 3.1 `<checkpoint_dir>/params/`
- 内容: orbax 格式权重 (含 `_sharding`, `array_metadatas/`, `d/`, `ocdbt.process_0/`)
- 加载点: `openpi/policies/policy_config.py:57` `_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16)`
- 缺则: `FileNotFoundError`

### 3.2 `<checkpoint_dir>/_CHECKPOINT_METADATA`
- 内容: orbax 校验 sidecar (列出 params/train_state/assets handler 类型)
- 缺则: orbax `Checkpoint not found` / handler mismatch
- 注: `train_state/` 不是必需 — 只读权重的推理可以不带; 想续训才需要

### 3.3 `norm_stats.json` — **推理永远从 ckpt-side 加载** (重点!)

policy_config.py 推理时**只看一个位置**: `<checkpoint_dir>/assets/<asset_id>/norm_stats.json`.

(2026-04-28 校正: 早先版本 README 把训练时的 `AssetsConfig(assets_dir=...)` 行为也写进了"推理查找优先级表", 错了.
推理 (`policy_config.create_trained_policy`) 直接调用 `_checkpoints.load_norm_stats(checkpoint_dir / "assets", asset_id)`,
**完全不**使用 `AssetsConfig.assets_dir`. 那个字段只在训练 `create_base_config` 时生效.)

`asset_id` 怎么决定:

| Config 写法 | asset_id 值 | ckpt-side 路径 |
|---|---|---|
| `AssetsConfig(asset_id="mixed_1")` 显式设了 | `"mixed_1"` (相对) | `<ckpt>/assets/mixed_1/norm_stats.json` |
| 没写 AssetsConfig (默认) | `repo_id` 全字 (**绝对路径**!) | pathlib `<ckpt>/assets/<absolute_repo_id>` 被吞 → 等价 **`<repo_id>/norm_stats.json`** |

两种实操落法 (二选一):

**(a) 显式 asset_id (相对)**: 把 norm_stats 直接放进 ckpt 内部 → `<ckpt>/assets/<asset_id>/norm_stats.json`
- 例: `mixed_1` 系列, mixed_gf0_*, visrobot01_only_* — 都用 `pi05_flatten_fold_awbc_from_official_mixed` config (asset_id="mixed_1")
  → ckpt 里必须有 `assets/mixed_1/norm_stats.json`

**(b) asset_id 默认 = repo_id 绝对**: 把 norm_stats 放在数据集路径 → `<repo_id>/norm_stats.json`
- 例: `mix_vis600` / `mixed_visrobot01` — 没设 AssetsConfig
  → `kai0/data/Task_A/self_built/mix_vis600/base/norm_stats.json` (与训练时 dataset 自带的同一份)

⚠️ **检查清单时必看**: 用 (a) 的话, ckpt 自带 `assets/<asset_id>/norm_stats.json` 才能跑; 用 (b) 的话, dataset 路径下要有 `norm_stats.json`.

⚠️ **用错 norm_stats 真机会乱跑** — 关节角被错误 mean/std 反归一化, 输出 garbage. 拉新 ckpt 时**必须**对齐 md5 与训练时落盘的那份.

缺则报: `FileNotFoundError: Norm stats file not found at: <expected path>` (照这个 path 去找哪个 asset_id, 然后把 norm_stats.json 放过去).

### 3.4 `kai0/src/openpi/training/config.py` 里有对位 TrainConfig
- 加载点: `policy_inference_node.py:540` `_config.get_config(config_name)`
- 决定: `model` (Pi0Config / Pi0RTCConfig)、`data` (LerobotAgilexDataConfig 决定 transform & norm_stats path)、`weight_loader`、`repo_id`
- ⚠️ **gf 上的 config.py ≠ sim01 上的 config.py**, 不会自动同步. 在 gf 注册的新 config 必须**手动 port 到 sim01 这份**, 否则推理直接 `ValueError: Config '<name>' not found.`
- sim01 路径: `/data1/tim/workspace/deepdive_kai0/kai0/src/openpi/training/config.py`

---

## 4. 命名约定

### 4.1 扁平拓扑 (A 类)
```
<run_label>_<milestone>     # 例: mixed_gf0_best_at_4k, mixed_gf0_step12999_final, visrobot01_only_2k_step1999_gf0
```
- `run_label`: 数据/训练源 (`mixed_gf0`, `visrobot01_only`, `pure_vis600`, `awbc_v2_robust` 等)
- `milestone`: `best_at_<step>` / `step<N>_final` / `step<N>` — 一眼看出是 best ckpt 还是末尾 ckpt, 是否对应同系列其它点

### 4.2 run-style 拓扑 (B 类)
```
<config_name>/<run_id>/<step>     # 复刻 openpi 训练时目录结构
```
- `config_name`: 与 sim01 `config.py` 里的 TrainConfig.name 完全一致
- `run_id`: 训练发起时定的 exp_name (例如 `mix_vis600_v1`, `p_unfreeze_20k_v1`)
- `step`: 整数 step 号 (例 `38000`, `4000`)

### 4.3 项目级 asset 目录
共享给多个 ckpt 当 init weight / norm_stats 来源的"训练资产", 单独放 `Task_*/<asset_id>/`:

```
/data1/DATA_IMP/checkpoints/Task_A/mixed_1/
├── norm_stats.json
└── params/                    # 用作 weight_loader 的 init 源
```

不要和实际 ckpt 实体混到同一层级.

---

## 5. 上架新 ckpt 的标准流程

### 5.1 拉 ckpt 到本地 (gf → sim01, ~12 GB 走 TOS, ~2 min)
```bash
# gf1 (port 11111):
ssh -p 11111 tim@<host> "echo tim | sudo cp /vePFS/.../<name>.tar /transfer-shanghai/KAI0/<name>.tar"

# sim01:
cd /data1/DATA_IMP/KAI0
/data1/miniconda3/bin/python from_tos_file.py \
    --object_key KAI0/<name>.tar \
    --file /data1/<name>.tar
```

### 5.2 落盘到规范位置 (直接到 checkpoints/, 不要走中转)

**A 类 (扁平)**:
```bash
mkdir -p /data1/DATA_IMP/checkpoints/<name>
tar -xf /data1/<name>.tar -C /data1/DATA_IMP/checkpoints/<name>/
```

**B 类 (run-style)**:
```bash
mkdir -p /data1/DATA_IMP/checkpoints/<config_name>/<run_id>
tar -xf /data1/<name>.tar -C /tmp/extract_<name>
# 假设 tar 解开是 params/ + assets/ + _CHECKPOINT_METADATA, 整体当作一个 step
mv /tmp/extract_<name> /data1/DATA_IMP/checkpoints/<config_name>/<run_id>/<step>
```

⚠️ **永远先 `mkdir -p <target_dir>` 再 `tar -xf -C <target_dir>/`**, 否则裸 params/assets 会污染上层目录 (历史踩坑 §8).

### 5.3 norm_stats 对位
```bash
# 来源: gf 上训练时落盘的 norm_stats.json (run-level), 或 dataset 自带 base/norm_stats.json
# 落到: config 实际查找的路径 (见 §3.3 表)
scp -P 11111 tim@<host>:<gf_path>/norm_stats.json <local_target_path>
md5sum <local_target_path>   # 必须等于 gf 源 md5
```

### 5.4 同步 config.py (如果是新 config)
```bash
# 把 gf 上的 TrainConfig 块 (用 grep -n 'name="<NAME>"' 找) port 到 sim01:
#   /data1/tim/workspace/deepdive_kai0/kai0/src/openpi/training/config.py
# 验证:
kai0/.venv/bin/python -c "from openpi.training import config as c; print(c.get_config('<NAME>').name)"
```

### 5.5 改 `start_scripts/start_autonomy_temp.sh`
- 注释掉旧 active launch 块
- 加新选项块, 注明: 数据源 / 训练 schedule / norm_stats md5 / 备注 (例如 "best step", "long-schedule final" 等)
- `checkpoint_dir:=kai0/checkpoints/<...>` 指向 §5.2 落盘的 ckpt 路径

### 5.6 清理
```bash
# TOS 对象 (root-owned, 必须从 gf1 fuse 删, 不然 sim01 sdk 报 AccessDenied):
ssh -p 11111 tim@<host> "echo tim | sudo rm /transfer-shanghai/KAI0/<name>.tar"
# 本地 tar:
rm /data1/<name>.tar
# gf 源 tar (不再需要分发):
ssh -p 11111 tim@<host> "rm /vePFS/tim/workspace/deepdive_kai0_tmp/data/<name>.tar"
```

---

## 6. 兜底 checklist (上线前过一遍)

- [ ] `<checkpoint_dir>/params/` 存在且 `_sharding` / `ocdbt.process_0/` 都齐
- [ ] `<checkpoint_dir>/_CHECKPOINT_METADATA` 存在
- [ ] `norm_stats.json` 在 config 期望的路径下 (扫一眼 config.py 找 `AssetsConfig` / `repo_id`, 决定查哪)
- [ ] `norm_stats.json` md5 与 gf 训练源一致
- [ ] sim01 `config.py` 里 `get_config('<name>')` 能调到; 没有就 port 过来
- [ ] `start_autonomy_temp.sh` 只有一个非注释 launch 行
- [ ] 真机就绪: 3 RealSense 在线 / 4 CAN up / GPU 显存 ≥ 12 GB

---

## 7. 已上架的 ckpt 索引 (上架时追加, 下架时移除)

"norm_stats 路径" 列写的是该 ckpt 实际 inference 读取的位置 (即 §3.3 公式套出的路径); "内容来源" 列写文件内容的 md5 出处, 用于追溯训练时 dataset 实际采的 stats.

### A. 扁平 (lightweight inference, asset_id="mixed_1" 显式) — §3.3 (a) 类

inference 读: `<ckpt>/assets/mixed_1/norm_stats.json` (config 都是 `pi05_flatten_fold_awbc_from_official_mixed`).

| 路径 | 内容来源 (md5) | 备注 |
|---|---|---|
| `mixed_gf0_best_at_4k` | `b206072c...` (Task_A/mixed_1/norm_stats.json) | gf0 mixed run early-stop |
| `mixed_gf0_step12999_final` | `b206072c...` 同上 | gf0 mixed run final |
| `visrobot01_only_best_step6000` | `b206072c...` 同上 (⚠️ 实际训练可能用纯 visrobot01 stats, 出问题再换) | gf0 visrobot01_only best |
| `visrobot01_only_2k_step1999_gf0` | `b206072c...` 同上 | 2k 短 schedule final |
| `pi05_flatten_fold_awbc_from_official_mixed/beta_official_v1/19999` | `b206072c...` 同上 | base AWBC official mixed full schedule |

### B. run-style (无 AssetsConfig, asset_id 默认 = repo_id 绝对) — §3.3 (b) 类

inference 读: `<repo_id>/norm_stats.json` (绝对路径, ckpt 内 assets/ 不参与).

| 路径 | inference 读取的 norm_stats | 备注 |
|---|---|---|
| `pi05_flatten_fold_mix_vis600/mix_vis600_v1/38000` | `kai0/data/Task_A/self_built/mix_vis600/base/norm_stats.json` (md5 `38bff549...`) | mix_vis600 best @ step 38000 |
| `pi05_flatten_fold_mixed_visrobot01/mixed_visrobot01_1500/49999` | `kai0/data/Task_A_mixed_gf1/base/norm_stats.json` (md5 `731fb5df...`) | xyh 在外部机器训, 50k schedule final |
| `pi05_pick_place_box_kai0_unfreeze_20k/p_unfreeze_20k_v1/4000` | (按复用 config 决议) | gf0 unfreeze 20k schedule, step 4000 best |

### Project assets (共享 init 源)

| 路径 | 内容 |
|---|---|
| `Task_A/mixed_1/` | `norm_stats.json` + `params/` (用作多个 fold/awbc config 的 weight_loader init + 默认 norm_stats) |

---

## 8. 历史踩坑 (避免重蹈)

- **2026-04-24 mixed_gf0_best_at_4k 上线**: tar 解开是裸 `params/assets/_CHECKPOINT_METADATA`, 没有外层目录, 直接污染上层根. → **现在固定 `tar -xf <tar> -C <target_subdir>/`, 永远先 mkdir 再 -C 进去**.
- **同 ckpt assets/ 是空的**: 训练只落 run-level norm_stats, 不在 per-step assets/. → 必须手动从 run-level 或 dataset 路径 copy norm_stats 到 config 期望的位置.
- **2026-04-25 norm_stats 错版本**: 同名 `norm_stats.json` 在 `kai0/data/Task_A/advantage/` (5356 B) 和 `kai0/checkpoints/Task_A/mixed_1/` (5343 B) **内容不同**. 用错的话关节会偏. → **md5 对齐训练时实际用的那份**, 不要凭文件名.
- **2026-04-27 mix_vis600 first launch crash**: gf config.py 注册了 `pi05_flatten_fold_mix_vis600`, sim01 没 port 过来 → `ValueError: Config not found`. → **新 config 永远要 port 到 sim01 config.py 并跑一次 `get_config(name)` 验证**.
- **TOS 对象 sudo cp 后 root-owned, sim01 sdk 删不掉**: AccessDenied. → 从 gf1 fuse 用 sudo rm 删, 不要在 sim01 走 SDK delete_object.
- **2026-04-27 整理布局**: 之前是"实体落 `/data1/DATA_IMP/KAI0/ckpt_downloads/<name>/` + symlink 在 `kai0/checkpoints/<name>`"两层结构, 不必要; 已合并为单层 — 全部 ckpt 直接落 `/data1/DATA_IMP/checkpoints/<name>/`, `kai0/checkpoints` 整体一个 symlink 指过来. ckpt_downloads/ 只剩 3 个孤儿 .tar 文件可以择机清理.
- **2026-04-28 mixed_1 baseline 启动炸 norm_stats**: 选 K (`pi05_flatten_fold_awbc_from_official_mixed` config + `Task_A/mixed_1` ckpt) 启动报 `FileNotFoundError: kai0/checkpoints/Task_A/mixed_1/assets/mixed_1/norm_stats.json`. mixed_1 ckpt 里 norm_stats 只在根目录 (`Task_A/mixed_1/norm_stats.json`), 没在 `assets/mixed_1/` 下. → 复制一份到 `assets/mixed_1/`, 解决. **同时校正 §3.3 文档**: 推理只从 ckpt-side `<ckpt>/assets/<asset_id>/` 加载, AssetsConfig.assets_dir 只在训练时用, 之前 README 那张优先级表把两边混了, 误导.
