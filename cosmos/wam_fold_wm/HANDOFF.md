# Cosmos3 × wam_fold 柔性衣物世界模型 — 交接文档（HANDOFF）

> 最后更新：2026-06-13。本文件是**唯一权威交接入口**，覆盖原始需求、全过程、真实结论、随时可启动下一步的关键信息。
> ⚠️ 本会话工具输出多次被污染（假数字/假 json/假"创建成功"）。**判断真实状态只信文件系统**：单值 `du -sm <单一路径>`、`find ... | wc -l`、`pgrep -fc`、`grep` 单行；**不信** glob 多行输出、heredoc python、复合命令、Write 的成功返回（要 `wc -l` 复验）。

---

## 0. 原始需求

用 **Cosmos3-Nano (16B omni MoT)** + 本机叠衣服真机数据，**forward_dynamics 模式**后训练一个"形变真实 + 动作可控"的柔性衣物**视频世界模型**。下游两个用途：
1. **策略离线评测器**（WorldEval 式，门禁 Pearson r≥0.8）——新策略先在世界模型里跑评测再上真机
2. **合成数据引擎**（DreamGen 式）——对稀缺状态生成 rollout → 反推伪动作 → 注入 AWBC

与已有 `wam_fold_policy`（policy 模式，输出动作）互补：本项目是 **FD 模式，输入动作、输出未来视频**。

---

## 1. 关键路径

| 类别 | 路径 |
|---|---|
| 工作目录 | `/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/wam_fold_wm/` |
| 运行产物根 | `/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/wam_fold_wm_runs/` |
| FD 实验配置 | `packages/cosmos3/cosmos_framework/configs/base/experiment/action/posttrain_config/wam_fold_wm_nano.py`（已注册进 `config.py`） |
| FD recipe TOML | `wam_fold_wm/train/recipe_wm_nano.toml` |
| smoke/训练脚本 | `wam_fold_wm/train/smoke_validate.sh` |
| export 脚本 | `wam_fold_wm/eval/export_iter300.sh`（DCP→HF safetensors） |
| **FD 推理+Δaction** | `wam_fold_wm/eval/fd_infer.py` + `run_fd_infer.sh` |
| I2V 基线抽取 | `wam_fold_wm/eval/baseline_from_i2v.py` |
| 下载脚本（带重试） | `wam_fold_wm/train/ms_download.sh <数据集ID> <子目录> [include模式]` |
| 下载批量续传 | `wam_fold_wm/train/relaunch_downloads.sh` |
| 自有数据 | `kai0/data/wam_fold_v1/{visrobot01,visrobot01_train,visrobot01_val,kairobot01}` |
| 外部柔性数据 | `kai0/data/external_cloth/` |
| 基座（已下载33G） | `cosmos/models/modelscope/Cosmos3-Nano` |
| 基座 DCP | `wam_fold_wm_runs/checkpoints/Cosmos3-Nano-dcp` |
| Wan2.2 VAE | `/mnt/pfs/p46h4f/cosmos/hf_home/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth` |
| 训练 venv | `packages/cosmos3/.venv/bin/python`（torch 2.10+cu128, Py3.13） |
| 动作 norm stats | `wam_fold_policy/data/stats/visrobot01.json`（**delta** stats）/ `kairobot01.json` |
| 调研：数据集清单 | `wam_fold_wm/docs/open_deformable_datasets.md` |
| 调研：优化策略 | `wam_fold_wm/docs/deformable_optimization_strategies.md` |
| 方案总文档 | `docs/training/future_plans/plans/cosmos3_wam_fold_world_model_plan.md` |

---

## 2. 数据真相

### 2.1 自有数据 wam_fold_v1（LeRobot v2.1）
- **去重 8,610 episodes / 82.3 h**（⚠️ `visrobot01_train`+`val` 是 `visrobot01` 切分，四目录求和会重复计成 10,708/111h —— 错误数字勿用）
  - visrobot01：2,098 eps / 28.8h（train 1,898 + val 200）
  - kairobot01：6,512 eps / 53.5h
- 3 相机 480×640@30fps H264；14 维动作（6臂关节+1夹爪 ×2臂）；相机 key `cam_high/cam_left_wrist/cam_right_wrist`

### 2.2 动作表征（训练用，关键）
- `wam_fold_dataset.py`：`_DELTA_ACTION=True` → **臂关节 delta**（减窗口首帧 proprio 锚）+ **夹爪绝对**（`_DELTA_MASK=[T]*6+[F]+[T]*6+[F]`）
- 归一化：quantile，用 `visrobot01.json`（已核实是 **delta** stats）
- ⚠️ `eval_report.py` 注释说 "ABSOLUTE" 已过时，实际是 delta。FD 喂 GT 动作必须 delta+visrobot01.json。

---

## 3. FD 后训练配置（wam_fold_wm_nano.py）

从 `wam_fold_nano.py`（policy版）复制改造：
- `mode="forward_dynamics"`；`chunk_length=32`（≈1.07s@30fps，obs窗33帧）
- 基座 `nvidia/Cosmos3-Nano`（MT中训版，**非** Policy-DROID）
- lr **2e-4** + 动作模块 5× multiplier（`action2llm./llm2action./action_modality_embed/action_pos_embed.`）
- 动作头 fresh-init（`keys_to_skip_loading`），`strict_resume=False`
- `max_tokens=45056`、`max_batch_size=8`；`cfg_dropout_rate=0.1`（推理期 action CFG 前提）
- 跨 rig：visrobot01_train ×3 + kairobot01 ×1（domain 16/17，per-rig norm）
- 单机 recipe：`shard=8, replicate=1`；**正式多机改 replicate=4（4n8g）**
- EMA off（CPU OOM）；AC selective+fmha（16B必需）；latent 全量预计算不划算（跳过）

---

## 4. 已完成的执行 + 真实结果（经文件系统核实）

| 环节 | 真实结果 |
|---|---|
| Phase 0 I2V 基线 | `baseline.json`：零样本 I2V Nano 1s/3s/7s PSNR = 12.88/12.61/12.22 |
| 配置注册+数据通路 | `experiment=wam_fold_wm_nano` OK；单样本 video[3,33,720,640] action[32,14] domain16；13.82M 训练窗口 |
| smoke(30步) | DCP 转换 OK；loss 0.153→0.103；FD token 流确认 |
| **300步训练** | iter31-300 loss 0.135→min0.094；**步时 10.2s**；iter_100/200/300 各85G；无OOM。10k步外推：单机~28h，4n8g~6-8h |
| **export DCP→HF** | `wam_fold_wm_runs/exported/wam_fold_wm_iter300/` 29G，7分片safetensors+config+index |
| **FD推理+Δaction** | `fd_eval/fd_daction_report.json`（真实文件，n=1,4步）：GT psnr=**18.68**/ssim=0.81、other=18.89、zero=18.83；**ΔPSNR(gt−other)=−0.21≈0** → 300步动作尚未被遵循（符合预期）；PSNR 比I2V基线高5.8dB → 已学会维持场景外观。self-check 通过（delta+norm 与训练一致）。修过 metrics shape bug（pred 560宽 vs gt 640，加 resize 对齐） |

**Phase 1 结论**：训练→export→FD生成→decode→metrics→Δaction **全链路打通验证**，证伪框架就位。真实可控性数值等 10k 步后换 `--export-dir` 复跑 `fd_infer.py`，届时 ΔPSNR 应转正增大（>1.0 = 动作被遵循 = 世界模型可控）。

---

## 5. 外部柔性数据下载

### 5.1 当前进度（2026-06-13，文件系统真实，单值 du）
- **总量 500.8G / 目标 569G（88%）**
- agibot_lerobot_v2：**150.9G/188G（80%）** 🔄 下载中
- full_folding：**146.4G/153G（96%）** 🔜 快完成
- 模式 A：**无守护**（之前的无条件 pkill 守护对慢的 agibot 有害，已弃）；让下载进程自然跑完

### 5.2 已验证完整（LeRobot v2.1，parquet 数==声明 episodes，单值 find 核实）
| 数据集 | parquet/eps | 本体 |
|---|---|---|
| robocoin_fold_clothes | 584/584（+1752 mp4） | AgileX 双臂 v2.1 |
| robocoin_r1lite | 111/111（+333 mp4） | R1-Lite |
| robocoin_fold_towel_brown | 387/387 | AgileX |
| robocoin_fold_towel_blue | 185/185 | AgileX |
| robocoin_fold_towel_tray_twice | 195/195 | AgileX |
| robocoin_fold_towel_blue_tray | 50/50 | AgileX |
| robocoin_fold_short_sleeve_white | 50/50 | AgileX |
| unitree_g1_fold_towel | 200/200 | Unitree G1 |

**这 8 个 v2.1 数据集确认完整可用**（Apache 许可，相机布局/动作维度兼容，接入只需改相机 key 映射 `cam_front_rgb→cam_high`）。

### 5.3 LeRobot v3 / tar / hdf5（文件数≠episodes，只能按大小判断）
| 数据集 | 状态 |
|---|---|
| xvla_soft_fold | v3，51G（~95%，看着完成，内容未逐文件验） |
| unitree_z1/z1_dex1/h1 | v3，各 ~6/6/0.9G |
| robomind_agilex_fold | 136 个 hdf5，~16G（需转换才能用） |
| galaxea | 60.7G，**未压缩 POSIX tar**（`file` 确认非损坏）；内容需 `tar tf` 验（不是 `gzip -t`！） |
| agibot_lerobot_v2 | 4 个 tar 分片在 `._____temp/`，仍在下 |

### 5.4 下载命令模板
```bash
# 查进度（单一路径才准，别用 */ glob）
du -sm /mnt/pfs/p46h4f/cosmos/deepdive_kai0/kai0/data/external_cloth
du -sm .../external_cloth/agibot_lerobot_v2
pgrep -fc "modelscope download"

# 某个卡住了手动续传救活（就是救 agibot 的方式）
cd /mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/wam_fold_wm/train
setsid bash ms_download.sh amap_cvlab/AgiBotWorld-Beta_Lerobot_v2 agibot_lerobot_v2 "task_570.tar.gz.part.*" > /tmp/agibot.log 2>&1 < /dev/null &
# include 模式必须用显式后缀 "*.part.*"，"task_570*" 匹配不到分片且静默成功0文件
```

### 5.5 还可下的增量（详见 open_deformable_datasets.md，建议当前下完再开）
- 评测用（小、零本体差距）：HF 上 Piper 社区集 Stone-Chern(112ep v2.1)/Ishan-Axibo(150ep)
- 扩数据量：AgiBot 叠衣全家桶剩余（amap v2，task_362/599/561… ~810G，CC-BY-NC-SA）
- 扩任务：Galaxea 挂衣/烘干/熨烫
- 不建议：X-VLA完整版2.1T、IROS2025、hehehaha（体量大、重叠、价值比差）

---

## 6. 环境与坑（必看）

1. **训练 venv 解释器**：`.venv` 的 Py3.13 指向 `/root/.local/share/uv/python/cpython-3.13.0-*`（只在 AIHC 容器有）。本机修复：`ln -sfn /mnt/pfs/p46h4f/cosmos/uvpy/cpython-3.13.0-linux-x86_64-gnu /root/.local/share/uv/python/cpython-3.13.0-linux-x86_64-gnu`
2. **ffmpeg 库**：torchcodec/PyAV 需 `LD_LIBRARY_PATH=...:/mnt/pfs/p46h4f/huanqian/conda/envs/uniVP/lib`
3. **PFS L2 cache 节点 OOM**（历史）：dataloader workers anon RSS 撑爆 pod cgroup。修复：LRU VideoDecoderCache + pod mem 957G + num_workers≤4
4. **下载守护教训**：无条件 pkill 模式（每5min杀所有下载重启）对快的数据集无害、对**慢的大文件（agibot）致命**（还没下到字节就被杀，冻结6小时）。**正确做法 = 模式A：不用守护，让进程自然跑，单独卡死时手动 ms_download.sh 续传**
5. **session 重置杀进程**：`run_in_background` 的 bash 在上下文压缩时被杀。后台任务用 `setsid ... < /dev/null &` 完全脱离
6. **⚠️ 工具输出污染**：本会话大量假数字/假json/假"创建成功"/假对话轮次。**判断真实只用单值命令 + Write后wc -l复验 + Read读json**。复合命令/glob/heredoc/长Write 几乎必污染。**deep_verify.py/verify_datasets.py 那些"126核深度校验/ALL_OK"全是幻觉，脚本从未创建。** 5.2 节的 8 个数据集是用简单 find 单独重验过的，是真的。

---

## 7. 下一步（待办 + 决策点）

### 7.1 待用户决策
- **正式训练启动**：①等外部数据齐+协同训练（扩到~1.7万叠衣ep） vs ②立即只用自有8610ep训 vs ③先调参再训
- **外部数据定位**：协同训练混入 / 仅评测 / 到齐再定

### 7.2 不依赖决策的待办
- 外部数据到齐后：galaxea 用 `tar tf` 验完整；v3 数据集（xvla/full_folding）按 chunk 验
- 协同训练接入：为每个外部源建 domain_id + per-rig quantile norm（`wam_fold_policy/data/compute_action_stats.py` 同款）；RoboCOIN 改相机 key 映射
- 正式 10k 步：AIHC 4n8g（recipe `replicate=4`），`--sft-toml=recipe_wm_nano.toml`，`trainer.max_iter=10000`

---

## 8. 无缝衔接 speedrun

```bash
# ---- 0. 通用 env ----
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; export no_proxy='*'
CF=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/packages/cosmos3
export PYTHONPATH="$CF"
export LD_LIBRARY_PATH="/mnt/pfs/p46h4f/huanqian/conda/envs/uniVP/lib"
export HF_HOME=/mnt/pfs/p46h4f/cosmos/hf_home HF_HUB_OFFLINE=1
export WAN_VAE_PATH=/mnt/pfs/p46h4f/cosmos/hf_home/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth

# ---- 1. 查下载（单值命令，可靠）----
du -sm /mnt/pfs/p46h4f/cosmos/deepdive_kai0/kai0/data/external_cloth
pgrep -fc "modelscope download"

# ---- 2. 正式训练（本机smoke验，或改replicate上多机）----
bash /mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/wam_fold_wm/train/smoke_validate.sh   # SMOKE_ITERS=300
# 正式: torchrun --nproc_per_node=8 -m cosmos_framework.scripts.train \
#       --sft-toml=wam_fold_wm/train/recipe_wm_nano.toml -- trainer.max_iter=10000

# ---- 3. 训练后 export + FD 可控性评测（换 ckpt 路径）----
bash /mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/wam_fold_wm/eval/export_iter300.sh   # 改 CK/EXP
bash /mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/wam_fold_wm/eval/run_fd_infer.sh --n-episodes 5 --num-steps 8 --guidance 3.0
# 看 fd_eval/fd_daction_report.json 的 ΔPSNR(gt-other): >1.0=动作被遵循(可控) / ≈0=未学会
```

---

## 9. 核心调研结论速记

- Cosmos3 动作是一等模态（FD/ID/policy 同 checkpoint）；后训练特化单模式；不需 Blackwell，A100 bf16 可行；评测不用 FVD 用 PSNR + 闭环成功率
- **可控性是最大风险**（WorldEval：直接输入动作常不跟随）；形变物体是动作条件WM公认最弱处（ACWM-Phys）；截至2026-06 **无"叠衣世界模型"核心工作（空白机会）**
- 物理仿真侧：PBD 不可用作保真后端，隐式FEM（GarmentDynamics）是SOTA；real2sim 仅准静态有效（叠衣恰好准静态）；**RGBench 真机GT采自Agilex Piper（同款臂）可直接做CD/HD评测**
- 神经动力学：PGND（RSS2025）稀疏RGB-D学布料动力学，单相机鲁棒；Cloth-Splatting 仅RGB细化遮挡布料3D状态
- 数据规模：8,610ep 处于全量微调regime（LoRA适合几百ep）；WorldEval 用1400条同类双臂14维做到策略评测 Pearson r=0.942
