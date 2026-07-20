# LaWAM 本地复现 — 文件存放位置报告(2026-07-12)

> 各部分放哪、多大、干什么。复现见 [[reference_lawam_repro_local]] / `../lmwm/docs/LAWAM_reproduce_and_kai0_sft_plan_2026-07-12.md`。

## 总览:三处物理位置

| 位置 | 内容 | 是否进 git |
|---|---|---|
| `deepdive_kai0/lmvla/lawam/` | LaWAM 代码 + 权重 + eval 结果 + 编排脚本 | ❌ 整个 `/lawam/` 已 gitignore(外部 repo + 大权重) |
| `deepdive_kai0/lmvla/LIBERO/` | LIBERO 模拟器(vanilla clone) | ❌ `/LIBERO/` 已 gitignore |
| `/vePFS/tim/robotwin_client_deps/` | RoboTwin 用的 client 依赖(--target, 不碰共享 env) | ❌ 在仓库外 |
| conda envs(`~/miniconda3/envs/`) | lawam / libero 两个 env | — |

---

## 1. `lmvla/lawam/`(主目录,复用 vendored LaWAM clone)

### 代码(LaWAM 本体, 来自 github.com/RLinf/LaWAM)
| 路径 | 说明 |
|---|---|
| `starVLA/` (1.2M) | LaWAM 模型/dataloader/训练循环/config(核心) |
| `latent_action_model/` | LaWM/LAM 世界模型代码 |
| `deployment/model_server/` | 策略服务器(eval 用, server_policy.py) |
| `examples/LIBERO/` `examples/Robotwin/` (834K) | 两个 benchmark 的 eval 脚本 + adapter |
| `train_lawam.sh` / `train_lawam_distributed.sh` | SFT 训练入口(单机/多机) |

### 权重(全下载, ~28G)
| 路径 | 大小 | 内容 |
|---|---|---|
| `results/Checkpoints/qwen3_weights/` | 4.0G | Qwen3-VL-2B-Instruct(base VLM) |
| `results/Checkpoints/pretrain/lawam_pretrain/` | 6.7G | SFT 初始化 ckpt |
| `results/Checkpoints/libero/lawam_libero_sft_release/` | 6.7G | LIBERO SFT ckpt(eval 用) |
| `results/Checkpoints/robotwin/lawam_robotwin_sft_release/` | 6.7G | RoboTwin SFT ckpt(eval 用) |
| `weights/dinov3-vitb16-pretrain-lvd1689m/` | 327M | DINOv3 编码器(**ModelScope 下**, HF gated) |
| `ckpts_dl/` | 2.8G | LAM ckpt(`pytorch_model.pt` + `dino_large_vae.yaml`) |
| `latent_action_model/logs/dino_large_vae/lam_release/` | (symlink) | → 指向 `ckpts_dl/`(config 期望路径) |

### Eval 结果(~1.7M)
| 路径 | 内容 |
|---|---|
| `results/eval_runs/libero/lawam_libero_sft/<时间戳>/` | LIBERO 各 suite 的 `summary.json` + `eval.log`(libero_10=98.0%) |
| `results/eval_runs/robotwin/lawam_robotwin_sft_release__demo_clean/<时间戳>/` | RoboTwin 各 task 的 `summary.json`(beat_block_hammer=90.0%) |

### 我加的编排脚本 + 文档(本 session 产出)
| 文件 | 用途 |
|---|---|
| `dl_weights.py` | HF 权重下载(hf-mirror) |
| `build_env.sh` / `build_libero_env.sh` / `install_libero_lean.sh` | env 搭建 |
| `run_libero_eval.sh` / `run_robotwin_eval.sh` | eval 启动 |
| `robotwin_python_wrapper.sh` | ROBOTWIN_PYTHON wrapper(注入 VK_ICD + client deps PYTHONPATH) |
| `*.log` | 各步骤日志(下载/装env/eval) |
| `STORAGE_LAYOUT.md` | 本报告 |

---

## 2. `lmvla/LIBERO/`(426M) — LIBERO 模拟器
vanilla LIBERO(ghfast.top 代理拉的 tarball 解压)。已 `pip install -e` 进 libero env(PEP420 命名空间,用 `libero_ns.pth`)。**已 patch 4 处 `torch.load` 加 `weights_only=False`**(torch2.6 兼容)。

## 3. `/vePFS/tim/robotwin_client_deps/`(8.9M) — RoboTwin client 依赖(tim-owned)
websockets/json_numpy/msgpack/rich/omegaconf/tyro/antlr4(4.9.3)/... + **`sitecustomize.py`**(warp.torch shim)。运行时由 wrapper 前置到 PYTHONPATH,**不碰 huanqian 共享 env**。

## 4. conda envs
| env | 大小 | 用途 |
|---|---|---|
| `~/miniconda3/envs/lawam` | 6.7G | 策略 server(torch2.6+cu124, starVLA) |
| `~/miniconda3/envs/libero` | 6.5G | LIBERO 模拟器 worker(torch2.6, robosuite, mujoco3.3.2) |
| `/vePFS/HuanQian/conda_envs/RoboTwin` | (共享,非我们) | RoboTwin 模拟器(sapien3.0.0b1, curobo) — 只用不改 |

---

## 5. 换机/清理提示
- **可删可重生成**:`results/eval_runs/`(结果小,建议留)、所有 `*.log`、`LIBERO/`(重拉)、`robotwin_client_deps/`(重装)。
- **大头**:`results/Checkpoints/`(25G)+ `ckpts_dl/`(2.8G)= 权重,删了要重下(hf-mirror + ModelScope)。
- **不在仓库**:整个 lawam/LIBERO/client_deps 都 gitignore,换机需重跑搭建脚本 + 重下权重。
