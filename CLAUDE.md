# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a workspace for reproducing and deploying **χ₀ (kai0)** — a resource-efficient robotic manipulation framework built on top of [openpi](https://github.com/Physical-Intelligence/openpi) (Physical Intelligence's π₀/π₀.₅ models). The project focuses on Task A (T-shirt flatten & fold) with dual-arm Agilex Piper robots.

The workspace has two main areas:
- **`kai0/`** — The core kai0 repository (a fork/clone of the kai0 project), containing model code, training scripts, and all three technical modules
- **Top-level scripts/docs/ros2_ws** — Local deployment scripts, ROS2 workspace, and reproduction guides specific to this setup

## Hardware Setup

- **sim01** (this machine): Dual RTX 5090 32GB — used for inference serving and IPC (cameras, CAN, ROS2, Piper SDK)
- **gf0/gf1** (remote training): 8×A100 80GB each at `14.103.44.161` (ports 55555/11111) — used for full fine-tuning, AWBC training, advantage estimator training
- External network access on gf0/gf1 uses SSH reverse tunnel proxy on port 29290

## Build & Development Commands

All commands below run from **`kai0/`** directory unless otherwise noted.

### Environment Setup
```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

Python version: 3.12 on sim01 (required for ROS2 Jazzy compatibility); `pyproject.toml` requires `>=3.11`. Package manager: `uv`. Virtual env at `kai0/.venv/`. The uv workspace includes `packages/openpi-client` as a member.

The top-level `install.sh` handles full-stack installation (sys deps, ROS2 Jazzy, Piper SDK, Python env, checkpoints). Supports `--skip-ros`, `--skip-venv`, `--skip-ckpt` flags to skip sections.

### Linting
```bash
uv run ruff check .          # lint
uv run ruff check --fix .    # lint + autofix
uv run ruff format .         # format
```

Ruff config: line-length=120, target py311. Excludes `docker/`, `third_party/`, `src/openpi/models_pytorch/transformers_replace/*`. Pre-commit hooks (in `kai0/.pre-commit-config.yaml`) run `uv-lock`, ruff lint (with `--fix`), and ruff format; `third_party/` is excluded. CI also runs these via `.github/workflows/pre-commit.yml`.

### Testing
```bash
uv run pytest --strict-markers -m "not manual"    # all non-manual tests
uv run pytest src/openpi/models/pi0_test.py       # single test file
uv run pytest -k "test_name"                      # single test by name
```

Test paths: `src/`, `scripts/`, `packages/`. Tests marked `manual` are excluded by default.

### Training (JAX)
```bash
# Compute normalization stats first
uv run python scripts/compute_norm_states_fast.py --config-name <config_name>

# Full fine-tuning (JAX, uses openpi training loop)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py <config_name> --exp_name=<name>
```

Config names (defined in `src/openpi/training/config.py`): `pi05_flatten_fold_normal`, `pi05_flatten_fold_awbc`, `pi05_tee_shirt_sort_awbc`, `pi05_hang_cloth_awbc`, etc.

### Training (PyTorch — Advantage Estimator)
```bash
# Single GPU
uv run python scripts/train_pytorch.py <config_name> --exp_name=<name> --save_interval 10000

# Multi-GPU DDP
uv run torchrun --standalone --nproc_per_node=8 scripts/train_pytorch.py <config_name> --exp_name=<name>
```

### Inference Server
```bash
uv run python scripts/serve_policy.py --config <config_name> --checkpoint <ckpt_path> --port 8000
```

## Architecture

### Core Source (`kai0/src/openpi/`)

- **`models/`** — JAX/Flax model definitions: `pi0.py` (base π₀), `pi0_fast.py` (π₀-FAST with FSQ tokenizer), `pi0_rtc.py` (real-time chunking variant), `gemma.py`/`gemma_fast.py` (language backbone), `siglip.py`/`vit.py` (vision encoders), `tokenizer.py` (action tokenization)
- **`models_pytorch/`** — PyTorch reimplementation for advantage estimator training (`pi0_pytorch.py`, `preprocessing_pytorch.py`), plus patched HuggingFace Transformers modules in `transformers_replace/`
- **`policies/`** — Robot-specific policy wrappers that handle observation preprocessing and action postprocessing: `agilex_policy.py`, `arx_policy.py`, `aloha_policy.py`, `droid_policy.py`, `libero_policy.py`. Config registry in `policy_config.py`
- **`training/`** — Training loop, data loading (`data_loader.py`, `advantage_dataset.py`), config registry (`config.py` — central config file for all train configs), checkpointing, optimizer, sharding, weight loaders
- **`serving/`** — WebSocket policy server (`websocket_policy_server.py`)
- **`transforms.py`** — Data transforms (image resizing, action normalization) applied per-policy

### Three Technical Modules

1. **Model Arithmetic** (`kai0/model_arithmetic/`) — Weight-space checkpoint merging. `arithmetic.py` (JAX) and `arithmetic_torch.py` (PyTorch) support 6 methods: average, inverse_loss, gradient_descent, adaptive_gradient_descent, greedy, manual weights
2. **Stage Advantage** (`kai0/stage_advantage/`) — Stage-aware advantage estimation pipeline: GT annotation → train estimator → predict advantage → discretize → AWBC training
3. **Train-Deploy Alignment** (`kai0/train_deploy_alignment/`) — Data augmentation (time scaling, space mirroring), DAgger data collection, inference with temporal smoothing/ensembling/RTC

### Deployment Scripts (top-level `scripts/`)

Local scripts for this specific deployment setup:
- `start_autonomy.sh` — Launch full autonomy (policy rollout) stack: cameras + arms + policy node
- `start_policy_node.sh` — Launch only policy_inference_node (--mode ros2/websocket/both), when other nodes already running
- `start_server_xla_cache.sh` — Start policy server with XLA cache
- `start_teleop.sh` — Launch teleoperation mode
- `toggle_execute.sh` — Toggle execution mode on/off
- `launch_3cam.py` — Launch 3-camera RealSense setup
- `test_integration_ros2.py`, `test_inference_parity.py` — End-to-end and parity tests
- `test_inference_server.py --check latency|quality|all` — Inference latency + quality benchmarking
- `test_hardware.py` — Hardware verification (cameras + arms)
- `test_cameras.py` — Camera diagnostics

### Piper Tools (`piper_tools/`)

CAN bus utilities for Agilex Piper arms: `setup_can.sh`, `activate_can.sh`, `find_all_can_port.sh`, `diagnose_can.sh`, `calibrate_can_mapping.py`, `verify_can_mapping.py`, `piper_ctrl_go_zero.py`, `piper_ctrl_gripper.py`.

### ROS2 Workspace (`ros2_ws/`)

ROS2 packages for Piper robot control (`ros2_ws/src/piper/`) and message definitions (`ros2_ws/src/piper_msgs/`). Built separately with `colcon build`.

## Key Configuration

- Training configs are all defined in `kai0/src/openpi/training/config.py` — this is the central place to set `repo_id` (dataset path), `weight_loader` (base checkpoint path), batch size, etc. Key dataclasses: `TrainConfig`, `DataConfig`, `AssetsConfig`
- Policy configs (observation/action specs per robot) are in `kai0/src/openpi/policies/policy_config.py`
- AWBC prompts must match training format exactly: `"<task>, Advantage: positive"` / `"<task>, Advantage: negative"`

### Hardware Configuration (`config/`)

- `config/pipers.yml` — Dual-arm CAN bus configuration: port mappings, feedback Hz, ROS2 topics for 4 Piper arms
- `config/cameras.yml` — RealSense camera inventory (D435 + 2× D405): serial numbers, resolution (640×480), 30fps, ROS2 topics
- `config/calibration.yml` — Hand-eye calibration transforms (T_world_camF, T_world_baseL/R, T_link6_camL/R) and camera intrinsics

### Data Transform Pipeline

Both training and inference apply transforms in this order:
1. `repack_transforms` — format conversion (dataset → internal representation)
2. `data_transforms` — robot-specific preprocessing (per-policy, defined in `policies/`)
3. `Normalize` — using norm_stats from checkpoint assets
4. `model_transforms` — image resize, tokenization, padding (per-model type)

At inference time, the inverse is applied to outputs (unnormalize → inverse data_transforms → inverse repack_transforms).

### Checkpoint Structure

```
checkpoints/<name>/
├── params/                           # JAX model parameters (sharded)
├── assets/<asset_id>/norm_stats.json # normalization statistics
├── model.safetensors                 # PyTorch checkpoint (if applicable)
└── wandb_id.txt                      # W&B run ID for resuming
```

`policy_config.create_trained_policy()` auto-detects JAX vs PyTorch by checking for `model.safetensors`.

### Environment Variables

| Variable | Purpose | Typical Value |
|----------|---------|---------------|
| `OPENPI_DATA_HOME` | Cache dir for downloaded checkpoints/data | `~/.cache/openpi` |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | JAX GPU memory fraction | `0.9` for training |
| `XLA_PYTHON_CLIENT_PREALLOCATE` | JAX memory preallocation | `false` (set in data_loader, model_arithmetic) |
| `JAX_COMPILATION_CACHE_DIR` | XLA compilation cache | `.xla_cache` (project-local in start_policy_node.sh) |
| `CUDA_VISIBLE_DEVICES` | GPU selection | Varies per script |
| `GIT_LFS_SKIP_SMUDGE` | Skip Git LFS downloads during install | `1` |

### Data Layout

Datasets follow LeRobot v2.1 format under `kai0/data/`:
```
Task_{A,B,C}/{base,dagger}/
├── data/chunk-000/episode_*.parquet   # obs state [N,14], actions [N,14]
├── videos/chunk-000/{camera}/episode_*.mp4  # 480×640, AV1, 30fps
└── meta/  # info.json, episodes.jsonl, tasks.jsonl
```

Camera keys: `top_head`, `hand_left`, `hand_right`. Observation state is 14-dim (dual arm joint angles + gripper open).

## Documentation (`docs/`)

Deployment and reproduction guides: `sim01_deployment.md` (full setup), `teleoperation_guide.md`, `inference_visualization.md`/`inference_visualization_mesh.md` (Rerun visualization), `taskA_master_plan.md` (Task A reproduction plan), `training_reproduction_log.md`.
