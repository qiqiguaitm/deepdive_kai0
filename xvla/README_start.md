# start_scripts/xvla/ — X-VLA 架构真机部署脚本

> X-VLA (Florence2 + SoftPromptedTransformer, PyTorch) 真机部署脚本目录.

## 脚本

| 脚本 | 用途 |
|---|---|
| `start_xvla_autonomy.sh` | EE6D 推理 stack: `server <ckpt>` 起 `kai0/scripts/serve_policy_xvla.py` (:8003, emit `action_kind=ee` 16D); `client` 复用 `../kai/start_autonomy.sh --execution-mode ee_pose` 驱动真机 (详见 `docs/deployment/inference/xvla_inference_bringup.md`) |

## 与 `start_scripts/kai/` 的关系

| 目录 | 架构 | 推理 stack |
|---|---|---|
| `kai/` | pi0 / openpi (JAX) | `serve_policy_v1.py` 系列, ROS2 + WebSocket / SHM |
| `xvla/` (本目录) | X-VLA Florence2 | `serve_policy_xvla.py` (server-only, emit `action_kind=ee`), client 复用 kai/ autonomy |

## 何时往这里放东西?

- X-VLA 架构的 server / autonomy / policy_node 脚本归这里 (如 `start_xvla_autonomy.sh`)
- 涉及 `xvla/X-VLA` (submodule) 的 deploy.py 调用脚本也归这里
- 不要把 pi0-based "xvla_*" 实验的 deploy 放这里 — 那些走 kai/
