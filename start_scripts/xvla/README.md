# start_scripts/xvla/ — X-VLA 架构真机部署脚本

> **预留位置**: 当前为空, 等待真 X-VLA 架构 (Florence2 + SoftPromptedTransformer, PyTorch) 落地真机部署后, 在这里放对应的 launch / autonomy / policy_node 脚本.

## 与 `start_scripts/kai/` 的关系

| 目录 | 架构 | 推理 stack |
|---|---|---|
| `kai/` | pi0 / openpi (JAX) | `serve_policy_v1.py` 系列, ROS2 + WebSocket / SHM |
| `xvla/` (本目录) | X-VLA Florence2 | TBD — 需评估 PyTorch model 与现有 ROS2 piper 控制流的接入方式 |

## 何时往这里放东西?

- Track X / X-VLA ckpt 准备真机部署时, 写 `start_xvla_autonomy.sh` 等放此处
- 涉及 `xvla/X-VLA` (submodule) 的 deploy.py 调用脚本也归这里
- 不要把 pi0-based "xvla_*" 实验的 deploy 放这里 — 那些走 kai/
