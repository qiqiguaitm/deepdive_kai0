# xvla/ — X-VLA 全管线代码集散地

> 2026-07-01 重构: 所有 XVLA 相关代码 (训练 / 部署 / 数据 / 评估 / 分析 / 推理) 统一收归本目录。

## 目录结构

```
xvla/
├── README.md                       ← 本文档
│
├── start_xvla_from_ckpt.sh         # 部署入口: 从 ckpt 一键起 server+client
├── start_xvla_stack.sh             # stack 启动器: server 后台 + client 前台
├── start_xvla_autonomy.sh          # autonomy 分体启动: server/client 两终端
├── xvla_repack_deploy_taskp.sh     # repack: state_dict.pt → 部署目录 + sidecar
├── xvla_taskp_local_5090.sh        # 训练启动器: 本地 5090 单卡 TaskP
│
├── launch/                         # 训练入口 (Python)
│   ├── xvla_train.py               #   主训练脚本 (configs + launch)
│   └── xvla_train_smoke.py         #   smoke test 辅助
│
├── data/                           # 数据工具 + 数据集
│   ├── joint_to_ee6d.py            #   14D joint → 20D EE6D 转换
│   ├── convert_xvla_action.py      #   action 格式转换
│   ├── multi_domain_dataset.py     #   多域数据集加载器
│   ├── self_built/                 #   EE6D 数据集 (TaskP_ee6d, TaskP_ee6d_continuous 等)
│   └── *.yaml                      #   数据集 manifest
│
├── eval/                           # 离线评估
│   ├── eval_xvla_ee6d.py           #   离线 EE6D MAE 评估
│   └── eval_pi05_fk_ee6d.py        #   pi05 FK→EE6D 对照评估
│
├── serve/                          # 推理服务
│   └── serve_policy_xvla.py        #   X-VLA WebSocket 推理 server
│
├── analysis/                       # trace / 性能分析
│   ├── analyze_pipeline_trace.py   #   pipeline 逐帧 trace 核验
│   ├── analyze_tracking.py         #   目标跟踪分析
│   └── analyze_cmd_vs_output.py    #   指令 vs 输出对照
│
├── X-VLA/                          # ★ upstream submodule (pristine, NEVER modify)
│   ├── models/                     #   Florence2 + SoftPromptedTransformer + action_hub
│   ├── datasets/                   #   HDF5/Parquet dataset loaders
│   ├── train.py / deploy.py        #   官方训练/部署入口
│   └── evaluation/                 #   官方 eval
│
├── docs/                           # 📚 XVLA 文档集散地
│   ├── deployment/inference/       #   部署文档
│   ├── training/analysis/          #   训练分析
│   ├── training/plans/             #   未来计划
│   ├── training/experiments/       #   实验记录
│   └── README.md                   #   文档索引
├── assets/                         # tokenizer (bart-large) + 基座 config
├── ckpts/                          # 训练产出的 checkpoint
├── ckpts_official/                 # 官方 ckpt (X-VLA-SoftFold)
├── scripts/                        # 历史实验 YAML
└── README_start.md                 # 旧 start_scripts/xvla 文档 (参考)
```

## 快速链接

| 你想做什么 | 入口 |
|---|---|
| 真机部署 X-VLA | `./xvla/start_xvla_from_ckpt.sh <ckpt_name> --execute` |
| 本地训练 TaskP | `BS=6 ./xvla/xvla_taskp_local_5090.sh full` |
| 转换数据 (joint→EE6D) | `python xvla/data/joint_to_ee6d.py --in_dir <raw> --out_dir <out> [--continuous]` |
| 离线 eval MAE | `python xvla/eval/eval_xvla_ee6d.py --ckpt <path>` |
| 分析 trace | `python xvla/analysis/analyze_pipeline_trace.py <trace_dir>` |

## 与旧路径的兼容

重构后旧路径保留 symlink → 新路径:

| 旧路径 | → 新路径 |
|---|---|
| `train_scripts/xvla/launch/` | → `xvla/launch/` |
| `train_scripts/xvla/data/` | → `xvla/data/` |
| `train_scripts/xvla/eval/` | → `xvla/eval/` |
| `start_scripts/xvla/` | → `xvla/` (top-level) |
| `kai0/scripts/serve_policy_xvla.py` | → `xvla/serve/serve_policy_xvla.py` |
| `docs/deployment/inference/xvla_*.md` | → `xvla/docs/deployment/inference/` |
| `docs/training/analysis/xvla_*.md` | → `xvla/docs/training/analysis/` |
| `docs/training/history/experiments/xvla_*.md` | → `xvla/docs/training/experiments/` |
| `docs/training/future_plans/plans/xvla_*.md` | → `xvla/docs/training/plans/` |

## 与 upstream 的关系

- **`X-VLA/` 是 git submodule** → `github.com/2toinf/X-VLA`
- 当前 pin: commit `ccd1992`
- **永远不要直接编辑 `X-VLA/` 内容** — 我们的扩展通过 `xvla/launch/`, `xvla/serve/` 等 wrapper 实现
