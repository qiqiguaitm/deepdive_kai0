# wam_fold_policy — Cosmos3-Nano-Policy → WAM dual-arm cloth-fold adaptation

Launchers, configs, and normalization stats for adapting **Cosmos3-Nano-Policy-DROID** (16B
omni MoT policy) into a **14-D dual-arm cloth-fold (`wam_fold`)** policy, trained jointly
**cross-rig** over two embodiment domains:

- **visrobot01** → domain `wam_fold` (has train/val split)
- **kairobot01** → domain `kairobot01` (single root, no split)

Both rigs run the same 14-D fold task (6 arm joints + 1 gripper × 2 arms) but with different
camera extrinsics / workspace, so they are distinct domains with **per-rig quantile
normalization**.

## What lives where

This directory holds the **launchers / configs / stats** only. The actual training & dataset
**code** lives in the cosmos3 framework package:

- recipe / experiment config: `packages/cosmos3/cosmos_framework/configs/base/experiment/action/posttrain_config/wam_fold_nano.py`
- dataset class: `packages/cosmos3/cosmos_framework/data/vfm/action/datasets/wam_fold_dataset.py`
  (its `_RIG_DEFAULTS` resolves stats to `data/stats/{visrobot01,kairobot01}.json` here)

### Layout

```
wam_fold_policy/
├── data/
│   ├── compute_action_stats.py     # compute 14-D action+state norm stats from parquet
│   └── stats/{visrobot01,kairobot01}.json   # per-rig quantile + mean/std stats
├── train/
│   ├── recipe_nano.toml            # SFT recipe (--sft-toml) for cosmos_framework.scripts.train
│   ├── env.sh                      # shared 2-node env preamble (sourced per rank)
│   ├── train_2node.sh              # 2-node (b0+b1) FSDP-16 full fine-tune
│   ├── train_single_node.sh        # single-node 8-GPU FSDP full fine-tune
│   └── smoke_validate.sh           # convert→DCP→train N steps→assert loss drops
├── eval/
│   ├── eval_report.py              # export HF model + roll out + 3-way MAE/video report
│   ├── shard.sh                    # one eval shard pinned to one GPU
│   ├── run_16gpu.sh                # 16-GPU sharded eval (b0 0-7 + b1 8-15) → aggregate
│   ├── run_single.sh               # single-process eval
│   └── validate_and_report.sh      # cross-rig smoke validation + 16-GPU eval
├── eval_i2v/                       # separate Cosmos3 image-to-video eval harness
└── setup/                          # env-build helpers (+ _archive/ one-off scripts)
```

## Outputs

All run outputs (checkpoints, exported model, train/smoke outputs, reports/shards/episodes/logs)
go to the **runs root** (NOT this dir):

```
RUNS = /mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/wam_fold_policy_runs
├── checkpoints/Cosmos3-Nano-Policy-DROID-dcp   # DCP warm-start ckpt
├── exported/Cosmos3-Nano-Policy-wam_fold       # exported HF model (eval target)
├── train_out_2node/   train_out_single/        # IMAGINAIRE_OUTPUT_ROOT per launcher
├── smoke_out/                                  # smoke-validation output
└── reports/                                    # report.html, summary.json, shards/, episodes/, *.log
```

## Run commands (run from anywhere; scripts use absolute paths)

1. **Compute norm stats** (per rig — pass N_EP, dataset root, out json):
   ```bash
   python3 data/compute_action_stats.py 300 \
     /mnt/pfs/p46h4f/cosmos/deepdive_kai0/kai0/data/wam_fold_v1/visrobot01_train \
     data/stats/visrobot01.json
   ```

2. **Smoke-validate** (convert Policy-DROID → DCP, train a few steps, assert loss drops):
   ```bash
   SMOKE_ITERS=12 NGPU=8 bash train/smoke_validate.sh
   ```

3. **Train** (writes to `$RUNS/train_out_{2node,single}`):
   ```bash
   # 2-node (run on b0); env.sh is sourced per rank inside the launcher
   MAXITER=50000 SAVEITER=1000 bash train/train_2node.sh
   # single node, 8 GPU
   NGPU=8 MAXITER=5000 SAVEITER=500 bash train/train_single_node.sh
   ```

4. **Export + eval + report** (export reads the latest train ckpt + config, writes HF model to
   `$RUNS/exported/...`, report to `$RUNS/reports/report.html`):
   ```bash
   # 16-GPU sharded across b0+b1
   NMETRIC=20 NVIZ=10 bash eval/run_16gpu.sh
   # or single-process
   bash eval/run_single.sh
   # cross-rig smoke validation + full 16-GPU eval, chained
   bash eval/validate_and_report.sh
   ```

The report is a 3-way action-MAE + video-metrics comparison (Cosmos3 vs GWP vs τ0/π0.5) at
`$RUNS/reports/report.html` with `summary.json` alongside.
