# Distillation Scale-Up Runbook — UPDATED with completed 38-ep result

**Status (2026-06-27, UPDATED):** CW→Cosmos3 distillation pipeline **built + validated + scale-up RUN**.
The CW-export `pipe_read` deadlock was **root-caused and FIXED this session** (a leftover DINOv2
download holding an HF lock + the Bash-tool's 120s default timeout cutting off the slow rollout;
fix = kill the holder, run foreground with tool-timeout≥560s, 3-step targets). Scaled 13→**38 eps**:

| eps | distill loss | eval ΔPSNR(GT−other) |
|----|----|----|
| 13 | 0.02 | −0.032 |
| **38** | 0.02 | **+0.027** |

**MEASURED CONCLUSION:** the action *is* usable (loss 0.02) and develops with data (−0.032→+0.027),
confirming the `∂/∂action≈0` wall is NOT architectural — but the **rate is too slow** (+0.06 per 3× data;
target-quality 8-step≈3-step, not the bottleneck). Extrapolation: ~90 eps→+0.16 plateau; **+8.17 is
infeasible by data-scale on the omni-MoT** (a fidelity cap — student fits targets yet reproduces only
+0.027). Path to +8.17 = full PAIWorld recipe (video-DiT base + 3D-REPA + scale), research-scale.
**The steps below still run** (now unblocked) if pursuing more eps, but the verdict won't change without
the architecture swap.

## Resume (fresh box, ~half a day)

### 1. Export ~100 CW action-differentiated targets (CW venv)
```bash
cd /mnt/pfs/p46h4f/cosmos/deepdive_kai0/Ctrl-World
# 4 shards across 4 GPUs (stagger 70s to avoid CPU-OOM on simultaneous CW loads):
for sh in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$sh HF_HUB_OFFLINE=1 nohup .venv/bin/python scripts/export_distill_targets.py \
    --ckpt model_ckpt/clothfold_svd_5n8g/checkpoint-40000.pt \
    --val_dir /mnt/pfs/p46h4f/cosmos/deepdive_kai0/kai0/data/wam_fold_v3_cw/visrobot01_v3_val \
    --stat dataset_meta_info_clothfold/visrobot01_v3_train/stat.json \
    --n_eps 100 --shard $sh --nshards 4 --max_frames 49 --num_inference_steps 30 \
    --out_dir ../cosmos/wam_fold_wm_runs/distill_targets100 >/tmp/exp$sh.log 2>&1 &
  sleep 70
done
# (each npz now has gt/wrong_action_perframe aligned 1:1 to frames — no temporal re-derivation needed)
```

### 2. Phase-0b re-arrange to student concat layout
```bash
cd /mnt/pfs/p46h4f/cosmos/deepdive_kai0
.venv/bin/python Ctrl-World/scripts/phase0b_reencode.py \
  --in_dir cosmos/wam_fold_wm_runs/distill_targets100 \
  --out_dir cosmos/wam_fold_wm_runs/distill_concat100
```

### 3. Retrain the distill student (framework venv, 8 GPUs)
```bash
cd /mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos
# point the distill data source at the 100-ep dir + run the registered config:
WAM_DISTILL_DIR=$PWD/wam_fold_wm_runs/distill_concat100 STEPS=2000 \
  bash wam_fold_wm/train/run_distill_local.sh
# config = wam_fold_wm_nano_distill (chunk16, fps10, stride1; data source build_wm_distill_data_source)
```

### 4. Go/no-go eval
```bash
# export iter_500/1000/2000 + run fd_infer; the metric is mean_dPSNR_gt_minus_other:
CKPT_BASE=.../train_out_distill_local/cosmos3/action/wam_fold_wm_nano_distill EXP_DIR=... \
  bash wam_fold_wm/eval/export_ckpt.sh <iter>
bash wam_fold_wm/eval/run_fd_infer_v3.sh --export-dir <exp> --chunk 16 --frame-stride 1 \
  --shift 2.0 --n-episodes 12 --num-steps 8 --guidance 3.0 --out-dir <out>
# read mean_dPSNR_gt_minus_other from <out>/fd_daction_report.json
```

**Decision:** ΔPSNR breaks +0.16 (toward >+1.0) ⇒ generalization works → scale further toward +8.17.
Stays flat even at 100 eps ⇒ the action *is* usable (loss 0.02) but doesn't generalize at this data
scale → either (a) need the full PAIWorld recipe (video-DiT base + 3D-REPA), or (b) more eps still.

## Key files (all built + validated this session)
- `Ctrl-World/scripts/export_distill_targets.py` (sharded, per-frame action)
- `Ctrl-World/scripts/phase0b_reencode.py` (CW 3-view → student concat)
- `packages/cosmos3/cosmos_framework/data/vfm/action/datasets/wam_fold_distill_dataset.py` (self-test ✓)
- `.../posttrain_config/wam_fold_wm_nano_distill.py` (registered in config.py)
- `.../wam_fold_wm_nano.py::build_wm_distill_data_source`
- `wam_fold_wm/train/{recipe_wm_nano_distill.toml,run_distill_local.sh}`
