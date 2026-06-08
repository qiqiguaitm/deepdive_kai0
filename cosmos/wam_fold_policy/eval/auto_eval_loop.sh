#!/usr/bin/env bash
# Autonomous per-checkpoint eval on b0(gpu0-7)+b1(gpu0-7). Polls the training checkpoint
# dir; for each new iter_* DCP it: (1) exports DCP->HF (--no-vit, 1 GPU), (2) runs a
# 16-shard rollout eval across both nodes, (3) aggregates -> reports/<iter>/report.html,
# (4) appends action/video metrics to eval_curve.csv (feeds curve_monitor.py).
# Keeps pace: ~40 min/eval < ~72 min/checkpoint.
set -uo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY LD_LIBRARY_PATH; export no_proxy='*'
ROOT=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos
CF=$ROOT/packages/cosmos3
RUNS=$ROOT/wam_fold_policy_runs
CKPTDIR=$RUNS/train_out_4n8g/cosmos3/action/wam_fold_nano/checkpoints
CFG=$RUNS/train_out_4n8g/cosmos3/action/wam_fold_nano/config.yaml
EXPORTED=$RUNS/exported
REPORTS=$RUNS/reports
SHARD=$ROOT/wam_fold_policy/eval/shard.sh
PY=$CF/.venv/bin/python
B1="ssh -p 429 -o BatchMode=yes -o StrictHostKeyChecking=no -o ServerAliveInterval=30 root@120.48.99.93"
NUM=16
NMETRIC=${NMETRIC:-16}; NVIZ=${NVIZ:-6}
DONE=$REPORTS/evaled.txt; touch "$DONE"
CURVE=$REPORTS/eval_curve.csv
[ -f "$CURVE" ] || echo "iter,action_mae,mae@1,mae@10,mae@24,mae@48,video_psnr,video_ssim" > "$CURVE"

# export env (mirror shard.sh, for cosmos_framework.scripts.export_model)
exp_env() { export PYTHONPATH="$CF" PATH=/mnt/pfs/p46h4f/cosmos/uvbin:$PATH \
  HF_ENDPOINT=https://hf-mirror.com HF_HOME=/mnt/pfs/p46h4f/cosmos/hf_home HF_HUB_OFFLINE=1 \
  WAN_VAE_PATH=/mnt/pfs/p46h4f/cosmos/hf_home/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth \
  TOKENIZERS_PARALLELISM=false PYTORCH_ALLOC_CONF=expandable_segments:True \
  LD_LIBRARY_PATH=/mnt/pfs/p46h4f/huanqian/conda/envs/uniVP/lib; }

echo "=== auto_eval_loop start $(date) | NMETRIC=$NMETRIC NVIZ=$NVIZ ==="
IDLE=0
while true; do
  newone=0
  for ck in $(ls -d "$CKPTDIR"/iter_* 2>/dev/null | sort -V); do
    N=$(basename "$ck")
    grep -qx "$N" "$DONE" && continue
    # checkpoint must be fully written (model subdir + a .metadata) — skip if still saving
    [ -d "$ck/model" ] || { echo "[$(date +%T)] $N not ready (no model/), wait"; continue; }
    step=$(echo "$N" | grep -oE '[0-9]+' | sed 's/^0*//'); step=${step:-0}
    OUT=$REPORTS/$N; mkdir -p "$OUT/shards" "$OUT/episodes"
    EXP=$EXPORTED/$N
    echo "[$(date +%T)] ===== EVAL $N (step $step) ====="
    newone=1; IDLE=0
    # 1) export DCP -> HF (once; --no-vit avoids needing full Qwen3-VL-8B shards)
    if [ ! -f "$EXP/config.json" ]; then
      echo "[$(date +%T)] export $N -> $EXP"
      ( exp_env; CUDA_VISIBLE_DEVICES=0 cd "$CF" && "$PY" -m cosmos_framework.scripts.export_model \
          --checkpoint-path "$ck" --config-file "$CFG" --no-vit -o "$EXP" ) > "$OUT/export.log" 2>&1
      if [ ! -f "$EXP/config.json" ]; then echo "[$(date +%T)] EXPORT FAILED $N (see $OUT/export.log)"; tail -5 "$OUT/export.log"; echo "$N" >> "$DONE"; continue; fi
    fi
    # 2) 16-shard eval: b0 gpu g -> shard g ; b1 gpu g -> shard 8+g
    ARGS="--no_export --export_dir $EXP --out_dir $OUT --config_file $CFG --no_lpips --max_full_windows 100000 --max_win_per_ep 4 --n_metric_eps $NMETRIC --n_viz_eps $NVIZ"
    rm -f "$OUT"/shards/shard_*.json
    for g in 0 1 2 3 4 5 6 7; do nohup bash "$SHARD" "$g" "$g" "$NUM" $ARGS > "$OUT/shard_$g.log" 2>&1 & done
    $B1 "for g in 0 1 2 3 4 5 6 7; do nohup bash $SHARD \$g \$((8+g)) $NUM $ARGS > $OUT/shard_b1_\$g.log 2>&1 & done" 2>&1 | grep -v OpenSSL | tail -1
    for t in $(seq 1 180); do n=$(ls "$OUT"/shards/shard_*.json 2>/dev/null | wc -l); [ "$n" -ge "$NUM" ] && break; sleep 30; done
    nf=$(ls "$OUT"/shards/shard_*.json 2>/dev/null | wc -l)
    echo "[$(date +%T)] $N shards complete: $nf/$NUM"
    # 3) aggregate -> report.html + summary.json
    ( exp_env; bash "$SHARD" 0 0 "$NUM" $ARGS --aggregate ) > "$OUT/aggregate.log" 2>&1
    grep -E "aggregate\]|cosmos mae" "$OUT/aggregate.log" | tail -2
    # 4) record metrics for the eval curve
    "$PY" - "$OUT/summary.json" "$step" "$CURVE" <<'PYEOF'
import json,sys
sf,step,curve=sys.argv[1],sys.argv[2],sys.argv[3]
try:
    d=json.load(open(sf)); rm=d.get("raw_mae",{}) or {}; v=d.get("video",{}) or {}
    def g(m,k):
        x=m.get(k); return "" if x is None else f"{x:.5f}"
    # summary.json raw_mae keys are "1"/"10"/"24"/"48" (NOT "mae@1"); video has psnr/ssim.
    row=[step, g(rm,"1"), g(rm,"1"), g(rm,"10"), g(rm,"24"), g(rm,"48"), g(v,"psnr"), g(v,"ssim")]
    open(curve,"a").write(",".join(str(x) for x in row)+"\n")
    print("[curve] appended step",step,"mae@1",g(rm,"1"),"mae@48",g(rm,"48"),"psnr",g(v,"psnr"))
except Exception as e:
    print("[curve] parse fail:",e)
PYEOF
    echo "$N" >> "$DONE"
    echo "[$(date +%T)] ===== DONE $N -> $OUT/report.html ====="
  done
  # stop when training job is gone AND no unevaled checkpoints remain
  if [ "$newone" -eq 0 ]; then
    IDLE=$((IDLE+1))
    jobst=$(aihc job get "$(cat $REPORTS/train_job.txt 2>/dev/null)" -p aihc-serverless -q aihcq-z4v1apdppzwy 2>/dev/null | grep -E "^    status:" | head -1 | awk '{print $2}')
    echo "[$(date +%T)] no new ckpt (idle=$IDLE, train=$jobst)"
    # Only terminate on a DEFINITIVE terminal job state — an empty/unknown jobst (transient
    # aihc API hiccup) must NOT be read as "training ended" (that false-terminated the loop).
    case "$jobst" in
      Failed|Succeeded|Stopped|ManualTermination|Terminated|Deleted)
        if [ "$IDLE" -ge 5 ]; then
          echo "=== training ended ($jobst) and all checkpoints evaled — AUTO_EVAL_DONE ==="; break
        fi ;;
    esac
    sleep 120
  fi
done
