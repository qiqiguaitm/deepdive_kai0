#!/usr/bin/env bash
# Periodic checkpoint eval for wam_fold_wm_nano (FD world model).
#
# What it does (every --interval seconds):
#   1. Scan $CKPT_BASE/checkpoints/iter_*/  for new DCP dirs
#   2. Skip iters already recorded in eval_results.jsonl
#   3. Export DCP → HF  (export_ckpt.sh)
#   4. Run fd_infer.py  (Δaction perturbation PSNR test)
#   5. Append one JSON line to eval_results.jsonl
#
# Usage (run on sim01 or gf0, keep in tmux):
#   bash watch_and_eval.sh [--interval 1800] [--n-episodes 5] [--num-steps 15]
#
# Notes:
#   - export + eval together take ~20–40 min per checkpoint on 1×A100/RTX5090
#   - Pass --every-n 5 to evaluate every 5th checkpoint (e.g. iter 500,2500,5000…)
#     to reduce total eval time during long runs
set -uo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; export no_proxy='*'

# ---- defaults ----
INTERVAL=1800    # poll every 30 min
N_EPS=5          # val episodes per eval
NUM_STEPS=15     # diffusion steps (fast; use 30 for final eval)
GUIDANCE=3.0
EVERY_N=1        # evaluate every N-th checkpoint (1 = all)

while [[ $# -gt 0 ]]; do
  case $1 in
    --interval)  INTERVAL=$2;  shift 2;;
    --n-episodes) N_EPS=$2;   shift 2;;
    --num-steps)  NUM_STEPS=$2; shift 2;;
    --guidance)   GUIDANCE=$2;  shift 2;;
    --every-n)    EVERY_N=$2;   shift 2;;
    *) echo "Unknown arg: $1" >&2; exit 1;;
  esac
done

# ---- paths ----
CF=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/packages/cosmos3
VENV=$CF/.venv; PY=$VENV/bin/python
RUNS=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/wam_fold_wm_runs
CKPT_BASE=$RUNS/train_out_5n8g/cosmos3/action/wam_fold_wm_nano
EVAL_DIR=$(dirname "$0")
RESULTS=$RUNS/eval_results.jsonl
REPORT_DIR=$RUNS/reports/fd_eval

export PYTHONPATH="$CF:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/mnt/pfs/p46h4f/huanqian/conda/envs/uniVP/lib"
export PATH=/mnt/pfs/p46h4f/cosmos/uvbin:$PATH
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/mnt/pfs/p46h4f/cosmos/hf_home
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WAN_VAE_PATH=/mnt/pfs/p46h4f/cosmos/hf_home/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
mkdir -p "$REPORT_DIR"

echo "[watch] polling $CKPT_BASE/checkpoints every ${INTERVAL}s"
echo "[watch] eval_results -> $RESULTS"

# ---- already-evaluated set ----
already_done() {
  local iter=$1
  [ -f "$RESULTS" ] && grep -q "\"iter\": $iter" "$RESULTS" 2>/dev/null
}

POLL_IDX=0
while true; do
  # find all iter_XXXXXXXXX dirs, extract iter number, sort ascending
  mapfile -t ITER_DIRS < <(
    find "$CKPT_BASE/checkpoints" -maxdepth 1 -name "iter_*" -type d 2>/dev/null \
    | sort
  )

  for ck_dir in "${ITER_DIRS[@]}"; do
    iter_str=$(basename "$ck_dir" | sed 's/iter_//')
    iter=$((10#$iter_str))   # strip leading zeros

    # every-N filter
    if (( EVERY_N > 1 && iter % (EVERY_N * 500) != 0 )); then
      continue
    fi

    already_done "$iter" && continue

    echo ""
    echo "=== [watch] NEW CKPT  iter=$iter  $(date '+%Y-%m-%d %H:%M:%S') ==="

    # 1) export DCP -> HF
    EXP=$RUNS/exported/wam_fold_wm_iter$iter
    if [ ! -f "$EXP/config.json" ]; then
      bash "$EVAL_DIR/export_ckpt.sh" "$iter" || {
        echo "[watch] export failed for iter=$iter; skip" >&2; continue
      }
    fi

    # 2) run fd_infer.py (delta-action perturbation PSNR test + comparison videos)
    EP_REPORT_DIR=$REPORT_DIR/iter$iter
    mkdir -p "$EP_REPORT_DIR"
    "$PY" "$EVAL_DIR/fd_infer.py" \
      --export-dir "$EXP" \
      --out-dir "$EP_REPORT_DIR" \
      --n-episodes "$N_EPS" \
      --num-steps "$NUM_STEPS" \
      --guidance "$GUIDANCE" \
      --iter "$iter" \
      --save-videos \
      --video-fps 10 \
      || { echo "[watch] fd_infer failed for iter=$iter" >&2; continue; }

    # 3) parse the report and append to eval_results.jsonl
    REPORT=$EP_REPORT_DIR/fd_daction_report.json
    if [ -f "$REPORT" ]; then
      "$PY" - "$iter" "$REPORT" "$RESULTS" << 'PY'
import sys, json
iter, rpt_path, out = int(sys.argv[1]), sys.argv[2], sys.argv[3]
rpt = json.load(open(rpt_path))
row = {"iter": iter, **rpt.get("aggregate", {}), "verdict": rpt.get("verdict", "")}
with open(out, "a") as f:
    f.write(json.dumps(row) + "\n")
print(f"[watch] appended iter={iter} -> {out}")
print(f"[watch] {row}")
PY
    fi

    # 4) generate report.html
    "$PY" "$EVAL_DIR/make_report.py" \
      --iter "$iter" \
      --report-dir "$EP_REPORT_DIR" \
      --log-file "$RUNS/train_out_5n8g/train_node0.log" \
      --eval-history "$RESULTS" \
      && echo "[watch] report: $EP_REPORT_DIR/report.html" \
      || echo "[watch] WARNING: report generation failed (non-fatal)" >&2
  done

  POLL_IDX=$((POLL_IDX + 1))
  echo "[watch] poll#$POLL_IDX done at $(date '+%H:%M:%S'), sleeping ${INTERVAL}s ..."
  sleep "$INTERVAL"
done
