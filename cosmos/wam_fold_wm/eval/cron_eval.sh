#!/usr/bin/env bash
# One-shot cron script: find the latest unevaluated checkpoint and eval it.
# Designed to run via crontab (e.g. every 30 min). Safe to run concurrently —
# a lockfile prevents double-running the same checkpoint.
#
# Crontab entry (run as root):
#   */30 * * * * bash /mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/wam_fold_wm/eval/cron_eval.sh \
#                >> /mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/wam_fold_wm_runs/cron_eval.log 2>&1
set -uo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; export no_proxy='*'

EVAL_DIR=$(dirname "$(readlink -f "$0")")
CF=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/packages/cosmos3
VENV=$CF/.venv; PY=$VENV/bin/python
RUNS=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/wam_fold_wm_runs
CKPT_BASE=$RUNS/train_out_5n8g/cosmos3/action/wam_fold_wm_nano
RESULTS=$RUNS/eval_results.jsonl
REPORT_DIR=$RUNS/reports/fd_eval
LOCK=$RUNS/.cron_eval.lock

N_EPS=${CRON_N_EPS:-5}
NUM_STEPS=${CRON_NUM_STEPS:-15}
GUIDANCE=${CRON_GUIDANCE:-3.0}
EVERY_N=${CRON_EVERY_N:-1}   # eval every N-th save_iter (save_iter=500; EVERY_N=2 → eval iter 1000,2000,…)

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

echo "[cron_eval] $(date '+%Y-%m-%d %H:%M:%S') start"

# ---- lockfile: prevent overlapping runs ----
if [ -f "$LOCK" ]; then
  pid=$(cat "$LOCK" 2>/dev/null || echo "")
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    echo "[cron_eval] another instance running (pid=$pid), exit"
    exit 0
  fi
  echo "[cron_eval] stale lockfile, removing"
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

already_done() {
  [ -f "$RESULTS" ] && grep -q "\"iter\": $1" "$RESULTS" 2>/dev/null
}

# ---- find all checkpoint iters, pick the latest unevaluated one ----
TARGET_ITER=""
while IFS= read -r ck_dir; do
  iter_str=$(basename "$ck_dir" | sed 's/iter_//')
  iter=$((10#$iter_str))
  # EVERY_N filter: only eval multiples of (EVERY_N * 500)
  if (( EVERY_N > 1 )); then
    step=$(( EVERY_N * 500 ))
    (( iter % step != 0 )) && continue
  fi
  already_done "$iter" && continue
  TARGET_ITER=$iter   # keep scanning; last (highest) unevaluated iter wins
done < <(find "$CKPT_BASE/checkpoints" -maxdepth 1 -name "iter_*" -type d 2>/dev/null | sort)

if [ -z "$TARGET_ITER" ]; then
  echo "[cron_eval] no new checkpoints to evaluate, done"
  exit 0
fi

ITER=$TARGET_ITER
ITER9=$(printf "%09d" "$ITER")
CK=$CKPT_BASE/checkpoints/iter_$ITER9
EXP=$RUNS/exported/wam_fold_wm_iter$ITER
EP_REPORT_DIR=$REPORT_DIR/iter$ITER
mkdir -p "$EP_REPORT_DIR"

echo "[cron_eval] evaluating iter=$ITER  ckpt=$CK"

# 1) export DCP → HF (idempotent)
if [ ! -f "$EXP/config.json" ]; then
  bash "$EVAL_DIR/export_ckpt.sh" "$ITER" || { echo "[cron_eval] export failed"; exit 1; }
fi

# 2) fd_infer: PSNR eval + comparison videos
"$PY" "$EVAL_DIR/fd_infer.py" \
  --export-dir "$EXP" \
  --out-dir "$EP_REPORT_DIR" \
  --n-episodes "$N_EPS" \
  --num-steps "$NUM_STEPS" \
  --guidance "$GUIDANCE" \
  --iter "$ITER" \
  --save-videos \
  --video-fps 10 \
  || { echo "[cron_eval] fd_infer failed for iter=$ITER"; exit 1; }

# 3) append aggregate to eval_results.jsonl
REPORT=$EP_REPORT_DIR/fd_daction_report.json
if [ -f "$REPORT" ]; then
  "$PY" - "$ITER" "$REPORT" "$RESULTS" << 'PY'
import sys, json
iter, rpt_path, out = int(sys.argv[1]), sys.argv[2], sys.argv[3]
rpt = json.load(open(rpt_path))
row = {"iter": iter, **rpt.get("aggregate", {}), "verdict": rpt.get("verdict", "")}
with open(out, "a") as f:
    f.write(json.dumps(row) + "\n")
print(f"[cron_eval] appended iter={iter} -> {out}\n[cron_eval] {row}")
PY
fi

# 4) generate report.html
"$PY" "$EVAL_DIR/make_report.py" \
  --iter "$ITER" \
  --report-dir "$EP_REPORT_DIR" \
  --log-file "$RUNS/train_out_5n8g/train_node0.log" \
  --eval-history "$RESULTS" \
  && echo "[cron_eval] report.html written: $EP_REPORT_DIR/report.html" \
  || echo "[cron_eval] WARNING: report generation failed (non-fatal)" >&2

echo "[cron_eval] done  iter=$ITER  $(date '+%H:%M:%S')"
