#!/usr/bin/env bash
# Cosmos3-Nano (BASE) -> wam_fold FORWARD-DYNAMICS world-model smoke validation.
# Mirrors wam_fold_policy/train/smoke_validate.sh: convert base->DCP -> train N steps
# -> assert min(loss)<loss[0]. Single node 8xA100 (local-first plan).
set -uo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; export no_proxy='*'
CF=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/packages/cosmos3
VENV=$CF/.venv
PY=$VENV/bin/python
MODEL=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/models/modelscope/Cosmos3-Nano
DCP=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/wam_fold_wm_runs/checkpoints/Cosmos3-Nano-dcp
OUT=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/wam_fold_wm_runs/smoke_out
LOG=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/wam_fold_wm_runs/reports
TOML=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/wam_fold_wm/train/recipe_wm_nano.toml
export PYTHONPATH="$CF:${PYTHONPATH:-}"
export IMAGINAIRE_OUTPUT_ROOT="$OUT"; mkdir -p "$OUT" "$LOG" "$(dirname $DCP)"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NGPU=${NGPU:-8}
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/mnt/pfs/p46h4f/huanqian/conda/envs/uniVP/lib"
export PATH=/mnt/pfs/p46h4f/cosmos/uvbin:$PATH
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/mnt/pfs/p46h4f/cosmos/hf_home
export HF_HUB_OFFLINE=1
export WAN_VAE_PATH="${WAN_VAE_PATH:-/mnt/pfs/p46h4f/cosmos/hf_home/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth}"
cd "$CF"

echo "=== [0] env check $(date +%H:%M:%S) ==="
[ -x "$PY" ] || { echo "FATAL: no venv python $PY"; exit 2; }
"$PY" -c "import torch;assert torch.cuda.is_available();print('torch',torch.__version__,'cuda ok',torch.cuda.device_count(),'gpus')" || { echo "FATAL: torch/cuda not ready"; exit 2; }

echo "=== [1] convert base Cosmos3-Nano diffusers -> DCP $(date +%H:%M:%S) ==="
if [ -d "$DCP" ] && find "$DCP" -name ".metadata" | grep -q .; then
  echo "DCP already present, skip convert"
else
  "$PY" -m cosmos_framework.scripts.convert_model_to_dcp --checkpoint-path "$MODEL" -o "$DCP" > "$LOG/convert_dcp.log" 2>&1
  rc=$?; echo "convert rc=$rc"; tail -15 "$LOG/convert_dcp.log"
  [ $rc -eq 0 ] || { echo "FATAL: convert failed (see convert_dcp.log)"; exit 3; }
fi

echo "=== [2] smoke train wam_fold_wm FD (${NGPU} GPU, ${SMOKE_ITERS:-30} steps) $(date +%H:%M:%S) ==="
export BASE_CKPT_DCP="$DCP"
PORT=$(( ( $$ % 20000 ) + 30000 ))
"$VENV/bin/torchrun" --nproc_per_node=$NGPU --master_port=$PORT \
  -m cosmos_framework.scripts.train --sft-toml="$TOML" -- \
  trainer.max_iter=${SMOKE_ITERS:-30} \
  > "$LOG/smoke_train.log" 2>&1
rc=$?; echo "train rc=$rc"
echo "--- last 25 lines ---"; tail -25 "$LOG/smoke_train.log"

echo "=== [3] validate loss trend $(date +%H:%M:%S) ==="
"$PY" - "$LOG/smoke_train.log" << 'PYEOF'
import re,sys
t=open(sys.argv[1]).read()
rx=re.compile(r"\[RANK\s+0\]\s+Iteration\s+\d+:.*?Loss:\s+([-+0-9.eE]+)")
v=[float(x) for x in rx.findall(t)]
print("parsed losses:",v)
if not v: print("VALIDATION: NO LOSSES PARSED (training may have errored)"); sys.exit(1)
ok = (min(v) < v[0]) if len(v)>1 else (v[0]==v[0])
print(f"loss[0]={v[0]:.4f} min={min(v):.4f} -> {'PASS' if ok else 'FAIL (no downward step)'}")
sys.exit(0 if ok else 1)
PYEOF
vrc=$?
echo "=== WM_SMOKE_VALIDATION_DONE rc=$vrc $(date +%H:%M:%S) ==="
