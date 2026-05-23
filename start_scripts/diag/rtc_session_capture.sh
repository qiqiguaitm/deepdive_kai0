#!/bin/bash
###############################################################################
# RTC session capture: 切到指定预设 → 等 N 秒 → 解析 autonomy 日志输出 JSON
#
# 配合 start_autonomy_v1.sh 在另一终端运行. 预设的 ros2 param set + 滚动窗口
# 解析 /tmp/v1_autonomy.log 内 [injected-*/baseline] + `infer Xms` 行.
#
# Usage:
#   ./rtc_session_capture.sh <preset> <duration_s> [log_path]
#
# Presets (V1 post-deploy):
#   M0         enable_rtc=true  rate=3   lat_k=8  exec_h=16 max_guid=0.5   (旧默认 baseline)
#   M1-10Hz    + inference_rate=10
#   M2-C       + latency_k=3 min_smooth=3 exec_h=6
#   rtc_off    enable_rtc=false (其他不动)
#
# 输出: logs/rtc_sweep_<date>/<preset>.json + stdout 一行摘要
###############################################################################
set -euo pipefail

PRESET="${1:-}"
DURATION="${2:-60}"
LOG_PATH="${3:-/tmp/v1_autonomy.log}"

if [ -z "$PRESET" ]; then
  echo "Usage: $0 <preset> <duration_s> [log_path]" >&2
  echo "  presets: M0 | M1-10Hz | M2-C | rtc_off" >&2
  exit 1
fi

NODE=/policy_inference

apply_preset() {
  case "$1" in
    M0)        params=(enable_rtc:true inference_rate:3.0  latency_k:8 min_smooth_steps:8 rtc_execute_horizon:16 rtc_max_guidance_weight:0.5) ;;
    M1-10Hz)   params=(enable_rtc:true inference_rate:10.0 latency_k:8 min_smooth_steps:8 rtc_execute_horizon:16 rtc_max_guidance_weight:0.5) ;;
    M2-C)      params=(enable_rtc:true inference_rate:10.0 latency_k:3 min_smooth_steps:3 rtc_execute_horizon:6  rtc_max_guidance_weight:0.5) ;;
    rtc_off)   params=(enable_rtc:false inference_rate:10.0 latency_k:3 min_smooth_steps:3 rtc_execute_horizon:6  rtc_max_guidance_weight:0.5) ;;
    *) echo "[FAIL] unknown preset: $1" >&2; exit 1 ;;
  esac
  echo "[$(date +%H:%M:%S)] Applying preset $PRESET:"
  for kv in "${params[@]}"; do
    k="${kv%%:*}"; v="${kv#*:}"
    echo "  $k = $v"
    ros2 param set "$NODE" "$k" "$v" >/dev/null
  done
}

# Pre-flight: node must be reachable
if ! ros2 node list 2>/dev/null | grep -q "policy_inference"; then
  echo "[FAIL] /policy_inference node not running — start_autonomy_v1.sh first" >&2
  exit 1
fi
if [ ! -f "$LOG_PATH" ]; then
  echo "[FAIL] log not found: $LOG_PATH" >&2
  exit 1
fi

DATE_TAG=$(date +%Y-%m-%d)
OUT_DIR="logs/rtc_sweep_${DATE_TAG}"
mkdir -p "$OUT_DIR"
OUT_JSON="$OUT_DIR/${PRESET}.json"

# Apply preset, mark log byte offset
apply_preset "$PRESET"
START_OFFSET=$(stat -c %s "$LOG_PATH")
START_TS=$(date +%s)
echo "[$(date +%H:%M:%S)] Capturing ${DURATION}s, log offset = $START_OFFSET"
sleep "$DURATION"
END_TS=$(date +%s)
END_OFFSET=$(stat -c %s "$LOG_PATH")
echo "[$(date +%H:%M:%S)] Done, log grew $((END_OFFSET-START_OFFSET)) bytes"

# Slice log + parse with python
python3 - "$LOG_PATH" "$START_OFFSET" "$END_OFFSET" "$PRESET" "$START_TS" "$END_TS" "$DURATION" "$OUT_JSON" << 'PYEOF'
import sys, re, json, statistics
log_path, start_off, end_off, preset, start_ts, end_ts, dur, out_json = sys.argv[1:9]
start_off, end_off = int(start_off), int(end_off)
start_ts, end_ts, dur = int(start_ts), int(end_ts), int(dur)

with open(log_path, 'rb') as f:
    f.seek(start_off)
    chunk = f.read(end_off - start_off).decode('utf-8', errors='replace')

# Patterns
re_infer = re.compile(r'infer (\d+)ms \| chunk=')
re_diag  = re.compile(r'\[(injected-(?:norm|raw)|baseline)\] d=(\d+) exec_h=(\d+) \| guid_MAE=([0-9.]+) free_MAE=([0-9.]+) ratio=([0-9.naninf-]+) \| \|prev\|=([0-9.]+) \|new\|=([0-9.]+)')

infers = [int(m.group(1)) for m in re_infer.finditer(chunk)]
diags  = []
for m in re_diag.finditer(chunk):
    try:
        ratio_str = m.group(6)
        ratio = float(ratio_str) if ratio_str not in ('nan','inf','-inf') else None
        diags.append({
            'inj': m.group(1),
            'd': int(m.group(2)),
            'exec_h': int(m.group(3)),
            'guid_MAE': float(m.group(4)),
            'free_MAE': float(m.group(5)),
            'ratio': ratio,
            'prev_abs': float(m.group(7)),
            'new_abs': float(m.group(8)),
        })
    except ValueError:
        continue

def pct(xs, p):
    if not xs: return None
    xs = sorted(xs); k = (len(xs)-1) * p / 100
    f, c = int(k), min(int(k)+1, len(xs)-1)
    return xs[f] if f==c else xs[f] + (k-f)*(xs[c]-xs[f])

result = {
    'preset': preset,
    'start_ts': start_ts,
    'end_ts': end_ts,
    'duration_s': dur,
    'n_infer': len(infers),
    'n_diag': len(diags),
    'effective_rate_hz': round(len(infers) / dur, 2) if dur > 0 else None,
    'infer_ms': {
        'P50': pct(infers, 50),
        'P95': pct(infers, 95),
        'P99': pct(infers, 99),
        'mean': round(statistics.mean(infers), 1) if infers else None,
        'std': round(statistics.stdev(infers), 1) if len(infers) > 1 else None,
    },
}
if diags:
    guid = [d['guid_MAE'] for d in diags]
    free = [d['free_MAE'] for d in diags]
    ratios = [d['ratio'] for d in diags if d['ratio'] is not None]
    result['chunk_mae'] = {
        'guid_MAE_mean': round(statistics.mean(guid), 4),
        'free_MAE_mean': round(statistics.mean(free), 4),
        'ratio_mean':    round(statistics.mean(ratios), 3) if ratios else None,
        'ratio_P50':     round(pct(ratios, 50), 3) if ratios else None,
        'rtc_active_pct': round(100 * sum(1 for d in diags if d['inj'].startswith('injected')) / len(diags), 1),
    }

with open(out_json, 'w') as f:
    json.dump(result, f, indent=2)

# Stdout one-line summary
ir = result['infer_ms']
cm = result.get('chunk_mae', {})
print(f"\n=== {preset} | {dur}s | {result['n_infer']} infer ({result['effective_rate_hz']} Hz eff)")
print(f"  infer_ms  P50={ir['P50']} P95={ir['P95']} mean={ir['mean']} std={ir['std']}")
if cm:
    print(f"  chunk MAE guid={cm['guid_MAE_mean']} free={cm['free_MAE_mean']} ratio={cm['ratio_mean']} (rtc_active={cm['rtc_active_pct']}%)")
print(f"  → {out_json}")
PYEOF
