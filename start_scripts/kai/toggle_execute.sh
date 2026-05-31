#!/bin/bash
###############################################################################
# Interactive keyboard control. Behavior depends on /tmp/kai0_deployment_mode:
#
#   "autonomy"  (default) → toggle /policy/execute (run vs observe-only)
#   "dagger"             → toggle /dagger/takeover (policy-run vs human-record).
#                          dagger_recorder_node handles the full takeover
#                          sequence: stop policy → align master to slave →
#                          switch master to 0xFA → start recording (and reverse).
#
# Usage: run in a separate terminal while the inference stack is running.
#   ./scripts/toggle_execute.sh
#
# Keys:
#   Enter/Space  → toggle
#   q/Esc        → force back to baseline (observe / policy-run)
#   Ctrl+C       exit
###############################################################################

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ROS2_WS="$PROJECT_ROOT/ros2_ws"

eval "$(conda shell.bash hook 2>/dev/null)" 2>/dev/null; conda deactivate 2>/dev/null || true
source /opt/ros/jazzy/setup.bash
[ -f "$ROS2_WS/install/setup.bash" ] && source "$ROS2_WS/install/setup.bash"

CYAN='\033[0;36m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

# ── Detect deployment mode ──
MARKER="/tmp/kai0_deployment_mode"
DEPLOY_MODE="autonomy"  # default
[ -f "$MARKER" ] && DEPLOY_MODE=$(cat "$MARKER" 2>/dev/null || echo "autonomy")

if [ "$DEPLOY_MODE" = "dagger" ]; then
    TOPIC="/dagger/takeover"
    BASELINE_LABEL="POLICY-RUN"
    ACTIVE_LABEL="RECORDING-HUMAN"
    HEADER="DAgger Takeover Control"
else
    TOPIC="/policy/execute"
    BASELINE_LABEL="OBSERVE"
    ACTIVE_LABEL="EXECUTE"
    HEADER="Policy Execute Control"
fi

state="$BASELINE_LABEL"
echo -e "${CYAN}$HEADER${NC}  (deploy_mode=${DEPLOY_MODE}, topic=${TOPIC})"
echo "  [Enter/Space] toggle  ($BASELINE_LABEL ↔ $ACTIVE_LABEL)"
echo "  [q/Esc]       → $BASELINE_LABEL"
echo "  [Ctrl+C]      exit"
echo ""
if [ "$DEPLOY_MODE" = "dagger" ]; then
    echo -e "${YELLOW}NOTE${NC}: 切换到 RECORDING-HUMAN 时，master 会自动驱动到 slave 位 (~2s)。"
    echo -e "      保持手在主臂外，等 [TAKEOVER] DONE 日志出现再拖动。"
    echo ""
fi
echo -e "State: ${YELLOW}$state${NC}"

publish() {
    local data=$1
    ros2 topic pub --once "$TOPIC" std_msgs/msg/Bool "{data: $data}" >/dev/null 2>&1
}

toggle() {
    if [ "$state" = "$BASELINE_LABEL" ]; then
        state="$ACTIVE_LABEL"
        publish true
        if [ "$DEPLOY_MODE" = "dagger" ]; then
            echo -e "State: ${RED}$state${NC}  (master 正在对齐 — keep clear)"
        else
            echo -e "State: ${GREEN}$state${NC}"
        fi
    else
        state="$BASELINE_LABEL"
        publish false
        echo -e "State: ${YELLOW}$state${NC}"
    fi
}

baseline() {
    if [ "$state" != "$BASELINE_LABEL" ]; then
        state="$BASELINE_LABEL"
        publish false
        echo -e "State: ${YELLOW}$state${NC}"
    fi
}

stty_orig=$(stty -g)
trap 'stty "$stty_orig"; publish false; echo; echo "exited"; exit 0' INT TERM EXIT
stty -echo -icanon min 1 time 0

while true; do
    ch=$(dd bs=1 count=1 2>/dev/null)
    case "$ch" in
        $'\n'|$'\r'|' ')
            toggle
            ;;
        q)
            baseline
            ;;
        $'\x1b')
            stty min 0 time 1
            extra=$(dd bs=1 count=2 2>/dev/null || true)
            stty min 1 time 0
            if [ -z "$extra" ]; then
                baseline
            fi
            ;;
    esac
done
