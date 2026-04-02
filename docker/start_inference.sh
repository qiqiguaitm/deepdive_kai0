#!/bin/bash
###############################################################################
# Full inference stack inside the IPC container (run inside container)
#
# Usage (inside container):
#   bash /workspace/docker/start_inference.sh [--dagger]
#
# Prerequisites:
#   - Policy server running on host: CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py ...
#   - CAN interfaces activated on HOST (before starting container):
#     cd /workspace/kai0/train_deploy_alignment/dagger/agilex && sudo ./activate_can_arms.sh
###############################################################################

set -e

DAGGER_MODE=false
if [[ "$1" == "--dagger" ]]; then
    DAGGER_MODE=true
fi

REPO="/workspace/kai0"
LOG_DIR="/workspace/logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p ${LOG_DIR}

# Source ROS
source /opt/ros/noetic/setup.bash

if [[ "${DAGGER_MODE}" == true ]]; then
    echo "=== DAgger mode: using ros_ws_dagger (master + slave) ==="
    source /ros_ws_dagger/devel/setup.bash
    PIPER_LAUNCH="start_ms_piper_new.launch"
else
    echo "=== Inference mode: using ros_ws (slave only) ==="
    source /ros_ws/devel/setup.bash
    PIPER_LAUNCH="start_ms_piper.launch mode:=1 auto_enable:=true"
fi

echo "Logs: ${LOG_DIR}"
echo ""

# Start tmux session
SESSION="kai0"
tmux kill-session -t ${SESSION} 2>/dev/null || true
tmux new-session -d -s ${SESSION}

# Pane 0: roscore
tmux send-keys -t ${SESSION} "source /opt/ros/noetic/setup.bash && roscore 2>&1 | tee ${LOG_DIR}/roscore.log" Enter
sleep 2

# Pane 1: RealSense cameras
tmux split-window -h -t ${SESSION}
tmux send-keys -t ${SESSION} "source /opt/ros/noetic/setup.bash && sleep 2 && roslaunch realsense2_camera rs_camera.launch 2>&1 | tee ${LOG_DIR}/realsense.log" Enter
# NOTE: Replace with your custom multi_camera.launch once serial numbers are configured:
# tmux send-keys -t ${SESSION} "roslaunch /workspace/kai0/my_multi_camera.launch 2>&1 | tee ${LOG_DIR}/realsense.log" Enter

# Pane 2: Piper arms
tmux split-window -v -t ${SESSION}
if [[ "${DAGGER_MODE}" == true ]]; then
    tmux send-keys -t ${SESSION} "source /ros_ws_dagger/devel/setup.bash && sleep 3 && roslaunch piper ${PIPER_LAUNCH} 2>&1 | tee ${LOG_DIR}/piper.log" Enter
else
    tmux send-keys -t ${SESSION} "source /ros_ws/devel/setup.bash && sleep 3 && roslaunch piper ${PIPER_LAUNCH} 2>&1 | tee ${LOG_DIR}/piper.log" Enter
fi

# Pane 3: Inference client (or DAgger)
tmux split-window -v -t ${SESSION}
if [[ "${DAGGER_MODE}" == true ]]; then
    tmux send-keys -t ${SESSION} "echo '=== DAgger client ready ===' && echo 'Run:' && echo 'cd ${REPO}/train_deploy_alignment/dagger/agilex && python3 agilex_openpi_dagger_collect.py --host localhost --port 8000 --ctrl_type joint --use_temporal_smoothing --chunk_size 50 --dataset_name my_dataset'" Enter
else
    tmux send-keys -t ${SESSION} "echo '=== Inference client ready ===' && echo 'Run:' && echo 'cd ${REPO}/train_deploy_alignment/inference/agilex/inference && python3 agilex_inference_openpi_temporal_smoothing.py --host localhost --port 8000 --ctrl_type joint --use_temporal_smoothing --chunk_size 50'" Enter
fi

echo ""
echo "=========================================="
echo " tmux session: ${SESSION}"
echo " Attach:  tmux attach -t ${SESSION}"
echo " Kill:    tmux kill-session -t ${SESSION}"
echo "=========================================="

tmux attach -t ${SESSION}
