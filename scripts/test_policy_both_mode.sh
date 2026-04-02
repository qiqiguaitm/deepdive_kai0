#!/bin/bash
eval "$(conda shell.bash hook 2>/dev/null)"; conda deactivate 2>/dev/null || true
source /opt/ros/jazzy/setup.bash
source /data1/tim/workspace/deepdive_kai0/ros2_ws/install/setup.bash
VENV=/data1/tim/workspace/deepdive_kai0/kai0/.venv/lib/python3.12/site-packages
export LD_LIBRARY_PATH=$(find $VENV/nvidia -name 'lib' -type d 2>/dev/null | tr '\n' ':')${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONPATH="${VENV}:/data1/tim/workspace/deepdive_kai0/kai0/src:${PYTHONPATH}"
export JAX_COMPILATION_CACHE_DIR=/tmp/xla_cache
export CUDA_VISIBLE_DEVICES=0
unset http_proxy https_proxy XLA_FLAGS
echo "=== Testing both mode ==="
exec ros2 run piper policy_inference_node.py --ros-args \
  -p mode:=both \
  -p checkpoint_dir:=/data1/tim/workspace/deepdive_kai0/kai0/checkpoints/Task_A/mixed_1 \
  -p config_name:=pi05_flatten_fold_normal \
  -p gpu_id:=0 -p ws_port:=8000 \
  -p img_front_topic:=/camera/camera/color/image_raw \
  -p img_left_topic:=/camera_l/camera/color/image_rect_raw \
  -p img_right_topic:=/camera_r/camera/color/image_rect_raw
