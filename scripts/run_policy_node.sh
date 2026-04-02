#!/bin/bash
eval "$(conda shell.bash hook 2>/dev/null)"; conda deactivate 2>/dev/null
source /opt/ros/jazzy/setup.bash
source /data1/tim/workspace/deepdive_kai0/ros2_ws/install/setup.bash
unset http_proxy https_proxy XLA_FLAGS
exec /data1/tim/workspace/deepdive_kai0/kai0/.venv/bin/python \
  /data1/tim/workspace/deepdive_kai0/ros2_ws/install/piper/lib/piper/policy_inference_node.py \
  --ros-args -p mode:=websocket -p host:=localhost -p port:=8000 \
  -p img_front_topic:=/camera_f/color/image_raw \
  -p img_left_topic:=/camera_l/color/image_rect_raw \
  -p img_right_topic:=/camera_r/color/image_rect_raw
