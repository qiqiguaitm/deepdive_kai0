#!/bin/bash
###############################################################################
# Launch kai0 IPC container (ROS Noetic + Piper + RealSense)
#
# Usage:
#   ./docker/run_ipc.sh          # interactive bash
#   ./docker/run_ipc.sh roscore  # run a specific command
###############################################################################

CONTAINER_NAME="kai0-ipc"
IMAGE_NAME="kai0-ipc:noetic"
WORKSPACE="/data1/tim/workspace/deepdive_kai0"

# Stop existing container if running
if docker ps -q -f name="^${CONTAINER_NAME}$" | grep -q .; then
    echo "Container ${CONTAINER_NAME} already running, attaching..."
    exec docker exec -it ${CONTAINER_NAME} bash
fi

# Remove stopped container with same name
docker rm -f ${CONTAINER_NAME} 2>/dev/null

docker run -it \
    --name ${CONTAINER_NAME} \
    --privileged \
    --net=host \
    --pid=host \
    -e DISPLAY=$DISPLAY \
    -e ROS_MASTER_URI=http://localhost:11311 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v ${WORKSPACE}:/workspace \
    -v /dev:/dev \
    -v /run/udev:/run/udev:ro \
    ${IMAGE_NAME} \
    ${@:-}
