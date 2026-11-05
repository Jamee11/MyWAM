#!/usr/bin/env bash
# Source only; reuse Kai0's environment without installing or changing anything in it.
source /opt/ros/noetic/setup.bash
source "$HOME/kai0_repo/kai0/train_deploy_alignment/inference/agilex/Piper_ros_private-ros-noetic/devel/setup.bash"
export PATH="$HOME/miniconda3/envs/kai0_inference0/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1
