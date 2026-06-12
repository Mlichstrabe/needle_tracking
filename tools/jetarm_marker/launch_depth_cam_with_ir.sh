#!/bin/bash
# JetArm: 启动深度相机并开启 IR 流（对比阶段 1 用）
# 依据: peripherals/launch/include/astra.launch.py  default enable_ir=false

source /opt/ros/humble/setup.bash
source "${HOME}/ros2_ws/install/setup.bash" 2>/dev/null || true

# enable_ir 在 astra.launch.py 里，需传给 include 的 astra launch
ros2 launch peripherals depth_camera.launch.py
# 若上式未开启 IR，改用（路径以 JetArm 上 ros2_ws 为准）:
# ros2 launch peripherals astra.launch.py camera_name:=depth_cam enable_ir:=true
