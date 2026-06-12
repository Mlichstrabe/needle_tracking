#!/bin/bash
# 在 JetArm 上录制「含 IR」的 modality 对比 bag（SSH 登录后执行）
# 用法: bash record_modality_bag.sh marker_rgb_ir_depth_01

set -euo pipefail
NAME="${1:-marker_rgb_ir_depth_01}"
BAG_DIR="${HOME}/ros2_ws/bags/${NAME}"

echo "录制到: ${BAG_DIR}"
mkdir -p "$(dirname "${BAG_DIR}")"

# 确认相机在跑；若未启动请先: ros2 launch peripherals depth_camera.launch.py
# 本脚本用 enable_ir:=true 单独起相机（若已有节点在跑，先停掉避免冲突）

source /opt/ros/humble/setup.bash
source "${HOME}/ros2_ws/install/setup.bash" 2>/dev/null || true

echo "检查 topic..."
ros2 topic list | grep depth_cam || true

ros2 bag record -o "${BAG_DIR}" \
  /depth_cam/rgb/image_raw \
  /depth_cam/rgb/camera_info \
  /depth_cam/ir/image_raw \
  /depth_cam/ir/camera_info \
  /depth_cam/depth/image_raw \
  /depth_cam/depth/camera_info

echo "完成。请 scp -r ${BAG_DIR} 到开发机 data/jetarm_marker/bags/"
