#!/bin/bash
# JetArm 上启动 depth_cam + IR TCP 转发（SSH 登录后执行）
set -eo pipefail

PORT="${1:-8765}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

set +u
source /opt/ros/humble/setup.bash
source "${HOME}/ros2_ws/install/setup.bash" 2>/dev/null || true
set -u

export need_compile=False
export CAMERA_TYPE=GEMINI

if ! ros2 topic list 2>/dev/null | grep -q '/depth_cam/ir/image_raw'; then
  echo "[start] 启动 depth_cam ..."
  pkill -f 'depth_camera.launch.py' 2>/dev/null || true
  pkill -f 'camera_container' 2>/dev/null || true
  sleep 2
  nohup ros2 launch peripherals depth_camera.launch.py depth_camera_name:=depth_cam \
    > /tmp/depth_cam.log 2>&1 &
  for _ in $(seq 1 20); do
    if ros2 topic list 2>/dev/null | grep -q '/depth_cam/ir/image_raw'; then
      break
    fi
    sleep 1
  done
fi

if ! ros2 topic list 2>/dev/null | grep -q '/depth_cam/ir/image_raw'; then
  echo "[error] IR topic 未出现，查看 /tmp/depth_cam.log"
  tail -20 /tmp/depth_cam.log 2>/dev/null || true
  exit 1
fi

echo "[start] IR topic 就绪，启动 stream server :${PORT}"
pkill -f 'ir_stream_server.py' 2>/dev/null || true
sleep 1
nohup python3 "${SCRIPT_DIR}/ir_stream_server.py" --port "${PORT}" > /tmp/ir_stream_server.log 2>&1 &
sleep 1
echo "[ok] ir_stream_server 已后台运行，Windows 端："
echo "  python tools/jetarm_marker/live_needle_gl.py --host 192.168.55.1 --port ${PORT}"
echo "  python tools/jetarm_marker/live_ir_markers.py --host 192.168.55.1 --port ${PORT}"
