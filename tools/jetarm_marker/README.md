# JetArm 4-Marker Needle — 主路径

针上 4 个反光 marker → 估计针尖/针轴 → 在 needle_tracking 3D 里显示针。

**旧 RGB 光流与模态探路脚本**已移至 `legacy/`，见 [docs/jetarm_marker/LEGACY_RGB.md](../../docs/jetarm_marker/LEGACY_RGB.md)。

## 环境

```bash
pip install -r requirements-jetarm-marker.txt
```

## 目录结构（主路径）

```text
tools/jetarm_marker/
  ir_marker_detect.py      # 共享 IR 检测 + ROM + m1 门控
  bracket_rom.py           # 支架边长 ROM 匹配
  ir_depth_stream_protocol.py  # TCP 流（JARM / JARD）
  live_pose_estimate.py    # 实时 3D 姿态
  needle_gl_view.py        # 3D 显示共用（回放 + 实时）
  live_needle_gl.py        # ★ 实时 3D 针体
  live_ir_markers.py       # 2D 调参预览
  detect_ir_markers.py     # 离线 bag IR 检测
  pose_from_ir_depth.py    # 离线 IR+depth 姿态
  replay_pose_csv.py       # 离线 CSV 3D 回放
  ir_stream_server.py      # JetArm 侧推流
  legacy/                  # 旧脚本归档
data/jetarm_marker/
  geometry/                # ROM、内参、scene_transform
  bags/                    # rosbag（本地，不入库）
```

## 1. JetArm 部署与启动

```powershell
.\tools\jetarm_marker\deploy_to_jetarm.ps1
ssh ubuntu@192.168.55.1 "bash /home/ubuntu/jetarm_marker_tools/start_live_ir_on_jetarm.sh"
```

## 2. 实时 3D 针体（优先）

```powershell
python tools\jetarm_marker\live_needle_gl.py --host 192.168.55.1

# 无硬件：用 bag 模拟
python tools\jetarm_marker\live_needle_gl.py --source bag data\jetarm_marker\bags\marker_move_rgb_ir_depth_01
```

- 流协议 **JARD**：IR JPEG + depth uint16 同帧
- 坐标系：**相机系 mm**（尚未接 CT / scene 配准）
- 针轴 m2→m1，针尖从 m1 外推 140 mm（`--tip-offset-mm`）

## 3. 实时 2D 调参

```powershell
python tools\jetarm_marker\live_ir_markers.py --host 192.168.55.1
python tools\jetarm_marker\live_ir_markers.py --source bag data\jetarm_marker\bags\marker_move_rgb_ir_depth_01
```

窗口：`candidate_count`、ROM、m1 门控；`q` 退出，`r` 重置，`+`/`-` 调阈值。

## 4. 离线 IR 链路

```powershell
# 1) bag 放到 data/jetarm_marker/bags/<name>/

# 2) IR 2D 检测
python tools\jetarm_marker\detect_ir_markers.py data\jetarm_marker\bags\marker_move_rgb_ir_depth_01

# 3) IR + depth → 针姿态 CSV
python tools\jetarm_marker\pose_from_ir_depth.py ^
  --bag data\jetarm_marker\bags\marker_move_rgb_ir_depth_01 ^
  --markers data\jetarm_marker\ir_detection\marker_move_rgb_ir_depth_01_ir_markers.csv

# 4) 3D 回放
python tools\jetarm_marker\replay_pose_csv.py data\jetarm_marker\ir_depth_pose\marker_move_rgb_ir_depth_01_ir_depth_pose.csv
```

### marker 约定

```text
m0 = 右侧球
m1 = 针尖方向球（下方）
m2 = 针尾方向球（上方）
m3 = 左侧球
针轴 = m2 → m1
m1 门控：|m2-m1| / |m0-m3| >= 0.55
```

### 3D 图例

```text
红色 = 针尖 | 绿色 = 针尾参考点
m0 蓝 | m1 黄 | m2 青 | m3 紫
浅色线 = m0-m3 与 m2-m1 支架
```

## 5. 状态与精度

链路已跑通，**刚体/精度未验收**。详见 [docs/jetarm_marker/STATUS.md](../../docs/jetarm_marker/STATUS.md)。
