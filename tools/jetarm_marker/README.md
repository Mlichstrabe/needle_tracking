# JetArm 4-Marker Needle V1 Offline Pipeline

目标：

```text
针上 4 个 marker -> 估计针尖和针轴 -> 在 needle_tracking 3D 前端显示一根针
```

当前策略是离线优先：先用 JetArm 录好的 bag 跑通算法链路，再考虑实时相机。

## 0. 环境

```bash
pip install -r requirements-jetarm-marker.txt
```

## 1. 准备 bag

把 JetArm 上的 ROS2 bag 整个目录复制到：

```text
data/jetarm_marker/bags/<bag_name>/
```

目录内必须有：

```text
metadata.yaml
*.db3
```

## 2. 探测和导出对比帧

```bash
python tools/jetarm_marker/bag_probe.py data/jetarm_marker/bags/marker_static_clean_01

python tools/jetarm_marker/export_modality_compare.py ^
  data/jetarm_marker/bags/marker_static_clean_01 ^
  --samples 12
```

输出：

```text
data/jetarm_marker/exports/<bag_name>/probe_summary.json
data/jetarm_marker/exports/<bag_name>/frames/*_compare.png
```

注意：当前 `marker_static_clean_01` 和 `marker_move_clean_01` 没有录 `/depth_cam/ir/image_raw`，所以这两包只能做 RGB + depth。

## 3. 初始化 4 个 marker

可以用 GUI 手动点：

```bash
python tools/jetarm_marker/init_markers.py ^
  data/jetarm_marker/exports/marker_static_clean_01/frames/frame_004_idx0173_rgb.png ^
  --bag marker_static_clean_01
```

也可以直接使用当前已经保存的初始化文件：

```text
data/jetarm_marker/inits/marker_static_clean_01_init.json
data/jetarm_marker/inits/marker_move_clean_01_init.json
```

当前约定的 marker 顺序：

```text
m0 = 右侧球
m1 = 上方球
m2 = 下方球
m3 = 左侧/针尖侧球
```

## 4. 2D marker 跟踪

### 4A. IR 亮球检测（当前推荐）

对新录的 RGB + IR + depth bag，优先用 IR 检测反光球：

```bash
python tools/jetarm_marker/detect_ir_markers.py ^
  data/jetarm_marker/bags/marker_static_rgb_ir_depth_01

python tools/jetarm_marker/detect_ir_markers.py ^
  data/jetarm_marker/bags/marker_move_rgb_ir_depth_01
```

输出：

```text
data/jetarm_marker/ir_detection/<bag_name>_ir_markers.csv
data/jetarm_marker/ir_detection/<bag_name>_ir_markers.mp4
data/jetarm_marker/ir_detection/<bag_name>_ir_markers_contact.jpg
data/jetarm_marker/ir_detection/<bag_name>_ir_markers.summary.json
```

当前已验证：

```text
marker_static_rgb_ir_depth_01: 49/49 valid
marker_move_rgb_ir_depth_01:   98/98 valid
```

检测策略：先找 IR 高亮候选，再从候选里选空间分布最像 4 个端点球的一组，避免把中心夹具反光当成 marker。

当前 IR 检测视频里的临时标号约定：

```text
m0 = 右侧球
m1 = 下方/针尖方向球
m2 = 上方/针尾方向球
m3 = 左侧球
```

因此当前离线针轴先用 `m2 -> m1` 近似，针尖从 `m1` 沿该方向外推。当前手工测得 `m1 -> 针尖` 约为 140 mm，所以 `pose_from_ir_depth.py` 默认使用 `--tip-offset-mm 140`。

已知问题：当 m1 不在视野内，或者 m1 与下方关节点混淆时，针尖会明显偏。当前 3D 姿态脚本会用 `m2-m1 / m0-m3 >= 0.55` 做几何门控，低于阈值的帧不输出有效针姿态。

### 4A-live. 实时 IR 预览（现场调机）

JetArm 经 USB 连接 PC（`192.168.55.1`），检测在 Windows 跑，JetArm 只转发 IR 流。

```powershell
# 同步 + 启动（JetArm 侧相机 + TCP:8765）
.\tools\jetarm_marker\deploy_to_jetarm.ps1
ssh ubuntu@192.168.55.1 "bash /home/ubuntu/jetarm_marker_tools/start_live_ir_on_jetarm.sh"

# Windows 预览
python tools\jetarm_marker\live_ir_markers.py --host 192.168.55.1

# 或用 bag 离线调参
python tools\jetarm_marker\live_ir_markers.py --source bag data\jetarm_marker\bags\marker_move_rgb_ir_depth_01
```

窗口显示：`candidate_count`、跟踪状态、`m1_gate PASS/FAIL`、几何比；m1 可疑时标红。`q` 退出，`r` 重置跟踪，`+`/`-` 调阈值。

### 4A-live-3d. 实时针体 3D（当前优先）

IR + depth → 针尖/针轴 → `GLVisualizationWidget`（相机系 mm，先不接 CT）：

```powershell
ssh ubuntu@192.168.55.1 "bash /home/ubuntu/jetarm_marker_tools/start_live_ir_on_jetarm.sh"
python tools\jetarm_marker\live_needle_gl.py --host 192.168.55.1

# 无硬件时用 bag 模拟
python tools\jetarm_marker\live_needle_gl.py --source bag data\jetarm_marker\bags\marker_move_rgb_ir_depth_01
```

### 4B. RGB 光流跟踪（旧验证路径）

静态 bag 已验证命令：

```bash
python tools/jetarm_marker/track_markers.py ^
  --bag data/jetarm_marker/bags/marker_static_clean_01 ^
  --init data/jetarm_marker/inits/marker_static_clean_01_init.json ^
  --start 173 ^
  --end 260
```

运动 bag 推荐用更宽松的 2D 形变阈值：

```bash
python tools/jetarm_marker/track_markers.py ^
  --bag data/jetarm_marker/bags/marker_move_clean_01 ^
  --init data/jetarm_marker/inits/marker_move_clean_01_init.json ^
  --start 0 ^
  --end 504 ^
  --jump-px 70 ^
  --rigid-tol 0.55 ^
  -o data/jetarm_marker/tracking/marker_move_clean_01_track2d_relaxed.csv ^
  --video data/jetarm_marker/tracking/marker_move_clean_01_track_preview_relaxed.mp4
```

输出：

```text
data/jetarm_marker/tracking/*_track2d*.csv
data/jetarm_marker/tracking/*_track_preview*.mp4
```

## 5. 深度融合和针姿态估计

### 5A. IR + depth 三维估计（当前推荐）

IR 和 depth 的图像分辨率、内参一致，当前可以直接用 IR marker 像素坐标查 depth，再反投影为相机坐标系下的 3D 点：

```bash
python tools/jetarm_marker/pose_from_ir_depth.py ^
  --bag data/jetarm_marker/bags/marker_static_rgb_ir_depth_01 ^
  --markers data/jetarm_marker/ir_detection/marker_static_rgb_ir_depth_01_ir_markers.csv ^
  -o data/jetarm_marker/ir_depth_pose/marker_static_rgb_ir_depth_01_ir_depth_pose.csv

python tools/jetarm_marker/pose_from_ir_depth.py ^
  --bag data/jetarm_marker/bags/marker_move_rgb_ir_depth_01 ^
  --markers data/jetarm_marker/ir_detection/marker_move_rgb_ir_depth_01_ir_markers.csv ^
  -o data/jetarm_marker/ir_depth_pose/marker_move_rgb_ir_depth_01_ir_depth_pose.csv
```

输出：

```text
data/jetarm_marker/ir_depth_pose/<bag_name>_ir_depth_pose.csv
data/jetarm_marker/ir_depth_pose/<bag_name>_ir_depth_pose.summary.json
```

当前已验证：

```text
marker_static_rgb_ir_depth_01:
  IR detection: 49/49 valid
  IR + depth pose: 0/49 valid after m1 geometry gate
  note: this bag likely did not contain a reliable true m1 view

marker_move_rgb_ir_depth_01:
  IR detection: 98/98 valid
  IR + depth pose: 69/98 valid after m1 geometry gate
  rejected by m1 geometry gate: 28 frames
  tip motion extent: about 125 mm
```

注意：`pose_from_ir_depth.py` 默认用 `--depth-half-window 13`，这是为了在反光球附近深度空洞较多时提高有效率。它适合当前“跑通链路”的阶段，但后面做精度时需要更严格的深度策略和实测 marker 球心局部几何。

### 5B. RGB + depth 三维估计（旧验证路径）

静态 bag：

```bash
python tools/jetarm_marker/pose_from_markers.py ^
  --bag data/jetarm_marker/bags/marker_static_clean_01 ^
  --track data/jetarm_marker/tracking/marker_static_clean_01_track2d.csv ^
  --geometry data/jetarm_marker/geometry/needle_geometry.json ^
  --scene data/jetarm_marker/geometry/scene_transform.json
```

运动 bag：

```bash
python tools/jetarm_marker/pose_from_markers.py ^
  --bag data/jetarm_marker/bags/marker_move_clean_01 ^
  --track data/jetarm_marker/tracking/marker_move_clean_01_track2d_relaxed.csv ^
  --geometry data/jetarm_marker/geometry/needle_geometry.json ^
  --scene data/jetarm_marker/geometry/scene_transform.json ^
  -o data/jetarm_marker/tracking/marker_move_clean_01_pose_relaxed.csv
```

输出：

```text
data/jetarm_marker/tracking/*_pose*.csv
data/jetarm_marker/tracking/*_pose*.summary.json
```

当前 `needle_geometry.json` 是 placeholder，`measured=false`。这能用于显示和链路验证，不能用于真实精度结论。

## 6. 在 3D 前端回放

当前推荐先回放 IR + depth 结果：

```bash
python tools/jetarm_marker/replay_pose_csv.py ^
  data/jetarm_marker/ir_depth_pose/marker_move_rgb_ir_depth_01_ir_depth_pose.csv
```

这个回放器会自动识别 `pose_from_ir_depth.py` 输出的 `tip_x_cam_mm / axis_x_cam` 字段，只播放 `valid=1` 的帧，因此被 m1 几何门控剔除的误识别帧不会显示。

3D 回放颜色约定：

```text
红色方块 = 估计针尖
绿色方块 = 沿针轴反方向延长得到的针尾参考点
m0 蓝色点 = 右侧 marker
m1 黄色点 = 针尖方向 marker
m2 青色点 = 针尾方向 marker
m3 紫色点 = 左侧 marker
浅色短线 = marker 架的 m0-m3 与 m2-m1 两条支架线
```

旧 RGB + depth 结果也仍然可以回放：

```bash
python tools/jetarm_marker/replay_pose_csv.py ^
  data/jetarm_marker/tracking/marker_static_clean_01_pose.csv
```

或：

```bash
python tools/jetarm_marker/replay_pose_csv.py ^
  data/jetarm_marker/tracking/marker_move_clean_01_pose_relaxed.csv
```

这个脚本复用现有 `GLVisualizationWidget`，不会修改原本 IMU 流程。

## 当前验证结果

**主路径（IR + depth）** — 见 `data/jetarm_marker/ir_depth_pose/*.summary.json`

```text
marker_static_rgb_ir_depth_01:
  IR detection: 49/49
  IR + depth pose: 0/49 valid (m1 geometry gate; likely no reliable m1 view)

marker_move_rgb_ir_depth_01:
  IR detection: 98/98
  IR + depth pose: 69/98 valid
  m1 gate rejects: 28 frames
  tip motion extent: ~125 mm
```

结论：**链路跑通，刚体/精度未验收**。完整状态见 [docs/jetarm_marker/STATUS.md](../../docs/jetarm_marker/STATUS.md)。

## 下一步（已决策）

**优先：实时 IR 2D 检测窗口** — 订阅 `/depth_cam/ir/image_raw`，现场看 m0–m3 与 m1 门控 pass/fail。

**不要**：继续盲录 bag、不要直接冲实时 3D。

顺序：稳定角度 → 2–3 段高质量 bag → 刚体定量 → 实时 3D → 相机→头模配准。

## 旧路径对照（非主叙事）

```text
marker_static_clean_01 (RGB LK): 88/88 pose, tip jitter ~0.53 mm — 无 IR，仅对照
marker_move_clean_01 (relaxed):   129/505 pose — 遮挡多，非 IR 主线
```
