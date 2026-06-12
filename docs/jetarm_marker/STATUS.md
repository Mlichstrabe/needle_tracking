# JetArm Marker V1 — 当前状态（Cursor 同步）

**工作目录（唯一有效）**

```text
C:\Users\Lu\PyCharmMiscProject\needle_tracking\needle_tracking 无配准
```

**不是** `nneedle_tracking 3-27-稳定备份\needle_tracking 无配准`。

**核心路径**

```text
tools/jetarm_marker/
data/jetarm_marker/
requirements-jetarm-marker.txt
```

**Agent 优先阅读**

```text
tools/jetarm_marker/README.md
tools/jetarm_marker/live_needle_gl.py
tools/jetarm_marker/detect_ir_markers.py
tools/jetarm_marker/pose_from_ir_depth.py
docs/jetarm_marker/LEGACY_RGB.md   # 旧 RGB 路径
```

**目录布局（2026-06 整理后）**

```text
tools/jetarm_marker/
  主路径：live_needle_gl, live_ir_markers, detect_ir_markers,
          pose_from_ir_depth, replay_pose_csv, ir_marker_detect,
          needle_gl_view, ir_depth_stream_protocol
  legacy/：track_markers, pose_from_markers, init_markers, run_pipeline 等
```

---

## 当前目标

用 JetArm 的 Orbbec/Gemini 深度相机，**临时代替 Aimooe**，验证：

```text
针上 4 个反光 marker -> 识别 marker -> 获取深度 -> 估计针轴和针尖 -> 在 needle_tracking 3D 前端显示一根针
```

与主程序 IMU「针尖钉 Entry」路径**并行实验**，不宣称临床或 Aimooe 级精度。

---

## 已完成

### 1. 相机数据链路

ROS2 已确认可用：

```text
/depth_cam/rgb/image_raw
/depth_cam/depth/image_raw
/depth_cam/ir/image_raw
/depth_cam/*/camera_info
/depth_cam/depth/points
```

### 2. 模态结论

**IR 对反光球效果明显优于 RGB** → **2D marker 检测主路线为 IR**（已决策，不再待定）。

### 3. 关键 bag（已拷本地）

```text
data/jetarm_marker/bags/marker_static_rgb_ir_depth_01
data/jetarm_marker/bags/marker_move_rgb_ir_depth_01
```

旧 clean bag（无 IR）仍保留作 RGB 光流对照，**非主路径**。

### 4. IR marker 检测

`tools/jetarm_marker/detect_ir_markers.py`

输出：

```text
data/jetarm_marker/ir_detection/*_ir_markers.csv
*_ir_markers.mp4
*_ir_markers_contact.jpg
```

| bag | IR 2D 检测 |
|-----|------------|
| marker_static_rgb_ir_depth_01 | 49/49 valid |
| marker_move_rgb_ir_depth_01 | 98/98 valid |

### 5. IR + depth 三维估计

`tools/jetarm_marker/pose_from_ir_depth.py`

逻辑：

```text
IR 2D marker 坐标
-> depth 图取中值深度
-> 反投影为 camera 坐标系 3D marker
-> m2 -> m1 作为针轴方向
-> m1 沿轴向外推 140 mm 作为针尖
```

### 6. m1 误识别门控

m1 易与下方关节点混淆，几何门控：

```text
|m2-m1| / |m0-m3| >= 0.55
```

不满足 → 该帧 `valid=0`，不参与 3D 回放。

### 7. 移动包当前结果

`marker_move_rgb_ir_depth_01`（见 `ir_depth_pose/*summary.json`）：

```text
总帧数：98
有效 3D 姿态帧：69
被 m1 几何门控剔除：28 帧
m1 -> 针尖距离：140 mm（手工测量）
针尖运动范围：约 124.56 mm
```

静态包：IR 检测 49/49，但 **3D 姿态 0/49**（likely 缺少可靠 m1 视角，非算法单独问题）。

### 8. 3D 离线回放

`tools/jetarm_marker/replay_pose_csv.py`

```powershell
cd "C:\Users\Lu\PyCharmMiscProject\needle_tracking\needle_tracking 无配准"
python tools\jetarm_marker\replay_pose_csv.py data\jetarm_marker\ir_depth_pose\marker_move_rgb_ir_depth_01_ir_depth_pose.csv
```

回放图例：

```text
红色方块 = 估计针尖
绿色方块 = 针尾参考点
蓝色 = m0 | 黄色 = m1 | 青色 = m2 | 紫色 = m3
浅色线 = m0-m3 与 m2-m1 支架线
```

---

## 当前定位

**链路跑通** ✅ — **姿态估计可靠** ❌

只能说：

```text
IR 检测可行
IR + depth 初步可行
刚体稳定性仍需验证
```

**不能**做精度结论。

### 主要问题

1. m1 易被下方关节点误识别  
2. m1 不完整入镜 → 针尖大偏差  
3. 4 marker 的 3D 相对关系不应随帧变化（当前未稳定）  
4. depth 在反光球附近可能空洞或混入背景  
5. **尚未**完成相机坐标系 → 人头模型坐标系配准  

---

## 实时 IR 2D 预览（已实现）

**架构**：JetArm 只跑相机 + `ir_stream_server.py`（TCP JPEG）；Windows 跑检测 + OpenCV 窗口。

### 启动步骤

```powershell
# 1. 同步脚本到 JetArm（首次或更新后）
.\tools\jetarm_marker\deploy_to_jetarm.ps1

# 2. SSH 启动相机 + IR 转发（JetArm 上执行，或由 deploy 提示的命令）
ssh ubuntu@192.168.55.1 "bash /home/ubuntu/jetarm_marker_tools/start_live_ir_on_jetarm.sh"

# 3. Windows 开预览窗口
cd "C:\Users\Lu\PyCharmMiscProject\needle_tracking\needle_tracking 无配准"
python tools\jetarm_marker\live_ir_markers.py --host 192.168.55.1
```

**离线 bag 调参**（同一 UI）：

```powershell
python tools\jetarm_marker\live_ir_markers.py --source bag data\jetarm_marker\bags\marker_move_rgb_ir_depth_01
```

**快捷键**：`q` 退出 · `r` 重置跟踪 · `+`/`-` 调阈值

**已验证**：USB 网段 `192.168.55.1:8765` 可收到 640×400 IR 帧（~30 Hz）。

### 模块

| 文件 | 作用 |
|------|------|
| `ir_marker_detect.py` | 共享检测 + ROM + m1 门控 |
| `ir_depth_stream_protocol.py` | TCP 流 JARM/JARD |
| `ir_stream_server.py` | JetArm 侧 IR+depth 推流 |
| `needle_gl_view.py` | 3D marker 叠加与 CSV 回放窗口 |
| `live_ir_markers.py` | Windows 2D 实时/离线预览 |
| `live_needle_gl.py` | Windows 实时 3D 针体 |
| `deploy_to_jetarm.ps1` | scp 到 `~/jetarm_marker_tools/` |

---

## 实时针体 3D 显示（优先）

```powershell
ssh ubuntu@192.168.55.1 "bash /home/ubuntu/jetarm_marker_tools/start_live_ir_on_jetarm.sh"
python tools\jetarm_marker\live_needle_gl.py --host 192.168.55.1
```

- 流协议 **JARD**：IR JPEG + depth uint16 同帧
- 坐标系：**相机系 mm**（尚未接 CT / scene 配准）
- 离线试：`python tools\jetarm_marker\live_needle_gl.py --source bag data\jetarm_marker\bags\marker_move_rgb_ir_depth_01`

## 下一步（现场调机）

**要做**：实时 GL 里看到针尖随手动；同时用 `live_ir_markers` 调阈值/ROM

现场移动针架时观察：

- m1 什么时候丢  
- m1 什么时候被关节点抢走  
- 哪个相机角度最稳定  
- 背景反光是否影响检测  

### 后续顺序（门控通过后再做）

```text
1. 找到稳定拍摄角度和距离
2. 再录 2-3 段高质量 bag 做定量验证
3. 批量评估 2D/3D marker 刚体稳定性
4. 通过后再接实时 3D
5. 最后做相机坐标系 -> 人头模型坐标系配准
```

---

## 旧路径（仅供参考）

已归档至 `tools/jetarm_marker/legacy/`，文档见 [LEGACY_RGB.md](LEGACY_RGB.md)。

---

## 文档索引

| 文档 | 用途 |
|------|------|
| 本文 | Cursor 会话 handoff / 进度真相 |
| [tools/jetarm_marker/README.md](../../tools/jetarm_marker/README.md) | 主路径命令 |
| [LEGACY_RGB.md](LEGACY_RGB.md) | 旧 RGB/LK 与模态探路 |
| [2026-06-11-jetarm-marker-v1-implementation.md](../superpowers/specs/2026-06-11-jetarm-marker-v1-implementation.md) | 九阶段实施表 |
