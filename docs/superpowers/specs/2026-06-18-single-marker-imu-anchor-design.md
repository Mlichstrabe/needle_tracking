# Single Marker + IMU Anchor Design

## 背景

当前 JetArm marker 实验线已经跑通 IR + depth 链路，但 4 个反光点在实时画面中并不稳定。多数情况下，系统能稳定捕捉到至少 1 个反光点；强行从不稳定的 4 点恢复完整视觉 6DoF 会导致针体漂移、跳变，甚至让错误点污染姿态估计。

本设计把职责拆开：

- JetArm IR + depth 相机负责稳定测量单个反光点的 3D 位移。
- IMU 负责针体姿态和针轴方向。
- 已知针具几何负责把单点锚点扩展为完整针体显示。

第一版目标是演示优先：在 3D 界面中看到整根针随单点平移，并按 IMU 姿态旋转。该方案不声明真实空间配准或视觉 6DoF 精度。

## 目标

实现一个独立演示工具：

```text
tools/jetarm_marker/live_anchor_imu_needle.py
```

该工具用一个稳定反光点（默认 m2）作为位移锚点，用 IMU 四元数作为姿态来源，在 3D 界面显示完整针体。

成功标准：

- 10 秒演示中，m2 大部分时间处于 tracking 状态。
- 前后移动针架时，m2 的 z 坐标连续变化。
- 左右/上下移动时，m2 的 x/y 坐标有合理变化。
- 转动针架时，针轴方向随 IMU 姿态变化。
- 短时丢点或 depth 空洞不会导致针体突然飞走。

## 非目标

第一版不做：

- CT / 头模 / scene 配准。
- 4 点视觉姿态估计。
- 用补齐点制造假的 4 点稳定。
- 将单点位移结果声明为完整视觉 6DoF。
- 直接替换主程序 IMU 主路径。

## 总体架构

```text
JetArm IR/depth
  -> 单点 m2 检测
  -> depth 中值窗口
  -> m2_camera_3d

IMU quaternion
  -> imu_kinematics
  -> needle_axis

几何配置
  -> m2_to_tip_mm
  -> needle_length_mm

融合输出
  -> tip_pos
  -> axis_dir
  -> tail_pos / imu_pos
  -> tracking_state
```

职责边界：

- 相机点只决定位移。
- IMU 只决定姿态。
- 几何配置决定 m2 到针尖、针尾和 IMU 盒的固定关系。

## 单点跟踪

默认锚点为 m2。第一版采用半自动初始化，不强依赖四点几何。

流程：

```text
IR frame
  -> bright blob candidates
  -> if previous_m2 exists:
       select nearest candidate within max_jump_px
     else:
       select highest-score candidate
  -> depth at selected m2
  -> m2_camera_3d
```

默认参数：

```text
max_jump_px = 80
max_hold_frames = 10
depth_half_window = 13
min_depth_pixels = 3
```

交互：

| 键 | 作用 |
|---|---|
| `space` | 重新选择当前最亮点为 m2 |
| `r` | 重置 m2 跟踪 |
| `+` / `-` | 调 IR 阈值 |
| `[` / `]` | 调 depth 中值窗口 |
| `h` | 开关 hold/coasting |
| `q` | 退出 |

## 状态机

| 状态 | 条件 | 行为 |
|---|---|---|
| `tracking` | IR 有 m2 且 depth 有效 | 更新 m2 位置，正常显示针体 |
| `coasting` | IR 短暂丢失或 depth 无效，且未超过 hold 帧数 | 保留上一帧位置，针体变黄并提示 hold |
| `lost` | 连续丢失超过 hold 帧数 | 停止更新位置，针体变灰或透明，等待重新捕捉 |

短时 hold 用于吸收反光球 depth 空洞和瞬时检测丢失。hold 时间必须有限，避免把旧位置伪装成有效测量。

## 姿态融合

第一版采用演示优先的坐标策略：相机系直接作为显示系使用，不做严格 scene 配准。

融合公式：

```text
axis = needle_axis_scene_normalized(quaternion)
tip = m2_pos - axis * m2_to_tip_mm
tail = tip - axis * needle_length_mm
```

其中：

- `m2_pos` 来自 IR + depth 反投影。
- `axis` 来自 IMU 四元数。
- `m2_to_tip_mm` 是配置值，第一版可先使用粗测值，后续再标定。
- `needle_length_mm` 复用当前针长默认值 162 mm。

如果未连接 IMU，工具支持 `--demo-axis`，用固定轴只验证单点位移是否稳定。

## 配置

新增配置建议：

```text
data/jetarm_marker/geometry/anchor_imu_needle.json
```

建议字段：

```json
{
  "anchor_marker": "m2",
  "m2_to_tip_mm": 140.0,
  "needle_length_mm": 162.0,
  "max_jump_px": 80.0,
  "max_hold_frames": 10,
  "depth_half_window": 13,
  "min_depth_pixels": 3
}
```

第一版暂不引入相机到 scene 的轴映射。后续若需要让显示方向更贴近实际，可扩展 `camera_to_scene_axes` 配置。

## 界面

独立窗口推荐布局：

```text
左：IR 画面，只突出当前 m2
中：depth 局部 / 数值状态
右：3D 针体
底部：m2、depth、IMU、tip、state 状态
```

底部状态栏至少显示：

```text
state=tracking/coasting/lost
m2=(x,y,z)mm
depth_pixels=n
imu=connected/demo/disconnected
axis=(x,y,z)
tip=(x,y,z)mm
```

视觉反馈：

- `tracking`：针体正常颜色。
- `coasting`：针体变黄。
- `lost`：针体变灰或透明。

## 运行方式

JetArm 推流：

```powershell
ssh ubuntu@192.168.55.1 "bash /home/ubuntu/jetarm_marker_tools/start_live_ir_on_jetarm.sh"
```

只测单点位移：

```powershell
python tools\jetarm_marker\live_anchor_imu_needle.py --host 192.168.55.1 --demo-axis
```

接入 IMU：

```powershell
python tools\jetarm_marker\live_anchor_imu_needle.py --host 192.168.55.1 --serial COMx
```

## 错误处理

| 问题 | 判定 | 行为 |
|---|---|---|
| IR 无候选点 | `candidate_count=0` | 进入 `coasting` 或 `lost` |
| m2 跳太远 | 距上一帧超过 `max_jump_px` | 拒绝该点，保持上一帧 |
| depth 空洞 | 有 IR 点但有效 depth 像素不足 | 进入 `coasting` |
| IMU 未连接 | 无 quaternion | 使用 `--demo-axis` 或固定上一姿态 |
| TCP 断流 | socket 异常 | 3 秒重连 |
| 串口断开 | DeviceManager disconnected | 位移继续，姿态固定/灰色提示 |

## 测试计划

### 1. 相机单点位移

```powershell
python tools\jetarm_marker\live_anchor_imu_needle.py --host 192.168.55.1 --demo-axis
```

检查：

- m2.z 随前后移动连续变化。
- m2.x / m2.y 随左右上下移动合理变化。
- `state` 大部分时间为 `tracking`。

### 2. IMU 姿态融合

```powershell
python tools\jetarm_marker\live_anchor_imu_needle.py --host 192.168.55.1 --serial COMx
```

检查：

- 移动针架时整根针平移。
- 转动针架时针轴旋转。
- 短时丢点时针体不突然飞走。

### 3. 10 秒演示

推荐动作：

```text
0-2s 静止
2-5s 前后移动
5-7s 左右移动
7-10s 原地旋转
```

验收以视觉连续性和状态栏数值合理为主，不做精度结论。

## 后续扩展

第一版稳定后再考虑：

- 标定 m2 到针尖的真实偏移。
- 加相机系到显示 scene 的轴映射。
- 用 2 个或更多稳定点辅助校正平移方向。
- 将该工具输出接入主窗口作为实验模式。
- 录制 10 秒演示数据并导出 3D 回放视频。
