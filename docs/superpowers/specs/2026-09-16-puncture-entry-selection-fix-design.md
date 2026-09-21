# 穿刺 Entry 选点修复设计

日期：2026-09-16  
状态：用户已确认，已实施

## 目标

恢复 CT 加载后在 3D 头部表面点击选择 Entry 的功能，并固定使用 CT 加载后生成的默认 Target，避免用户进入额外的 Target 选点流程。

## 方案

- `PuncturePointSelector` 优先使用现有 trimesh 射线检测；检测到缺少 `rtree` 时切换到内置的分块 Möller–Trumbore NumPy 射线检测。
- 保持 Entry 的 `point_selected(point, normal)` 信号和主窗口后续针尖/方向/Target 更新逻辑不变。
- 隐藏“自选 Target”和“重新选择 Target”入口；Target 仍由 CT 加载后的默认点自动设置。
- 手动 Target 信号即使被外部触发，也不再进入选择流程。

## 验收

1. CT 加载成功后，Entry 选择按钮可进入选点模式。
2. 点击 3D 头部表面后，Entry 坐标和法线显示并退出选点模式。
3. Target 显示默认点，不出现用户自选 Target 入口。
4. 无 `rtree` 时不再出现 `ModuleNotFoundError`，窗口可正常启动和关闭。
