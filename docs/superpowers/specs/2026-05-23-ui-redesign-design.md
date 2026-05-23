# UI 重设计规格 — 穿刺训练模式

**日期**: 2026-05-23  
**范围**: PyQt5 主窗口与侧栏，不改 3D/串口业务逻辑。

## 场景

术者/学员在较暗环境、27 寸屏、高 DPI（如 2560×1600 @150%）下完成：加载 CT → 选 Entry → 连 IMU → 对准 Target → 可选路径引导。

## 设计决策

1. **统一暗色 product UI**：全窗深色，消除右栏浅色块；主色 `#5ec8f0`，成功 `#6ee7a0`，危险 `#ff7b7b`。
2. **按训练流程排右栏**：① 穿刺点 → ② 对准圆盘 → ③ 路径引导；顶栏说明步骤链。
3. **样式集中化**：`styles/stylesheet.qss` + `ui/widgets/ui_helpers.py`（variant / role）；面板减少内联 `setStyleSheet`。
4. **可读性**：等宽数字（tabular）、状态卡片、主按钮 32px 高、侧栏可滚动。

## 未改

- `gl_widget` 渲染、DICOM/串口协议、对齐算法与定时器间隔。

## 文件

| 文件 | 变更 |
|------|------|
| `styles/stylesheet.qss` | 设计 token、WorkflowCard、AlignmentStatus |
| `ui/widgets/ui_helpers.py` | 新建 |
| `ui/widgets/panels.py` | 面板样式与穿刺/IMU/设备 |
| `ui/widgets/simulation_panel.py` | 流程第 3 步 |
| `ui/main_window.py` | 顶栏 + 右栏流程布局 |
