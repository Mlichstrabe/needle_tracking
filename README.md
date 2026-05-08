# needle_tracking（无配准版）说明文档

本目录是 `needle_tracking` 项目的一个可运行快照（**无配准**版本）。程序为 **PyQt5 桌面应用**，集成了：

- **串口 IMU 设备接入**（四元数/欧拉角等数据）
- **针具姿态与位置计算**（针尖方向、IMU 位置）
- **三维 OpenGL 可视化**（`GLVisualizationWidget`）
- **三视图投影面板**（`ProjectionPanel`）
- **穿刺训练流程**（`PunctureSession` / `PunctureMonitor`）
- **模拟控制**（`SimulationPanel` / `SimulationManager`）
- **CT/DICOM 导入与显示**（`DicomModelLoader` + `CTModelPanel`）
- **穿刺点选择**（`PuncturePointPanel` + `PuncturePointSelector`）

> 备注：仓库根目录可能存在多个日期版本文件夹；本 README 只描述本目录这一套（`needle_tracking 无配准`）。

---

## 目录结构（关键模块）

```text
.
├─ main.py                      # 程序入口：导入检查 + QApplication + MainWindow
├─ config/
│  └─ settings.py               # 串口/针长/EKF 等配置（部分在本版本中仅用于打印或默认值）
├─ core/
│  ├─ device_manager.py         # 串口读取线程 + 信号 data_received/connected/...
│  ├─ puncture_monitor.py       # 穿刺监控：对齐/偏差/阈值判定（供会话使用）
│  ├─ puncture_session.py       # 穿刺训练状态机：对齐→锁定→进针→判定
│  ├─ simulation_manager.py     # 仿真管理
│  ├─ dicom_loader.py           # DICOM/CT 导入（异步 + 进度信号）
│  ├─ trajectory_tracker.py     # 轨迹跟踪（若被 UI 使用）
│  └─ ekf_filter.py             # 姿态滤波（本版本主窗口未直接使用，但入口会导入探测）
├─ ui/
│  ├─ main_window.py            # 主窗口：把所有模块“粘”在一起（核心数据流都在这里）
│  └─ widgets/
│     ├─ gl_widget.py           # OpenGL 3D 视图：针、CT 模型、轨迹等渲染
│     ├─ projection_views.py    # 三视图投影面板
│     ├─ panels.py              # 左/右侧面板（串口、IMU、针配置、CT、穿刺点等）
│     ├─ simulation_panel.py    # 模拟控制面板
│     ├─ puncture_panel.py / puncture_*（若存在） # 穿刺相关 UI
│     └─ puncture_point_selector.py # 3D 里选点逻辑（与 gl_widget 联动）
├─ styles/
│  └─ stylesheet.qss            # Qt 样式表（存在则 main.py 会加载）
└─ assets/
   └─ sounds/alarm.wav          # 可能用于报警/提示音
```

---

## 运行方式

### 1）准备 Python 环境

推荐使用虚拟环境（venv/conda 均可）。最低建议：

- Python 3.8+（越新越好）
- Windows 10/11（串口设备常见）

安装常见依赖（按实际 import 为准）：

```bash
pip install PyQt5 numpy pyserial
```

若使用 DICOM/CT 功能，通常还需要：

```bash
pip install pydicom
```

若 `gl_widget.py` 依赖 OpenGL/渲染库（按实际 import 安装）：

```bash
pip install PyOpenGL PyOpenGL_accelerate
```

> 注意：本项目未提供统一的 `requirements.txt`（至少在当前快照目录中未见），依赖需以实际 `import` 报错为准补齐。

### 2）配置串口（可选）

默认串口参数在 `config/settings.py`，但本版本的 `DeviceManager` 支持在 UI 中动态选择端口与波特率。

### 3）启动

在本目录下执行：

```bash
python main.py
```

`main.py` 会先进行一系列导入检查（打印 `[1] ... [8]`），随后创建 `QApplication`、加载 `styles/stylesheet.qss`（若存在），并创建 `MainWindow`。

---

## 主流程概览（你改功能时最该先读哪里）

### 必读文件

- **`ui/main_window.py`**：UI 组合、信号连接、定时器、数据处理路径几乎都在这里
- **`core/device_manager.py`**：串口读数据、产出 `data_received`（IMU 数据字典）
- **`ui/widgets/gl_widget.py`**：3D 绘制与交互（CT 模型显示、针位置更新、固定针尖点等）
- **`core/puncture_session.py`**：穿刺训练状态机与判定逻辑（对齐/锁定/进针/完成）
- **`core/dicom_loader.py`**：CT 导入（成功/失败/进度信号）

---

## 数据流图（核心：设备 → 姿态/位置 → 可视化/训练）

下图为本版本的关键数据与控制信号流向（可直接在支持 Mermaid 的 Markdown 渲染器中显示）。

```mermaid
flowchart LR
  %% =========================
  %% 启动与 UI 组装
  %% =========================
  subgraph Boot[启动]
    A[main.py] -->|创建 QApplication\n加载 stylesheet.qss| B[MainWindow()]
  end

  subgraph UI[UI 主窗口 ui/main_window.py]
    B --> C[_init_core_components()]
    B --> D[_init_ui(): QSplitter 左/中/右]
    B --> E[_connect_signals()]
    B --> F[_init_timers(): update_timer 60Hz\npanel_update_timer 30Hz]
  end

  %% =========================
  %% 核心组件
  %% =========================
  subgraph Core[核心模块 core/*]
    DM[DeviceManager\n(串口线程)]:::core
    PM[PunctureMonitor\n(阈值/偏差)]:::core
    PS[PunctureSession\n(穿刺状态机)]:::core
    SM[SimulationManager]:::core
    DL[DicomModelLoader\n(CT 导入)]:::core
  end

  %% =========================
  %% UI 组件 widgets
  %% =========================
  subgraph Widgets[UI 组件 ui/widgets/*]
    Conn[DeviceConnectionPanel]:::ui
    IMU[IMUDataPanel]:::ui
    Needle[NeedleConfigPanel]:::ui
    CTP[CTModelPanel]:::ui
    GL[GLVisualizationWidget\n(3D)]:::ui
    Proj[ProjectionPanel\n(三视图)]:::ui
    Sim[SimulationPanel]:::ui
    PPP[PuncturePointPanel]:::ui
    Sel[PuncturePointSelector]:::ui
  end

  %% =========================
  %% 信号连接（控制流）
  %% =========================
  Conn -->|connect_clicked / disconnect_clicked| DM
  DM -->|connected / disconnected| UIState[_on_connection_changed()]:::fn
  DM -->|error_occurred| Err[_on_device_error()]:::fn
  DM -->|data_received(dict)| OnData[_on_device_data()]:::fn

  %% =========================
  %% 数据处理（姿态→方向→位置→渲染）
  %% =========================
  OnData -->|quaternion,euler| Filter{_filter_mode?\n'normal'/'stable'}:::fn
  Filter -->|stable| SF[_apply_smart_filter()]:::fn
  Filter -->|normal| Calc[_calculate_positions_fast()]:::fn
  SF --> Calc

  Calc -->|imu_pos, tip_pos| GL
  OnData --> Dir[_update_needle_direction_fast()]:::fn
  Dir -->|needle_direction| GL

  %% 针尖固定逻辑（穿刺点已选时）
  GL -->|get_fixed_tip_position()| Calc

  %% =========================
  %% 面板刷新（显示流）
  %% =========================
  UIState --> IMU
  F -->|panel_update_timer| Panels[_update_panels()]:::fn
  Panels --> IMU
  Panels --> Needle

  %% =========================
  %% CT 导入与显示
  %% =========================
  CTP -->|load_clicked/clear_clicked| CTHandlers[_on_ct_load/_on_ct_clear]:::fn
  CTHandlers --> DL
  DL -->|progress_updated| CTProg[_on_ct_progress]:::fn
  DL -->|loading_finished(model_data)| CTDone[_on_ct_loaded]:::fn
  DL -->|loading_failed| CTFail[_on_ct_failed]:::fn
  CTDone --> GL
  CTP -->|visibility_changed| GL
  PPP -->|显示/隐藏（CT 导入后可见）| UI

  %% =========================
  %% 穿刺点选择（交互流）
  %% =========================
  PPP -->|start_selection/reselect| SelHandlers[_on_start_selection/_on_reselect_puncture_point]:::fn
  SelHandlers --> Sel
  Sel -->|point, normal| PointChosen[_on_puncture_point_selected]:::fn
  PointChosen --> GL

  %% =========================
  %% 穿刺训练会话与仿真
  %% =========================
  Sim -->|simulation_started/stopped| SimHandlers[_on_simulation_started/_on_simulation_stopped]:::fn
  Sim -->|target_direction_changed| TargetDir[_on_target_direction_changed]:::fn
  Sim -->|orientation_locked| Locked[_on_orientation_locked]:::fn

  PM --> PS
  PS -->|phase_changed/depth_changed/\nwarning/result| SessionUI[_on_phase/_on_depth/_on_result]:::fn
  PS -->|simulated_tip_moved| GL

  %% 定时器驱动的周期性逻辑
  F -->|update_timer| Tick[_on_update_tick]:::fn
  Tick -->|对齐监控/偏差计算等| PS

  classDef core fill:#1b3b5a,stroke:#3aa0ff,color:#fff;
  classDef ui fill:#2b1b5a,stroke:#b09cff,color:#fff;
  classDef fn fill:#0f172a,stroke:#94a3b8,color:#e2e8f0;
```

---

## 关键计算说明（定位坐标/方向问题时用）

### 姿态数据来源

- `DeviceManager` 通过串口读取 IMU 数据并解析为 `dict`，至少包含：
  - `quaternion`: `(q0,q1,q2,q3)`
  - `euler`: `(roll,pitch,yaw)`（单位依解析器实现）

### 从四元数到针方向

`MainWindow._update_needle_direction_fast()` 与 `_calculate_positions_fast()` 都使用类似逻辑：

1. 取一个固定“针体坐标系”的向量 `needle_body`（代码里是一个 45° 的组合向量）。
2. 用四元数旋转得到 `raw_x/raw_y/raw_z`。
3. 做轴映射得到 `tip_x/tip_y/tip_z`，再归一化得到 `needle_direction`。

### 针尖固定（穿刺点模式）

`_calculate_positions_fast()` 会向 `gl_widget` 询问 `get_fixed_tip_position()`：

- 若存在固定针尖点：`tip_pos` 直接等于该点（Entry/选点处）
- 若不存在：默认 `tip_pos = [0,0,0]`
- IMU 位置通过 `tip_pos - direction * needle_length` 计算

这套逻辑让你在穿刺训练中可以“固定针尖”，只让 IMU（针尾）随着姿态变化移动。

---

## 常见问题（FAQ）

### 1）终端出现 `profile.ps1` 执行策略报错

这是 **PowerShell 执行策略**问题，与项目代码无关。可用以下方式之一解决：

- 放开当前用户策略（推荐）：`Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`
- 启动 PowerShell 时禁用 profile：`powershell -NoProfile`

### 2）串口连不上 / 报 COM 口占用

- 检查串口号是否正确、设备是否被其他程序占用
- 确认波特率与设备一致
- Windows 下可在设备管理器确认 COM 口号

### 3）3D 不显示 / OpenGL 报错

通常是 OpenGL 依赖或显卡驱动问题，建议：

- 安装 `PyOpenGL`（必要时加 `PyOpenGL_accelerate`）
- 更新显卡驱动
- 先确认 UI 能启动，再逐步排查 `gl_widget.py` 的 import/初始化

---

## 开发建议（你准备改功能时）

- **先在 `ui/main_window.py` 找入口**：大部分“某个按钮 → 某个功能”的连接都在 `_connect_signals()`。
- **改穿刺流程**：优先改 `core/puncture_session.py`（状态机与判定），UI 只负责显示与触发。
- **改坐标/方向**：集中在 `MainWindow._calculate_positions_fast()` 与 `GLVisualizationWidget` 的坐标系映射。
- **改 CT 导入**：看 `core/dicom_loader.py` 的信号与 `MainWindow._on_ct_*` 的处理。

