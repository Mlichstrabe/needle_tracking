# needle_tracking（无配准版）

PyQt5 桌面应用：串口 IMU 姿态 + CT 三维显示 + 穿刺点选点与对准训练。**不做 CT–患者空间配准**；连接设备后针尖锚定在 Entry，仅跟踪姿态。

## 功能概览

- 串口 IMU（JY901S 协议）：四元数 / 欧拉角
- DICOM 文件夹 → 头部网格 → OpenGL 显示
- 在 CT 表面选择 **Entry（穿刺点）**，默认 **Target** 为模型中心偏移
- 连接 IMU 后：针尖固定于 Entry，3D 显示针体；**对准引导罗盘** + 穿刺面板角度误差
- 可选 **穿刺路径引导**：预设方向线、姿态锁定（右侧模拟面板）

## 目录结构

```text
.
├── main.py                 # 入口
├── core/
│   ├── device_manager.py   # 串口与帧解析
│   ├── dicom_loader.py     # CT 加载（后台线程）
│   └── imu_kinematics.py   # 四元数 → 针轴、IMU 位置
├── ui/
│   ├── main_window.py      # 主窗口与数据流
│   └── widgets/
│       ├── gl_widget.py           # 3D 视图
│       ├── panels.py              # 设备 / IMU / CT / 穿刺 / 对准罗盘
│       ├── simulation_panel.py    # 路径引导与姿态锁定
│       └── puncture_point_selector.py
└── styles/stylesheet.qss
```

## 依赖与运行

```bash
pip install PyQt5 numpy pyserial pydicom scikit-image trimesh pyqtgraph PyOpenGL
python main.py
```

建议 Python 3.8+，Windows 10/11（串口）。

## 典型流程

1. 左侧：加载 DICOM → 右侧：开始选择穿刺点 → 在 3D 模型上点击 Entry  
2. 左侧：选择串口并连接设备  
3. 调整针体对准 Target（红球）；查看对准罗盘与穿刺面板角度  
4. （可选）右侧启动「穿刺路径引导」并锁定姿态  

## 关键实现说明

- **针长**：默认 162 mm（`MainWindow.needle_length`），用于由针尖反推 IMU 位置。  
- **针轴**：`imu_kinematics.needle_axis_scene_normalized()`，IMU 体坐标参考方向写死在模块内。  
- **Target 点**：加载 CT 后为 `center + [30, -25, 60]`（mm），尚未提供 UI 重选。  

## 开发入口

| 需求 | 文件 |
|------|------|
| 串口 / 校准 | `core/device_manager.py` |
| 姿态与针几何 | `core/imu_kinematics.py` |
| 主流程与信号 | `ui/main_window.py` |
| 3D 渲染 | `ui/widgets/gl_widget.py` |
| CT 导入 | `core/dicom_loader.py` |

## 常见问题

**串口连不上**：确认 COM 口与波特率 115200，端口未被占用。  

**窗口过大**：启动时会按屏幕可用区域自动缩放（约 88%）。  

**OpenGL 异常**：安装/更新 `PyOpenGL` 与显卡驱动。  

**PowerShell profile 报错**：与项目无关，可用 `powershell -NoProfile` 或调整执行策略。
