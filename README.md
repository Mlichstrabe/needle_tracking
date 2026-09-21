# needle_tracking（无配准版）

PyQt5 桌面应用：串口 IMU 姿态 + CT 三维显示 + 穿刺点选点与对准训练。**不做 CT–患者空间配准**；穿刺训练中针尖以 Entry 为起点，姿态观察中针尖固定在场景原点。

## 功能概览

- 串口 IMU（JY901S 协议）：四元数 / 欧拉角
- DICOM 文件夹 → 头部网格 → OpenGL 显示
- 在 CT 表面选择 **Entry（穿刺点）**，默认 **Target** 为模型中心偏移
- 穿刺训练连接 IMU 后：针尖固定于 Entry，3D 显示针体；**对准引导罗盘** + 穿刺面板角度误差
- 姿态观察模式：红色针尖固定在场景原点，绿色 IMU 端按 IMU 姿态和针长绕原点移动
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
pip install -r requirements.txt
python main.py
```

建议 Python 3.8+，Windows 10/11（串口）。

## 100米激光模块独立读取

### 主窗口相对深度面板

左侧可折叠准备区中新增“实时穿刺深度”卡片，默认收起；展开后可查看测距状态、选择串口、连接模块并记录零点，不占用右侧入针点操作区。右侧固定实时工作区只保留对准 HUD，Entry/Target 和路径引导在其下方独立滚动。顶栏“测距”状态同步显示未连接、连接中、正常或异常。选择与 IMU 不同的串口，点击连接测距；以 9600 baud 请求 10 Hz 连续输出，沿用独立采集工具的日志。针尖就位、姿态和固定参考靶面保持不变后，等待 10 个读数（跨度至少 0.5 秒、极差不超过 5 mm）再记录零点。

相对深度 = 零点距离中值 − 当前原始距离；不做分段补偿或平滑，不截断负值或超深值。此面板独立显示，不驱动 3D 或替换原有模拟进针。错误、超过 0.5 秒无数据、超过 30 mm 的相邻读数跳变会清除零点。0–500 mm 仅为软件合理性检查，不能代表已经验证的物理工作范围。重新连接后需重新记录零点。连接前关闭占用该串口的独立工具。

`python tools/laser100m_read.py --port COM3` 按9600、8N1循环发送单次测量请求，
显示毫米距离、设备错误或超时，Ctrl+C结束并释放串口。依赖 `pyserial`。
`--list-ports` 列出串口；`--duration 15` 限时运行15秒。
默认将时间戳、校验通过的距离/错误、响应等待时间及原始字节保存至 `output/sensor_checks/` CSV。
响应等待时间包含串口和主机开销，不等于纯测量耗时。脚本不更改模块配置，不接主窗口。
`ERR--15` 表示超量程，不能当作距离；实际更新速度取决于单次测量响应时间。

静态精度测试使用：

```powershell
python tools/laser100m_range_test.py --port COM3
```

脚本默认设置20 Hz连续测量，并依次引导测试100、200、300 mm，每点在收到首帧后采集30秒。
它将所有有效距离、模块错误码和原始帧保存到 `rangefinder_test_logs/` 的CSV，
同时用JSON保存实际帧率、帧间隔、均值、标准差、峰峰值、偏差、MAE和RMSE。
单点快速测试可使用 `--reference-mm 100 --duration 10`。
默认软件补偿为 `--offset-mm 3.8`：补偿后距离=原始距离+3.8 mm；可设3、3.5、4或0，
不写入模块硬件距离修正。CSV的 `distance_mm` 始终保留原始值，另存 `corrected_distance_mm` 和 `offset_mm`。
JSON的 `statistics` 保持原始统计，`corrected_statistics` 保存补偿统计，
`offset_comparison` 在同一批样本上比较0、+3、+3.5、+4 mm及指定补偿。
错误帧不生成补偿距离，汇总显示有效回复比例；该比例不代表丢包率。
如需探测近距离下限，使用 `--distances 50,60,70,80`。参考距离以尺具为准，
不要通过调整靶面让传感器读数等于参考距离。补偿值仍需不同距离和重复测量验证。

## 典型流程

### 激光模块实时曲线（独立窗口）

STP-23L自由移动测试：`python tools/stp23l_live.py --port COM3`。
用 `--list-ports` 查询串口。采用230400波特率，被动接收上电输出，不发送100米模块命令。
默认补偿0，显示每包12个点中非负距离的中位数；界面频率为包频率，并非120 Hz点频率。
完整12点及质量字段保存至 `rangefinder_test_logs/stp23l_live_*.csv`。
支持 `--max-mm 1000 --window-s 30`，停止/关闭窗口释放串口；无需按阶段操作。

运行 `python tools/laser100m_live.py --port COM3`，点击“开始采集”，前后移动平整靶面，
实时查看原始距离和默认+2 mm补偿值（显示距离=原始距离+2 mm）。窗口默认显示最近15秒、0～500 mm，
可用 `--max-mm 1000`、`--window-s 30`、`--offset-mm 0` 修改。
开始时默认请求10 Hz连续输出并显示设置应答状态。全部回复自动保存到
`rangefinder_test_logs/laser100m_live_*.csv`；停止或关闭窗口会保存并释放串口。
超出显示范围的数字仍显示并记录，曲线断开；错误帧不生成距离，超时不保持旧数值。
曲线无平滑滤波，回复频率为电脑接收频率。该观察工具不能单独给出动态精度或运动延迟。
依赖 `PyQt5 pyqtgraph numpy pyserial`。运行前关闭占用同一串口的测试脚本。
100米实时脚本启动时先设置连续输出时间间隔为0，再请求10 Hz，各自等待并显示确认；
可用 `--frequency 5` 或 `--frequency 20` 做同条件对比。界面同时统计超过0.25秒的间断次数和最大间断。
设置命令及应答写入同名 `.settings.json`，CSV的 `reply_gap_s` 记录相邻回复的主机接收间隔。
界面可输入尺具测得的真实距离并点击“记录当前对照点”。脚本用点击时最新的有效补偿后距离计算
有符号误差、绝对误差和相对误差，保存到 `rangefinder_test_logs/distance_comparison_*.csv`；
超过0.5秒的旧数据不能记录，对照CSV同时保存原始值、固定偏置、测量时刻和数据年龄。
这是一次配置优化试验，不能保证消除模块内部失锁/重测暂停；不插值、不重复发送测量命令填补空白。
该设置不修改STP采集流程；STP仍被动接收。

1. 左侧：加载 DICOM → 右侧：开始选择穿刺点 → 在 3D 模型上点击 Entry  
2. 左侧：选择串口并连接设备  
3. 系统使用 CT 加载后的默认 Target（红球），调整针体对准并查看对准罗盘与穿刺面板角度
4. （可选）右侧启动「穿刺路径引导」并锁定姿态  

## 关键实现说明

- **针长**：默认 200 mm（`config/imu_geometry.json` → `MainWindow.needle_length`），用于由针尖反推 IMU 位置。  
- **针轴**：`imu_kinematics.needle_axis_scene_normalized()`，IMU 体坐标参考方向由 `imu_geometry.json` 配置。  
- **Target 点**：加载 CT 后默认为右侧豆状核区偏移；可在 3D 视图中点击选择自定义 Target，或点击「使用默认」。

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
