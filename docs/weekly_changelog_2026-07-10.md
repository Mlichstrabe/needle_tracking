# 版本对比文档 — 本周改动 (2026-07-05 → 2026-07-10)

> **基线**: `bc991a9` (2026-07-05) "Fix CT entry picking, viewport, and JetArm marker pipeline"  
> **当前**: `f8385e1` (2026-07-10) "revert: remove needle arrow marker from 3D view"  
> **涉及 commits**: 4 个 (09c1f9d, c50852c, 7e54213, f8385e1)

---

## 变更摘要

| 指标 | 数值 |
|------|------|
| 修改文件 | **17 个** (11 修改 + 3 新增 + 3 仅配置) |
| 新增代码行 | **+1329** |
| 删除代码行 | **-598** |
| 核心模块改动 | `device_manager.py` (重写), `imu_kinematics.py` (重构), `main_window.py` (大幅增强) |

---

## 一、架构级变更

### 1.1 新增基础设施模块

| 文件 | 用途 |
|------|------|
| `core/logging_config.py` | 统一日志系统：控制台 + 文件双输出、级别过滤、第三方库噪音抑制 |
| `core/project_paths.py` | 项目路径解析单例，支持 PyInstaller (`sys._MEIPASS`) 打包 |
| `docs/project_evaluation_report.md` | 357 行项目全面评估报告 (P0/P1/P2/P3 问题分类) |

**影响**: 
- 从此所有模块使用 `logger.info/debug/error` 替代裸 `print()`
- 路径不再依赖 `Path(__file__).resolve().parents[N]` 魔数
- 日志文件写入项目根目录 `app.log`

### 1.2 `DeviceManager` 全面重写

```
core/device_manager.py: 411 行变更 (-80% 代码翻新)
```

| 改动点 | 旧版 | 新版 |
|--------|------|------|
| **线程模型** | 无锁共享状态（`is_connected`、`_gyro_calib_samples` 裸读写） | `threading.Lock` 保护所有跨线程属性 |
| **异常处理** | `except Exception` 裸捕获 5 处 | `except (SerialException, OSError, ValueError, struct.error)` 精确捕获 |
| **串口 buffer** | `bytearray.pop(0)` 逐字节 O(n) 跳过 | `buffer.find(0x55)` + 切片一次性跳过，4KB 上限 |
| **校准超时** | 无限等待 150 样本 | 5 秒超时后用已有样本完成 |
| **日志** | `print()` | `logger.info/warning/error` |
| **类型注解** | 无 | `from __future__ import annotations` + 完整类型注解 |

**影响**: 
- **线程崩溃风险消除** (P0 级修复)
- 串口噪声不再无限堆积内存
- 校准失败不会静默挂起

---

## 二、功能新增与变更

### 2.1 CT 加载固定路径

**文件**: `ui/widgets/panels.py`

```python
# 旧: QFileDialog.getExistingDirectory → 用户手动选文件夹
# 新: 直接加载固定路径
default_folder = r"C:\Users\Lu\PyCharmMiscProject\needle_tracking\CT6"
```

**影响**: 点击「选择 DICOM 文件夹」立即加载 CT6，跳过交互对话框。

### 2.2 入针完成弹窗

**文件**: `ui/main_window.py`

进针到达 Target 时新增 `QMessageBox.information` 弹窗，显示 Entry→Target 距离和实际进针深度。带 `_closing` 标志防护防止窗口销毁后崩溃。

### 2.3 弹窗样式修复 (QMessageBox)

**文件**: `styles/stylesheet.qss` (+45 行)

```css
QMessageBox { background: #121a26; ... }
QMessageBox QLabel { color: #d4dee8; ... }
QMessageBox QPushButton { ... }
```

**影响**: 之前系统浅色对话框 + QSS 白色文字 = 不可读；现在统一暗色主题。

### 2.4 进针计时改用真实时间

**文件**: `ui/main_window.py` — `_puncture_tick()`

| 旧 | 新 |
|----|-----|
| `dt = 0.01` (硬编码假设 10ms tick) | `dt = now - self._puncture_last_tick` (真实 `perf_counter` 差) |
| 无上限 | `dt = min(dt, 0.1)` 防止异常帧跳变 |

**影响**: 系统高负载时进针速度不再失真。

### 2.5 UI 面板刷新频率降低

**文件**: `ui/main_window.py` — `_init_timers()`

```python
# panel_update_timer: 33ms → 100ms (30fps → 10fps)
```

数据显示无需高频刷新，节省 ~60% 无效 `setText` 调用。

### 2.6 IMU 几何状态封装

**文件**: `core/imu_kinematics.py`

8 个模块级 `global` 变量 → `_State` 类单例，保留向后兼容的函数接口。

**新增**: `get_state()` / `reset_state()` 公开 API。

### 2.7 配置路径统一

`imu_geometry_config.py` 和 `gl_widget.py` 中原有的 `Path(__file__).resolve().parents[N]` 改为引用 `core/project_paths.CONFIG_DIR`。

### 2.8 UI 辅助函数优化

**文件**: `ui/widgets/ui_helpers.py`

- 颜色值提取为命名常量: `COLOR_SIDEBAR_BG`, `COLOR_INPUT_BG`
- `set_label_role()` / `set_button_variant()` 增加当前值检查，仅在值变化时触发 `unpolish/polish`

---

## 三、Bug 修复列表

| # | 严重度 | 描述 | 修复方式 |
|---|--------|------|---------|
| B1 | 🔴 P0 | `DeviceManager` 多线程数据竞争（`_gyro_calib_samples` 在两个线程间无锁并发读写） | 添加 `threading.Lock` |
| B2 | 🔴 P0 | 5 处 `except Exception` 裸捕获吞异常，连接失败时串口句柄泄露 | 精确捕获 + 异常时清理 `self.serial = None` |
| B3 | 🔴 P0 | `QTimer.singleShot(3000, self._finish_calibration)` — 窗口关闭后回调访问已销毁 UI 导致崩溃 | 添加 `_closing` 标志防护 |
| B4 | 🔴 P0 | `hasattr(self, "target_direction_world")` — 属性未初始化时对准逻辑静默跳过 | `__init__` 中显式初始化为 `None` |
| B5 | 🟡 P1 | 串口 buffer `pop(0)` 逐字节 O(n) 跳过，噪声数据堆积时严重拖慢 | `buffer.find()` + 切片 + 4KB 上限 |
| B6 | 🟡 P1 | 进针定时器假设精确 10ms tick，实际间隔波动导致进度不准 | `time.perf_counter()` 真实时间差 |
| B7 | 🟡 P1 | CT 重复加载时旧 `QThread` 泄露（`moveToThread` 未 `quit/wait`） | `_on_ct_load` 前检查并清理 `ct_loader_thread` |
| B8 | 🟡 P1 | QMessageBox 弹窗文字白色（QSS 全局 `QLabel color` 覆盖系统对话框样式） | 添加 `QMessageBox` 暗色 QSS 规则 |
| B9 | 🟢 P2 | `panels.py` 串口列表 combo 的 `addItem(text, data)` 字段不一致 | 统一为 `display` 和 `device` 参数 |
| B10 | 🟢 P3 | README 中针长 "162mm" 与实际 "200mm" 不一致 | 修正 |
| B11 | 🟢 P3 | `requirements.txt` 仅含单条 `rtree` 依赖 | 补全 10+ 库及版本约束 |
| B12 | 🟢 P3 | `gl_widget.py` 多处注释掉的 `print`/`set_preset_path` 残留 | 清理 |

---

## 四、配置与依赖变更

### 4.1 `requirements.txt` 重写

```
旧: rtree>=1.0.0
新: PyQt5>=5.15, numpy>=1.24, pyserial>=3.5, pydicom>=3.0,
    scikit-image>=0.20, pyqtgraph>=0.13, PyOpenGL>=3.1,
    trimesh>=4.0, rtree>=1.0, scipy>=1.10, lz4>=4.0
```

### 4.2 `.gitignore` 更新

新增忽略: `.workbuddy/`, `*.log`

### 4.3 `main.py` 入口变更

新增 `core.logging_config.setup_logging(level=DEBUG)` 调用。

### 4.4 未修改的模块

以下核心模块本周无任何改动: `core/dicom_loader.py`(仅 2 行注释变更), `ui/widgets/simulation_panel.py`, `ui/widgets/alignment_hud.py`, `ui/widgets/workflow_stepper.py`, `core/marker_tracker.py`, `core/marker_pose.py`

---

## 五、回滚记录

### 针体运动箭头 (已回滚)

在 commit `7e54213` 中向 `gl_widget.py` 添加了:
- `_create_cone_mesh()` 锥体生成器
- `_rotation_matrix_from_to()` 旋转矩阵计算
- `_update_arrow_transform()` 箭头变换更新

在 commit `f8385e1` 中完整回滚 (~90 行代码移除)。

---

## 六、潜在影响分析

### ⚠️ 高风险

1. **DeviceManager 重写**: 虽然增强了线程安全，但 API 行为有细微变化：
   - `connect()` 失败时 `self.serial` 现在显式设为 `None`
   - `disconnect()` 使用 `_lock` 上下文保护清除操作
   - 建议: 完整走一遍连接→校准→断开流程验证

2. **Logging 引入**: `main.py` 必须在 `QApplication` 之前调用 `setup_logging()`，若顺序错误日志将不生效。

### ⚡ 中风险

3. **panel_update_timer 降频**: 从 30fps 降至 10fps，IMU 数据显示可能有轻微延迟，但 FPS 计数器本身是每秒刷新，不受影响。

4. **CT 路径硬编码**: `C:\Users\Lu\PyCharmMiscProject\needle_tracking\CT6` — 换机器或路径变更需修改代码。

### ✅ 低风险

5. **color tokens + property 缓存**: 纯优化，不影响功能。
6. **README 更新**: 纯文档。
7. **`_State` 封装**: 函数签名和调用方式完全兼容。

---

## 七、未覆盖的遗留问题

以下问题在评估报告中已识别但未在本周修复:

| 问题 | 优先级 | 原因 |
|------|--------|------|
| MainWindow 1285 行单一文件 | P2 | 需拆分为多个 controller，属于架构重构 |
| panels.py 5 个不相关类混在一起 | P2 | 同属架构重构 |
| 无单元测试 | P3 | 需要确定测试框架和策略 |
| PuncturePointSelector 屏幕坐标→世界射线依赖 pyqtgraph 内部 API | P2 | 需要验证不同 pyqtgraph 版本的兼容性 |
