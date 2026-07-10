"""设备管理器 - 线程安全 + 低延迟帧解析。

串口线程（daemon）读取原始字节 → 帧解析 → 信号发送回主线程。
所有跨线程共享状态由 _lock 保护。
"""
from __future__ import annotations

import logging
import struct
import threading
import time
from collections import deque
from typing import Any, Dict, Optional

import numpy as np
import serial
from PyQt5.QtCore import QObject, pyqtSignal

logger = logging.getLogger(__name__)

# ── 帧常量 ──────────────────────────────────────────────
FRAME_HEAD = 0x55
FRAME_LEN = 11
# 校准样本数 ≈ 3 秒 @ 50 Hz
GYRO_CALIB_SAMPLES = 150
GYRO_CALIB_TIMEOUT = 5.0  # 秒
# buffer 上限，防止串口噪声无限堆积
BUFFER_MAX_BYTES = 4096


class DeviceManager(QObject):
    connected = pyqtSignal()
    disconnected = pyqtSignal()
    data_received = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, port: Optional[str] = None, baudrate: int = 115200):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.serial: Optional[serial.Serial] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # ── 线程安全锁 ──
        self._lock = threading.Lock()
        self.is_connected = False

        # 数据缓存（仅串口线程写入，主线程通过信号拷贝读取）
        self._data_cache: Dict[str, Optional[np.ndarray]] = {
            "acc": None,
            "gyro": None,
            "euler": None,
            "quaternion": None,
            "mag": None,
        }

        # ── 陀螺仪零偏校准（锁保护）──
        self._gyro_bias = np.zeros(3)
        self._gyro_calib_samples: list = []
        self._gyro_calib_done = False
        self._gyro_calib_started_at: float = 0.0

    # ═══════════════════════════════════════════════════════
    #  连接 / 断开
    # ═══════════════════════════════════════════════════════

    def connect(self, port: Optional[str] = None, baudrate: Optional[int] = None) -> bool:
        """连接设备。port/baudrate 可选覆盖构造时的默认值。"""
        if port is not None:
            self.port = port
        if baudrate is not None:
            self.baudrate = baudrate

        if self.port is None:
            self.error_occurred.emit("未指定串口")
            return False

        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=0.5,
            )
        except (serial.SerialException, OSError, ValueError) as exc:
            logger.error("串口打开失败 port=%s: %s", self.port, exc)
            self.error_occurred.emit(f"无法打开 {self.port}: {exc}")
            self.serial = None
            return False

        with self._lock:
            self.is_connected = True
        self._running = True

        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

        self.connected.emit()
        logger.info("设备已连接: %s @ %d", self.port, self.baudrate)
        return True

    def disconnect(self) -> None:
        """断开设备，等待串口线程结束。"""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2)

        ser = self.serial
        if ser is not None:
            try:
                if ser.is_open:
                    ser.close()
            except (serial.SerialException, OSError):
                logger.warning("串口关闭时出错", exc_info=True)
            self.serial = None

        with self._lock:
            self.is_connected = False
            self._gyro_calib_done = False
            self._gyro_calib_samples.clear()

        self.disconnected.emit()
        logger.info("设备已断开")

    # ═══════════════════════════════════════════════════════
    #  陀螺仪零偏校准（线程安全）
    # ═══════════════════════════════════════════════════════

    def start_gyro_bias_calibration(self) -> None:
        """开始陀螺仪零偏采集（需保持静止）。"""
        with self._lock:
            self._gyro_calib_samples = []
            self._gyro_calib_done = False
            self._gyro_calib_started_at = time.monotonic()
        logger.info("陀螺仪零偏校准开始，请保持传感器静止 %.1f 秒...", GYRO_CALIB_TIMEOUT)

    def _feed_gyro_for_calib(self, gyro: np.ndarray) -> None:
        """向校准器送入一帧陀螺数据（串口线程调用）。"""
        with self._lock:
            if self._gyro_calib_done:
                return
            self._gyro_calib_samples.append(gyro.copy())

            # 超时检查
            elapsed = time.monotonic() - self._gyro_calib_started_at
            if elapsed > GYRO_CALIB_TIMEOUT:
                logger.warning("陀螺校准超时 (%.1fs)，用已有 %d 样本完成校准",
                               elapsed, len(self._gyro_calib_samples))
                self._finish_gyro_bias_calibration()
                return

            if len(self._gyro_calib_samples) >= GYRO_CALIB_SAMPLES:
                self._finish_gyro_bias_calibration()

    def _finish_gyro_bias_calibration(self) -> None:
        """完成陀螺仪零偏校准（持有锁时调用）。"""
        if not self._gyro_calib_samples:
            logger.warning("陀螺校准: 无有效样本")
            return
        samples = np.array(self._gyro_calib_samples)
        self._gyro_bias = np.mean(samples, axis=0)
        self._gyro_calib_done = True
        logger.info(
            "陀螺仪零偏校准完成: [%.4f, %.4f, %.4f] rad/s (%d 样本)",
            self._gyro_bias[0], self._gyro_bias[1], self._gyro_bias[2],
            len(self._gyro_calib_samples),
        )

    def get_gyro_bias(self) -> Optional[np.ndarray]:
        """获取陀螺仪零偏。返回 None 表示未校准。"""
        with self._lock:
            if self._gyro_calib_done:
                return self._gyro_bias.copy()
        return None

    def get_calibrated_gyro(self, raw_gyro: np.ndarray) -> np.ndarray:
        """获取去偏置后的陀螺仪数据。"""
        with self._lock:
            if self._gyro_calib_done:
                return raw_gyro - self._gyro_bias
        return raw_gyro

    # ═══════════════════════════════════════════════════════
    #  串口读取 (daemon 线程)
    # ═══════════════════════════════════════════════════════

    def _read_loop(self) -> None:
        """数据读取循环 — daemon 线程。"""
        buffer = bytearray()
        ser = self.serial  # 线程启动时 ser 一定非 None

        while self._running:
            try:
                n_waiting = ser.in_waiting
                if n_waiting > 0:
                    chunk = ser.read(n_waiting)
                    buffer.extend(chunk)

                    # 防止噪声导致 buffer 无限堆积
                    if len(buffer) > BUFFER_MAX_BYTES:
                        logger.warning("串口 buffer 溢出 (%d bytes)，丢弃旧数据", len(buffer))
                        buffer = buffer[-BUFFER_MAX_BYTES:]

                    # 逐帧解析（一次切片跳过无效字节，避免 O(n) pop）
                    while len(buffer) >= FRAME_LEN:
                        if buffer[0] != FRAME_HEAD:
                            # 寻找下一个帧头
                            idx = buffer.find(FRAME_HEAD, 1)
                            if idx == -1:
                                buffer.clear()
                                break
                            buffer = buffer[idx:]
                            continue

                        data_type = buffer[1]
                        frame = buffer[:FRAME_LEN]

                        # 校验
                        if (sum(frame[:10]) & 0xFF) != frame[10]:
                            buffer = buffer[1:]
                            continue

                        buffer = buffer[FRAME_LEN:]

                        parsed = self._parse_frame(data_type, frame)
                        if parsed is not None:
                            # pyqtSignal.emit 内部是线程安全的
                            self._data_cache.update(parsed)

                            if "quaternion" in parsed:
                                self._emit_data()

            except (serial.SerialException, OSError) as exc:
                logger.error("串口读取错误: %s", exc)
                if not self._running:
                    break
                # 短暂等待后重试
                time.sleep(0.05)
            except Exception:
                logger.exception("串口读取未预期错误")

    def _parse_frame(self, data_type: int, frame: bytearray) -> Optional[Dict[str, np.ndarray]]:
        """解析单帧数据，返回 {key: ndarray} 或 None。"""
        try:
            if data_type == 0x51:  # 加速度
                ax = struct.unpack("<h", frame[2:4])[0] / 32768.0 * 16 * 9.8
                ay = struct.unpack("<h", frame[4:6])[0] / 32768.0 * 16 * 9.8
                az = struct.unpack("<h", frame[6:8])[0] / 32768.0 * 16 * 9.8
                return {"acc": np.array([ax, ay, az])}

            elif data_type == 0x52:  # 陀螺仪
                gx = struct.unpack("<h", frame[2:4])[0] / 32768.0 * 2000 * np.pi / 180
                gy = struct.unpack("<h", frame[4:6])[0] / 32768.0 * 2000 * np.pi / 180
                gz = struct.unpack("<h", frame[6:8])[0] / 32768.0 * 2000 * np.pi / 180
                gyro = np.array([gx, gy, gz])
                # 送入零偏校准器
                with self._lock:
                    if not self._gyro_calib_done:
                        # 放锁后调用，避免死锁（_feed_gyro_for_calib 内部也获取锁）
                        pass
                if not self._gyro_calib_done:
                    self._feed_gyro_for_calib(gyro)
                return {"gyro": gyro}

            elif data_type == 0x53:  # 欧拉角
                roll = struct.unpack("<h", frame[2:4])[0] / 32768.0 * 180
                pitch = struct.unpack("<h", frame[4:6])[0] / 32768.0 * 180
                yaw = struct.unpack("<h", frame[6:8])[0] / 32768.0 * 180
                return {"euler": np.array([roll, pitch, yaw])}

            elif data_type == 0x54:  # 磁力计
                mx = struct.unpack("<h", frame[2:4])[0]
                my = struct.unpack("<h", frame[4:6])[0]
                mz = struct.unpack("<h", frame[6:8])[0]
                return {"mag": np.array([mx, my, mz])}

            elif data_type == 0x59:  # 四元数
                q0 = struct.unpack("<h", frame[2:4])[0] / 32768.0
                q1 = struct.unpack("<h", frame[4:6])[0] / 32768.0
                q2 = struct.unpack("<h", frame[6:8])[0] / 32768.0
                q3 = struct.unpack("<h", frame[8:10])[0] / 32768.0
                return {"quaternion": np.array([q0, q1, q2, q3])}

        except (struct.error, IndexError) as exc:
            logger.error("帧解析错误 type=0x%02X: %s", data_type, exc)

        return None

    def _emit_data(self) -> None:
        """发送完整数据包 — 深拷贝后通过信号传递到主线程。"""
        cache = self._data_cache
        if cache["quaternion"] is None and cache["euler"] is None:
            return

        data: Dict[str, Optional[np.ndarray]] = {}
        for k, v in cache.items():
            if v is None:
                data[k] = None
            elif isinstance(v, list):
                data[k] = v[:]
            else:
                data[k] = v.copy()

        self.data_received.emit(data)

    # ═══════════════════════════════════════════════════════
    #  传感器校准命令
    # ═══════════════════════════════════════════════════════

    def _write_command(self, cmd: bytes, name: str) -> bool:
        """统一写命令入口，带错误处理。"""
        with self._lock:
            if not self.is_connected:
                logger.warning("未连接，无法执行 %s", name)
                return False
        try:
            self.serial.write(cmd)  # type: ignore[union-attr]
        except (serial.SerialException, OSError) as exc:
            logger.error("%s 失败: %s", name, exc)
            return False
        logger.info("%s 已发送", name)
        return True

    def calibrate_acceleration(self) -> bool:
        """加速度校准"""
        return self._write_command(b"\xFF\xAA\x01\x00\x00", "加速度校准")

    def calibrate_magnetic_start(self) -> bool:
        """开始磁场校准"""
        return self._write_command(b"\xFF\xAA\x01\x07\x00", "磁场校准开始")

    def calibrate_magnetic_end(self) -> bool:
        """结束磁场校准"""
        return self._write_command(b"\xFF\xAA\x01\x00\x00", "磁场校准结束")
