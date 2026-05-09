"""设备管理器 - 稳定版本"""
import serial
import struct
import threading
from PyQt5.QtCore import QObject, pyqtSignal
import numpy as np


class DeviceManager(QObject):
    connected = pyqtSignal()
    disconnected = pyqtSignal()
    data_received = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, port=None, baudrate=115200):  # ← 改这里：port默认None
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self._running = False
        self._thread = None
        self.is_connected = False

        # 数据缓存
        self._data_cache = {
            'acc': None,
            'gyro': None,
            'euler': None,
            'quaternion': None,
            'mag': None
        }

        # 时间戳缓存
        self._last_timestamp = {
            'acc': None,
            'gyro': None,
            'euler': None,
            'quaternion': None,
            'mag': None
        }

        # ====== 陀螺仪零偏校准 ======
        self._gyro_bias = np.zeros(3)
        self._gyro_calib_samples = []
        self._gyro_calib_done = False

    def connect(self, port=None, baudrate=None):  # ← 改这里：允许动态指定
        """连接设备"""
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
                timeout=0.5
            )
            self.is_connected = True
            self._running = True

            self._thread = threading.Thread(target=self._read_loop, daemon=True)
            self._thread.start()

            self.connected.emit()
            print(f"✓ 设备已连接: {self.port}")
            return True

        except Exception as e:
            print(f"✗ 连接失败: {e}")
            self.error_occurred.emit(str(e))
            return False

    # ===== 以下代码保持100%不变 =====
    def disconnect(self):
        """断开设备"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

        if self.serial and self.serial.is_open:
            self.serial.close()

        self.is_connected = False
        self._gyro_calib_done = False
        self._gyro_calib_samples = []
        self.disconnected.emit()
        print("✓ 设备已断开")

    # ====== 陀螺仪零偏校准 ======

    def start_gyro_bias_calibration(self):
        """开始陀螺仪零偏采集（需保持静止）"""
        self._gyro_calib_samples = []
        self._gyro_calib_done = False
        print("✓ 陀螺仪零偏校准开始，请保持传感器静止3秒...")

    def _feed_gyro_for_calib(self, gyro):
        """向校准器送入一帧陀螺数据"""
        if self._gyro_calib_done:
            return
        self._gyro_calib_samples.append(gyro.copy())
        if len(self._gyro_calib_samples) >= 150:  # ~3秒 @ 50Hz
            self._finish_gyro_bias_calibration()

    def _finish_gyro_bias_calibration(self):
        """完成陀螺仪零偏校准"""
        if not self._gyro_calib_samples:
            return
        samples = np.array(self._gyro_calib_samples)
        self._gyro_bias = np.mean(samples, axis=0)
        self._gyro_calib_done = True
        print(f"✓ 陀螺仪零偏校准完成: "
              f"[{self._gyro_bias[0]:.4f}, {self._gyro_bias[1]:.4f}, {self._gyro_bias[2]:.4f}] rad/s")

    def get_gyro_bias(self):
        """获取陀螺仪零偏"""
        return self._gyro_bias.copy() if self._gyro_calib_done else None

    def get_calibrated_gyro(self, raw_gyro):
        """获取去偏置后的陀螺仪数据"""
        if self._gyro_calib_done:
            return raw_gyro - self._gyro_bias
        return raw_gyro

    def _read_loop(self):
        """数据读取循环"""
        buffer = bytearray()

        while self._running:
            try:
                if self.serial.in_waiting > 0:
                    chunk = self.serial.read(self.serial.in_waiting)
                    buffer.extend(chunk)

                    while len(buffer) >= 11:
                        if buffer[0] != 0x55:
                            buffer.pop(0)
                            continue

                        if len(buffer) < 11:
                            break

                        data_type = buffer[1]
                        frame = buffer[:11]

                        checksum = sum(frame[:10]) & 0xFF
                        if checksum != frame[10]:
                            buffer.pop(0)
                            continue

                        buffer = buffer[11:]

                        parsed = self._parse_frame(data_type, frame)
                        if parsed:
                            self._data_cache.update(parsed)

                            if 'quaternion' in parsed:
                                self._emit_data()

            except Exception as e:
                print(f"[Device] 读取错误: {e}")

    def _parse_frame(self, data_type, frame):
        """解析数据帧"""
        try:
            if data_type == 0x51:
                ax = struct.unpack('<h', frame[2:4])[0] / 32768.0 * 16 * 9.8
                ay = struct.unpack('<h', frame[4:6])[0] / 32768.0 * 16 * 9.8
                az = struct.unpack('<h', frame[6:8])[0] / 32768.0 * 16 * 9.8
                return {'acc': np.array([ax, ay, az])}

            elif data_type == 0x52:
                gx = struct.unpack('<h', frame[2:4])[0] / 32768.0 * 2000 * np.pi / 180
                gy = struct.unpack('<h', frame[4:6])[0] / 32768.0 * 2000 * np.pi / 180
                gz = struct.unpack('<h', frame[6:8])[0] / 32768.0 * 2000 * np.pi / 180
                gyro = np.array([gx, gy, gz])
                # 送入零偏校准器（仅首次连接时）
                if not self._gyro_calib_done:
                    self._feed_gyro_for_calib(gyro)
                return {'gyro': gyro}

            elif data_type == 0x53:
                roll = struct.unpack('<h', frame[2:4])[0] / 32768.0 * 180
                pitch = struct.unpack('<h', frame[4:6])[0] / 32768.0 * 180
                yaw = struct.unpack('<h', frame[6:8])[0] / 32768.0 * 180
                return {'euler': np.array([roll, pitch, yaw])}

            elif data_type == 0x54:
                mx = struct.unpack('<h', frame[2:4])[0]
                my = struct.unpack('<h', frame[4:6])[0]
                mz = struct.unpack('<h', frame[6:8])[0]
                return {'mag': np.array([mx, my, mz])}

            elif data_type == 0x59:
                q0 = struct.unpack('<h', frame[2:4])[0] / 32768.0
                q1 = struct.unpack('<h', frame[4:6])[0] / 32768.0
                q2 = struct.unpack('<h', frame[6:8])[0] / 32768.0
                q3 = struct.unpack('<h', frame[8:10])[0] / 32768.0
                return {'quaternion': np.array([q0, q1, q2, q3])}

        except Exception as e:
            print(f"[Device] 解析错误: {e}")

        return None

    def _emit_data(self):
        """发送完整数据包（去掉冗余滤波，只做透传）"""
        if self._data_cache['quaternion'] is None and self._data_cache['euler'] is None:
            return

        # 发送数据（不滤波，由主窗口统一处理）
        data = {k: (v[:] if isinstance(v, list) else (v.copy() if v is not None else None))
                for k, v in self._data_cache.items()}
        self.data_received.emit(data)

    def calibrate_acceleration(self):
        """加速度校准"""
        if not self.is_connected:
            return False
        try:
            self.serial.write(b'\xFF\xAA\x01\x00\x00')
            print("✓ 加速度校准命令已发送")
            return True
        except:
            return False

    def calibrate_magnetic_start(self):
        """开始磁场校准"""
        if not self.is_connected:
            return False
        try:
            self.serial.write(b'\xFF\xAA\x01\x07\x00')
            print("✓ 磁场校准开始")
            return True
        except:
            return False

    def calibrate_magnetic_end(self):
        """结束磁场校准"""
        if not self.is_connected:
            return False
        try:
            self.serial.write(b'\xFF\xAA\x01\x00\x00')
            print("✓ 磁场校准结束")
            return True
        except:
            return False
