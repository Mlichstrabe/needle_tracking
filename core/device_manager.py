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

        self._filtered_quat = None
        self._filter_alpha = 0.2

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
        self.disconnected.emit()
        print("✓ 设备已断开")

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
                return {'gyro': np.array([gx, gy, gz])}

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
        """发送完整数据包（带简单滤波）"""
        if self._data_cache['quaternion'] is None and self._data_cache['euler'] is None:
            return

        # ====== 四元数滤波 ======
        q = self._data_cache.get('quaternion')
        if q is not None:
            q = np.asarray(q, dtype=float)

            if self._filtered_quat is None:
                self._filtered_quat = q.copy()
            else:
                # 确保最短路径（避免符号突变）
                dot = np.dot(q, self._filtered_quat)
                if dot < 0:
                    q = -q

                # 一阶低通滤波
                alpha = self._filter_alpha
                self._filtered_quat = (1 - alpha) * self._filtered_quat + alpha * q

                # 归一化
                norm = np.linalg.norm(self._filtered_quat)
                if norm > 0.001:
                    self._filtered_quat /= norm

            # 用滤波后的值替换
            self._data_cache['quaternion'] = self._filtered_quat.copy()

        # 发送数据
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
