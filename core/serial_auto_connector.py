"""双串口协议探测。

探测只打开端口、读取/发送短帧后立即释放，不把端口描述或 VID/PID 当作设备身份。
"""

from __future__ import annotations

import time
import threading

import serial
from PyQt5.QtCore import QObject, pyqtSignal
from serial.tools import list_ports


def _looks_like_imu(data: bytes) -> bool:
    for start in range(max(0, len(data) - 10)):
        frame = data[start : start + 11]
        if len(frame) == 11 and frame[0] == 0x55 and (sum(frame[:10]) & 0xFF) == frame[10]:
            return True
    return False


def _looks_like_range(data: bytes) -> bool:
    return b"\x80\x06\x83" in data


def _range_command(*values: int) -> bytes:
    payload = bytes(values)
    return payload + bytes(((-sum(payload)) & 0xFF,))


def probe_port(port, baudrate, send=None, timeout=0.8, predicate=None):
    try:
        # 与正式测距采集一致：打开前关闭 DTR/RTS，避免 USB 串口复位时漏掉首帧。
        with serial.Serial(port=None, baudrate=baudrate, timeout=0.05, write_timeout=0.5) as device:
            device.port = port
            device.dtr = False
            device.rts = False
            device.open()
            if send:
                device.reset_input_buffer()
                device.write(send)
                device.flush()
            deadline = time.monotonic() + timeout
            data = bytearray()
            while time.monotonic() < deadline:
                if device.in_waiting:
                    data.extend(device.read(min(device.in_waiting, 4096)))
                    if predicate and predicate(bytes(data)):
                        return True
                else:
                    time.sleep(0.01)
            return bool(predicate and predicate(bytes(data)))
    except (serial.SerialException, OSError, ValueError):
        return False


class SerialAutoConnector(QObject):
    finished = pyqtSignal(object)
    status = pyqtSignal(str)

    def __init__(self, occupied=None, probe_range=True, parent=None):
        super().__init__(parent)
        self.occupied = set(occupied or [])
        self.probe_range = bool(probe_range)
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        ports = [p.device for p in sorted(list_ports.comports(), key=lambda item: item.device)]
        ports = [p for p in ports if p not in self.occupied]
        imu_ports = []
        range_ports = []
        for port in ports:
            if self._stop_event.is_set():
                break
            self.status.emit(f"正在验证 {port} 协议…")
            if probe_port(port, 115200, timeout=0.7, predicate=_looks_like_imu):
                imu_ports.append(port)
                continue
            if self.probe_range and probe_port(
                    port,
                    9600,
                    send=_range_command(0x80, 0x06, 0x03),
                    timeout=3.0,
                    predicate=_looks_like_range,
                ):
                    range_ports.append(port)
        result = {
            "imu": imu_ports[0] if len(imu_ports) == 1 else None,
            "range": range_ports[0] if len(range_ports) == 1 else None,
            "imu_matches": imu_ports,
            "range_matches": range_ports,
        }
        self.finished.emit(result)
