#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""后台读 IMU 串口（与 core.device_manager 协议一致），供三窗混合姿态。"""
from __future__ import annotations

import struct
import threading
import time
from typing import Any, Dict, List, Optional

import serial
import serial.tools.list_ports


def probe_imu_port(
    candidates: Optional[List[str]] = None, wait_s: float = 1.2
) -> Optional["ImuSerialReader"]:
    """在候选串口上找能收到 0x59 四元数的口，成功则返回已连接的 reader。"""
    if candidates is None:
        candidates = [p.device for p in serial.tools.list_ports.comports()]
    for port in candidates:
        reader = ImuSerialReader(port)
        if not reader.start():
            continue
        t0 = time.time()
        ok = False
        while time.time() - t0 < wait_s:
            if reader.get_quaternion() is not None:
                ok = True
                break
            time.sleep(0.05)
        if ok:
            print(f"[IMU] 自动选用 {port}")
            return reader
        reader.stop()
    return None


class ImuSerialReader:
    def __init__(self, port: str, baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self._ser: Optional[serial.Serial] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest: Dict[str, Any] = {
            "quaternion": None,
            "euler": None,
            "timestamp": 0.0,
        }
        self.connected = False

    def start(self) -> bool:
        try:
            self._ser = serial.Serial(self.port, self.baudrate, timeout=0.5)
        except Exception as exc:
            print(f"[IMU] 连接失败 {self.port}: {exc}")
            return False
        self._running = True
        self.connected = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[IMU] 已连接 {self.port}")
        return True

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._ser and self._ser.is_open:
            self._ser.close()
        self.connected = False

    def get_quaternion(self) -> Optional[list]:
        with self._lock:
            q = self._latest.get("quaternion")
            if q is None:
                return None
            if time.time() - self._latest["timestamp"] > 0.6:
                return None
            return list(q)

    def _loop(self) -> None:
        buf = bytearray()
        while self._running and self._ser and self._ser.is_open:
            try:
                chunk = self._ser.read(256)
                if not chunk:
                    continue
                buf.extend(chunk)
                while len(buf) >= 11:
                    if buf[0] != 0x55:
                        del buf[0]
                        continue
                    if len(buf) < 11:
                        break
                    frame = bytes(buf[:11])
                    checksum = sum(frame[:10]) & 0xFF
                    if checksum != frame[10]:
                        del buf[0]
                        continue
                    del buf[:11]
                    self._parse(frame)
            except Exception:
                time.sleep(0.05)

    def _parse(self, packet: bytes) -> None:
        if len(packet) != 11:
            return
        cmd = packet[1]
        data = packet[2:10]
        with self._lock:
            if cmd == 0x59:
                q0, q1, q2, q3 = struct.unpack("<hhhh", data)
                scale = 1.0 / 32768.0
                self._latest["quaternion"] = [
                    q0 * scale,
                    q1 * scale,
                    q2 * scale,
                    q3 * scale,
                ]
                self._latest["timestamp"] = time.time()
            elif cmd == 0x53:
                roll, pitch, yaw = struct.unpack("<hhh", data)
                s = 180.0 / 32768.0
                self._latest["euler"] = [roll * s, pitch * s, yaw * s]
                self._latest["timestamp"] = time.time()