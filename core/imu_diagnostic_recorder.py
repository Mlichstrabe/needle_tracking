"""临时姿态诊断记录：保留触发前数据，并在结束时写出可分析的 CSV。"""

from __future__ import annotations

import csv
import json
import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path


FIELDS = (
    "t_from_trigger_ms", "device_emit_ns", "ui_receive_ns", "ui_done_ns",
    "ui_queue_delay_ms", "ui_processing_ms",
    "q_w", "q_x", "q_y", "q_z",
    "used_q_w", "used_q_x", "used_q_y", "used_q_z",
    "gyro_x_rad_s", "gyro_y_rad_s", "gyro_z_rad_s",
    "acc_x_m_s2", "acc_y_m_s2", "acc_z_m_s2",
    "roll_deg", "pitch_deg", "yaw_deg",
    "axis_x", "axis_y", "axis_z",
    "tip_x_mm", "tip_y_mm", "tip_z_mm",
    "imu_x_mm", "imu_y_mm", "imu_z_mm",
    "smoothing_enabled", "translation_locked",
    "range_raw_mm", "range_filtered_mm", "range_baseline_mm", "range_valid",
    "measured_displacement_mm", "applied_displacement_mm",
)


class ImuDiagnosticRecorder:
    """由 UI 线程调用；空闲时只保留最近 pre_seconds 的帧。"""

    def __init__(self, output_dir: Path, pre_seconds=10.0, max_samples=10000, clock_ns=None):
        self.output_dir = Path(output_dir)
        self.pre_ns = int(pre_seconds * 1_000_000_000)
        self.max_samples = int(max_samples)
        self.clock_ns = clock_ns or time.monotonic_ns
        self._buffer = deque()
        self._rows = []
        self._trigger_ns = None
        self._metadata = {}

    @property
    def active(self):
        return self._trigger_ns is not None

    @property
    def sample_count(self):
        return len(self._rows)

    def push(self, sample: dict):
        """追加一帧；返回记录是否达到样本上限。"""
        row = dict(sample)
        frame_ns = int(row["device_emit_ns"])
        if self.active:
            self._rows.append(row)
            return len(self._rows) >= self.max_samples
        self._buffer.append(row)
        cutoff = frame_ns - self.pre_ns
        while self._buffer and int(self._buffer[0]["device_emit_ns"]) < cutoff:
            self._buffer.popleft()
        return False

    def start(self, metadata=None):
        if self.active:
            return False
        self._trigger_ns = int(self.clock_ns())
        cutoff = self._trigger_ns - self.pre_ns
        self._rows = [row.copy() for row in self._buffer
                      if int(row["device_emit_ns"]) >= cutoff]
        self._metadata = dict(metadata or {})
        self._metadata["trigger_local_time"] = datetime.now().astimezone().isoformat()
        self._metadata["pre_seconds"] = self.pre_ns / 1_000_000_000
        return True

    def clear_buffer(self):
        """断开或更换 IMU 后丢弃旧设备的预触发帧。"""
        self._buffer.clear()

    def stop(self, reason="manual"):
        """写出 CSV 与同名 JSON；写失败时保留内存记录以供重试。"""
        if not self.active:
            return None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stem = datetime.now().strftime("imu_event_%Y%m%d_%H%M%S_%f")
        csv_path = self.output_dir / f"{stem}.csv"
        json_path = self.output_dir / f"{stem}.json"
        csv_tmp = self.output_dir / f"{stem}.csv.tmp"
        json_tmp = self.output_dir / f"{stem}.json.tmp"
        try:
            with csv_tmp.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
                writer.writeheader()
                for sample in self._rows:
                    row = dict(sample)
                    row["t_from_trigger_ms"] = round(
                        (int(sample["device_emit_ns"]) - self._trigger_ns) / 1_000_000, 3
                    )
                    writer.writerow(row)
            metadata = dict(self._metadata)
            metadata.update({"trigger_monotonic_ns": self._trigger_ns,
                             "stop_reason": reason, "sample_count": len(self._rows)})
            with json_tmp.open("w", encoding="utf-8") as handle:
                json.dump(metadata, handle, ensure_ascii=False, indent=2)
            os.replace(csv_tmp, csv_path)
            os.replace(json_tmp, json_path)
        except Exception:
            csv_tmp.unlink(missing_ok=True)
            json_tmp.unlink(missing_ok=True)
            raise
        self._rows = []
        self._trigger_ns = None
        self._metadata = {}
        return csv_path
