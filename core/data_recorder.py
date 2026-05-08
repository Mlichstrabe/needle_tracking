"""数据记录器"""
import os
import time
from datetime import datetime


class DataRecorder:
    def __init__(self, output_dir="recordings"):
        self.output_dir = output_dir
        self._file = None
        self._is_recording = False
        self._start_time = None
        self._record_count = 0

    @property
    def is_recording(self):
        return self._is_recording

    @property
    def record_count(self):
        return self._record_count

    @property
    def elapsed_time(self):
        if self._start_time:
            return time.time() - self._start_time
        return 0

    def start(self, filename=None):
        """开始记录"""
        if self._is_recording:
            return False

        os.makedirs(self.output_dir, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}.csv"

        filepath = os.path.join(self.output_dir, filename)
        self._file = open(filepath, 'w', encoding='utf-8')
        self._file.write("Time,Tip_X,Tip_Y,Tip_Z,Roll,Pitch,Yaw\n")

        self._is_recording = True
        self._start_time = time.time()
        self._record_count = 0
        return filepath

    def record(self, tip_position, euler_angles):
        """记录一帧数据"""
        if not self._is_recording or not self._file:
            return

        elapsed = time.time() - self._start_time
        x, y, z = tip_position
        roll, pitch, yaw = euler_angles

        self._file.write(f"{elapsed:.3f},{x:.4f},{y:.4f},{z:.4f},{roll:.2f},{pitch:.2f},{yaw:.2f}\n")
        self._record_count += 1

    def stop(self):
        """停止记录"""
        if self._file:
            self._file.close()
            self._file = None
        self._is_recording = False
        return self._record_count
