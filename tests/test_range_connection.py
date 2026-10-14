import queue
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from tools import laser100m_live


class _SilentSerial:
    def __init__(self):
        self.in_waiting = 0
        self.writes = []
        self.port = None
        self.dtr = None
        self.rts = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def open(self):
        return None

    def reset_input_buffer(self):
        return None

    def write(self, value):
        self.writes.append(bytes(value))

    def flush(self):
        return None

    def read(self, _size):
        time.sleep(0.003)
        return b""


class RangeConnectionTests(unittest.TestCase):
    def test_silent_module_retries_once_then_finishes_with_diagnostics(self):
        serial_device = _SilentSerial()
        args = SimpleNamespace(
            port="COM_TEST", frequency=10, offset_mm=0.0,
            offset_full_until_mm=60.0, offset_zero_from_mm=70.0,
            max_mm=500.0,
        )
        events = queue.Queue()
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(laser100m_live.serial, "Serial", return_value=serial_device), \
                patch.object(laser100m_live, "CONFIG_ACK_TIMEOUT_S", 0.01), \
                patch.object(laser100m_live, "FIRST_FRAME_TIMEOUT_S", 0.05), \
                patch.object(laser100m_live, "FIRST_FRAME_RETRY_AFTER_S", 0.02):
            args.log_dir = directory
            laser100m_live.capture(args, events, threading.Event())

        captured = []
        while not events.empty():
            captured.append(events.get_nowait())
        finished = next(event for event in captured if event[0] == "finished")
        self.assertIn("首帧超时", finished[2])
        self.assertIn("收到 0 字节", finished[2])
        self.assertEqual(serial_device.writes.count(bytes.fromhex("80 06 03 77")), 2)


if __name__ == "__main__":
    unittest.main()
