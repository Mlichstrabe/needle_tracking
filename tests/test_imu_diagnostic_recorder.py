import csv
import json
import tempfile
import unittest
from pathlib import Path

from core.imu_diagnostic_recorder import ImuDiagnosticRecorder


class ImuDiagnosticRecorderTests(unittest.TestCase):
    def test_trigger_keeps_last_ten_seconds_and_writes_timing(self):
        now = [20_000_000_000]
        with tempfile.TemporaryDirectory() as directory:
            recorder = ImuDiagnosticRecorder(Path(directory), clock_ns=lambda: now[0])
            for seconds in (5, 9, 12, 19):
                recorder.push({"device_emit_ns": seconds * 1_000_000_000,
                               "q_w": float(seconds)})
            self.assertTrue(recorder.start({"port": "COM6"}))
            self.assertEqual(recorder.sample_count, 2)
            recorder.push({"device_emit_ns": 21_000_000_000, "q_w": 21.0})
            path = recorder.stop()
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([float(row["q_w"]) for row in rows], [12.0, 19.0, 21.0])
            self.assertEqual([float(row["t_from_trigger_ms"]) for row in rows],
                             [-8000.0, -1000.0, 1000.0])
            metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["sample_count"], 3)
            self.assertEqual(metadata["port"], "COM6")
            self.assertFalse(recorder.active)

    def test_buffer_clears_after_device_change(self):
        now = [2_000_000_000]
        with tempfile.TemporaryDirectory() as directory:
            recorder = ImuDiagnosticRecorder(Path(directory), clock_ns=lambda: now[0])
            recorder.push({"device_emit_ns": 1_000_000_000, "q_w": 1.0})
            recorder.clear_buffer()
            recorder.start()
            self.assertEqual(recorder.sample_count, 0)


if __name__ == "__main__":
    unittest.main()
