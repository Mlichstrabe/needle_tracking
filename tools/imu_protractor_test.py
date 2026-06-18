"""
已弃用量角器手输 — 请运行:

    python tools/imu_accuracy_test.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "imu_accuracy_test.py"
    spec = importlib.util.spec_from_file_location("imu_accuracy_test", target)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    mod.main()
