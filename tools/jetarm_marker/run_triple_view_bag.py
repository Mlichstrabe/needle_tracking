#!/usr/bin/env python
"""Launcher: IR + depth + 3D triple view from bag (hardcoded args)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.argv = ["live_triple_view.py", "--source", "bag", "data/jetarm_marker/bags/marker_move_rgb_ir_depth_01"]
from tools.jetarm_marker.live_triple_view import main
raise SystemExit(main())
