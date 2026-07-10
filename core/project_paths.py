"""项目路径解析 — 统一入口，支持 PyInstaller 打包。

使用方式::

    from core.project_paths import ROOT, CONFIG_DIR, STYLES_DIR
"""
from __future__ import annotations

import sys
from pathlib import Path


def _resolve_root() -> Path:
    """自动选择：PyInstaller 用 sys._MEIPASS，开发环境用 __file__。"""
    if getattr(sys, "frozen", False):
        # PyInstaller 打包
        return Path(getattr(sys, "_MEIPASS", sys.executable)).parent
    return Path(__file__).resolve().parents[1]


ROOT: Path = _resolve_root()
CONFIG_DIR: Path = ROOT / "config"
STYLES_DIR: Path = ROOT / "styles"
DATA_DIR: Path = ROOT / "data"
LOGS_DIR: Path = ROOT / "logs"
