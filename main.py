"""程序入口"""
import os
import sys

# Windows 默认控制台为 GBK，print 含 ✓ 等字符会 UnicodeEncodeError
if sys.platform == "win32":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

# 启用高DPI缩放（必须在QApplication创建之前）
QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

from ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    base_font = app.font()
    if base_font.pointSize() > 0:
        base_font.setPointSize(max(base_font.pointSize(), 10))
    else:
        base_font.setPointSizeF(max(base_font.pointSizeF(), 10.0))
    app.setFont(base_font)

    style_path = os.path.join(_ROOT, "styles", "stylesheet.qss")
    if os.path.isfile(style_path):
        with open(style_path, encoding="utf-8") as f:
            app.setStyleSheet(f.read())

    window = MainWindow()
    window.show()
    window.raise_()
    window.activateWindow()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
