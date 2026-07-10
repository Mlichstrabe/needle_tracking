"""统一日志配置。

在 main.py 入口调用 ``setup_logging()``，后续所有模块通过
``logging.getLogger(__name__)`` 获取 logger，无需再手动配置。
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

LOG_FMT = "%(asctime)s [%(levelname)-5s] %(name)s %(message)s"
LOG_DATE_FMT = "%H:%M:%S"


def setup_logging(
    level: int = logging.DEBUG,
    log_dir: str | Path | None = None,
    log_file: str = "app.log",
) -> None:
    """初始化根 logger：控制台 + 可选文件。

    Parameters
    ----------
    level : int
        全局最低日志级别。默认 DEBUG。
    log_dir : str | Path | None
        日志文件目录；为 None 则仅输出到控制台。
    log_file : str
        日志文件名。
    """
    root = logging.getLogger()
    root.setLevel(level)

    # 避免重复添加 handler
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(level)
        console.setFormatter(logging.Formatter(LOG_FMT, datefmt=LOG_DATE_FMT))
        root.addHandler(console)

        if log_dir is not None:
            log_path = Path(log_dir) / log_file
            log_path.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(str(log_path), encoding="utf-8")
            fh.setLevel(level)
            fh.setFormatter(logging.Formatter(LOG_FMT, datefmt=LOG_DATE_FMT))
            root.addHandler(fh)
            logging.info("日志文件: %s", log_path)

    # 降低第三方库的日志噪音
    for noisy in ("PIL", "trimesh", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
