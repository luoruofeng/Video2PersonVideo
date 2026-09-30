"""安装器图形入口（``Video2PersonVideo-Setup.exe``）。

打包成 Windows 窗口程序后**没有控制台**（``sys.stdout`` 是 ``None``），
所以这里额外挂一个文件日志：用户遇到问题把
``%LOCALAPPDATA%\\Video2PersonVideo\\installer.log`` 发过来就能定位。
"""

from __future__ import annotations

import logging
import multiprocessing
import sys
from pathlib import Path

from ..utils.app_paths import register_bundled_tools
from ..utils.logger import DATE_FORMAT, LOG_FORMAT, get_logger, setup_logging
from . import paths


def log_file() -> Path:
    """安装器日志路径（也是"查看日志"按钮指向的文件）。"""
    return paths.log_path()


def configure_logging(level: str = "INFO") -> Path | None:
    """配置日志：有控制台就照常输出，没有就写文件（并总是写文件）。"""
    setup_logging(level)
    logger = get_logger()

    target: Path | None = None
    try:
        target = log_file()
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        target = None

    # 没有控制台（窗口程序）时，把 stdout handler 摘掉，避免 emit 时抛异常
    if sys.stdout is None or sys.stderr is None:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)

    if target is not None:
        attached = any(
            isinstance(handler, logging.FileHandler)
            and Path(getattr(handler, "baseFilename", "")) == target
            for handler in logger.handlers
        )
        if not attached:
            file_handler = logging.FileHandler(target, encoding="utf-8")
            file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
            logger.addHandler(file_handler)
    logger.info("安装器启动：%s", target or "（无日志文件）")
    return target


def main(argv: list[str] | None = None) -> int:
    """入口：解析参数 → 决定走界面还是命令行。"""
    register_bundled_tools()
    configure_logging()
    from .cli import main as cli_main  # noqa: PLC0415 - 日志配好之后再导入

    return cli_main(argv)


if __name__ == "__main__":  # pragma: no cover
    multiprocessing.freeze_support()
    raise SystemExit(main())
