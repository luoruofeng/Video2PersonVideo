"""统一日志配置。

命令行与 GUI 共用同一套 logger；GUI 通过 :func:`attach_handler` 把日志转发到界面。
"""

from __future__ import annotations

import contextlib
import logging
import sys

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"
DATE_FORMAT = "%H:%M:%S"

_ROOT_NAME = "video2personvideo"
_configured = False


def _make_stream_handler() -> logging.StreamHandler:
    """构造 stdout handler。"""
    _configure_stdio()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    return handler


def _configure_stdio() -> None:
    """把 stdout/stderr 统一成 UTF-8。

    Windows 上中文默认走 GBK：一旦输出被重定向（管道、写日志文件、CI 抓取），
    中文（包括 tqdm 的进度条文案）就会乱码。统一成 UTF-8 并容错替换最稳妥。
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            with contextlib.suppress(ValueError, OSError):  # pragma: no cover - 取决于运行环境
                reconfigure(encoding="utf-8", errors="replace")


def setup_logging(level: str | int = "INFO") -> logging.Logger:
    """配置根 logger（重复调用只更新级别），返回项目根 logger。"""
    global _configured

    root = logging.getLogger(_ROOT_NAME)
    if not _configured:
        root.handlers.clear()
        root.addHandler(_make_stream_handler())
        root.propagate = False
        _configured = True

    root.setLevel(_normalize_level(level))
    return root


def _normalize_level(level: str | int) -> int:
    if isinstance(level, int):
        return level
    resolved = logging.getLevelName(str(level).upper())
    return resolved if isinstance(resolved, int) else logging.INFO


def attach_handler(handler: logging.Handler) -> logging.Logger:
    """把额外的 handler（例如 GUI 的日志控件）挂到项目 logger 上。"""
    root = setup_logging()
    root.addHandler(handler)
    return root


def detach_handler(handler: logging.Handler) -> None:
    """摘掉之前用 :func:`attach_handler` 挂上的 handler。"""
    logging.getLogger(_ROOT_NAME).removeHandler(handler)


def get_logger(name: str | None = None) -> logging.Logger:
    """获取子 logger，例如 ``get_logger(__name__)``。"""
    if not name or name in {"__main__", _ROOT_NAME}:
        return logging.getLogger(_ROOT_NAME)
    if name.startswith(f"{_ROOT_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT_NAME}.{name}")
