"""安装器的图形界面（PySide6）。

界面与逻辑严格分开：所有耗时操作都在 :class:`~QThread` 里跑
（硬件自检、断点续传下载、pip 安装、回滚删除），
主线程只负责刷新控件，所以"下一个几 GB 的包"的时候窗口依然能拖动、能点取消。
"""

from __future__ import annotations

__all__: list[str] = []
