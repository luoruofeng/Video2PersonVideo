"""``python -m video2personvideo.installer``：安装 / 卸载入口。

已安装的程序就用这个入口卸载（开始菜单与「应用和功能」里的卸载命令都是它），
所以卸载器不依赖最初那个 ``Setup.exe`` 还在不在。
"""

from __future__ import annotations

import multiprocessing
import sys

from .app import main


def _run() -> int:
    multiprocessing.freeze_support()
    return main()


if __name__ == "__main__":
    sys.exit(_run())
