"""PyInstaller 打包入口：安装器（``Video2PersonVideo-Setup.exe``）。

与 ``entrypoint.py``（主程序入口）一样，打包时不能直接把包内的 ``__main__.py``
当脚本用（包内相对导入会失效），所以提供一个位于包外的入口。
"""

from __future__ import annotations

import multiprocessing
import sys

from video2personvideo.installer.app import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
