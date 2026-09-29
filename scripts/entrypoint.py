"""PyInstaller 打包入口脚本。

打包时不能直接把 ``src/video2personvideo/__main__.py`` 当脚本用（包内相对导入
会失效），因此提供一个位于包外的入口，由 spec 文件指定。
"""

from __future__ import annotations

import multiprocessing
import sys

from video2personvideo.cli import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
