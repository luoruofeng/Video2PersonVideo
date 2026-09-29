"""``python -m video2personvideo`` 入口。"""

from __future__ import annotations

import multiprocessing

from .cli import main

if __name__ == "__main__":
    # 打包成 exe 后避免多进程重复启动（ultralytics 内部可能用到多进程）
    multiprocessing.freeze_support()
    raise SystemExit(main())
