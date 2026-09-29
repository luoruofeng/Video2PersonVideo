"""Video2PersonVideo：基于 Ultralytics YOLO 的视频人像构图裁剪工具。

用 YOLO 找到画面中的主要人物，把视频裁剪成用户指定的长宽比，
输出以人物为中心的「主播半身」构图；支持批量处理与向导式图形界面。
"""

from __future__ import annotations

__version__ = "0.2.0"
__all__ = ["__version__"]
