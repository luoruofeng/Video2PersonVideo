"""可视化安装器：把项目当成普通 Windows 软件那样"先安装、再使用"。

安装器与主程序是**两个独立程序**：

* 主程序（``v2pv`` / ``video2personvideo``）负责裁剪视频；
* 安装器（``Video2PersonVideo-Setup.exe``）负责把主程序装到用户机器上 ——
  内嵌一个独立的 Python 运行时、按显卡挑一套 PyTorch、下好 YOLO 权重、
  建好快捷方式并登记卸载信息。

分成两个程序的好处是**安装过程本身不依赖目标机器上有没有 Python**，
而且 PyTorch 那 1~3 GB 的 wheel 可以在安装阶段就断点续传地下好，
用户第一次打开软件就是"能用"的状态，而不是先等一个自检页下载半小时。

安装过程按 :class:`~video2personvideo.installer.stages.Stage` 分成若干步，
每一步的进度与结果都写进 :class:`~video2personvideo.installer.journal.InstallJournal`
（安装目录下的 ``install.json``），所以任何一步中断都能：

* **继续安装**：从记录的阶段接着走，已下好的文件只校验不重下；
* **重新安装**：把记录之后的阶段重置，从头走一遍（下载缓存仍可复用）；
* **卸载回滚**：把已经写进安装目录的内容、快捷方式与注册表项清干净。
"""

from __future__ import annotations

from .journal import InstallJournal, StageRecord
from .stages import STAGES, Stage, StageInfo, stage_info

__all__ = [
    "STAGES",
    "InstallJournal",
    "Stage",
    "StageInfo",
    "StageRecord",
    "stage_info",
]
