"""图形界面主窗口入口。

向导式实现见 :mod:`video2personvideo.gui.wizard`，这里保留 ``MainWindow``
别名，兼容既有导入路径与打包配置。
"""

from __future__ import annotations

from .wizard import WizardWindow

#: 兼容旧名字
MainWindow = WizardWindow

__all__ = ["MainWindow", "WizardWindow"]
