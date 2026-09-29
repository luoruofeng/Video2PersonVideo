"""GUI 启动入口。"""

from __future__ import annotations

import sys

from . import theme


def main(argv: list[str] | None = None) -> int:
    """启动 Qt 应用，返回退出码。"""
    # 高 DPI 策略必须在 QApplication 创建之前设置
    theme.enable_high_dpi()

    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as exc:  # pragma: no cover
        print(
            f"未安装 PySide6（{exc}），请执行：pip install -r requirements-gui.txt",
            file=sys.stderr,
        )
        return 1

    from ..utils.logger import setup_logging
    from .setup_dialog import maybe_show_setup
    from .wizard import WizardWindow

    setup_logging("INFO")
    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName(theme.APP_NAME)
    app.setOrganizationName(theme.ORG_NAME)
    theme.apply_theme(app)

    # 首次进入程序：先做一次显卡自检，决定装哪一套 PyTorch / YOLO
    # （用户点「稍后再说」也能继续用，主界面上随时可以再做一次）
    try:
        maybe_show_setup()
    except Exception:  # noqa: BLE001 - 自检失败不该挡住主界面
        import logging

        logging.getLogger("video2personvideo.gui.app").exception("首次运行自检失败")

    window = WizardWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
