"""卸载对话框（开始菜单 / 控制面板 / 安装器里的「卸载」都走这里）。

卸载有一件容易踩坑的事：从开始菜单点卸载时，**卸载器自己就跑在安装目录里的
那个 Python 上**，Windows 不允许删除正在使用的 exe。所以这里把"删目录"交给
:func:`video2personvideo.installer.engine.rollback_install` 处理 ——
删不掉的部分会安排一个后台脚本在进程退出后清理，界面上明确告诉用户"退出后自动清理"，
而不是让用户看到一个删一半的目录。
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from ...gui import theme
from .. import paths, windows
from ..engine import RollbackReport, mark_uninstalled
from .pages import _label, _paragraph, restyle
from .worker import RollbackWorker

WINDOW_PREFERRED = (760, 580)
WINDOW_MINIMUM = (560, 420)


class UninstallDialog(QDialog):
    """确认 + 执行卸载。"""

    def __init__(self, install_dir: str | Path, *, parent=None, quiet: bool = False) -> None:
        super().__init__(parent)
        self.install_dir = Path(install_dir)
        self.uninstalled = False
        self.quiet = quiet

        self._thread: QThread | None = None
        self._worker: RollbackWorker | None = None

        self.setWindowTitle(f"卸载 {paths.APP_DISPLAY_NAME}")
        theme.apply_window_size(self, preferred=WINDOW_PREFERRED, minimum=WINDOW_MINIMUM)

        layout = QVBoxLayout(self)
        layout.setSpacing(theme.SPACING)

        self.headline = _label(f"卸载 {paths.APP_DISPLAY_NAME}", object_name="headline")
        layout.addWidget(self.headline)

        self.summary = _paragraph("")
        layout.addWidget(self.summary)

        self.size_label = _label("", object_name="status", wrap=True)
        layout.addWidget(self.size_label)

        self.cache_check = QCheckBox("同时删除已下载的安装包（PyTorch 等，重装时需要重新下载）")
        self.cache_check.setChecked(True)
        layout.addWidget(self.cache_check)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # 不确定进度：删除过程很快但没有百分比
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setObjectName("logView")
        self.log_view.setVisible(False)
        layout.addWidget(self.log_view, stretch=1)

        row = QHBoxLayout()
        row.addStretch(1)
        self.cancel_button = QPushButton("取消")
        self.cancel_button.clicked.connect(self.reject)
        self.confirm_button = QPushButton("卸载")
        self.confirm_button.setObjectName("danger")
        self.confirm_button.clicked.connect(self._on_confirm)
        row.addWidget(self.cancel_button)
        row.addWidget(self.confirm_button)
        layout.addLayout(row)

        self._refresh()

    def _refresh(self) -> None:
        exists = self.install_dir.exists()
        size = windows.directory_size(self.install_dir) if exists else 0
        cache = paths.cache_dir()
        cache_size = windows.directory_size(cache) if cache.exists() else 0

        if not exists and not size:
            self.summary.setText(
                f"安装目录不存在或已被删除：{self.install_dir}\n"
                "仍然可以清理快捷方式与注册表里的卸载信息。"
            )
        else:
            self.summary.setText(
                f"将删除安装目录：{self.install_dir}\n"
                "并移除开始菜单 / 桌面快捷方式与「应用和功能」里的登记。"
            )
        self.size_label.setText(
            f"安装目录占用约 {paths.format_size(size)}　·　下载缓存约 {paths.format_size(cache_size)}"
        )
        self.cache_check.setEnabled(cache_size > 0)

    # ------------------------------------------------------------------ 执行
    @Slot()
    def _on_confirm(self) -> None:
        if self._thread is not None:
            return
        remove_cache = self.cache_check.isChecked()
        self.confirm_button.setEnabled(False)
        self.cache_check.setEnabled(False)
        self.progress.setVisible(True)
        self.log_view.setVisible(True)
        self.summary.setText("正在卸载…")

        mark_uninstalled(self.install_dir)
        worker = RollbackWorker(self.install_dir, remove_cache=remove_cache)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.logMessage.connect(self._append_log)
        worker.completed.connect(self._on_completed)
        self._worker, self._thread = worker, thread
        thread.start()

    def _append_log(self, text: str) -> None:
        for line in str(text).splitlines() or [""]:
            self.log_view.appendPlainText(line)

    @Slot(object)
    def _on_completed(self, report: RollbackReport) -> None:
        self._stop_thread()
        self.progress.setVisible(False)
        self.uninstalled = True
        self.headline.setText("卸载完成")
        restyle(self.headline, "badgeOk")
        extra = ""
        if report.scheduled:
            extra = "\n安装目录被正在运行的程序占用，已安排在本次卸载程序退出后自动清理。"
        self.summary.setText(f"{report.message}{extra}")
        self.cache_check.setVisible(False)
        self.confirm_button.setText("关闭")
        restyle(self.confirm_button, "primary")
        self.confirm_button.setEnabled(True)
        try:
            self.confirm_button.clicked.disconnect()
        except (RuntimeError, TypeError):  # pragma: no cover - 没连上也无所谓
            pass
        self.confirm_button.clicked.connect(self.accept)

    def _stop_thread(self) -> None:
        thread, self._thread = self._thread, None
        self._worker = None
        if thread is None:
            return
        thread.quit()
        thread.wait(30_000)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt 命名约定
        if self._thread is not None:
            event.ignore()
            QMessageBox.information(self, "正在卸载", "卸载正在进行中，请稍候…")
            return
        super().closeEvent(event)

    def reject(self) -> None:
        if self._thread is not None:
            return
        super().reject()


def run_uninstall_gui(install_dir: str | Path, *, quiet: bool = False) -> bool:
    """打开卸载窗口，返回是否真的卸载了。"""
    from PySide6.QtWidgets import QApplication  # noqa: PLC0415

    theme.enable_high_dpi()
    app = QApplication.instance() or QApplication([])
    app.setApplicationName(paths.APP_DISPLAY_NAME)
    app.setOrganizationName(paths.APP_DISPLAY_NAME)
    theme.apply_theme(app)
    dialog = UninstallDialog(install_dir, quiet=quiet)
    dialog.exec()
    return bool(dialog.uninstalled)


__all__ = ["UninstallDialog", "run_uninstall_gui"]
