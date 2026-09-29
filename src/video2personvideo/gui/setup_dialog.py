"""首次运行的环境自检对话框（也用于主界面里的「环境自检」按钮）。

对话框只做两件事：把 :class:`~video2personvideo.gui.pages.setup_page.SetupPage`
摆进窗口，以及管理后台 ``QThread``（页面本身不碰线程）。
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings, QThread, QUrl, Slot
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from ..utils.env_check import environment_ready
from ..utils.logger import get_logger
from . import theme
from .controller import SetupRequest, SetupResult, SetupWorker
from .pages.setup_page import SetupPage

logger = get_logger(__name__)

#: 首次运行自检是否已完成（``QSettings`` 键）
SETTINGS_KEY = "setup/completed"

#: 需要下载大文件时的提示（关闭对话框前确认）
CLOSE_WHILE_RUNNING = (
    "下载还在进行中。\n\n"
    "关闭会暂停下载，但已经下好的部分会保留下来——下次打开程序点「继续下载」"
    "会从断点接着下，不会白下。\n\n确定要关闭吗？"
)


def setup_settings() -> QSettings:
    return QSettings(theme.ORG_NAME, theme.APP_NAME)


def setup_completed(settings: QSettings | None = None) -> bool:
    """是否已经做过首次运行的环境自检。"""
    try:
        store = settings or setup_settings()
        return bool(store.value(SETTINGS_KEY, False, type=bool))
    except Exception as exc:  # noqa: BLE001 - 没有配置后端时按"没做过"处理
        logger.debug("读取自检状态失败：%s", exc)
        return False


def mark_setup_completed(settings: QSettings | None = None) -> None:
    """记下"已经自检过"，下次启动不再自动弹出。"""
    try:
        store = settings or setup_settings()
        store.setValue(SETTINGS_KEY, True)
        store.sync()
    except Exception as exc:  # noqa: BLE001
        logger.debug("写入自检状态失败：%s", exc)


class SetupDialog(QDialog):
    """环境自检 / 依赖下载对话框。"""

    def __init__(self, parent=None, *, auto_detect: bool = True, first_run: bool = False) -> None:
        super().__init__(parent)
        self._thread: QThread | None = None
        self._worker: SetupWorker | None = None
        self._pending_detect = bool(auto_detect)
        #: 用户换过方案后还没解析完就点了「开始下载」——解析完自动接着下
        self._pending_request: SetupRequest | None = None
        #: 极端情况下超时未退出的线程：留个引用，避免 QThread 被提前析构
        self._orphan_threads: list[QThread] = []

        self.setWindowTitle(
            "首次运行 · 环境自检与依赖下载" if first_run else "环境自检与依赖下载"
        )
        # 窗口大小按屏幕可用区域收缩（高缩放比下逻辑分辨率会小很多）
        theme.apply_window_size(self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            theme.SPACING_LARGE, theme.SPACING_LARGE, theme.SPACING_LARGE, theme.SPACING_LARGE
        )
        layout.setSpacing(theme.SPACING)

        self.page = SetupPage()
        # 自检页是三步向导，每步内容不长；放进滚动区兜底，窗口再小也只是出现滚动条
        layout.addWidget(theme.scrollable(self.page), stretch=1)

        footer = QHBoxLayout()
        self.footer_hint = QLabel(
            "点「下一步」一路走完：检查电脑 → 确认方案 → 开始下载。"
            "装完需要重启程序才会用上 GPU。"
        )
        self.footer_hint.setObjectName("hint")
        footer.addWidget(self.footer_hint, stretch=1)
        self.skip_button = QPushButton("稍后再说")
        self.skip_button.clicked.connect(self.reject)
        self.done_button = QPushButton("完成")
        self.done_button.setObjectName("primary")
        self.done_button.setEnabled(False)
        self.done_button.clicked.connect(self.accept)
        footer.addWidget(self.skip_button)
        footer.addWidget(self.done_button)
        layout.addLayout(footer)

        self.page.startRequested.connect(self._on_start)
        self.page.pauseRequested.connect(self._on_pause)
        self.page.refreshRequested.connect(self._on_refresh)
        self.page.backendChanged.connect(self._on_backend_changed)
        self.page.directoryRequested.connect(self._open_directory)
        self.page.directoryChanged.connect(self._on_directory_changed)

    # ------------------------------------------------------------- 生命周期
    def showEvent(self, event) -> None:  # noqa: N802 - Qt 命名约定
        super().showEvent(event)
        if self._pending_detect:
            self._pending_detect = False
            self._start_detect(force=False)

    def accept(self) -> None:
        self._teardown()
        super().accept()

    def reject(self) -> None:
        if self._thread is not None and not self._confirm_close():
            return
        self._stop_worker()
        self._teardown()
        super().reject()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt 命名约定
        if self._thread is not None and not self._confirm_close():
            event.ignore()
            return
        self._stop_worker()
        self._teardown()
        super().closeEvent(event)

    # ----------------------------------------------------------------- 任务
    def _start_detect(self, *, force: bool) -> None:
        self._spawn(mode="detect", force=force)

    def _spawn(
        self,
        *,
        mode: str,
        force: bool = False,
        backend_key: str | None = None,
        request: SetupRequest | None = None,
        directory=None,
    ) -> None:
        if self._thread is not None:
            return

        thread = QThread(self)
        worker = SetupWorker()
        worker.moveToThread(thread)
        worker.force = force
        worker.backend_key = backend_key or ""
        worker.request = request
        worker.directory = Path(directory) if directory else self.page.directory

        worker.detected.connect(self.page.set_profile)
        worker.recommended.connect(self.page.set_recommendation)
        worker.planReady.connect(self.page.set_plan)
        worker.cacheState.connect(self.page.set_cache_state)
        worker.stage.connect(self.page.set_stage)
        worker.progress.connect(self.page.set_progress)
        worker.totalProgress.connect(self.page.set_total_progress)
        worker.log.connect(self.page.append_log)
        worker.finished.connect(self._on_finished)
        worker.failed.connect(self._on_failed)
        worker.taskFinished.connect(self._on_task_finished)

        starters = {
            "detect": worker.start_detect,
            "plan": worker.start_load_plan,
            "run": worker.start_run,
        }
        thread.started.connect(starters[mode])

        self._thread, self._worker = thread, worker
        if mode == "run":
            self.page.set_running(True)
            self.page.reset_progress()
        thread.start()

    @Slot(object)
    def _on_start(self, request: SetupRequest) -> None:
        if self._thread is not None:
            return
        plan = self.page.plan
        if plan is None or plan.backend.key != request.backend_key:
            # 还没解析过这个方案（或用户刚换过）：先解析地址，解析完自动接着下载
            self._pending_request = request
            self._spawn(mode="plan", backend_key=request.backend_key)
            return
        self._pending_request = None
        self._spawn(mode="run", request=request)

    @Slot()
    def _on_pause(self) -> None:
        self._stop_worker()
        self.page.append_log("暂停请求已发出，等待当前数据块写完…")

    @Slot()
    def _on_refresh(self) -> None:
        if self._thread is not None:
            return
        self.page.reset_progress()
        self._start_detect(force=True)

    @Slot(str)
    def _on_backend_changed(self, backend_key: str) -> None:
        if self._thread is not None:
            return
        self._spawn(mode="plan", backend_key=backend_key)

    @Slot(object)
    def _on_directory_changed(self, directory) -> None:
        """换了下载目录：重新解析地址（plan.directory 决定文件落到哪）。"""
        if self._thread is not None:
            return
        self._spawn(mode="plan", backend_key=self.page.backend_key(), directory=directory)

    @Slot(object)
    def _on_finished(self, result: SetupResult) -> None:
        self.page.set_finished(result)
        self.done_button.setEnabled(True)
        if result.installed:
            self.done_button.setText("完成（重启后生效）")
            self.footer_hint.setText("安装完成：重启程序后即可使用 GPU 加速。")
        elif result.ok and result.skipped_install:
            self.footer_hint.setText(
                "安装包已下载完成：" + (result.install_message or "可用 pip 手动安装。")
            )
        elif result.failed:
            self.footer_hint.setText("有文件下载失败，可点「重试下载」（已下好的部分不会重复下载）。")

    @Slot(str)
    def _on_failed(self, message: str) -> None:
        self._pending_request = None
        self.page.set_running(False)
        self.page.set_stage("自检 / 下载失败")
        self.page.append_log(f"错误：{message}")
        self.done_button.setEnabled(True)
        QMessageBox.warning(
            self,
            "环境自检未完成",
            f"{message}\n\n可以点「重新自检」再试一次，或点「稍后再说」先跳过。",
        )

    @Slot(str)
    def _on_task_finished(self, mode: str) -> None:
        """每个后台任务结束后收线程；解析完地址后接着执行待跑的下载。"""
        self._teardown()
        if mode == "detect":
            self.done_button.setEnabled(True)
            return
        pending = self._pending_request
        if mode == "plan" and pending is not None:
            self._pending_request = None
            self._spawn(mode="run", request=pending)

    # ------------------------------------------------------------- 内部工具
    def _stop_worker(self) -> None:
        if self._worker is not None:
            self._worker.request_stop()

    def _teardown(self) -> None:
        thread, self._thread = self._thread, None
        self._worker = None
        if thread is None:
            return
        thread.quit()
        # 下载循环每读一个数据块就会检查停止标志，正常几百毫秒内就退出了；
        # 极端情况（网络卡死）最多等一个下载超时。
        if not thread.wait(30_000):  # pragma: no cover - 依赖真实网络
            logger.warning("下载线程未在 30 秒内退出，已挂起等待其自行结束")
            self._orphan_threads.append(thread)

    def _confirm_close(self) -> bool:
        answer = QMessageBox.question(
            self,
            "正在下载",
            CLOSE_WHILE_RUNNING,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _open_directory(self, directory) -> None:
        target = str(directory)
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(target)):
            QMessageBox.information(self, "提示", f"无法打开目录：{target}")


def maybe_show_setup(parent=None, *, force: bool = False) -> bool:
    """首次运行时弹出环境自检；做完 / 用户手动完成后记录状态。

    返回 ``True`` 表示用户完成了自检（可以启动主界面继续用）。

    两种情况直接跳过下载页：

    - 已经自检过（``QSettings`` 里记着）；
    - 本机环境本来就齐全（PyTorch 装好了、默认 YOLO 权重也在本地）。

    后一种**不写标记**：以后哪天把依赖卸了，下一次启动还会重新弹出自检页。
    主界面上的「环境自检」按钮走 ``force=True``，任何时候都能强制打开。
    """
    if not force and setup_completed():
        return True
    if not force and environment_ready():
        logger.info("本机已具备 PyTorch 与 YOLO 权重，跳过首次运行自检页")
        return True

    dialog = SetupDialog(parent, first_run=not force)
    accepted = bool(dialog.exec())
    if accepted:
        mark_setup_completed()
    return accepted


__all__ = [
    "SETTINGS_KEY",
    "SetupDialog",
    "mark_setup_completed",
    "maybe_show_setup",
    "setup_completed",
    "setup_settings",
]
