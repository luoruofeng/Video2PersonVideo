"""安装向导主窗口：把五个页面、后台线程与"退出语义"串起来。

退出语义是这个窗口最重要的一块逻辑：安装可能要下一个多小时，
用户在任意时刻关窗口 / 点取消时，界面必须：

1. **告诉他在哪一步**（以及这一步能不能只做一半）；
2. **给出三种明确的退出方式**：继续安装 / 暂停并退出（保留断点）/ 回滚并退出（删掉已写入的文件）；
3. **真的按他选的方式执行**：暂停要等后台线程安全停下来再关窗口；
   回滚要先停下、再删干净，最后才退出。
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, Qt, Slot
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ...gui import theme
from ...utils.logger import get_logger
from .. import paths, windows
from ..engine import (
    InstallEngine,
    InstallOutcome,
    RollbackReport,
    describe_exit,
    estimate_download_bytes,
)
from ..journal import InstallJournal, InstallProbe, probe_install
from ..options import InstallOptions, default_options
from ..stages import Stage, stage_info
from .pages import (
    STEP_SUBTITLES,
    STEP_TITLES,
    ComponentsPage,
    ExitDialog,
    FinishPage,
    LocationPage,
    ProgressPage,
    TextDialog,
    WelcomePage,
)
from .worker import InstallWorker, ProbeWorker, RollbackWorker

logger = get_logger(__name__)

PAGE_WELCOME = 0
PAGE_LOCATION = 1
PAGE_COMPONENTS = 2
PAGE_PROGRESS = 3
PAGE_FINISH = 4

#: 窗口尺寸
WINDOW_PREFERRED = (1040, 780)
WINDOW_MINIMUM = (860, 620)


class InstallerWindow(QDialog):
    """安装向导。"""

    def __init__(
        self,
        options: InstallOptions | None = None,
        *,
        parent: QWidget | None = None,
        repair: bool = False,
        auto_probe: bool = True,
    ) -> None:
        super().__init__(parent)
        self.options = options or default_options()
        self._probe: InstallProbe = probe_install(self.options.install_dir)
        self._journal: InstallJournal | None = self._probe.journal
        self._engine: InstallEngine | None = None
        self._thread: QThread | None = None
        self._worker: object | None = None
        self._probe_thread: QThread | None = None
        self._closing = False
        self._rollback_after = False
        self._installed = False
        self._finished_outcome: InstallOutcome | None = None

        self.setWindowTitle(f"安装 {paths.APP_DISPLAY_NAME}")
        theme.apply_window_size(self, preferred=WINDOW_PREFERRED, minimum=WINDOW_MINIMUM)

        self._build_ui()
        self._wire_pages()

        if repair and self._journal is not None:
            self._journal.reset_all()
            self._journal.save()
        self._load_state()
        if auto_probe:
            self._start_probe()

    # ------------------------------------------------------------------ 构建
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(
            theme.SPACING_LARGE, theme.SPACING_LARGE, theme.SPACING_LARGE, theme.SPACING_LARGE
        )
        outer.setSpacing(theme.SPACING)

        header = QVBoxLayout()
        header.setSpacing(2)
        header.addWidget(_label(f"安装 {paths.APP_DISPLAY_NAME}", object_name="headline"))
        header.addWidget(
            _label(
                "自带独立 Python 环境 · 按显卡自动挑选 PyTorch · 全程断点续传",
                object_name="hint",
            )
        )
        outer.addLayout(header)

        body = QHBoxLayout()
        body.setSpacing(theme.SPACING_LARGE)

        self.step_list = QListWidget()
        self.step_list.setObjectName("stepList")
        self.step_list.setFixedWidth(220)
        self.step_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.step_list.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        for index, (title, subtitle) in enumerate(zip(STEP_TITLES, STEP_SUBTITLES, strict=False)):
            item = QListWidgetItem(f"{index + 1}. {title}\n     {subtitle}")
            self.step_list.addItem(item)
        body.addWidget(self.step_list)

        self.stack = QStackedWidget()
        self.welcome_page = WelcomePage()
        self.location_page = LocationPage()
        self.components_page = ComponentsPage()
        self.progress_page = ProgressPage()
        self.finish_page = FinishPage()
        for page in (
            self.welcome_page,
            self.location_page,
            self.components_page,
            self.progress_page,
            self.finish_page,
        ):
            self.stack.addWidget(theme.scrollable(page))
        body.addWidget(self.stack, stretch=1)
        outer.addLayout(body, stretch=1)

        footer = QHBoxLayout()
        self.footer_hint = QLabel("")
        self.footer_hint.setObjectName("hint")
        self.footer_hint.setWordWrap(True)
        footer.addWidget(self.footer_hint, stretch=1)

        self.cancel_button = QPushButton("取消")
        self.cancel_button.clicked.connect(self._on_cancel)
        self.back_button = QPushButton("上一步")
        self.back_button.clicked.connect(self._on_back)
        self.next_button = QPushButton("下一步")
        self.next_button.setObjectName("primary")
        self.next_button.clicked.connect(self._on_next)
        footer.addWidget(self.cancel_button)
        footer.addWidget(self.back_button)
        footer.addWidget(self.next_button)
        outer.addLayout(footer)

    def _wire_pages(self) -> None:
        self.welcome_page.agreementChanged.connect(self._refresh_buttons)
        self.welcome_page.resumeRequested.connect(self._on_resume)
        self.welcome_page.reinstallRequested.connect(self._on_reinstall)
        self.welcome_page.uninstallRequested.connect(self._on_uninstall)
        self.location_page.optionsChanged.connect(self._refresh_buttons)
        self.components_page.optionsChanged.connect(self._refresh_buttons)
        self.progress_page.cancelRequested.connect(self._on_cancel)
        self.finish_page.launchRequested.connect(self._on_launch)
        self.finish_page.openFolderRequested.connect(self._on_open_folder)
        self.finish_page.viewLogRequested.connect(self._on_view_log)

    # ------------------------------------------------------------------ 状态
    def _load_state(self) -> None:
        """把"已安装 / 未完成"的状态灌进界面。"""
        probe = self._probe
        self.welcome_page.set_install_state(probe)

        options = self.options
        if probe.journal is not None:
            self._journal = probe.journal
            options = InstallOptions.from_dict(probe.journal.options) or options
            if probe.journal.backend_key:
                options.backend_key = probe.journal.backend_key
            options.install_dir = Path(probe.journal.install_dir or options.install_dir)
            self.options = options
        self.location_page.load(options)
        self.components_page.load(options)

        if probe.incomplete and probe.journal is not None:
            stage = probe.journal.interrupted_stage()
            text = probe.journal.summary()
            if stage is not None:
                text = f"{text}（上次退出位置：{stage_info(stage).title}）"
            self.footer_hint.setText(text)

    def _start_probe(self) -> None:
        """后台跑一次硬件自检。"""
        if self._probe_thread is not None:
            return
        thread = QThread(self)
        worker = ProbeWorker()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.detected.connect(self.components_page.set_profile)
        worker.recommended.connect(self.components_page.set_recommendation)
        worker.failed.connect(self._on_probe_failed)
        worker.completed.connect(self._on_probe_completed)
        self._probe_thread = thread
        self._probe_worker = worker
        thread.start()

    @Slot(str)
    def _on_probe_failed(self, message: str) -> None:
        self.components_page.set_problem(
            f"硬件自检未完成（{message}）。可以继续安装，默认按 CPU 方案。"
        )

    @Slot()
    def _on_probe_completed(self) -> None:
        thread, self._probe_thread = self._probe_thread, None
        if thread is not None:
            thread.quit()
            thread.wait(5000)
        # 自检完把推荐结果落到选项里（用户改过就尊重用户的选择）
        if self.options.backend_key in {"", "cpu"}:
            recommendation = self.components_page.recommendation()
            if recommendation is not None:
                self.options.backend_key = recommendation.backend.key
                self.components_page.load(self.options)
        self._refresh_buttons()

    # ------------------------------------------------------------------ 导航
    @property
    def _page_index(self) -> int:
        return self.stack.currentIndex()

    def _show(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        for row in range(self.step_list.count()):
            item = self.step_list.item(row)
            font = item.font()
            font.setBold(row == index)
            item.setFont(font)
            item.setForeground(_step_color(row == index))
        self._refresh_buttons()

    def _refresh_buttons(self) -> None:
        index = self._page_index
        self.back_button.setVisible(index in {PAGE_LOCATION, PAGE_COMPONENTS})
        self.back_button.setEnabled(index in {PAGE_LOCATION, PAGE_COMPONENTS})
        self.cancel_button.setVisible(index != PAGE_PROGRESS)
        self.cancel_button.setText("取消")
        self.cancel_button.setEnabled(True)
        self.next_button.setVisible(index != PAGE_FINISH)

        if index == PAGE_WELCOME:
            self.next_button.setText("下一步")
            self.next_button.setEnabled(self.welcome_page.agreed())
            self.footer_hint.setText("勾选许可条款后即可继续。")
        elif index == PAGE_LOCATION:
            ok, note = self.location_page.validate()
            self.next_button.setText("下一步")
            self.next_button.setEnabled(ok)
            self.footer_hint.setText(note)
        elif index == PAGE_COMPONENTS:
            ok, note = self.location_page.validate()
            self.next_button.setText("开始安装")
            self.next_button.setEnabled(ok)
            prefix = "点「开始安装」立刻开始下载。" if ok else "还不能开始："
            self.footer_hint.setText(prefix + (note if not ok else ""))
        elif index == PAGE_PROGRESS:
            self.footer_hint.setText("安装过程中可以随时取消：断点与已下载的文件都会保留。")
        elif index == PAGE_FINISH:
            self.footer_hint.setText("")

    # ------------------------------------------------------------------ 动作
    @Slot()
    def _on_next(self) -> None:
        index = self._page_index
        if index == PAGE_WELCOME:
            self._show(PAGE_LOCATION)
        elif index == PAGE_LOCATION:
            self._show(PAGE_COMPONENTS)
        elif index == PAGE_COMPONENTS:
            self._start_install()
        elif index == PAGE_PROGRESS:
            self._show(PAGE_FINISH)

    @Slot()
    def _on_back(self) -> None:
        index = self._page_index
        if index == PAGE_LOCATION:
            self._show(PAGE_WELCOME)
        elif index == PAGE_COMPONENTS:
            self._show(PAGE_LOCATION)

    @Slot()
    def _on_cancel(self) -> None:
        if self._page_index == PAGE_PROGRESS:
            self._request_exit()
            return
        self._close_without_install()

    def _close_without_install(self) -> None:
        if self._journal is None and not self._probe.installed:
            self.reject()
            return
        answer = QMessageBox.question(
            self,
            "退出安装",
            "退出安装程序？已经下载的安装包会保留，下次可以继续。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self._closing = True
            self.reject()

    @Slot()
    def _on_resume(self) -> None:
        self._load_state()
        self._show(PAGE_LOCATION)
        self.footer_hint.setText("将从上次中断的位置继续安装，已下载的内容不会重复下载。")

    @Slot()
    def _on_reinstall(self) -> None:
        answer = QMessageBox.question(
            self,
            "重新安装",
            "重新安装会重跑全部步骤（已经下载好的安装包仍会复用，不会重复下载）。\n继续吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        if self._journal is not None:
            self._journal.reset_all()
            self._journal.save()
        self._show(PAGE_LOCATION)
        self.footer_hint.setText("将从头执行全部步骤。")

    @Slot()
    def _on_uninstall(self) -> None:
        from .uninstall_dialog import UninstallDialog  # noqa: PLC0415

        target = Path(self.options.install_dir)
        dialog = UninstallDialog(target, parent=self)
        dialog.exec()
        if dialog.uninstalled:
            self._probe = probe_install(target)
            self._journal = self._probe.journal
            self.welcome_page.set_install_state(self._probe)
            self.footer_hint.setText("卸载完成。")

    # ------------------------------------------------------------------ 安装
    def _start_install(self) -> None:
        if self._thread is not None:
            return
        self.location_page.apply(self.options)
        self.components_page.apply(self.options)
        self.options.install_dir = self.location_page.install_dir()

        # 目录变了就当一次全新的安装（状态文件在新目录里）
        if self._journal is None or not _same_path(self._journal.install_dir, self.options.install_dir):
            self._journal = InstallJournal.load_for(self.options.install_dir)

        expected = estimate_download_bytes(self.options)
        self.progress_page.reset()
        self.progress_page.append_log(f"安装目录：{self.options.install_dir}")
        self.progress_page.append_log(
            f"预计下载约 {paths.format_size(expected)}（真实体积以官方索引为准）"
        )
        self.progress_page.append_log(self.options.describe())
        self._show(PAGE_PROGRESS)
        self.next_button.setEnabled(False)
        self.next_button.setVisible(False)
        self.cancel_button.setVisible(True)
        self.cancel_button.setText("取消安装…")
        self._closing = False
        self._rollback_after = False
        self._finished_outcome = None

        engine = InstallEngine(
            self.options,
            journal=self._journal,
            profile=self.components_page.profile(),
            payload_root=None,
        )
        worker = InstallWorker(engine)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.stageChanged.connect(self._on_stage_changed)
        worker.stageProgress.connect(self.progress_page.set_stage_progress)
        worker.totalProgress.connect(self.progress_page.set_total_progress)
        worker.logMessage.connect(self.progress_page.append_log)
        worker.downloadStats.connect(self.progress_page.set_download_stats)
        worker.completed.connect(self._on_install_completed)
        self._engine, self._worker, self._thread = engine, worker, thread
        thread.start()

    @Slot(object, str, str)
    def _on_stage_changed(self, stage: Stage, status: str, message: str) -> None:
        self.progress_page.set_stage(stage, status, message)
        if status == "failed":
            self.progress_page.set_finished(failed=True)

    @Slot(object)
    def _on_install_completed(self, outcome: InstallOutcome) -> None:
        self._finished_outcome = outcome
        self._stop_thread()
        self._journal = outcome.journal

        if outcome.cancelled:
            self.progress_page.set_finished(cancelled=True)
            self.progress_page.append_log("安装已暂停。")
        elif outcome.ok:
            self._installed = True
            self.progress_page.append_log("安装完成。")

        if self._rollback_after:
            self._rollback_after = False
            self._run_rollback()
            return

        if self._closing:
            self.accept()
            return

        self._show_finish(outcome)

    def _show_finish(self, outcome: InstallOutcome) -> None:
        details = _journal_details(outcome.journal)
        log = self.progress_page.log_text()
        self._log_text = log
        if outcome.ok:
            self.finish_page.set_success(
                Path(outcome.journal.install_dir),
                details,
                can_launch=self.options.launch_after_install,
            )
        elif outcome.cancelled:
            stage = outcome.stage or Stage.PREPARE
            self.finish_page.set_cancelled(Path(outcome.journal.install_dir), stage_info(stage).title)
        else:
            self.finish_page.set_failed(outcome.message, Path(outcome.journal.install_dir))

        self._show(PAGE_FINISH)
        self.next_button.setVisible(True)
        self.next_button.setText("继续安装" if not outcome.ok else "完成")
        self.next_button.setEnabled(True)
        self.cancel_button.setVisible(False)

        if outcome.ok and self.options.launch_after_install:
            self._on_launch()

    @Slot()
    def _on_launch(self) -> None:
        if windows.launch_application(self.options.install_dir):
            self.progress_page.append_log("已启动程序。")
        else:
            QMessageBox.information(
                self,
                "无法自动启动",
                "已经安装完成，请从开始菜单或安装目录里的 Video2PersonVideo.cmd 启动。",
            )

    @Slot()
    def _on_open_folder(self) -> None:
        target = self.options.install_dir
        if not windows.open_directory(target):
            QMessageBox.information(self, "提示", f"无法打开目录：{target}")

    @Slot()
    def _on_view_log(self) -> None:
        dialog = TextDialog("安装日志", getattr(self, "_log_text", self.progress_page.log_text()), self)
        dialog.exec()

    # ------------------------------------------------------------------ 退出
    def _request_exit(self) -> None:
        """用户在安装过程中要求退出：先讲清楚，再按选择执行。"""
        engine, worker = self._engine, self._worker
        if engine is None or worker is None:
            self.reject()
            return

        stage = engine.current_stage
        text = describe_exit(
            stage,
            downloaded_bytes=engine.downloaded_bytes,
            total_bytes=engine.total_bytes,
            install_dir_exists=Path(self.options.install_dir).exists(),
        )
        dialog = ExitDialog(text, can_rollback=True, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return  # 选择"继续安装"

        choice = dialog.choice()
        engine.cancel()
        self.cancel_button.setEnabled(False)
        self.footer_hint.setText("正在安全暂停：等当前数据块写完、断点落盘…")
        self.progress_page.append_log("已请求暂停，正在等待当前操作安全结束…")

        self._closing = True
        self._rollback_after = choice == ExitDialog.ROLLBACK
        if isinstance(worker, InstallWorker):
            worker.cancel()

    def _run_rollback(self) -> None:
        self.progress_page.append_log("开始回滚：删除已写入的文件与快捷方式…")
        worker = RollbackWorker(self.options.install_dir, remove_cache=False)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.logMessage.connect(self.progress_page.append_log)
        worker.completed.connect(self._on_rollback_completed)
        self._worker, self._thread = worker, thread
        thread.start()

    @Slot(object)
    def _on_rollback_completed(self, report: RollbackReport) -> None:
        self._stop_thread()
        self.progress_page.append_log(report.message)
        QMessageBox.information(
            self,
            "已回滚",
            f"{report.message}\n\n已经下载的安装包保留在：{report.cache_path}\n"
            "下次安装可以直接复用，不会重新下载。",
        )
        self.accept()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt 命名约定
        if self._thread is not None and not self._closing:
            event.ignore()
            self._request_exit()
            return
        self._stop_thread()
        super().closeEvent(event)

    def reject(self) -> None:
        if self._thread is not None and not self._closing:
            self._request_exit()
            return
        self._stop_thread()
        super().reject()

    def _stop_thread(self) -> None:
        thread, self._thread = self._thread, None
        self._worker = None
        self._engine = None
        if thread is None:
            return
        thread.quit()
        if not thread.wait(30_000):  # pragma: no cover - 依赖真实网络
            logger.warning("后台线程未在 30 秒内退出，已挂起等待其自行结束")


def _label(text: str, *, object_name: str = "") -> QLabel:
    widget = QLabel(text)
    if object_name:
        widget.setObjectName(object_name)
    widget.setWordWrap(True)
    return widget


def _step_color(active: bool):
    from PySide6.QtGui import QColor  # noqa: PLC0415

    palette = theme.active()
    return QColor(palette.primary if active else palette.text_muted)


def _same_path(left, right) -> bool:
    try:
        return Path(left).resolve() == Path(right).resolve()
    except OSError:  # pragma: no cover
        return str(left) == str(right)


def _journal_details(journal: InstallJournal) -> str:
    """把状态文件里的阶段结果整理成"安装明细"。"""
    lines: list[str] = []
    for info in journal.records.values():
        if info.status not in {"done", "skipped"}:
            continue
        title = stage_info(info.stage).title
        text = info.message or info.detail or _status_text(info.status)
        lines.append(f"· {title}：{text}")
    return "\n".join(lines) or "—"


def _status_text(status: str) -> str:
    return {"done": "完成", "skipped": "已跳过"}.get(status, status)


def run_installer_gui(options: InstallOptions, *, repair: bool = False) -> int:
    """打开安装向导（供 ``app`` / ``cli`` 调用）。"""
    from PySide6.QtWidgets import QApplication  # noqa: PLC0415

    theme.enable_high_dpi()
    app = QApplication.instance() or QApplication([])
    app.setApplicationName(paths.APP_DISPLAY_NAME)
    app.setOrganizationName(paths.APP_DISPLAY_NAME)
    theme.apply_theme(app)
    window = InstallerWindow(options, repair=repair)
    return int(window.exec())


__all__ = ["InstallerWindow", "run_installer_gui"]
