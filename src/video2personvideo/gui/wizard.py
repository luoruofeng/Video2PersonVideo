"""向导式主窗口：选择输入 → 选择比例 → 执行处理 → 结果汇总。"""

from __future__ import annotations

from dataclasses import replace

from PySide6.QtCore import QSettings, QThread, Slot
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..config import AppConfig, load_config
from ..core.batch import BatchProgress, BatchResult, Task, check_writable
from ..core.ratio import AspectRatio, RatioError, parse_ratio
from ..utils.logger import get_logger, setup_logging
from . import theme
from .controller import BatchWorker
from .pages import InputPage, RatioPage, ResultPage, RunPage
from .preview_dialog import PreviewDialog

logger = get_logger(__name__)

STEP_INPUT = 0
STEP_RATIO = 1
STEP_RUN = 2
STEP_RESULT = 3


class WizardWindow(QMainWindow):
    """四步向导主窗口。"""

    def __init__(self, cfg: AppConfig | None = None) -> None:
        super().__init__()
        setup_logging("INFO")

        self._base_cfg = cfg or load_config()
        self._thread: QThread | None = None
        self._worker: BatchWorker | None = None

        self.setWindowTitle("Video2PersonVideo · 视频人像构图")
        # 按屏幕可用区域决定窗口大小：Windows 150% 缩放下桌面的逻辑分辨率会小很多
        # （1920×1080 → 1280×720），写死 1000×760 会撑出屏幕、把底部按钮顶掉。
        theme.apply_window_size(self)

        self._build_ui()
        self._restore_settings()
        self._update_nav()

    # ------------------------------------------------------------ 界面构建
    def _build_ui(self) -> None:
        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(theme.SPACING_LARGE, theme.SPACING_LARGE,
                                  theme.SPACING_LARGE, theme.SPACING_LARGE)
        layout.setSpacing(theme.SPACING)

        layout.addLayout(self._build_step_bar())

        self.input_page = InputPage()
        self.ratio_page = RatioPage()
        self.run_page = RunPage()
        self.result_page = ResultPage()
        try:
            self.ratio_page.set_ratio(self._base_cfg.resolve_ratio())
        except RatioError as exc:
            logger.warning("配置中的比例非法（%s），回退到默认比例", exc)
        self.ratio_page.set_size_guard(self._base_cfg.size_guard)
        self.ratio_page.set_speaker_tracking(self._base_cfg.speaker_tracking)
        self.ratio_page.set_multi_person(self._base_cfg.multi_person)
        self.ratio_page.set_multi_person_stable(self._base_cfg.multi_person_stable_camera)
        self.ratio_page.set_no_person_mode(self._base_cfg.no_person_mode)

        self.stack = QStackedWidget()
        for page in (self.input_page, self.ratio_page, self.run_page, self.result_page):
            # 每页各自套一层滚动区（表头 / 底部按钮固定在外层）：
            # 窗口再小、系统缩放比再高，也只会出现滚动条，不会把控件压扁、把文字裁掉。
            self.stack.addWidget(theme.scrollable(page))
        layout.addWidget(self.stack, stretch=1)

        layout.addLayout(self._build_nav_row())
        self.setCentralWidget(central)

        self.input_page.selectionChanged.connect(self._update_nav)
        self.ratio_page.previewRequested.connect(self._show_preview)
        self.run_page.cancelRequested.connect(self._cancel)
        self.result_page.restartRequested.connect(self._restart)

        # 兼容旧接口（脚本 / 测试常用）
        self.start_button = self._next_button
        self.stop_button = self.run_page.cancel_button
        self.progress_bar = self.run_page.overall_bar
        self.status_label = self.run_page.frame_label
        self.log_view = self.run_page.log_view
        self.source_edit = self.input_page.path_edit

    def _build_step_bar(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self._chips: list[QLabel] = []
        for title in theme.STEP_TITLES:
            chip = QLabel(title)
            chip.setObjectName("stepChip")
            self._chips.append(chip)
            row.addWidget(chip)
        row.addStretch(1)
        return row

    def _build_nav_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self._back_button = QPushButton("上一步")
        self._back_button.clicked.connect(self._go_back)
        self._setup_button = QPushButton("环境自检")
        self._setup_button.setToolTip(
            "重新检测显卡型号，并按结果下载 / 安装对应的 PyTorch 与 YOLO 依赖\n"
            "（首次进入程序时也会自动弹出）"
        )
        self._setup_button.clicked.connect(self._open_setup)
        self._next_button = QPushButton("下一步")
        self._next_button.setObjectName("primary")
        self._next_button.clicked.connect(self._go_next)
        row.addWidget(self._back_button)
        row.addWidget(self._setup_button)
        row.addStretch(1)
        row.addWidget(self._next_button)
        return row

    # --------------------------------------------------------------- 导航
    def _update_nav(self) -> None:
        index = self.stack.currentIndex()
        for position, chip in enumerate(self._chips):
            chip.setObjectName("stepChipActive" if position == index else "stepChip")
            chip.style().unpolish(chip)
            chip.style().polish(chip)

        running = self._thread is not None
        # 处理中不让改环境，避免"边跑边装依赖"这种说不清的状态
        self._setup_button.setVisible(index != STEP_RUN)
        self._setup_button.setEnabled(not running)
        if index == STEP_INPUT:
            self._back_button.setVisible(False)
            self._next_button.setVisible(True)
            self._next_button.setText("下一步")
            self._next_button.setEnabled(self.input_page.is_valid())
        elif index == STEP_RATIO:
            self._back_button.setVisible(True)
            self._back_button.setText("上一步")
            self._back_button.setEnabled(not running)
            self._next_button.setVisible(True)
            self._next_button.setText("开始处理")
            self._next_button.setEnabled(not running)
        elif index == STEP_RUN:
            self._back_button.setVisible(False)
            self._next_button.setVisible(False)
        else:
            self._back_button.setVisible(True)
            self._back_button.setText("重新开始")
            self._back_button.setEnabled(True)
            self._next_button.setVisible(False)

    def _go_back(self) -> None:
        index = self.stack.currentIndex()
        if index == STEP_RATIO:
            self.stack.setCurrentIndex(STEP_INPUT)
        elif index == STEP_RESULT:
            self._restart()
        self._update_nav()

    def _go_next(self) -> None:
        index = self.stack.currentIndex()
        if index == STEP_INPUT:
            if not self.input_page.is_valid():
                QMessageBox.warning(self, "还没有输入", "请先选择要处理的视频或文件夹。")
                return
            self._save_settings()
            self.stack.setCurrentIndex(STEP_RATIO)
            self._update_nav()
        elif index == STEP_RATIO:
            self._start()

    def _restart(self) -> None:
        self.run_page.reset()
        self.stack.setCurrentIndex(STEP_INPUT)
        self._update_nav()

    # --------------------------------------------------------------- 配置
    def _collect_config(self) -> AppConfig:
        """把界面上当前的选择整理成一份 :class:`AppConfig`。"""
        if not self.input_page.is_valid():
            raise ValueError("请先选择要处理的视频或文件夹")

        ratio = self.ratio_page.current_ratio()
        tuning = self.ratio_page.tuning()
        cfg = replace(
            self._base_cfg,
            source=self.input_page.path,
            output=None,
            aspect_ratio=ratio.name,
            target_width=ratio.target_width,
            target_height=ratio.target_height,
            batch_recursive=self.input_page.recursive(),
            size_guard=self.ratio_page.size_guard(),
            speaker_tracking=self.ratio_page.speaker_tracking(),
            multi_person=self.ratio_page.multi_person(),
            # 「分屏窗口稳定跟随」一个开关同时管两件事：镜头按秒封顶 + 窗口固定跟人
            multi_person_stable_camera=self.ratio_page.multi_person_stable(),
            multi_person_seat_lock=self.ratio_page.multi_person_stable(),
            no_person_mode=self.ratio_page.no_person_mode(),
            show=False,
            **tuning,
        )
        cfg.validate()
        return cfg

    # --------------------------------------------------------------- 处理
    def _start(self) -> None:
        if self._thread is not None:
            return

        sources = self.input_page.sources
        try:
            cfg = self._collect_config()
        except (ValueError, RatioError) as exc:
            QMessageBox.warning(self, "参数有误", str(exc))
            return

        try:
            for directory in sorted({cfg.output_path_for(item).parent for item in sources}, key=str):
                check_writable(directory)
        except RuntimeError as exc:
            QMessageBox.critical(self, "输出目录不可用", str(exc))
            return

        self.run_page.reset()
        self.stack.setCurrentIndex(STEP_RUN)
        self._update_nav()
        self.run_page.append_log(f"开始处理 {len(sources)} 个视频…")

        self._thread = QThread(self)
        self._worker = BatchWorker(cfg, sources)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.taskUpdated.connect(self._on_task_updated)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.log.connect(self.run_page.append_log)
        self._thread.start()

    def _cancel(self) -> None:
        if self._worker is None:
            return
        self._worker.request_stop()
        self.run_page.append_log("已请求取消，正在等待当前帧处理完成…")
        self.run_page.cancel_button.setEnabled(False)

    @Slot(object)
    def _on_progress(self, progress: BatchProgress) -> None:
        self.run_page.set_progress(progress)

    @Slot(object)
    def _on_task_updated(self, task: Task) -> None:
        logger.debug("任务状态更新：%s -> %s", task.source, task.status)

    @Slot(object)
    def _on_finished(self, result: BatchResult) -> None:
        self._teardown()
        self.run_page.set_finished_message(
            f"处理完成：成功 {len(result.succeeded)} / 失败 {len(result.failed)} / "
            f"跳过 {len(result.skipped)}"
        )
        self.result_page.set_result(result)
        self.stack.setCurrentIndex(STEP_RESULT)
        self._update_nav()

        if result.failed:
            QMessageBox.warning(
                self,
                "部分任务失败",
                f"共 {len(result.failed)} 个视频处理失败，详见结果列表。",
            )

    @Slot(str)
    def _on_failed(self, message: str) -> None:
        self._teardown()
        self.run_page.set_finished_message("处理失败")
        QMessageBox.critical(self, "处理失败", message)
        self.stack.setCurrentIndex(STEP_RATIO)
        self._update_nav()

    def _teardown(self) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(5000)
            self._thread = None
        self._worker = None

    # --------------------------------------------------------------- 预览
    def _show_preview(self) -> None:
        sources = self.input_page.sources
        if not sources:
            QMessageBox.information(self, "还没有输入", "请先在第一步选择视频文件或文件夹。")
            return
        try:
            cfg = self._collect_config()
        except (ValueError, RatioError) as exc:
            QMessageBox.warning(self, "参数有误", str(exc))
            return

        dialog = PreviewDialog(cfg, sources[0], self)
        if dialog.exec():
            self.ratio_page.set_tuning(dialog.tuning())
            self.ratio_page.set_multi_person(dialog.multi_person())
            self.ratio_page.set_no_person_mode(dialog.no_person_mode())
            QMessageBox.information(self, "已应用", "预览参数已同步到设置，可以开始处理了。")

    # ----------------------------------------------------------- 环境自检
    def _open_setup(self) -> None:
        """重新做一次显卡自检 / 依赖下载（首次运行时也会自动弹出）。"""
        from .setup_dialog import SetupDialog, mark_setup_completed

        dialog = SetupDialog(self, first_run=False)
        if dialog.exec():
            mark_setup_completed()

    # --------------------------------------------------------------- 记忆
    def _settings(self) -> QSettings:
        return QSettings(theme.ORG_NAME, theme.APP_NAME)

    def _restore_settings(self) -> None:
        try:
            settings = self._settings()
            last_path = settings.value("input/last_path", "", type=str)
            ratio_name = settings.value("ratio/name", "", type=str)
            width = settings.value("ratio/width", 0, type=int)
            height = settings.value("ratio/height", 0, type=int)
            # 用 contains 判断"有没有记忆过"：直接把默认值写成 None 会被 Qt 转成 False
            has_multi_person = bool(settings.contains("multi/person"))
            multi_person = bool(settings.value("multi/person", True, type=bool))
            has_multi_stable = bool(settings.contains("multi/stable"))
            multi_stable = bool(settings.value("multi/stable", True, type=bool))
            has_no_person = bool(settings.contains("noPerson/mode"))
            no_person_mode = str(settings.value("noPerson/mode", "", type=str))
        except Exception as exc:  # noqa: BLE001 - 没有配置后端时忽略
            logger.debug("读取界面记忆失败：%s", exc)
            return

        if last_path:
            try:
                self.input_page.set_path(last_path)
            except Exception as exc:  # noqa: BLE001
                logger.debug("恢复上次路径失败：%s", exc)

        # 没记忆过就沿用配置默认值（默认开启），只覆盖用户真正选过的
        if has_multi_person:
            self.ratio_page.set_multi_person(multi_person)
        if has_multi_stable:
            self.ratio_page.set_multi_person_stable(multi_stable)
        if has_no_person and no_person_mode:
            self.ratio_page.set_no_person_mode(no_person_mode)

        if ratio_name:
            try:
                ratio = parse_ratio(ratio_name, target=(width, height) if width and height else None)
                self.ratio_page.set_ratio(ratio)
            except RatioError as exc:
                logger.debug("恢复上次比例失败：%s", exc)

    def _save_settings(self) -> None:
        try:
            settings = self._settings()
            ratio: AspectRatio = self.ratio_page.current_ratio()
            path = self.input_page.path
            settings.setValue("input/last_path", str(path) if path else "")
            settings.setValue("ratio/name", ratio.name)
            settings.setValue("ratio/width", ratio.target_width)
            settings.setValue("ratio/height", ratio.target_height)
            settings.setValue("multi/person", self.ratio_page.multi_person())
            settings.setValue("multi/stable", self.ratio_page.multi_person_stable())
            settings.setValue("noPerson/mode", self.ratio_page.no_person_mode())
            settings.sync()
        except Exception as exc:  # noqa: BLE001
            logger.debug("保存界面记忆失败：%s", exc)

    # --------------------------------------------------------------- 收尾
    def closeEvent(self, event) -> None:  # noqa: N802 - Qt 命名约定
        if self._worker is not None:
            self._worker.request_stop()
        self._teardown()
        self._save_settings()
        super().closeEvent(event)


__all__ = ["WizardWindow"]
