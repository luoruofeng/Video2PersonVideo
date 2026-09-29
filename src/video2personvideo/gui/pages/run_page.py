"""向导步骤 3：执行与进度（两级进度条 + 状态信息 + 实时日志 + 取消）。"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...core.batch import BatchProgress
from ...core.framing import mode_label
from .. import theme


def _format_seconds(seconds: float) -> str:
    if seconds <= 0:
        return "—"
    minutes, remain = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}小时{minutes}分"
    if minutes:
        return f"{minutes}分{remain}秒"
    return f"{remain}秒"


class RunPage(QWidget):
    """执行页。"""

    cancelRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.SPACING)

        progress_group = QGroupBox("进度")
        box = QVBoxLayout(progress_group)
        box.setSpacing(theme.SPACING_SMALL)

        box.addWidget(QLabel("整体进度"))
        self.overall_bar = QProgressBar()
        self.overall_bar.setRange(0, 100)
        self.overall_bar.setValue(0)
        box.addWidget(self.overall_bar)

        box.addWidget(QLabel("当前文件"))
        self.file_bar = QProgressBar()
        self.file_bar.setRange(0, 100)
        self.file_bar.setValue(0)
        box.addWidget(self.file_bar)

        info = QGridLayout()
        info.setHorizontalSpacing(theme.SPACING_LARGE)
        info.setVerticalSpacing(theme.SPACING_SMALL)
        self.file_label = QLabel("—")
        self.frame_label = QLabel("—")
        self.mode_label = QLabel("—")
        self.speed_label = QLabel("—")
        for column, (title, widget) in enumerate(
            (
                ("当前文件", self.file_label),
                ("帧进度", self.frame_label),
                ("裁剪模式", self.mode_label),
                ("速度 / 剩余", self.speed_label),
            )
        ):
            caption = QLabel(title)
            caption.setObjectName("hint")
            info.addWidget(caption, 0, column)
            info.addWidget(widget, 1, column)
        box.addLayout(info)

        action_row = QHBoxLayout()
        action_row.addStretch(1)
        self.cancel_button = QPushButton("取消处理")
        self.cancel_button.setObjectName("danger")
        self.cancel_button.clicked.connect(self.cancelRequested.emit)
        action_row.addWidget(self.cancel_button)
        box.addLayout(action_row)

        layout.addWidget(progress_group)

        log_group = QGroupBox("运行日志")
        log_box = QVBoxLayout(log_group)
        log_box.setSpacing(theme.SPACING_SMALL)

        log_row = QHBoxLayout()
        self.log_check = QCheckBox("显示日志")
        self.log_check.setChecked(True)
        self.log_check.toggled.connect(self._toggle_log)
        log_row.addWidget(self.log_check)
        log_row.addStretch(1)
        self.copy_button = QPushButton("复制全部")
        self.copy_button.clicked.connect(self._copy_log)
        log_row.addWidget(self.copy_button)
        log_box.addLayout(log_row)

        self.log_view = QPlainTextEdit()
        self.log_view.setObjectName("logView")
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        self.log_view.setMinimumHeight(200)
        log_box.addWidget(self.log_view)
        layout.addWidget(log_group, stretch=1)

    # ----------------------------------------------------------------- 接口
    def reset(self) -> None:
        self.overall_bar.setValue(0)
        self.file_bar.setValue(0)
        self.file_label.setText("—")
        self.frame_label.setText("—")
        self.mode_label.setText("—")
        self.speed_label.setText("—")
        self.log_view.clear()

    def set_progress(self, progress: BatchProgress) -> None:
        self.overall_bar.setValue(int(round(progress.overall * 100)))
        self.file_bar.setValue(int(round(progress.file_fraction * 100)))
        self.file_label.setText(
            f"{progress.source.name if progress.source else '—'} "
            f"（{progress.file_index}/{progress.file_count}）"
        )
        self.frame_label.setText(progress.message or "—")
        self.mode_label.setText(mode_label(progress.mode) if progress.mode else "—")
        speed = f"{progress.fps:.1f} FPS" if progress.fps else "—"
        self.speed_label.setText(f"{speed} · 剩余 {_format_seconds(progress.eta)}")

    def append_log(self, text: str) -> None:
        self.log_view.appendPlainText(text)

    def set_finished_message(self, text: str) -> None:
        self.frame_label.setText(text)
        self.mode_label.setText("—")
        self.speed_label.setText("—")

    # ------------------------------------------------------------- 内部逻辑
    def _toggle_log(self, visible: bool) -> None:
        self.log_view.setVisible(visible)

    def _copy_log(self) -> None:
        clipboard = QGuiApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(self.log_view.toPlainText())
