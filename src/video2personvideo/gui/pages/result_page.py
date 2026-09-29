"""向导步骤 4：结果汇总（成功 / 失败列表 + 打开输出文件夹）。"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...core.batch import BatchResult, status_label
from ...core.processor import format_bytes
from .. import theme


class ResultPage(QWidget):
    """结果页。"""

    restartRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._output_dir: Path | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.SPACING)

        group = QGroupBox("处理结果")
        box = QVBoxLayout(group)

        self.summary_label = QLabel("尚无结果")
        self.summary_label.setWordWrap(True)
        box.addWidget(self.summary_label)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["源文件", "输出", "状态", "说明"])
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        box.addWidget(self.table, stretch=1)

        action_row = QHBoxLayout()
        self.restart_button = QPushButton("再处理一批")
        self.restart_button.clicked.connect(self.restartRequested.emit)
        self.open_button = QPushButton("打开输出文件夹")
        self.open_button.setObjectName("primary")
        self.open_button.clicked.connect(self._open_folder)
        action_row.addWidget(self.restart_button)
        action_row.addStretch(1)
        action_row.addWidget(self.open_button)
        box.addLayout(action_row)

        layout.addWidget(group, stretch=1)

    # ----------------------------------------------------------------- 接口
    @property
    def output_directory(self) -> Path | None:
        return self._output_dir

    def set_result(self, result: BatchResult, output_directory: Path | None = None) -> None:
        self._output_dir = output_directory or _first_output_dir(result)

        text = (
            f"共 {len(result.tasks)} 个任务：成功 {len(result.succeeded)} · "
            f"跳过 {len(result.skipped)} · 失败 {len(result.failed)} · "
            f"取消 {len(result.cancelled)}，总耗时 {result.elapsed:.1f}s"
        )
        source_bytes, output_bytes = result.volume
        if output_bytes:
            text += f"\n输出体积合计 {format_bytes(output_bytes)}"
            if source_bytes:
                text += f"（原视频合计 {format_bytes(source_bytes)}，{output_bytes / source_bytes:.2f}×）"
        self.summary_label.setText(text)

        self.table.setRowCount(len(result.tasks))
        for row, task in enumerate(result.tasks):
            detail = task.error
            if not detail and task.result is not None:
                size = task.result.output_size
                detail = f"{size[0]}×{size[1]} · {task.result.frames_processed} 帧"
                if task.result.output_bytes:
                    detail += f" · {format_bytes(task.result.output_bytes)}"
                if task.result.size_guard:
                    detail += "（已自动压缩）"
            values = (
                task.source.name,
                str(task.output),
                status_label(task.status),
                detail or "—",
            )
            for column, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setToolTip(text)
                self.table.setItem(row, column, item)

    # ------------------------------------------------------------- 内部逻辑
    def _open_folder(self) -> None:
        target = self._output_dir
        if target is None or not target.exists():
            QMessageBox.information(self, "提示", "输出目录还不存在。")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))


def _first_output_dir(result: BatchResult) -> Path | None:
    for task in result.tasks:
        return task.output.parent
    return None
