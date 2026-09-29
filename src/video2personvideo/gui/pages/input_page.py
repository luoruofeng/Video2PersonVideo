"""向导步骤 1：选择输入（视频文件 或 视频文件夹），支持拖拽。"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from ...core.batch import VIDEO_EXTENSIONS, discover_videos
from .. import theme

#: 文件对话框过滤器
VIDEO_FILTER = (
    "视频文件 ("
    + " ".join(f"*{ext}" for ext in VIDEO_EXTENSIONS)
    + ");;所有文件 (*.*)"
)


class InputPage(QWidget):
    """输入选择页。"""

    selectionChanged = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._sources: list[Path] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.SPACING)

        drop_hint = QLabel("把视频文件或文件夹拖到这里，或点击下面的「浏览…」选择。")
        drop_hint.setObjectName("hint")
        layout.addWidget(drop_hint)

        group = QGroupBox("输入路径")
        box = QVBoxLayout(group)
        box.setSpacing(theme.SPACING_SMALL)

        mode_row = QHBoxLayout()
        self.file_radio = QRadioButton("单个视频文件")
        self.folder_radio = QRadioButton("视频文件夹")
        self.file_radio.setChecked(True)
        self.file_radio.toggled.connect(self._on_mode_changed)
        mode_row.addWidget(self.file_radio)
        mode_row.addSpacing(theme.SPACING_LARGE)
        mode_row.addWidget(self.folder_radio)
        mode_row.addStretch(1)
        box.addLayout(mode_row)

        path_row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("选择要处理的视频文件或文件夹…")
        self.path_edit.textChanged.connect(self._refresh)
        self.browse_button = QPushButton("浏览…")
        self.browse_button.clicked.connect(self._browse)
        path_row.addWidget(self.path_edit, stretch=1)
        path_row.addWidget(self.browse_button)
        box.addLayout(path_row)

        option_row = QHBoxLayout()
        self.recursive_check = QCheckBox("递归遍历子文件夹")
        self.recursive_check.setChecked(True)
        self.recursive_check.toggled.connect(self._refresh)
        option_row.addWidget(self.recursive_check)
        option_row.addStretch(1)
        box.addLayout(option_row)

        self.summary_label = QLabel("尚未选择输入")
        self.summary_label.setObjectName("hint")
        box.addWidget(self.summary_label)

        self.list_view = QListWidget()
        self.list_view.setMinimumHeight(180)
        box.addWidget(self.list_view, stretch=1)

        layout.addWidget(group, stretch=1)

    # ----------------------------------------------------------------- 接口
    @property
    def sources(self) -> list[Path]:
        return list(self._sources)

    @property
    def path(self) -> Path | None:
        text = self.path_edit.text().strip()
        return Path(text) if text else None

    def is_valid(self) -> bool:
        return bool(self._sources)

    def set_path(self, path: str | Path | None) -> None:
        """外部设置路径（拖拽 / 记忆恢复），并自动切换单选模式。"""
        if path is None:
            self.path_edit.clear()
            return
        target = Path(path)
        if target.is_dir():
            self.folder_radio.setChecked(True)
        elif target.suffix:
            self.file_radio.setChecked(True)
        self.path_edit.setText(str(target))
        self._refresh()

    def recursive(self) -> bool:
        return self.recursive_check.isChecked()

    # ------------------------------------------------------------- 内部逻辑
    def _on_mode_changed(self) -> None:
        self.path_edit.clear()
        self._refresh()

    def _browse(self) -> None:
        start = str(self.path.parent if self.path else Path.cwd())
        if self.folder_radio.isChecked():
            selected = QFileDialog.getExistingDirectory(self, "选择视频文件夹", start)
        else:
            selected, _ = QFileDialog.getOpenFileName(self, "选择输入视频", start, VIDEO_FILTER)
        if selected:
            self.set_path(selected)

    def _refresh(self) -> None:
        self.list_view.clear()
        self._sources = []
        path = self.path
        if path is None:
            self.summary_label.setText("尚未选择输入")
            self.selectionChanged.emit()
            return
        if not path.exists():
            self.summary_label.setText(f"路径不存在：{path}")
            self.selectionChanged.emit()
            return

        try:
            found = discover_videos(path, recursive=self.recursive_check.isChecked())
        except (OSError, RuntimeError) as exc:  # pragma: no cover - 磁盘异常
            self.summary_label.setText(f"读取失败：{exc}")
            self.selectionChanged.emit()
            return

        self._sources = found
        if path.is_file():
            self.summary_label.setText(f"已选择单个视频：{path.name}")
        elif found:
            self.summary_label.setText(
                f"共发现 {len(found)} 个视频（已自动忽略 *_person.* 产物）"
            )
        else:
            self.summary_label.setText("未在该文件夹中发现可处理的视频")

        for item in found[:300]:
            self.list_view.addItem(item.name if path.is_dir() else str(item))
        self.selectionChanged.emit()

    # ------------------------------------------------------------- 拖拽支持
    def dragEnterEvent(self, event) -> None:  # noqa: N802 - Qt 命名约定
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802 - Qt 命名约定
        urls = event.mimeData().urls()
        if not urls:
            return
        local = urls[0].toLocalFile()
        if local:
            self.set_path(local)
            event.acceptProposedAction()
