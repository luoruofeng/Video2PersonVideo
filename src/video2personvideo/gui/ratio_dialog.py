"""比例选择对话框：网格卡片 + 自定义比例输入，带实时校验与分辨率预估。"""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..core.ratio import PRESET_RATIOS, AspectRatio, RatioError, parse_ratio
from . import theme
from .ratio_grid import RatioGrid


class RatioDialog(QDialog):
    """模态对话框：选一个预置比例，或输入自定义 ``W:H``。"""

    def __init__(self, current: AspectRatio | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("选择目标长宽比")
        # 尺寸全部按屏幕可用区域收缩（高缩放比下逻辑分辨率会小很多），
        # 卡片网格自己会按可用宽度重排列数，不会把卡片挤在一起。
        min_width, _ = theme.clamp_size((620, 480), (460, 320), screen=self.screen())
        self.setMinimumWidth(min_width)

        self._ratio: AspectRatio = current or PRESET_RATIOS[1]

        layout = QVBoxLayout(self)
        layout.setContentsMargins(theme.SPACING_LARGE, theme.SPACING_LARGE,
                                  theme.SPACING_LARGE, theme.SPACING_LARGE)
        layout.setSpacing(theme.SPACING)

        # 内容区放进滚动区，按钮行固定在外层：屏幕再矮也能操作
        content = QWidget()
        content_box = QVBoxLayout(content)
        content_box.setContentsMargins(0, 0, 0, 0)
        content_box.setSpacing(theme.SPACING)

        hint = QLabel("输出视频的宽高比由此决定；裁剪框会以主要人物为中心保持该比例。")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        content_box.addWidget(hint)

        self.grid = RatioGrid(self._ratio)
        self.grid.ratioChanged.connect(self._on_preset_chosen)
        content_box.addWidget(self.grid)

        content_box.addWidget(self._build_custom_row())

        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.preview_label.setWordWrap(True)
        content_box.addWidget(self.preview_label)

        content_box.addStretch(1)
        layout.addWidget(theme.scrollable(content), stretch=1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("确定")
        buttons.button(QDialogButtonBox.StandardButton.Ok).setObjectName("primary")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._sync_spinboxes(self._ratio)
        self._update_preview()

        # 初始尺寸按「内容自然大小」算（滚动区的 sizeHint 很小，不能直接用它），
        # 再压进屏幕可用区域：屏幕够大时刚好放下、不出滚动条，屏幕小就滚动。
        natural = QSize(
            max(self.minimumWidth(), content.sizeHint().width() + 2 * theme.SPACING_LARGE),
            content.sizeHint().height()
            + buttons.sizeHint().height()
            + 3 * theme.SPACING
            + 2 * theme.SPACING_LARGE,
        )
        self.resize(
            *theme.clamp_size(
                (natural.width(), natural.height()),
                (self.minimumWidth(), 320),
                screen=self.screen(),
            )
        )

    # ------------------------------------------------------------- 子控件
    def _build_custom_row(self) -> QWidget:
        container = QWidget(self)
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(theme.SPACING_SMALL)

        row.addWidget(QLabel("自定义比例"))
        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 100)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(1, 100)
        for spin in (self.width_spin, self.height_spin):
            # 不再写死宽度：按控件自身 sizeHint（含上下箭头）留位，字号变大也不裁字
            spin.setMinimumWidth(spin.sizeHint().width())
            spin.valueChanged.connect(self._validate_custom)
        row.addWidget(self.width_spin)
        row.addWidget(QLabel(":"))
        row.addWidget(self.height_spin)

        self.apply_button = QPushButton("应用")
        self.apply_button.clicked.connect(self._apply_custom)
        row.addWidget(self.apply_button)

        self.custom_hint = QLabel()
        self.custom_hint.setObjectName("hint")
        row.addWidget(self.custom_hint, stretch=1)
        return container

    # ------------------------------------------------------------- 交互
    def _sync_spinboxes(self, ratio: AspectRatio) -> None:
        for spin, value in ((self.width_spin, ratio.w), (self.height_spin, ratio.h)):
            spin.blockSignals(True)
            spin.setValue(max(1, min(int(value), 100)))
            spin.blockSignals(False)

    def _on_preset_chosen(self, ratio: AspectRatio) -> None:
        self._ratio = ratio
        self._sync_spinboxes(ratio)
        self._update_preview()

    def _custom_candidate(self) -> AspectRatio | None:
        try:
            return parse_ratio(f"{self.width_spin.value()}:{self.height_spin.value()}")
        except RatioError:
            return None

    def _validate_custom(self) -> None:
        candidate = self._custom_candidate()
        if candidate is None:
            self.custom_hint.setText("比例非法（不能为 0，且需在 0.1~10 之间）")
            self.custom_hint.setStyleSheet(f"color: {theme.COLOR_DANGER};")
        else:
            self.custom_hint.setText(f"输出 {candidate.target_width}×{candidate.target_height}")
            self.custom_hint.setStyleSheet(f"color: {theme.COLOR_TEXT_MUTED};")

    def _apply_custom(self) -> None:
        candidate = self._custom_candidate()
        if candidate is None:
            self._validate_custom()
            return
        self._ratio = candidate
        self.grid.set_current(candidate)
        self._update_preview()

    def _update_preview(self) -> None:
        ratio = self._ratio
        self.preview_label.setText(
            f"当前选择：{ratio.name}    预估输出分辨率：{ratio.target_width} × {ratio.target_height}"
        )
        self._validate_custom()

    # ------------------------------------------------------------- 结果
    def selected_ratio(self) -> AspectRatio:
        return self._ratio
