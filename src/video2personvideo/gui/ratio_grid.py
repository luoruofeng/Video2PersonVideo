"""比例选择用的网格卡片控件：用图形化缩略直观展示比例形状。

尺寸策略：卡片大小由**字体度量**算出来（不写死像素），列数由可用宽度决定。
这样无论系统缩放比例是 100% 还是 200%、用户字号是大是小，卡片里的
「比例名 + 分辨率」都不会被裁掉，也不会撑出窗口。
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFontMetrics, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QGridLayout,
    QSizePolicy,
    QToolButton,
    QWidget,
)

from ..core.ratio import PRESET_RATIOS, AspectRatio, parse_ratio
from . import theme

#: 卡片上的缩略图尺寸
ICON_SIZE = QSize(72, 46)
#: 卡片的最小尺寸（字体更小也不会缩到看不清；字体更大时按度量自动放大）
MIN_CARD_SIZE = QSize(126, 112)
#: 每行最多排几张卡片（宽屏下多排几列，能显著降低纵向高度，少出滚动条）
MAX_CARD_COLUMNS = 6
#: 宽度不足时逐级减少列数，最终兜底列数
MIN_CARD_COLUMNS = 1
#: 还没拿到真实宽度时的起始列数（按原来的观感，4 列 3 行）
DEFAULT_CARD_COLUMNS = 4
#: 与 QSS 里 ``QToolButton#ratioCard`` 的 padding / 选中边框保持一致
CARD_PADDING = 6
CARD_BORDER = 2


def make_ratio_icon(ratio: AspectRatio, size: QSize = ICON_SIZE) -> QIcon:
    """按比例画一个矩形缩略图，作为卡片图标。"""
    pixmap = QPixmap(size)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    padding = 6
    available_w = size.width() - padding * 2
    available_h = size.height() - padding * 2
    if ratio.value >= available_w / available_h:
        width = float(available_w)
        height = width / ratio.value
    else:
        height = float(available_h)
        width = height * ratio.value
    left = (size.width() - width) / 2
    top = (size.height() - height) / 2

    painter.setPen(QPen(QColor(theme.COLOR_PRIMARY), 2))
    painter.drawRect(int(left), int(top), max(int(width), 2), max(int(height), 2))
    painter.setBrush(QColor(theme.COLOR_CARD_SELECTED))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawRect(int(left) + 2, int(top) + 2, max(int(width) - 3, 1), max(int(height) - 3, 1))
    painter.end()
    return QIcon(pixmap)


def card_size_for(text: str, font) -> QSize:
    """按字体度量算卡片尺寸：保证缩略图与多行文字都放得下。

    字体变大（或系统缩放比例变化导致度量不同）时卡片跟着变大，
    而不是把文字压在固定尺寸里裁掉。
    """
    metrics = QFontMetrics(font)
    lines = text.split("\n") or [""]
    text_width = max(metrics.horizontalAdvance(line) for line in lines)
    text_height = metrics.lineSpacing() * len(lines)
    chrome = 2 * (CARD_PADDING + CARD_BORDER)
    width = max(MIN_CARD_SIZE.width(), text_width + chrome + 6)
    height = max(MIN_CARD_SIZE.height(), ICON_SIZE.height() + text_height + chrome + 8)
    return QSize(width, height)


class RatioCard(QToolButton):
    """单张比例卡片（可选中）。"""

    def __init__(self, ratio: AspectRatio, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ratio = ratio
        self.setObjectName("ratioCard")
        self.setCheckable(True)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        self.setIcon(make_ratio_icon(ratio))
        self.setIconSize(ICON_SIZE)
        self.setText(f"{ratio.name}\n{ratio.target_width}×{ratio.target_height}")
        self.setToolTip(f"{ratio.name} · 输出 {ratio.target_width}×{ratio.target_height}")
        self._apply_size()

    def _apply_size(self) -> None:
        self.setFixedSize(card_size_for(self.text(), self.font()))

    def changeEvent(self, event) -> None:  # noqa: N802 - Qt 命名约定
        super().changeEvent(event)
        if event.type() == QEvent.Type.FontChange:
            # 字号变了（换主题 / 系统缩放变化）就重新按度量定尺寸
            self._apply_size()


class RatioGrid(QWidget):
    """预置比例的网格选择器，单选。

    列数按自身可用宽度动态决定：窗口窄就少排几列（卡片不变形），
    宽了就多排几列（少占纵向空间、少出滚动条）。
    """

    ratioChanged = Signal(object)

    def __init__(self, current: AspectRatio | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._cards: list[RatioCard] = []
        self._columns = 0

        self._grid = QGridLayout(self)
        self._grid.setSpacing(theme.SPACING)
        self._grid.setContentsMargins(0, 0, 0, 0)
        for index, ratio in enumerate(PRESET_RATIOS):
            card = RatioCard(ratio)
            self._group.addButton(card, index)
            card.clicked.connect(lambda _=False, item=ratio: self._on_clicked(item))
            self._cards.append(card)

        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        # 构造时还没有真实宽度，先按默认列数排；真实宽度会由 resizeEvent 校正
        self._relayout(DEFAULT_CARD_COLUMNS)

        self._current: AspectRatio | None = None
        self.set_current(parse_ratio(current.name) if current else PRESET_RATIOS[1])

    # ----------------------------------------------------------------- 布局
    def _card_width(self) -> int:
        return self._cards[0].width() if self._cards else MIN_CARD_SIZE.width()

    def _column_count(self, width: int | None = None) -> int:
        """按可用宽度算能排几列（至少 1 列，最多 ``MAX_CARD_COLUMNS`` 列）。"""
        available = self.width() if width is None else width
        spacing = self._grid.spacing()
        step = self._card_width() + spacing
        if available <= 0 or step <= 0:  # 还没布局过：先用默认列数
            return DEFAULT_CARD_COLUMNS
        columns = (available + spacing) // step
        return max(MIN_CARD_COLUMNS, min(MAX_CARD_COLUMNS, int(columns)))

    def _grid_size(self, columns: int) -> QSize:
        """按列数算网格整体尺寸。"""
        columns = max(MIN_CARD_COLUMNS, columns)
        card_w = self._card_width()
        card_h = self._cards[0].height() if self._cards else MIN_CARD_SIZE.height()
        spacing = self._grid.spacing()
        rows = (len(self._cards) + columns - 1) // columns if self._cards else 1
        return QSize(
            columns * card_w + (columns - 1) * spacing,
            rows * card_h + (rows - 1) * spacing,
        )

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt 命名约定
        """最小宽度只按 **一张卡片** 算。

        这一点很关键：如果最小宽度按「当前列数」算（6 列 = 806px），父布局就
        永远不允许窗口变窄，``resizeEvent`` 也就永远不会把列数降下来——列数会
        被自己锁死，窄窗口只能出现横向滚动条。所以高度按当前排列给，宽度放开。
        """
        natural = self._grid_size(self._columns or DEFAULT_CARD_COLUMNS)
        return QSize(self._card_width(), natural.height())

    def _relayout(self, columns: int) -> None:
        columns = max(MIN_CARD_COLUMNS, columns)
        if columns == self._columns:
            return
        self._columns = columns
        for card in self._cards:
            self._grid.removeWidget(card)
        for index, card in enumerate(self._cards):
            self._grid.addWidget(card, index // columns, index % columns)
        # 排列变了，尺寸需求跟着变，通知父布局重新排版
        self._grid.invalidate()
        self.updateGeometry()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt 命名约定
        super().resizeEvent(event)
        self._relayout(self._column_count(event.size().width()))

    # ----------------------------------------------------------------- 接口
    @property
    def current_ratio(self) -> AspectRatio:
        return self._current  # type: ignore[return-value]

    def set_current(self, ratio: AspectRatio) -> None:
        """选中某个比例；不在预置表内时全部取消勾选。"""
        self._current = ratio
        matched = False
        for card in self._cards:
            selected = card.ratio.name == ratio.name
            card.setChecked(selected)
            matched = matched or selected
        if not matched:
            self._group.setExclusive(False)
            for card in self._cards:
                card.setChecked(False)
            self._group.setExclusive(True)

    def _on_clicked(self, ratio: AspectRatio) -> None:
        self._current = ratio
        self.ratioChanged.emit(ratio)
