"""构图参数可视化示意图。

「选择比例」页里的构图参数（人物最小占比 / 头顶留白 / 镜头跟随）都是纯数字，
不看效果很难判断该往哪边调。本模块用一块**自绘控件**把这些参数的效果实时画出来：

* **左 · 源画面示意**：画出两个示意人物，用虚线标出「人物最小占比」阈值，并用
  蓝框标出裁剪范围 —— 拖动滑块时能立刻看到谁会被当成背景忽略、取景框怎么变；
  开启「多人分屏」时额外用橙色编号框标出每人各自的上半身小窗口；
* **右 · 输出画面示意**：按当前比例画出裁剪结果，并高亮「头顶留白」；
* **下 · 镜头跟随示意**：用**真实的 :class:`BoxSmoother`** 模拟人物走动时镜头的
  横向轨迹 —— 镜头曲线越平，画面越稳、越不容易看晕。

全部是纯几何重算（不读视频、不做检测），所以拖动滑块时即时刷新。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import lru_cache

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QWidget

from ..core.framing import (
    DEFAULT_FRAMING_PARAMS,
    DEFAULT_MIN_PERSON_HEIGHT_RATIO,
    MODE_FIT,
    MODE_HALFBODY,
    MODE_TILES,
    BBox,
    CropBox,
    center_box,
    compute_target_box,
    compute_upper_body_box,
    mode_label,
)
from ..core.layout import (
    DEFAULT_MULTI_PERSON_POLICY,
    cell_ratio,
    compute_cell_rects,
    fit_rect,
    tiles_layout,
)
from ..core.ratio import PRESET_RATIOS, AspectRatio
from ..core.smoothing import (
    CAMERA_PRESETS,
    DEFAULT_CAMERA_PRESET,
    DEFAULT_NO_PERSON_MODE,
    NO_PERSON_MODES,
    BoxSmoother,
    camera_preset,
    no_person_mode_label,
)
from . import theme

#: 示意图里源画面固定按 16:9 画（绝大多数素材如此）
SOURCE_W = 1280.0
SOURCE_H = 720.0
#: 示意主角：高度占源画面 74%（高于默认阈值 0.73，且仍在「半身」档 —— 头顶留白才看得出来）
MAIN_HEIGHT_RATIO = 0.74
MAIN_CENTER_X = 0.56
#: 示意路人：高度占源画面 73.5%（比主角略矮，但仍在默认阈值 0.73 之上 ——
#: 把「最小占比」往上抬一档（0.74）就能看到它被当成背景忽略）
BYSTANDER_HEIGHT_RATIO = 0.735
BYSTANDER_CENTER_X = 0.17
#: 人物脚底所在的画面高度比例（站在画面偏下位置）
GROUND_RATIO = 0.94
#: 示意人物的宽度 / 高度（真实人体框约 0.3~0.4）
PERSON_ASPECT = 0.34

#: 镜头跟随模拟：采样点数、人物横向走动幅度（源画面像素）、模拟取景框高度
FOLLOW_SAMPLES = 96
FOLLOW_AMPLITUDE = 300.0
FOLLOW_BOX_HEIGHT = 360.0


@dataclass(frozen=True, slots=True)
class _Scene:
    """一帧示意场景：源画面里的两个人物 + 按当前参数算出的取景框。"""

    ratio: AspectRatio
    box: CropBox
    main_bbox: BBox
    bystander_bbox: BBox
    main_valid: bool
    bystander_valid: bool
    #: 多人分屏的 ``(人物框, 上半身取景框)`` 列表；未开启或不足两人时为空
    windows: tuple[tuple[BBox, CropBox], ...]
    #: 没有主要人物时的显示方式（``fit`` / ``tiles`` / ``scan`` / ``center``）
    no_person: str = DEFAULT_NO_PERSON_MODE
    #: 全画面适配 / 全景那一格在输出画布上的像素矩形（画布比例坐标系下）
    band: tuple[int, int, int, int] | None = None
    #: 「全景 + 特写」的 ``(远景小人物取景框, 它在画布上的格子)`` 列表
    tiles: tuple[tuple[CropBox, tuple[int, int, int, int]], ...] = ()


def _person_bbox(height_ratio: float, center_x_ratio: float) -> BBox:
    """按"占画面高度的比例 + 横向位置"造一个示意人物框（脚底落在同一水平线上）。"""
    height = SOURCE_H * height_ratio
    width = height * PERSON_ASPECT
    bottom = SOURCE_H * GROUND_RATIO
    center = SOURCE_W * center_x_ratio
    return (center - width / 2.0, bottom - height, center + width / 2.0, bottom)


@lru_cache(maxsize=16)
def follow_tracks(key: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """用真实的平滑器模拟"人物走动时镜头怎么动"。

    :return: ``(人物横向位置序列, 镜头横向位置序列)``，均为源画面像素坐标。
    """
    ratio = PRESET_RATIOS[0]  # 16:9，只用来给平滑器一个合法比例
    smoother = BoxSmoother(ratio, params=camera_preset(key).params)
    box_w = FOLLOW_BOX_HEIGHT * ratio.value
    center_y = (SOURCE_H - FOLLOW_BOX_HEIGHT) / 2.0
    base_x = SOURCE_W / 2.0

    persons: list[float] = []
    cameras: list[float] = []
    for index in range(FOLLOW_SAMPLES):
        t = index / float(FOLLOW_SAMPLES - 1)
        # 人物先向右走、再走回来（正弦），模拟"在画面里来回走动"
        person_x = base_x + math.sin(2.0 * math.pi * t) * FOLLOW_AMPLITUDE
        persons.append(person_x)
        target = CropBox(person_x - box_w / 2.0, center_y, box_w, FOLLOW_BOX_HEIGHT)
        cameras.append(smoother.update(target, (SOURCE_W, SOURCE_H)).cx)
    return tuple(persons), tuple(cameras)


# --------------------------------------------------------------------- 绘制工具
def _fit_rect(container: QRectF, aspect: float, *, top_pad: float = 0.0) -> QRectF:
    """在 ``container`` 内按 ``aspect`` 等比放下一个矩形（居中，顶部留 ``top_pad``）。"""
    available = QRectF(
        container.left(),
        container.top() + top_pad,
        container.width(),
        container.height() - top_pad,
    )
    if available.width() <= 2.0 or available.height() <= 2.0 or aspect <= 0.0:
        return available
    if available.width() / available.height() > aspect:
        height = available.height()
        width = height * aspect
    else:
        width = available.width()
        height = width / aspect
    return QRectF(
        available.left() + (available.width() - width) / 2.0,
        available.top() + (available.height() - height) / 2.0,
        width,
        height,
    )


def _draw_person(painter: QPainter, rect: QRectF, color: QColor, *, dashed: bool = False) -> None:
    """画一个示意人物（圆头 + 圆角躯干）。"""
    if rect.width() < 2.0 or rect.height() < 3.0:
        return
    width = rect.width()
    head_radius = max(width / 2.0, 1.0)
    pen = QPen(color, 1.4)
    if dashed:
        pen.setStyle(Qt.PenStyle.DashLine)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush if dashed else QBrush(color))
    painter.drawEllipse(QPointF(rect.center().x(), rect.top() + head_radius), head_radius, head_radius)

    body = QRectF(
        rect.left(),
        rect.top() + head_radius * 2.0,
        width,
        max(rect.height() - head_radius * 2.0, 1.0),
    )
    radius = min(width * 0.35, body.height() * 0.35)
    painter.drawRoundedRect(body, radius, radius)


def _bbox_rect(mapper, bbox: BBox) -> QRectF:
    """把源画面坐标下的人物框映射到控件坐标。"""
    x1, y1, x2, y2 = bbox
    return QRectF(mapper(x1, y1), mapper(x2, y2)).normalized()


def _box_rect(mapper, box: CropBox) -> QRectF:
    return _bbox_rect(mapper, (box.x, box.y, box.x + box.w, box.y + box.h))


def _canvas_rect(
    frame: QRectF, out_size: tuple[int, int], rect: tuple[int, int, int, int]
) -> QRectF:
    """把"输出画布上的像素矩形"映射到控件里的画布矩形。"""
    out_w, out_h = max(int(out_size[0]), 1), max(int(out_size[1]), 1)
    x, y, width, height = (float(value) for value in rect)
    return QRectF(
        frame.left() + x / out_w * frame.width(),
        frame.top() + y / out_h * frame.height(),
        width / out_w * frame.width(),
        height / out_h * frame.height(),
    )


def _region_mapper(source: CropBox, rect: QRectF):
    """源画面坐标 → 控件里某个矩形（整幅画面放进某一格时用）。"""

    def to_widget(x: float, y: float) -> QPointF:
        return QPointF(
            rect.left() + (x - source.x) / source.w * rect.width(),
            rect.top() + (y - source.y) / source.h * rect.height(),
        )

    return to_widget


class FramingSketch(QWidget):
    """构图参数示意图（纯自绘，参数一变就重画）。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._ratio: AspectRatio = PRESET_RATIOS[1]
        self._min_person_ratio = DEFAULT_MIN_PERSON_HEIGHT_RATIO
        self._headroom = DEFAULT_FRAMING_PARAMS.headroom
        self._follow_key = DEFAULT_CAMERA_PRESET
        self._multi_person = True
        self._no_person = DEFAULT_NO_PERSON_MODE
        policy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)
        self.setToolTip(
            "示意图随下方参数实时变化：\n"
            "· 左：源画面里的人物是否达到「人物最小占比」阈值（未达到会被忽略）；\n"
            "  橙色编号框是开启「多人分屏」后每人各自的小窗口\n"
            "· 右：按当前比例裁剪出来的画面，高亮区域是「头顶留白」；\n"
            "  画面里没有主要人物时，这里显示「无人物显示」选定的那一档效果\n"
            "· 下：镜头跟随档位决定了人物走动时镜头跟着动多少（越平越稳）"
        )

    # ----------------------------------------------------------------- 接口
    def configure(
        self,
        *,
        ratio: AspectRatio,
        min_person_ratio: float,
        headroom: float,
        follow_key: str,
        multi_person: bool,
        no_person_mode: str = DEFAULT_NO_PERSON_MODE,
    ) -> None:
        """写入当前参数并重画（不读视频、不做检测，调用成本极低）。"""
        ratio_changed = ratio.name != self._ratio.name
        self._ratio = ratio
        self._min_person_ratio = float(min_person_ratio)
        self._headroom = float(headroom)
        self._follow_key = follow_key if follow_key in CAMERA_PRESETS else DEFAULT_CAMERA_PRESET
        self._multi_person = bool(multi_person)
        self._no_person = (
            no_person_mode if no_person_mode in NO_PERSON_MODES else DEFAULT_NO_PERSON_MODE
        )
        if ratio_changed:
            # 高度是按比例算出来的：换比例后要让布局重新问一次
            self.updateGeometry()
        self.update()

    # ----------------------------------------------------------------- 尺寸
    def hasHeightForWidth(self) -> bool:  # noqa: N802 - Qt 命名约定
        return True

    def _frame_height(self, width: float) -> float:
        """给定宽度时两块示意图能有多高（其余高度留给标题与下方的镜头曲线）。"""
        usable = max(float(width) - 4.0, 120.0)
        span = SOURCE_W / SOURCE_H + max(self._ratio.value, 0.05)
        return min(max(usable * 0.58 / span, 110.0), 260.0)

    def heightForWidth(self, width: int) -> int:  # noqa: N802 - Qt 命名约定
        line = float(theme.ui_metrics().line_height)
        return int(self._frame_height(width) + 2.0 * line + 42.0 + theme.SPACING + 10.0)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt 命名约定
        return QSize(560, self.heightForWidth(560))

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt 命名约定
        return QSize(340, self.heightForWidth(340))

    # ----------------------------------------------------------------- 场景
    def _scene(self) -> _Scene:
        ratio = self._ratio
        params = replace(DEFAULT_FRAMING_PARAMS, headroom=self._headroom)
        frame_size = (SOURCE_W, SOURCE_H)
        out_size = ratio.target_size
        main_bbox = _person_bbox(MAIN_HEIGHT_RATIO, MAIN_CENTER_X)
        bystander_bbox = _person_bbox(BYSTANDER_HEIGHT_RATIO, BYSTANDER_CENTER_X)
        # 阈值 ≤ 高度即"够大"，与"低于阈值算背景"是同一件事的两种读法
        threshold = float(self._min_person_ratio)
        main_valid = threshold <= MAIN_HEIGHT_RATIO
        bystander_valid = threshold <= BYSTANDER_HEIGHT_RATIO

        windows: tuple[tuple[BBox, CropBox], ...] = ()
        if self._multi_person and main_valid and bystander_valid:
            windows = tuple(
                (bbox, compute_upper_body_box(bbox, frame_size, ratio, params))
                for bbox in (main_bbox, bystander_bbox)
            )

        # 没有主要人物时的显示方式：全画面适配 / 全景+特写要画出"整幅画面放哪儿"
        band: tuple[int, int, int, int] | None = None
        tiles: tuple[tuple[CropBox, tuple[int, int, int, int]], ...] = ()
        if main_valid:
            box = compute_target_box(main_bbox, frame_size, ratio, params)
        elif self._no_person in (MODE_FIT, MODE_TILES):
            box = CropBox(0.0, 0.0, SOURCE_W, SOURCE_H)
            band = fit_rect(frame_size, out_size)
            if self._no_person == MODE_TILES and bystander_valid:
                gap = DEFAULT_MULTI_PERSON_POLICY.gap_px(out_size)
                layout = tiles_layout(out_size, frame_size, 1, gap=gap)
                if layout is not None:
                    rects = compute_cell_rects(layout, out_size, gap)
                    band = rects[0]
                    tiles = (
                        (
                            compute_upper_body_box(
                                bystander_bbox, frame_size, cell_ratio(rects[1]), params
                            ),
                            rects[1],
                        ),
                    )
        else:
            box = center_box(frame_size, ratio)

        return _Scene(
            ratio=ratio,
            box=box,
            main_bbox=main_bbox,
            bystander_bbox=bystander_bbox,
            main_valid=main_valid,
            bystander_valid=bystander_valid,
            windows=windows,
            no_person=self._no_person,
            band=band,
            tiles=tiles,
        )

    # ----------------------------------------------------------------- 绘制
    def paintEvent(self, event) -> None:  # noqa: N802 - Qt 命名约定
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        colors = theme.active()
        metrics = theme.ui_metrics()

        area = QRectF(self.rect()).adjusted(1.0, 1.0, -1.0, -1.0)
        if area.width() < 160.0 or area.height() < 130.0:
            painter.end()
            return

        scene = self._scene()
        line = float(metrics.line_height)
        gap = float(theme.SPACING)
        strip_h = float(metrics.line_height + 42)
        top_h = area.height() - strip_h - gap
        if top_h < line + 56.0:
            # 空间不够就只画上面两块图，别把三块挤成一片
            strip_h = 0.0
            top_h = area.height()

        source_area, output_area = self._split_row(area, scene, top_h, line, gap)
        self._paint_source(painter, source_area, scene, colors, metrics)
        self._paint_output(painter, output_area, scene, colors, metrics)
        if strip_h > 0.0:
            strip_area = QRectF(area.left(), area.top() + top_h + gap, area.width(), strip_h)
            self._paint_follow(painter, strip_area, colors, metrics)
        painter.end()

    def _split_row(
        self, area: QRectF, scene: _Scene, top_h: float, line: float, gap: float
    ) -> tuple[QRectF, QRectF]:
        """把上排分成「源画面 / 输出画面」两栏。

        两块图都按各自的比例尽量占满行高；当行宽有富余时把整体居中，
        避免出现"小小一张图飘在大片空白里"的观感。
        """
        usable_h = max(top_h - line - 2.0, 1.0)
        aspect = SOURCE_W / SOURCE_H
        target = max(scene.ratio.value, 0.05)
        need_source = max(
            usable_h * aspect, float(theme.text_width_px("源画面（原视频）") + 8)
        )
        need_output = max(
            usable_h * target, float(theme.text_width_px(self._output_title(scene)) + 8)
        )

        total = need_source + need_output + gap
        if total <= area.width():
            left = area.left() + (area.width() - total) / 2.0
        else:
            # 挤不下：按比例分栏，两块图各自再等比缩小
            share = aspect / (aspect + target)
            need_source = max((area.width() - gap) * share, 1.0)
            need_output = max(area.width() - gap - need_source, 1.0)
            left = area.left()
        return (
            QRectF(left, area.top(), need_source, top_h),
            QRectF(left + need_source + gap, area.top(), need_output, top_h),
        )

    def _output_title(self, scene: _Scene) -> str:
        if scene.main_valid:
            return f"输出画面 · {scene.ratio.name} · {mode_label(scene.box.mode)}"
        if scene.band is not None:
            return f"输出画面 · {scene.ratio.name} · 无人物：{no_person_mode_label(scene.no_person)}"
        return f"输出画面 · {scene.ratio.name} · 画面居中"

    # ------------------------------------------------------------- 无人物输出
    def _paint_no_person_output(
        self, painter: QPainter, area: QRectF, scene: _Scene, colors, line: float
    ) -> None:
        """没有主要人物时的输出示意：整幅画面放进画布（+ 可选的特写窗口）。"""
        frame = _fit_rect(area, scene.ratio.value, top_pad=line + 2.0)
        if frame.width() < 24.0 or frame.height() < 24.0 or scene.band is None:
            return

        # 底色（真实渲染里是同一帧的模糊放大版）
        background = QColor(colors.border_soft)
        background.setAlpha(120)
        painter.setPen(QPen(QColor(colors.border), 1.2))
        painter.setBrush(QBrush(background))
        painter.drawRect(frame)

        band = _canvas_rect(frame, scene.ratio.target_size, scene.band)
        painter.setPen(QPen(QColor(colors.border), 1.0))
        painter.setBrush(QBrush(QColor(colors.surface)))
        painter.drawRect(band)

        painter.save()
        painter.setClipRect(band)
        to_widget = _region_mapper(scene.box, band)
        for bbox, valid in (
            (scene.bystander_bbox, scene.bystander_valid),
            (scene.main_bbox, scene.main_valid),
        ):
            if valid:
                continue
            painter.setOpacity(0.5)
            _draw_person(painter, _bbox_rect(to_widget, bbox), QColor(colors.text_muted))
            painter.setOpacity(1.0)
        painter.restore()

        if scene.tiles:
            for order, (window_box, cell) in enumerate(scene.tiles, start=1):
                rect = _canvas_rect(frame, scene.ratio.target_size, cell)
                painter.setPen(QPen(QColor(colors.warning), 1.4, Qt.PenStyle.DashLine))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(rect)
                painter.save()
                painter.setClipRect(rect)
                _draw_person(
                    painter,
                    _bbox_rect(_region_mapper(window_box, rect), scene.bystander_bbox),
                    QColor(colors.text_muted),
                )
                painter.restore()
                painter.setPen(QPen(QColor(colors.warning)))
                painter.drawText(
                    QRectF(rect.left() + 4.0, rect.top() + 2.0, rect.width() - 8.0, line),
                    int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop),
                    f"特写 {order}",
                )
        elif scene.no_person == MODE_TILES:
            painter.setPen(QPen(QColor(colors.warning)))
            painter.drawText(
                QRectF(frame.left() + 4.0, band.bottom() + 2.0, frame.width() - 8.0, line),
                int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                "没有够大的远景小人物 → 全部按全画面适配",
            )

    def _paint_source(
        self, painter: QPainter, area: QRectF, scene: _Scene, colors, metrics
    ) -> None:
        line = float(metrics.line_height)
        painter.setPen(QPen(QColor(colors.text_muted)))
        painter.drawText(
            QRectF(area.left(), area.top(), area.width(), line),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            "源画面（原视频）",
        )

        frame = _fit_rect(area, SOURCE_W / SOURCE_H, top_pad=line + 2.0)
        if frame.width() < 24.0 or frame.height() < 24.0:
            return
        painter.setPen(QPen(QColor(colors.border), 1.2))
        painter.setBrush(QBrush(QColor(colors.surface)))
        painter.drawRect(frame)

        def to_widget(x: float, y: float) -> QPointF:
            return QPointF(
                frame.left() + x / SOURCE_W * frame.width(),
                frame.top() + y / SOURCE_H * frame.height(),
            )

        # 阈值线：人物矮于这条线就会被当成背景忽略
        if self._min_person_ratio > 0.0:
            threshold_y = frame.top() + (GROUND_RATIO - self._min_person_ratio) * frame.height()
            pen = QPen(QColor(colors.warning), 1.2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawLine(
                QPointF(frame.left(), threshold_y), QPointF(frame.right(), threshold_y)
            )
            painter.setPen(QPen(QColor(colors.warning)))
            painter.drawText(
                QRectF(frame.left() + 4.0, threshold_y - line, frame.width() - 8.0, line),
                int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                f"人物最小占比 {self._min_person_ratio:.2f}（以下忽略）",
            )

        # 两个示意人物（未达阈值的人物画成虚线淡色）
        for bbox, valid, tag in (
            (scene.bystander_bbox, scene.bystander_valid, "路人"),
            (scene.main_bbox, scene.main_valid, "主角"),
        ):
            rect = _bbox_rect(to_widget, bbox)
            if valid:
                color = QColor(colors.primary if tag == "主角" else colors.text_muted)
                _draw_person(painter, rect, color)
            else:
                painter.setOpacity(0.55)
                _draw_person(painter, rect, QColor(colors.danger), dashed=True)
                painter.setOpacity(1.0)
            if valid and tag == "主角":
                label_color = colors.primary
            elif valid:
                label_color = colors.text_muted
            else:
                label_color = colors.danger
            painter.setPen(QPen(QColor(label_color)))
            painter.drawText(
                QRectF(rect.left() - 6.0, rect.top() - line, rect.width() + 12.0, line),
                int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom),
                tag if valid else f"{tag}被忽略",
            )

        # 取景框（= 右侧输出画面的范围）
        box_rect = _box_rect(to_widget, scene.box)
        fill = QColor(colors.primary)
        fill.setAlpha(36)
        painter.setPen(QPen(QColor(colors.primary), 1.6, Qt.PenStyle.DashLine))
        painter.setBrush(QBrush(fill))
        painter.drawRect(box_rect)

        # 多人分屏：每人一个上半身小窗口（与左侧取景框同框显示，便于对照）
        for order, (_bbox, window_box) in enumerate(scene.windows, start=1):
            window_rect = _box_rect(to_widget, window_box)
            painter.setPen(QPen(QColor(colors.warning), 1.6, Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(window_rect)
            painter.setPen(QPen(QColor(colors.warning)))
            painter.drawText(
                QRectF(window_rect.left() + 4.0, window_rect.top() + 2.0, 24.0, line),
                int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop),
                str(order),
            )

    def _paint_output(
        self, painter: QPainter, area: QRectF, scene: _Scene, colors, metrics
    ) -> None:
        line = float(metrics.line_height)
        painter.setPen(QPen(QColor(colors.text_muted)))
        painter.drawText(
            QRectF(area.left(), area.top(), area.width(), line),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            self._output_title(scene),
        )
        if scene.band is not None:
            self._paint_no_person_output(painter, area, scene, colors, line)
            return
        self._paint_single_output(painter, area, scene, colors, line)

    def _paint_single_output(
        self, painter: QPainter, area: QRectF, scene: _Scene, colors, line: float
    ) -> None:
        frame = _fit_rect(area, scene.ratio.value, top_pad=line + 2.0)
        if frame.width() < 24.0 or frame.height() < 24.0:
            return

        painter.setPen(QPen(QColor(colors.border), 1.2))
        painter.setBrush(QBrush(QColor(colors.surface)))
        painter.drawRect(frame)

        def to_widget(x: float, y: float) -> QPointF:
            return QPointF(
                frame.left() + (x - scene.box.x) / scene.box.w * frame.width(),
                frame.top() + (y - scene.box.y) / scene.box.h * frame.height(),
            )

        # 头顶留白：只对半身构图有意义
        if scene.main_valid and scene.box.mode == MODE_HALFBODY:
            head_y = frame.top() + (scene.main_bbox[1] - scene.box.y) / scene.box.h * frame.height()
            band = QRectF(frame.left(), frame.top(), frame.width(), max(head_y - frame.top(), 0.0))
            fill = QColor(colors.warning)
            fill.setAlpha(46)
            painter.setPen(QPen(QColor(colors.warning), 1.0, Qt.PenStyle.DashLine))
            painter.setBrush(QBrush(fill))
            painter.drawRect(band)
            if band.height() > line:
                painter.setPen(QPen(QColor(colors.warning)))
                painter.drawText(
                    band.adjusted(4.0, 0.0, -4.0, 0.0),
                    int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                    f"头顶留白 {self._headroom:.2f}",
                )

        painter.save()
        painter.setClipRect(frame)
        for bbox, valid, tag in (
            (scene.bystander_bbox, scene.bystander_valid, "路人"),
            (scene.main_bbox, scene.main_valid, "主角"),
        ):
            if not valid:
                continue
            _draw_person(
                painter,
                _bbox_rect(to_widget, bbox),
                QColor(colors.primary if tag == "主角" else colors.text_muted),
            )
        if not scene.main_valid:
            # 无人可用 → 退回画面正中，示意人物画淡色虚线说明"不会被跟"
            painter.setOpacity(0.5)
            for bbox in (scene.bystander_bbox, scene.main_bbox):
                _draw_person(painter, _bbox_rect(to_widget, bbox), QColor(colors.danger), dashed=True)
            painter.setOpacity(1.0)
        painter.restore()

    def _paint_follow(self, painter: QPainter, area: QRectF, colors, metrics) -> None:
        line = float(metrics.line_height)
        if self._follow_key in CAMERA_PRESETS:
            label = CAMERA_PRESETS[self._follow_key].label
        else:
            label = "自定义（按标准档示意）"
        persons, cameras = follow_tracks(self._follow_key)
        person_travel = max(persons) - min(persons)
        camera_travel = max(cameras) - min(cameras)
        percent = camera_travel / person_travel * 100.0 if person_travel > 0.0 else 0.0

        painter.setPen(QPen(QColor(colors.text_muted)))
        painter.drawText(
            QRectF(area.left(), area.top(), area.width() * 0.5, line),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            f"镜头跟随 · {label}（纵轴 = 横向位置）",
        )
        painter.drawText(
            QRectF(area.left() + area.width() * 0.5, area.top(), area.width() * 0.5, line),
            int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
            f"─ 人物    ─ 镜头（约为人物移动的 {percent:.0f}%）",
        )

        plot = QRectF(
            area.left(), area.top() + line + 2.0, area.width(), area.height() - line - 4.0
        )
        if plot.width() < 60.0 or plot.height() < 20.0:
            return

        painter.setPen(QPen(QColor(colors.border_soft), 1.0))
        painter.setBrush(QBrush(QColor(colors.surface)))
        painter.drawRect(plot)

        # 纵轴按人物实际走动范围贴边展开，档位之间的差别才看得出来
        low = min(min(persons), min(cameras)) - 40.0
        high = max(max(persons), max(cameras)) + 40.0
        span = max(high - low, 1.0)

        def to_point(index: int, value: float) -> QPointF:
            x = plot.left() + index / float(len(persons) - 1) * plot.width()
            y = plot.bottom() - (value - low) / span * plot.height()
            return QPointF(x, y)

        painter.setPen(QPen(QColor(colors.text_muted), 1.4))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPolyline(QPolygonF([to_point(i, v) for i, v in enumerate(persons)]))

        painter.setPen(QPen(QColor(colors.primary), 2.0))
        painter.drawPolyline(QPolygonF([to_point(i, v) for i, v in enumerate(cameras)]))
