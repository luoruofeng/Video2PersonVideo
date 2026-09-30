"""画面里没有主要人物时的显示方式：全画面适配与「全景 + 特写」。

这里只解决**无人物帧**这一种情况的"看什么、怎么排"，不做时序平滑
（过渡进度由 :class:`~video2personvideo.core.smoothing.BoxSmoother` 负责）：

* ``fit`` **全画面适配**：整幅画面等比缩小、完整放进画布。一帧看全、镜头完全不移动，
  是最保守也最不容易看晕的一档（也是默认档）；横屏素材转竖屏时上下空出来的部分
  由裁剪层用同一帧的模糊放大版填满，不会出现黑边；
* ``tiles`` **全景 + 特写**：上方通栏窗口放整幅画面（"主屏幕"保住了），
  下方纵向排列若干**次要小人物**（高度占比低于 ``min_person_height_ratio``、
  又高于 ``secondary_ratio`` 的那些人）的半身特写 —— 于是纵向排列的每一格都承载
  **不同的信息**（环境 + 被主角阈值过滤掉的人），而不是同一画面的复制。

窗口怎么排由 :func:`~video2personvideo.core.layout.tiles_layout` 生成：
首行高度 = 通栏宽度 ÷ 源画面比例，所以全景那一格里的画面正好填满该格。

特写窗口在过渡的后半段淡入（:data:`FADE_START` 起），于是观感是"画面缓缓收进上格、
下面的特写顺势露出"，而不是"啪"地换一个版式。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .crop import WindowSlice
from .framing import (
    DEFAULT_FRAMING_PARAMS,
    DEFAULT_MIN_PERSON_HEIGHT_RATIO,
    CropBox,
    FramingParams,
    compute_upper_body_box,
)
from .layout import (
    DEFAULT_MULTI_PERSON_POLICY,
    MIN_CELL_PX,
    MultiPersonPolicy,
    TransitionFrame,
    cell_ratio,
    compute_cell_rects,
    tiles_layout,
)
from .pose import build_anchors
from .ratio import AspectRatio
from .subject import Detection

#: 默认的"次要小人物"下限（高度占画面比例）：低于它的连特写也不给 ——
#: 再小就只剩马赛克，放大了反而更糊
DEFAULT_SECONDARY_PERSON_RATIO = 0.12

#: 默认最多给几个小人物开特写窗口（再多，竖屏里每格就只剩一条）
DEFAULT_TILES_MAX = 2

#: 特写窗口开始淡入的过渡进度（0~1）：前半段只做"拉远"，后半段画面已基本收进上格，
#: 此时特写再露出，观感更顺
FADE_START = 0.55


@dataclass(frozen=True, slots=True)
class NoPersonDisplay:
    """无人物帧的显示外观（显示方式本身由 ``SmoothingParams.no_person_mode`` 决定）。"""

    #: 画布底色是否用"同一帧的模糊放大版"（全画面适配时比纯色 / 黑边自然得多）
    blur: bool = True
    #: 「全景 + 特写」最多给几个小人物开特写窗口
    tiles_max: int = DEFAULT_TILES_MAX
    #: 「次要小人物」的高度下限（占画面比例）：低于它的不给特写
    secondary_ratio: float = DEFAULT_SECONDARY_PERSON_RATIO

    def describe(self) -> str:
        return (
            f"无人物显示：模糊背景 {'开' if self.blur else '关'}，"
            f"最多 {int(self.tiles_max)} 个特写窗口（小人物下限 {self.secondary_ratio:.2f}）"
        )


def whole_frame_box(frame_size: tuple[float, float]) -> CropBox:
    """整幅画面（"不发生裁剪"的取景框）。"""
    return CropBox(0.0, 0.0, float(frame_size[0]), float(frame_size[1]))


def fade_alpha(transition_progress: float) -> float:
    """特写窗口的不透明度：过渡前半段为 0，后半段线性淡入到 1。"""
    value = min(max(float(transition_progress), 0.0), 1.0)
    if value <= FADE_START:
        return 0.0
    return min((value - FADE_START) / (1.0 - FADE_START), 1.0)


class TilesPlanner:
    """「全景 + 特写」的排布：谁进特写、每格多大、过渡时怎么淡入。"""

    def __init__(
        self,
        ratio: AspectRatio,
        *,
        params: FramingParams = DEFAULT_FRAMING_PARAMS,
        policy: MultiPersonPolicy | None = None,
        min_person_height_ratio: float = DEFAULT_MIN_PERSON_HEIGHT_RATIO,
        secondary_ratio: float = DEFAULT_SECONDARY_PERSON_RATIO,
        max_details: int = DEFAULT_TILES_MAX,
        annotate: bool = False,
    ) -> None:
        self.ratio = ratio
        self.out_size: tuple[int, int] = ratio.target_size
        self.params = params
        self.policy = policy or DEFAULT_MULTI_PERSON_POLICY
        self.min_person_height_ratio = float(min_person_height_ratio)
        self.secondary_ratio = float(secondary_ratio)
        self.max_details = max(int(max_details), 1)
        self.annotate = bool(annotate)

        self._details: list[Detection] = []
        self._rects: tuple[tuple[int, int, int, int], ...] = ()
        self._cache_key: tuple[tuple[float, float], int] | None = None

    # ------------------------------------------------------------------ 选人
    def update(
        self, detections: Sequence[Detection], frame_size: tuple[float, float]
    ) -> int:
        """关键帧：重新挑"次要小人物"（够大才给特写），返回挑中的人数。"""
        frame_h = float(frame_size[1])
        ceiling = self.min_person_height_ratio * frame_h
        floor = max(self.secondary_ratio, 0.0) * frame_h
        candidates = [
            item
            for item in detections
            if item.width > 0 and item.height > 0 and floor <= item.height < ceiling
        ]
        candidates.sort(key=lambda item: item.height, reverse=True)
        self._details = candidates[: self.max_details]
        return len(self._details)

    @property
    def details(self) -> int:
        """当前有几个特写窗口。"""
        return len(self._details)

    # ------------------------------------------------------------------ 排布
    def dst_rect(self, frame_size: tuple[float, float]) -> tuple[int, int, int, int] | None:
        """全景窗口在画布上的格子；没有空间（或没有小人物）时返回 ``None``。

        返回 ``None`` 时调用方应退回"全画面适配"：宁可画面小一点，
        也不要为了塞特写把全景挤成一条。
        """
        rects = self._resolve(frame_size)
        return rects[0] if rects else None

    def windows(
        self, frame_size: tuple[float, float], transition: TransitionFrame
    ) -> list[WindowSlice] | None:
        """本帧要贴的窗口（**先特写、后全景**，顺序即贴图顺序）。

        全景窗口最后贴，是为了在过渡期间让它盖住下面的特写：画面缓缓上移、
        特写随之露出，比"版式瞬间切换"自然。
        """
        rects = self._resolve(frame_size)
        if not rects:
            return None

        alpha = fade_alpha(transition.progress)
        windows: list[WindowSlice] = []
        for detection, rect in zip(self._details, rects[1:], strict=True):
            box = compute_upper_body_box(
                detection.bbox,
                frame_size,
                cell_ratio(rect),
                self.params,
                build_anchors(detection.keypoints, detection.bbox),
            )
            windows.append(WindowSlice(box, rect, detection.bbox, fit=False, alpha=alpha))
        windows.append(
            WindowSlice(whole_frame_box(frame_size), transition.dst, fit=True, alpha=1.0)
        )
        return windows

    def describe(self) -> str:
        return f"全景 + 特写（当前 {self.details} 个特写窗口）"

    # ------------------------------------------------------------------ 内部
    def _resolve(
        self, frame_size: tuple[float, float]
    ) -> tuple[tuple[int, int, int, int], ...]:
        """（并按需重建）当前帧尺寸与人数下的格子划分。"""
        key = (float(frame_size[0]), float(frame_size[1])), self.details
        if key == self._cache_key:
            return self._rects
        self._cache_key = key
        self._rects = ()

        if self.details <= 0:
            return self._rects

        gap = self.policy.gap_px(self.out_size)
        layout = tiles_layout(
            self.out_size, frame_size, self.details, gap=gap, min_cell_px=MIN_CELL_PX
        )
        if layout is None:
            return self._rects
        self._rects = compute_cell_rects(layout, self.out_size, gap)
        return self._rects


__all__ = [
    "DEFAULT_SECONDARY_PERSON_RATIO",
    "DEFAULT_TILES_MAX",
    "FADE_START",
    "NoPersonDisplay",
    "TilesPlanner",
    "fade_alpha",
    "whole_frame_box",
]
