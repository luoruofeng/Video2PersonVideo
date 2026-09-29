"""多人分屏合成（时序部分）：把"画面里的多个人"变成"一帧里的多个上半身窗口"。

和单窗口裁剪相比，这里多出三件事：

1. **选人**：过滤掉过小的人物（背景里的小人、路人），剩下的按主角打分取前
   ``max_persons`` 个 —— 保证屏幕上不会挤满不认识的人；
2. **排座**：默认让"正在说话的人"（如果有判定结果）坐 1 号窗口，其余人按空间位置
   （先上后下、先左后右）依次入座，于是**左的人在左窗口**，画面不会莫名跳来跳去；
3. **追踪**：每个人用自己的"座位"记住上一帧的位置与平滑器，靠 IoU / 中心距离做
   跨帧关联 —— 每个人的镜头都独立平滑，不会因为排在旁边的人动了而跟着晃。

窗口形状由 :mod:`~video2personvideo.core.layout` 的网格决定，每个窗口**按自己的
宽高比**取景（``cell_ratio``），所以窄窗口切出来仍是等比的上半身，不会拉伸变形。

取景框本身由 :func:`~video2personvideo.core.framing.compute_upper_body_box` 生成：
它不看人物在画面里的大小分档，**一律按"恰好装下上半身"取景**，因此人物在画面里
占比很小时也只会切出上半身，而不会把腿部与左右大片非人物场景一起框进来。
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import NamedTuple

import numpy as np

from ..utils.logger import get_logger
from .crop import WindowSlice, compose_multi_frame
from .framing import (
    DEFAULT_FRAMING_PARAMS,
    CropBox,
    FramingParams,
    compute_upper_body_box,
    interpolate_boxes,
)
from .layout import (
    DEFAULT_MULTI_PERSON_POLICY,
    MIN_CELL_PX,
    GridLayout,
    MultiPersonPolicy,
    cell_ratio,
    compute_cell_rects,
)
from .pose import Keypoints, build_anchors
from .ratio import AspectRatio
from .smoothing import DEFAULT_SMOOTHING_PARAMS, BoxSmoother, SmoothingParams
from .subject import (
    DEFAULT_WEIGHTS,
    BBox,
    Detection,
    SubjectWeights,
    filter_small,
    iou,
    score_detection,
)

logger = get_logger(__name__)

#: 跨帧关联一个人所需的 IoU 下限（低于它才考虑用中心距离兜底）
MATCH_IOU = 0.15

#: 中心距离兜底：距离小于"半个框"时也认为同一个人（人动得快时 IoU 会掉到 0）
MATCH_DISTANCE = 0.5

#: 判断"同一水平带"用的高度比例：两个人中心高度差小于它时算同一排
ROW_BAND_RATIO = 0.30


@dataclass(slots=True)
class PersonSlot:
    """一个人物在分屏里的"座位"：跨帧保持同一条轨迹与同一个平滑器。"""

    bbox: BBox
    smoother: BoxSmoother
    keypoints: Keypoints | None = None
    #: 连续多少帧没有匹配上（超过保持帧数就释放座位）
    missing: int = 0

    @property
    def box(self) -> CropBox | None:
        return self.smoother.box


@dataclass(slots=True)
class MultiPlan:
    """多人分屏一帧的渲染计划（取景框已完成平滑 / 插值）。"""

    boxes: list[CropBox]
    subjects: list[BBox | None]
    layout: GridLayout
    rects: tuple[tuple[int, int, int, int], ...]

    @property
    def count(self) -> int:
        """本帧实际显示了几个窗口。"""
        return len(self.boxes)


class _Candidate(NamedTuple):
    """一位候选人物的打分与归属（``slot`` 为空表示这帧才出现的新人）。"""

    detection: Detection
    score: float
    bonus: float
    slot: PersonSlot | None
    row_band: int
    center_x: float


class MultiPersonComposer:
    """多人分屏的决策与渲染：选人 → 排座 → 逐人取景 → 拼接。"""

    def __init__(
        self,
        ratio: AspectRatio,
        *,
        params: FramingParams = DEFAULT_FRAMING_PARAMS,
        smoother_params: SmoothingParams | None = None,
        weights: SubjectWeights = DEFAULT_WEIGHTS,
        policy: MultiPersonPolicy | None = None,
        min_person_height_ratio: float = 0.08,
        annotate: bool = False,
    ) -> None:
        self.ratio = ratio
        self.params = params
        self.smoother_params = smoother_params or DEFAULT_SMOOTHING_PARAMS
        #: 源视频帧率（校准每个窗口的镜头"每秒感受"），探测到元信息后由 set_fps 补上
        self.fps: float | None = None
        self.weights = weights
        self.policy = policy or DEFAULT_MULTI_PERSON_POLICY
        self.min_person_height_ratio = float(min_person_height_ratio)
        self.annotate = bool(annotate)

        self.out_size: tuple[int, int] = ratio.target_size
        self.gap = self.policy.gap_px(self.out_size)

        self._slots: list[PersonSlot] = []
        #: 上一个 / 当前关键帧的**未平滑**目标取景框（填充帧在两者之间插值）
        self._prev_targets: list[CropBox] = []
        self._cur_targets: list[CropBox] = []
        self._cur_subjects: list[BBox | None] = []
        self._active_slots: list[PersonSlot] = []
        self._cur_layout: GridLayout | None = None
        self._cur_rects: tuple[tuple[int, int, int, int], ...] = ()
        self._layout_cache: dict[tuple[str, int], tuple[GridLayout, tuple[tuple[int, int, int, int], ...]]] = {}

        self._stable_count = 0
        self._pending_count = 0
        self._pending_frames = 0

        #: 统计：真正走多人分屏的帧数 / 各布局用了多少帧 / 单帧最多几个窗口
        self.multi_frames = 0
        self.layout_counts: Counter = Counter()
        self.windows_peak = 0

    def set_fps(self, fps: float | None) -> None:
        """告知源视频真实帧率：已有座位的平滑器与后续新座位都按它校准镜头速度。"""
        if fps is not None and float(fps) <= 0.0:
            return
        self.fps = None if fps is None else float(fps)
        self.smoother_params = replace(self.smoother_params, fps=self.fps)
        for slot in self._slots:
            slot.smoother.set_fps(self.fps)

    # ------------------------------------------------------------------ 关键帧
    def update(
        self,
        frame_size: tuple[float, float],
        detections: Sequence[Detection],
        bonuses: Mapping[BBox, float] | None = None,
    ) -> MultiPlan | None:
        """关键帧：重新选人排座，返回本帧的渲染计划。

        画面里不足两个主要人物时返回 ``None``（交给单窗口逻辑），
        但座位与轨迹会保留，人物重新出现时能接着平滑。
        """
        self._prev_targets = self._cur_targets

        candidates = filter_small(detections, frame_size, self.min_person_height_ratio)
        if len(candidates) < 2:
            self._reset_round()
            self._advance_missing()
            return None

        matches = self._match(candidates)
        ranked = self._rank(candidates, matches, frame_size, bonuses)
        ceiling = self._max_fitting_count(len(ranked))
        count = min(self._stabilize(ceiling), ceiling)

        layout, rects = self._layout_for(count)
        slots = self._assign(ranked[:count])

        targets: list[CropBox] = []
        subjects: list[BBox | None] = []
        for slot, rect in zip(slots, rects, strict=True):
            slot.smoother.ratio = cell_ratio(rect)  # 窗口换形状时，窗口自己的比例也要换
            anchors = build_anchors(slot.keypoints, slot.bbox)
            targets.append(
                compute_upper_body_box(
                    slot.bbox, frame_size, slot.smoother.ratio, self.params, anchors
                )
            )
            subjects.append(slot.bbox)

        boxes = [
            slot.smoother.update(target, frame_size)
            for slot, target in zip(slots, targets, strict=True)
        ]

        self._cur_targets = targets
        self._cur_subjects = subjects
        self._active_slots = slots
        self._cur_layout = layout
        self._cur_rects = rects
        self.windows_peak = max(self.windows_peak, len(boxes))
        return MultiPlan(boxes=boxes, subjects=subjects, layout=layout, rects=rects)

    # ------------------------------------------------------------------ 填充帧
    def interpolate(
        self, t: float, frame_size: tuple[float, float]
    ) -> MultiPlan | None:
        """填充帧：在上一关键帧与本关键帧之间插值，再走各自座位的平滑。"""
        if not self._cur_targets or not self._active_slots or self._cur_layout is None:
            return None

        boxes: list[CropBox] = []
        for position, (slot, target) in enumerate(
            zip(self._active_slots, self._cur_targets, strict=True)
        ):
            previous = self._prev_targets[position] if position < len(self._prev_targets) else None
            ratio = cell_ratio(self._cur_rects[position])
            mixed = interpolate_boxes(previous, target, t, ratio)
            if mixed is None:  # pragma: no cover - target 一定非空
                mixed = target
            boxes.append(slot.smoother.update(mixed, frame_size))

        return MultiPlan(
            boxes=boxes,
            subjects=list(self._cur_subjects),
            layout=self._cur_layout,
            rects=self._cur_rects,
        )

    # ------------------------------------------------------------------ 渲染
    def windows(self, plan: MultiPlan) -> list[WindowSlice]:
        """把渲染计划摊平成"从哪切 / 贴到哪 / 是谁"的窗口列表。"""
        return [
            WindowSlice(box, rect, subject)
            for box, rect, subject in zip(plan.boxes, plan.rects, plan.subjects, strict=True)
        ]

    def render(self, frame: np.ndarray, plan: MultiPlan) -> np.ndarray:
        """按渲染计划把源帧拼成输出帧（尺寸恒等于 ``ratio.target_size``）。"""
        return compose_multi_frame(
            frame,
            self.windows(plan),
            self.out_size,
            background=self.policy.background,
            annotate=self.annotate,
        )

    def note_frame(self, plan: MultiPlan) -> None:
        """记一笔统计（由流水线在真正输出多人帧时调用）。"""
        self.multi_frames += 1
        self.layout_counts[plan.layout.describe()] += 1

    @property
    def slots(self) -> int:
        """当前还在跟踪的座位数（调试 / 测试用）。"""
        return len(self._slots)

    def describe(self) -> str:
        return self.policy.describe()

    # ------------------------------------------------------------------ 内部
    def _reset_round(self) -> None:
        """本帧不足两人：清掉本轮目标，但保留座位轨迹。"""
        self._cur_targets = []
        self._cur_subjects = []
        self._active_slots = []
        self._cur_layout = None
        self._cur_rects = ()
        self._stable_count = 0
        self._pending_count = 0
        self._pending_frames = 0

    def _advance_missing(self) -> None:
        """没有任何候选时推进各座位的"缺席"计数，超时的释放。"""
        for slot in self._slots:
            slot.missing += 1
        self._prune_slots()

    def _prune_slots(self) -> None:
        keep = max(int(self.smoother_params.hold_frames), 0)
        self._slots = [slot for slot in self._slots if slot.missing <= keep]

    def _rank(
        self,
        candidates: Sequence[Detection],
        matches: Mapping[int, PersonSlot],
        frame_size: tuple[float, float],
        bonuses: Mapping[BBox, float] | None,
    ) -> list[_Candidate]:
        """打分并按策略排出窗口顺序。

        打分沿用单窗口那套「面积 + 居中 + 时序 + 置信度」，只是"时序连续"改成
        用**这个人自己座位上一帧的位置**（比"跟上一帧主角的 IoU"更准确）。
        额外偏置来自说话人判定，量纲一致（≥1.0 即必定坐在 1 号窗口）。
        """
        frame_h = float(frame_size[1])
        band = max(frame_h * ROW_BAND_RATIO, 1.0)
        ranked: list[_Candidate] = []
        for position, detection in enumerate(candidates):
            slot = matches.get(position)
            bonus = float(bonuses.get(tuple(detection.bbox), 0.0)) if bonuses else 0.0
            score = score_detection(
                detection,
                frame_size,
                prev_bbox=slot.bbox if slot is not None else None,
                weights=self.weights,
                bonus=bonus,
            )
            center_x, center_y = detection.center
            ranked.append(
                _Candidate(
                    detection=detection,
                    score=score,
                    bonus=bonus,
                    slot=slot,
                    row_band=int(center_y / band),
                    center_x=center_x,
                )
            )

        limit = self.policy.limit()
        chosen = sorted(ranked, key=lambda item: item.score, reverse=True)[:limit]
        if self.policy.order == "score":
            return chosen

        spatial = sorted(chosen, key=lambda item: (item.row_band, item.center_x))
        speaker = max(chosen, key=lambda item: item.bonus)
        if speaker.bonus > 0.0:
            # 正在说话的人坐 1 号窗口（通常是最大 / 最靠上的那个），其余按空间顺序
            return [speaker, *(item for item in spatial if item is not speaker)]
        return spatial

    def _max_fitting_count(self, available: int) -> int:
        """在"每个窗口都不至于小到看不清"的前提下，最多能同时显示几个人。

        输出分辨率偏低（或人多窗口密）时，与其把画面切成看不清的小方块，
        不如少显示几个人 —— 每个人的上半身还能看清楚。下限是 2 人。
        """
        ceiling = min(int(available), self.policy.limit())
        for count in range(ceiling, 2, -1):
            if self._min_cell_side(count) >= MIN_CELL_PX:
                return count
        return min(ceiling, 2)

    def _min_cell_side(self, count: int) -> int:
        """某个布局里最窄格子的短边（像素）。"""
        _, rects = self._layout_for(count)
        return min(min(rect[2], rect[3]) for rect in rects)

    def _stabilize(self, count: int) -> int:
        """人数滞回：连续 ``switch_hold`` 帧都是新人数才真的换布局。"""
        hold = max(int(self.policy.switch_hold), 1)
        if count == self._stable_count:
            self._pending_count = count
            self._pending_frames = 0
            return self._stable_count
        if count == self._pending_count:
            self._pending_frames += 1
        else:
            self._pending_count = count
            self._pending_frames = 1
        if self._stable_count == 0 or self._pending_frames >= hold:
            self._stable_count = count
            self._pending_frames = 0
        return self._stable_count

    def _match(self, candidates: Sequence[Detection]) -> dict[int, PersonSlot]:
        """把本帧的候选人物关联到已有座位（IoU 优先，中心距离兜底，贪心匹配）。"""
        pairs: list[tuple[float, int, int]] = []
        for position, detection in enumerate(candidates):
            for index, slot in enumerate(self._slots):
                affinity = _affinity(detection, slot)
                if affinity >= MATCH_IOU:
                    pairs.append((affinity, position, index))
        pairs.sort(reverse=True)

        matched: dict[int, PersonSlot] = {}
        taken: set[int] = set()
        for _, position, index in pairs:
            if position in matched or index in taken:
                continue
            matched[position] = self._slots[index]
            taken.add(index)
        return matched

    def _assign(self, chosen: Sequence[_Candidate]) -> list[PersonSlot]:
        """把选中的候选人放进座位（能复用上一帧的座位就复用）。"""
        slots: list[PersonSlot] = []
        for item in chosen:
            slot = item.slot
            if slot is None:
                slot = PersonSlot(
                    bbox=item.detection.bbox,
                    smoother=BoxSmoother(
                        self.ratio, params=self.smoother_params, fps=self.fps
                    ),
                )
                self._slots.append(slot)
            slot.bbox = item.detection.bbox
            slot.keypoints = item.detection.keypoints
            slot.missing = 0
            slots.append(slot)

        used = {id(slot) for slot in slots}
        for slot in self._slots:
            if id(slot) not in used:
                slot.missing += 1
        self._prune_slots()
        return slots

    def _layout_for(self, count: int) -> tuple[GridLayout, tuple[tuple[int, int, int, int], ...]]:
        key = (self.ratio.name, count)
        cached = self._layout_cache.get(key)
        if cached is None:
            layout = self.policy.layout_for(self.ratio, count)
            cached = (layout, compute_cell_rects(layout, self.out_size, self.gap))
            self._layout_cache[key] = cached
        return cached


def plan_static_multi(
    detections: Sequence[Detection],
    frame_size: tuple[float, float],
    ratio: AspectRatio,
    *,
    params: FramingParams = DEFAULT_FRAMING_PARAMS,
    policy: MultiPersonPolicy | None = None,
    min_person_height_ratio: float = 0.08,
) -> list[WindowSlice] | None:
    """无时序的多人分屏规划：单帧选人 → 排窗口。

    与 :class:`MultiPersonComposer` 的区别是不做轨迹关联与时序平滑 ——
    预览只想看"这一帧会排成什么样"。画面里不足两个主要人物时返回 ``None``。
    """
    active = policy or DEFAULT_MULTI_PERSON_POLICY
    candidates = filter_small(detections, frame_size, min_person_height_ratio)
    if len(candidates) < 2:
        return None

    band = max(float(frame_size[1]) * ROW_BAND_RATIO, 1.0)
    ordered = sorted(candidates, key=lambda item: (int(item.center[1] / band), item.center[0]))[
        : active.limit()
    ]

    layout = active.layout_for(ratio, len(ordered))
    out_size = ratio.target_size
    rects = compute_cell_rects(layout, out_size, active.gap_px(out_size))

    windows: list[WindowSlice] = []
    for detection, rect in zip(ordered, rects, strict=True):
        windows.append(
            WindowSlice(
                compute_upper_body_box(
                    detection.bbox,
                    frame_size,
                    cell_ratio(rect),
                    params,
                    build_anchors(detection.keypoints, detection.bbox),
                ),
                rect,
                detection.bbox,
            )
        )
    return windows


def compose_static_multi(
    frame: np.ndarray,
    detections: Sequence[Detection],
    ratio: AspectRatio,
    *,
    params: FramingParams = DEFAULT_FRAMING_PARAMS,
    policy: MultiPersonPolicy | None = None,
    min_person_height_ratio: float = 0.08,
    annotate: bool = False,
) -> np.ndarray | None:
    """无时序的多人分屏（构图预览用）：规划 + 拼接一步到位。"""
    active = policy or DEFAULT_MULTI_PERSON_POLICY
    frame_size = (float(frame.shape[1]), float(frame.shape[0]))
    windows = plan_static_multi(
        detections,
        frame_size,
        ratio,
        params=params,
        policy=active,
        min_person_height_ratio=min_person_height_ratio,
    )
    if windows is None:
        return None
    return compose_multi_frame(
        frame, windows, ratio.target_size, background=active.background, annotate=annotate
    )


def _affinity(detection: Detection, slot: PersonSlot) -> float:
    """一个人与一个座位的"像不像同一个人"：IoU 优先，中心距离兜底。"""
    overlap = iou(detection.bbox, slot.bbox)
    if overlap >= MATCH_IOU:
        return overlap

    center_x, center_y = detection.center
    slot_x, slot_y = (
        (slot.bbox[0] + slot.bbox[2]) / 2.0,
        (slot.bbox[1] + slot.bbox[3]) / 2.0,
    )
    scale = max(
        detection.width,
        detection.height,
        abs(slot.bbox[2] - slot.bbox[0]),
        abs(slot.bbox[3] - slot.bbox[1]),
        1.0,
    )
    distance = math.hypot(center_x - slot_x, center_y - slot_y) / scale
    if distance < MATCH_DISTANCE:
        # 中心距离越近越像，但始终略低于"IoU 达标"的强度，避免抢走真正的重叠匹配
        return max(overlap, 1.0 - distance)
    return overlap


__all__ = [
    "MATCH_IOU",
    "MultiPersonComposer",
    "MultiPlan",
    "PersonSlot",
    "compose_static_multi",
    "plan_static_multi",
]
