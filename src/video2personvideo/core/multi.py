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

还有第四件事：**每个窗口都必须"正常速度播放"**。窗口只有一格大、放大倍数高，
整幅画面里"跟得紧"的镜头放到小窗口里就变成了飞快地平移 / 推拉（看着像快进），
所以每个窗口的镜头改用一套按秒封顶、更懒、更柔和的参数，并且窗口"认人"
（不因说话人变化而互换内容）、换人时直接切镜头而不是横扫 ——
详见 :class:`WindowCamera` 与 :class:`MultiPersonComposer`。

第五件事是**人数抖动不能把画面晃出"闪烁"**：某一帧漏检一个人时，窗口既不消失、
也不重排、更不会退回单窗口 —— 那几帧由 :class:`PersonSlot` 继续占着自己的位置
（人物框保持不动、画面继续播放），布局只在连续多帧都确认人数真的变了之后才改
（加窗口 `switch_hold` 帧、撤窗口 `hold_frames` 帧）。
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
    DEFAULT_MIN_PERSON_HEIGHT_RATIO,
    CropBox,
    FramingParams,
    compute_upper_body_box,
    interpolate_boxes,
)
from .layout import (
    DEFAULT_MULTI_PERSON_POLICY,
    MAX_WINDOWS,
    MIN_CELL_PX,
    GridLayout,
    MultiPersonPolicy,
    cell_ratio,
    compute_cell_rects,
)
from .pose import Keypoints, build_anchors
from .ratio import AspectRatio
from .smoothing import (
    DEFAULT_SMOOTHING_PARAMS,
    DEFAULT_WINDOW_PAN_SPEED,
    BoxSmoother,
    SmoothingParams,
    window_smoothing_params,
)
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


@dataclass(frozen=True, slots=True)
class WindowCamera:
    """分屏小窗口的镜头策略（"每个窗口都按正常速度播放"的那几项）。

    三件事互不相干、可单独关掉，见 :class:`MultiPersonComposer` 的说明：

    * ``stable``：窗口镜头换成 :func:`~video2personvideo.core.smoothing.
      window_smoothing_params` 的"按秒封顶 + 更懒 + 更柔和"一套（默认开）；
      关掉 = 沿用用户「镜头跟随」档位的原参数（旧行为，小窗口里会显得很急）；
    * ``pan_speed``：稳镜头的平移上限 —— **每秒最多平移"窗口自身的几倍"**
      （默认 :data:`~video2personvideo.core.smoothing.DEFAULT_WINDOW_PAN_SPEED`
      = 半格/秒，越小越稳）；
    * ``seat_lock``：窗口"认人" —— 人物一旦入座就固定待在自己的窗口，不因为
      说话人变化 / 位置微动而跟别人换窗口（默认开）；关掉 = 每关键帧按
      说话人 / 空间顺序重排座位（旧行为）。
    """

    stable: bool = True
    pan_speed: float = DEFAULT_WINDOW_PAN_SPEED
    seat_lock: bool = True

    def describe(self) -> str:
        if not self.stable:
            return "沿用镜头档位（小窗口里会比较急）"
        text = f"稳镜头：每秒最多平移 {self.pan_speed:.2f} 个窗口宽度"
        text += "，入座后固定跟人" if self.seat_lock else "，座位按说话人 / 位置重排"
        return text


@dataclass(slots=True)
class PersonSlot:
    """一个人物在分屏里的"座位"：跨帧保持同一条轨迹与同一个平滑器。"""

    bbox: BBox
    smoother: BoxSmoother
    keypoints: Keypoints | None = None
    #: 连续多少帧没有匹配上（超过保持帧数就释放座位）
    missing: int = 0
    #: 当前占用的窗口编号（1 起，对应 ``rects`` 的下标 + 1）；0 = 还没入座
    seat: int = 0

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
    """一位候选人物的打分与归属（``slot`` 为空表示这帧才出现的新人）。

    ``frozen=True`` 表示"这一帧并没有检到这个人"：人物框沿用座位上的上一帧结果，
    只为把已经确认的布局撑住（见 :meth:`MultiPersonComposer._seat_plan`）。
    """

    detection: Detection
    score: float
    bonus: float
    slot: PersonSlot | None
    row_band: int
    center_x: float
    #: 本帧漏检、只是用上一帧的人物框把窗口撑住（窗口不动、画面继续播放）
    frozen: bool = False


class MultiPersonComposer:
    """多人分屏的决策与渲染：选人 → 排座 → 逐人取景 → 拼接。

    **每个窗口都要"正常速度播放"**：小窗口面积小、放大倍数高，同样的镜头运动在
    屏幕上比整幅画面显眼得多，所以这里和单窗口模式有两处刻意的差别（都可通过
    :class:`WindowCamera` 关掉，关掉即回到旧行为）：

    * **镜头更慢更稳**（``camera.stable``）：每个窗口的平滑参数由用户档位经
      :func:`~video2personvideo.core.smoothing.window_smoothing_params` 推导 ——
      平移 / 缩放按"每秒最多移动窗口自身的 ``camera.pan_speed`` 倍"封顶，
      死区更大（人物在窗口里晃动时镜头完全不动），平滑与加速度更柔和；
    * **窗口认人**（``camera.seat_lock``）：人物一旦入座就固定待在自己的窗口，
      说话人变化 / 位置微动都不会让两个窗口的内容互换；
    * **换人就位**：万不得已要换人（原人物离开、窗口数变化）时，窗口**直接就位**
      到新构图（切镜头），而不是以最高速度从旧人物"扫"过去 —— 后者正是观众看到的
      "画面在快进"；
    * **漏检不停格**（``switch_hold`` / ``hold_frames``）：某一帧没检到人
      （漏检、转身、短暂遮挡）**不会**让窗口消失、重排或退回单窗口 —— 布局要连续
      ``switch_hold`` 帧（撤窗口则要连续 ``hold_frames`` 帧）确认才真的改变，
      中间这几帧由"还在跟踪的座位"顶上（人物框保持不动、画面继续播放）。
      否则人数每帧抖一下，输出就会在「两格分屏」与「三格分屏」（或「分屏」与
      「整屏单人」）之间来回跳 —— 观众看到的就是画面不停地闪。
    """

    def __init__(
        self,
        ratio: AspectRatio,
        *,
        params: FramingParams = DEFAULT_FRAMING_PARAMS,
        smoother_params: SmoothingParams | None = None,
        weights: SubjectWeights = DEFAULT_WEIGHTS,
        policy: MultiPersonPolicy | None = None,
        min_person_height_ratio: float = DEFAULT_MIN_PERSON_HEIGHT_RATIO,
        min_person_sharpness: float = 0.0,
        annotate: bool = False,
        camera: WindowCamera | None = None,
    ) -> None:
        self.ratio = ratio
        self.params = params
        #: 用户档位（不含帧率）：窗口参数由它推导，故单独留一份基准
        self._base_params = replace(smoother_params or DEFAULT_SMOOTHING_PARAMS, fps=None)
        self.smoother_params = self._base_params
        #: 本片的分屏镜头策略（默认"稳镜头 + 窗口认人"）
        self.camera = camera or WindowCamera()
        #: 分屏窗口实际使用的平滑参数（稳定镜头开关关闭时 = 用户档位）
        self.window_params = self._window_params()
        #: 源视频帧率（校准每个窗口的镜头"每秒感受"），探测到元信息后由 set_fps 补上
        self.fps: float | None = None
        self.weights = weights
        self.policy = policy or DEFAULT_MULTI_PERSON_POLICY
        self.min_person_height_ratio = float(min_person_height_ratio)
        #: 主要人物的清晰度下限（与"够大"并列；见 ``core.sharpness``）
        self.min_person_sharpness = float(min_person_sharpness)
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
        self.smoother_params = replace(self._base_params, fps=self.fps)
        self.window_params = self._window_params()
        for slot in self._slots:
            # 座位上的平滑器先按窗口参数校准帧率，位置状态保持不变
            slot.smoother.set_params(self.window_params)

    def _window_params(self) -> SmoothingParams:
        """本帧起分屏窗口使用的平滑参数（"稳镜头"开关关闭时 = 用户档位）。"""
        if not self.camera.stable:
            return self.smoother_params
        return window_smoothing_params(self.smoother_params, pan_speed=self.camera.pan_speed)

    # ------------------------------------------------------------------ 关键帧
    def update(
        self,
        frame_size: tuple[float, float],
        detections: Sequence[Detection],
        bonuses: Mapping[BBox, float] | None = None,
    ) -> MultiPlan | None:
        """关键帧：重新选人排座，返回本帧的渲染计划。

        画面里不足两个主要人物时返回 ``None``（交给单窗口逻辑），但**不是立刻** ——
        先用上一帧的布局把这几帧撑住（本帧漏检的人保持自己的窗口不动、画面继续播放），
        只有连续多帧都凑不满才真的退出。否则"两人画面里有一人偶尔漏检"会让输出在
        「整屏单人」与「两格分屏」之间每帧来回跳，看上去就是画面不停地闪。

        座位与轨迹会保留，人物重新出现时能接着平滑。
        """
        self._prev_targets = self._cur_targets

        candidates = filter_small(
            detections, frame_size, self.min_person_height_ratio, self.min_person_sharpness
        )
        matches = self._match(candidates)
        ranked = self._rank(candidates, matches, frame_size, bonuses)

        # 本帧"够大够清晰"的人数决定想要几个窗口（不足两人 = 0 = 本轮退出分屏）。
        # 滞回在这里，不拿本帧的人数去夹它：夹一下就等于没有滞回 ——
        # 人数在 2 / 3 之间抖动时，布局会跟着 2 窗 ↔ 3 窗 每帧重排（整块画面闪）。
        wanted = self._max_fitting_count(len(ranked)) if len(ranked) >= 2 else 0
        count = min(self._stabilize(wanted), self.policy.limit())
        seats = self._seat_plan(ranked, count)
        if len(seats) < count:
            # 座位已被回收（这个人确实走了很久）：按真正坐得下的窗口数重排一次
            seats = self._seat_plan(ranked, len(seats))
        if len(seats) < 2:
            # 退出分屏：推进各座位的缺席计数，超时的释放
            self._advance_missing()
            self._reset_round()
            return None
        count = len(seats)

        layout, rects = self._layout_for(count)
        slots, snapped = self._assign(seats)

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
            (
                # 换人 / 换座位：直接切到新构图，不从旧人物身上"扫"过去
                slot.smoother.snap_to(target, frame_size)
                if snap
                else slot.smoother.update(target, frame_size)
            )
            for slot, target, snap in zip(slots, targets, snapped, strict=True)
        ]
        if any(snapped):
            # 插值基准也跟着就位：否则接下来的填充帧仍会从"上一个窗口的内容"滑过来
            previous = list(self._prev_targets)
            self._prev_targets = [
                target if snap or position >= len(previous) else previous[position]
                for position, (target, snap) in enumerate(zip(targets, snapped, strict=True))
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
        """本轮不做分屏（凑不满两个窗口）时推进各座位的"缺席"计数，超时的释放。"""
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

        if not ranked:  # 本帧一个人都没有（退出分屏的路上会走到这里）
            return []
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
        """人数滞回：连续若干帧都是新人数才真的换布局（``count = 0`` 表示退出分屏）。

        加窗口比撤窗口"急"：新出现的窗口只要 :attr:`MultiPersonPolicy.switch_hold` 帧
        确认（路人一闪而过不会立刻占一个格子），而**撤掉窗口**（人数少了一个 / 全走了）
        要等 :attr:`SmoothingParams.hold_frames` 帧 —— 漏检、转身、短暂遮挡都会让人数
        瞬间少一个，若立刻跟着改布局，观众看到的就是"画面在闪"。这与单窗口模式
        "丢失目标先保持上一帧"是同一套语义，也让布局与"座位保持多久"同步。
        """
        count = max(int(count), 0)
        if count == self._stable_count:
            self._pending_count = count
            self._pending_frames = 0
            return self._stable_count
        if count == self._pending_count:
            self._pending_frames += 1
        else:
            self._pending_count = count
            self._pending_frames = 1
        if self._stable_count == 0 or self._pending_frames >= self._hold_rounds(count):
            self._stable_count = count
            self._pending_frames = 0
        return self._stable_count

    def _hold_rounds(self, count: int) -> int:
        """这次人数变化需要连续多少帧确认（撤窗口比加窗口保守得多）。"""
        hold = max(int(self.policy.switch_hold), 1)
        if count >= self._stable_count:
            return hold
        return max(int(self.smoother_params.hold_frames), hold)

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

    def _seat_plan(
        self, ranked: Sequence[_Candidate], count: int
    ) -> list[tuple[int, _Candidate]]:
        """把候选人安排进 1~``count`` 号窗口，返回 ``[(座位号, 候选人), ...]``（按座位号升序）。

        默认（``camera.seat_lock``）**每个窗口认人**：只要这个人还在画面里，他就一直待在
        自己的窗口 —— 说话人换了、位置微动了都不会跟别人换窗口，于是窗口内容始终是同一个
        人，观众不会看到两个窗口忽然互换（那正是"画面像快进"的来源）。

        只有座位真的空出来（人物离开、窗口数变多）时，才由 ``ranked`` 的顺序
        （先说话人、再按画面位置）从最小的空位开始补 —— 新来的人自然坐到空位上。
        关掉 ``seat_lock`` 则退回旧行为：完全按 ``ranked`` 顺序占 1~``count`` 号窗口。

        本帧凑不满 ``count`` 个人时（有人漏检），空位由**还在跟踪的座位**顶上
        （``frozen``：人物框沿用座位上的上一帧结果，窗口不动、画面继续播放）——
        漏检一两帧不会让布局跳一下。跟踪也已超时的座位不算，所以这一档最多撑
        ``hold_frames`` 帧，与布局滞回的"撤窗口"门槛一致。
        """
        limit = min(int(count), self.policy.limit())
        if limit < 1:
            return []
        if not self.camera.seat_lock:
            seats = [(seat, item) for seat, item in enumerate(ranked[:limit], start=1)]
            seats += self._standby_seats(seats, limit)
            return sorted(seats)

        seated: dict[int, _Candidate] = {}
        used: set[int] = set()
        # ① 老面孔待在自己的窗口里
        for item in ranked:
            slot = item.slot
            if slot is None or id(slot) in used:
                continue
            if 1 <= slot.seat <= limit and slot.seat not in seated:
                seated[slot.seat] = item
                used.add(id(slot))
        # ② 新人（或原座位已不在窗口数内的人）坐进最小的空位
        for item in ranked:
            if item.slot is not None and id(item.slot) in used:
                continue
            free = _first_free_seat(seated, limit)
            if free is None:
                break
            seated[free] = item
            if item.slot is not None:
                used.add(id(item.slot))
        # ③ 还空着的窗口：用"本帧漏检、但仍在保持期内"的座位顶上（窗口冻结）
        if len(seated) < limit:
            for slot in self._standby_slots(used):
                own = slot.seat if 1 <= slot.seat <= limit and slot.seat not in seated else 0
                free = own or _first_free_seat(seated, limit)
                if free is None:
                    break
                seated[free] = self._frozen_candidate(slot)
                used.add(id(slot))
        return sorted(seated.items())

    def _standby_seats(
        self, seats: list[tuple[int, _Candidate]], limit: int
    ) -> list[tuple[int, _Candidate]]:
        """关掉「窗口认人」时给空位补上"本帧漏检、但仍在保持期内"的座位（窗口冻结）。

        座位顺序仍是旧的"按 ``ranked`` 顺序占 1~N 号窗口"，只是不再因为有人漏检
        而把窗口数缩下来（那同样会让画面闪）。
        """
        used = {id(item.slot) for _, item in seats if item.slot is not None}
        extra: list[tuple[int, _Candidate]] = []
        for slot in self._standby_slots(used):
            if len(seats) + len(extra) >= limit:
                break
            extra.append((len(seats) + len(extra) + 1, self._frozen_candidate(slot)))
        return extra

    def _standby_slots(self, used: set[int]) -> list[PersonSlot]:
        """本帧没检到、但还在保持期内的座位（越新越优先顶上）。"""
        keep = max(int(self.smoother_params.hold_frames), 0)
        standby = [
            slot
            for slot in self._slots
            if id(slot) not in used and slot.missing <= keep and slot.box is not None
        ]
        standby.sort(key=lambda slot: (slot.missing, slot.seat if slot.seat else MAX_WINDOWS))
        return standby

    def _frozen_candidate(self, slot: PersonSlot) -> _Candidate:
        """本帧漏检的人：沿用座位上的上一帧人物框，让窗口先撑住（镜头不动、画面继续播放）。"""
        detection = Detection(bbox=slot.bbox, confidence=0.0, keypoints=slot.keypoints)
        return _Candidate(
            detection=detection,
            score=0.0,
            bonus=0.0,
            slot=slot,
            row_band=0,
            center_x=detection.center[0],
            frozen=True,
        )

    def _assign(
        self, plan: Sequence[tuple[int, _Candidate]]
    ) -> tuple[list[PersonSlot], list[bool]]:
        """按排座结果把候选人放进座位；第二个返回值表示该窗口是否要"直接就位"。

        需要就位的两种情况：新入座的人（``slot is None``）、以及窗口换了占用者
        （``slot.seat`` 与目标座位不一致）—— 后者若走平滑，就是一次横跨画面的高速
        平移，看起来像"快进"，因此一律切镜头。

        ``frozen`` 的人（本帧漏检）只把"缺席"记上：人物框、窗口位置都保持上一帧的样子，
        这样画面既不跳也不闪；缺席累计超过 ``hold_frames`` 的座位会被回收。
        """
        slots: list[PersonSlot] = []
        snapped: list[bool] = []
        for seat, item in plan:
            slot = item.slot
            if slot is None:
                slot = PersonSlot(
                    bbox=item.detection.bbox,
                    smoother=BoxSmoother(
                        self.ratio, params=self.window_params, fps=self.fps
                    ),
                    seat=seat,
                )
                self._slots.append(slot)
                snapped.append(True)
            else:
                snapped.append(slot.seat != seat)
                slot.seat = seat
            if item.frozen:
                slot.missing += 1
            else:
                slot.bbox = item.detection.bbox
                slot.keypoints = item.detection.keypoints
                slot.missing = 0
            slots.append(slot)

        used = {id(slot) for slot in slots}
        for slot in self._slots:
            if id(slot) not in used:
                slot.missing += 1
        self._prune_slots()
        return slots, snapped

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
    min_person_height_ratio: float = DEFAULT_MIN_PERSON_HEIGHT_RATIO,
    min_person_sharpness: float = 0.0,
) -> list[WindowSlice] | None:
    """无时序的多人分屏规划：单帧选人 → 排窗口。

    与 :class:`MultiPersonComposer` 的区别是不做轨迹关联与时序平滑 ——
    预览只想看"这一帧会排成什么样"。画面里不足两个主要人物时返回 ``None``。
    """
    active = policy or DEFAULT_MULTI_PERSON_POLICY
    candidates = filter_small(
        detections, frame_size, min_person_height_ratio, min_person_sharpness
    )
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
    min_person_height_ratio: float = DEFAULT_MIN_PERSON_HEIGHT_RATIO,
    min_person_sharpness: float = 0.0,
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
        min_person_sharpness=min_person_sharpness,
    )
    if windows is None:
        return None
    return compose_multi_frame(
        frame, windows, ratio.target_size, background=active.background, annotate=annotate
    )


def _first_free_seat(seated: Mapping[int, _Candidate], limit: int) -> int | None:
    """1~``limit`` 号窗口里最小的空位；坐满了返回 ``None``。"""
    for seat in range(1, int(limit) + 1):
        if seat not in seated:
            return seat
    return None


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
    "WindowCamera",
    "compose_static_multi",
    "plan_static_multi",
]
