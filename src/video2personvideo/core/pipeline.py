"""单视频裁剪流水线：检测 → 选主角 → 构图 → 平滑 → 裁剪。

这个模块把 M1 的各个纯函数模块串起来，并实现 M1-7 的两项加速：

* **抽帧检测 + 插值追踪**：每 ``detect_interval`` 帧跑一次 YOLO，
  中间帧在相邻两次检测结果之间做线性插值；
* **批量帧推理**：攒够 ``infer_batch`` 个关键帧后一次性送进模型
  （``detect_boxes_batch``），CPU/GPU 都能减少每帧的调用开销。

输出尺寸只在初始化时确定一次，全程不变（不变量 2）。
帧的发射顺序与逐帧检测时完全一致，只是延迟可能增加 ``infer_batch`` 个关键帧。

多人物场景下可挂上"跟随正在说话的人"（可选）：
关键帧上先让 :class:`~video2personvideo.core.mouth.MouthActivityAnalyzer` 量出
每个人的嘴动强度，再由 :class:`~video2personvideo.core.speaker.ActiveSpeakerSelector`
结合语音能量包络判定说话人，判定结果作为主角打分的偏置传进去（见 ``bonuses``）。
这两件东西都是可选的，缺失时行为与以前完全一致。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from ..utils.logger import get_logger
from .crop import WindowSlice, crop_frame
from .framing import (
    DEFAULT_FRAMING_PARAMS,
    DEFAULT_MIN_PERSON_HEIGHT_RATIO,
    MODE_MULTI,
    CropBox,
    FramingParams,
    compute_target_box,
    interpolate_boxes,
    mode_label,
)
from .layout import MultiPersonPolicy
from .mouth import FaceLocator, MouthActivityAnalyzer
from .multi import MultiPersonComposer, MultiPlan
from .pose import build_anchors
from .ratio import AspectRatio
from .smoothing import BoxSmoother
from .speaker import ActiveSpeakerSelector
from .subject import DEFAULT_WEIGHTS, SubjectWeights, filter_small, select_subject

logger = get_logger(__name__)


@dataclass(slots=True)
class FrameOutcome:
    """一帧的最终产出。"""

    index: int
    frame: np.ndarray
    person_count: int
    has_person: bool
    mode: str
    box: CropBox | None
    subject_bbox: tuple[float, float, float, float] | None = None
    #: 多人分屏时的各窗口（空列表 = 本帧是单窗口 / 兜底画面）
    windows: list[WindowSlice] = field(default_factory=list)

    @property
    def mode_label(self) -> str:
        return mode_label(self.mode)

    @property
    def is_multi(self) -> bool:
        """本帧是否走了多人分屏。"""
        return bool(self.windows)

    @property
    def window_count(self) -> int:
        """本帧显示了几个窗口（单窗口帧为 1，无人兜底帧为 0）。"""
        return len(self.windows) if self.windows else (1 if self.box is not None else 0)


@dataclass(slots=True)
class PipelineStats:
    """流水线统计（写进 ``ProcessResult`` 供摘要展示）。

    "保持 / 回中"帧数由 :class:`~video2personvideo.core.smoothing.BoxSmoother`
    维护，通过 :class:`CropPipeline` 的 ``hold_frames`` / ``center_frames`` 属性读取，
    这里不重复记账，避免两份真相。
    """

    frames: int = 0
    detected_frames: int = 0
    interpolated_frames: int = 0
    #: 实际发起过多少次推理（批量推理时小于 ``detected_frames``）
    inference_calls: int = 0
    #: 单次推理最多算了几帧
    max_batch: int = 0
    #: 判定出"正在说话的人"并据此选主角的关键帧数（未启用说话人跟随时恒为 0）
    speaker_frames: int = 0
    #: 真正以多人分屏输出的帧数（未启用多人分屏、或画面里只有一个主要人物时恒为 0）
    multi_frames: int = 0
    #: 单帧最多同时显示了几个窗口
    windows_peak: int = 0
    modes: Counter = field(default_factory=Counter)

    def as_dict(self) -> dict[str, int]:
        return {mode: count for mode, count in self.modes.items() if count}


#: 队列元素类型：关键帧（需要推理） / 填充帧（靠插值）
_KEY = "key"
_FILL = "fill"


class CropPipeline:
    """把一帧帧原图变成一帧帧构图好的输出图。"""

    def __init__(
        self,
        ratio: AspectRatio,
        detector,
        *,
        params: FramingParams = DEFAULT_FRAMING_PARAMS,
        smoother: BoxSmoother | None = None,
        weights: SubjectWeights = DEFAULT_WEIGHTS,
        min_person_height_ratio: float = DEFAULT_MIN_PERSON_HEIGHT_RATIO,
        detect_interval: int = 1,
        infer_batch: int = 1,
        crop: bool = True,
        annotate: bool = False,
        speaker: ActiveSpeakerSelector | None = None,
        mouth: MouthActivityAnalyzer | None = None,
        speaker_weight: float = 0.0,
        multi: MultiPersonPolicy | None = None,
    ) -> None:
        self.ratio = ratio
        self.detector = detector
        self.params = params
        self.weights = weights
        self.min_person_height_ratio = float(min_person_height_ratio)
        self.detect_interval = max(int(detect_interval), 1)
        self.infer_batch = max(int(infer_batch), 1)
        self.crop = bool(crop)
        self.annotate = bool(annotate)
        self.out_size: tuple[int, int] = ratio.target_size
        self.smoother = smoother or BoxSmoother(ratio)
        self.stats = PipelineStats()
        #: 说话人跟随（可选）：判定器 + 嘴动测量器 + 偏置权重（0 = 不使用偏置）
        self.speaker = speaker
        self.mouth = (
            mouth
            if mouth is not None
            else (MouthActivityAnalyzer(face_locator=FaceLocator()) if speaker else None)
        )
        self.speaker_weight = max(float(speaker_weight), 0.0)

        #: 多人分屏（可选）：画面里有两个及以上主要人物时，每人一个上半身小窗口
        self.multi_policy = multi if (multi is not None and multi.enabled) else None
        self.composer: MultiPersonComposer | None = (
            MultiPersonComposer(
                ratio,
                params=params,
                smoother_params=self.smoother.params,
                weights=weights,
                policy=self.multi_policy,
                min_person_height_ratio=self.min_person_height_ratio,
                annotate=self.annotate,
            )
            if self.multi_policy is not None
            else None
        )

        self._prev_detect_index: int | None = None
        self._prev_target: CropBox | None = None
        self._prev_bbox: tuple[float, float, float, float] | None = None
        self._prev_count = 0
        #: 最近一次**入队**的关键帧下标（决定抽帧节奏，与是否已推理无关）
        self._last_key_index: int | None = None
        #: 待定稿的帧队列，元素为 ``(kind, index, frame)``
        self._queue: list[tuple[str, int, np.ndarray]] = []

    # ------------------------------------------------------------- 主流程
    def process(self, frame: np.ndarray, index: int) -> list[FrameOutcome]:
        """喂入一帧，返回**本帧之前所有已可以定稿**的输出帧（可能为空）。"""
        kind = _KEY if self._should_detect(index) else _FILL
        if kind == _KEY:
            self._last_key_index = index
        self._queue.append((kind, index, frame))
        if kind == _KEY and self._key_count() >= self.infer_batch:
            return self._consume()
        return []

    def flush(self) -> list[FrameOutcome]:
        """视频结束时定稿尾部：剩余关键帧补跑一次推理，纯填充帧沿用最后目标。"""
        outcomes = self._consume(force=True)
        outcomes.extend(self._render_tail(index, frame) for _, index, frame in self._queue)
        self._queue.clear()
        return outcomes

    @property
    def hold_frames(self) -> int:
        return self.smoother.holds

    @property
    def center_frames(self) -> int:
        return self.smoother.centers

    # ------------------------------------------------------------- 帧率校准
    def set_fps(self, fps: float | None) -> None:
        """告知源视频真实帧率，用于校准镜头跟随的"每秒感受"。

        单窗口与每个多人窗口的平滑器都会更新。应在**探测到视频元信息之后、
        开始逐帧处理之前**调用（构造流水线时还不知道帧率）；``None`` / 非法帧率
        一律保持原状，因此忘记调用只会退化成"按 30fps 标定"，不会出错。
        """
        self.smoother.set_fps(fps)
        if self.composer is not None:
            self.composer.set_fps(fps)

    # ------------------------------------------------------------- 内部逻辑
    def _should_detect(self, index: int) -> bool:
        if self.detect_interval <= 1 or self._last_key_index is None:
            return True
        return index - self._last_key_index >= self.detect_interval

    def _key_count(self) -> int:
        return sum(1 for kind, _, _ in self._queue if kind == _KEY)

    def _detect_batch(self, frames: list[np.ndarray]) -> list[list]:
        """批量推理；检测器没实现批量接口时自动退化为逐帧调用。"""
        batch_fn = getattr(self.detector, "detect_boxes_batch", None)
        if callable(batch_fn):
            results = list(batch_fn(frames))
        else:  # pragma: no cover - 仅在自定义检测器时走到
            results = [self.detector.detect_boxes(frame) for frame in frames]

        self.stats.inference_calls += 1
        self.stats.max_batch = max(self.stats.max_batch, len(frames))
        return results

    def _consume(self, *, force: bool = False) -> list[FrameOutcome]:
        """把队列里"到齐的关键帧"及其之间的填充帧定稿。

        只有到齐 ``infer_batch`` 个关键帧才推理；``force=True`` 时（视频结束）
        把剩下的关键帧一起推理，保证不漏检。
        """
        key_positions = [pos for pos, (kind, _, _) in enumerate(self._queue) if kind == _KEY]
        if not key_positions or (not force and len(key_positions) < self.infer_batch):
            return []

        cut = key_positions[-1] + 1
        segment, self._queue = self._queue[:cut], self._queue[cut:]

        key_frames = [frame for kind, _, frame in segment if kind == _KEY]
        results = self._detect_batch(key_frames)

        outcomes: list[FrameOutcome] = []
        pending_fill: list[tuple[int, np.ndarray]] = []
        result_index = 0

        for kind, index, frame in segment:
            if kind == _FILL:
                pending_fill.append((index, frame))
                continue

            detections = results[result_index]
            result_index += 1
            frame_size = (float(frame.shape[1]), float(frame.shape[0]))
            count = len(detections)
            self.stats.detected_frames += 1

            # 说话人偏置只算一次，单窗口与多人分屏共用（顺带避免重复推进嘴动基准）
            bonuses = self._speaker_bonuses(frame, index, detections, frame_size)
            plan = (
                self.composer.update(frame_size, detections, bonuses)
                if self.composer is not None
                else None
            )

            subject = None
            target = None
            if plan is None:
                subject = select_subject(
                    detections,
                    frame_size,
                    prev_bbox=self._prev_bbox,
                    weights=self.weights,
                    min_height_ratio=self.min_person_height_ratio,
                    bonuses=bonuses,
                )
                target = (
                    compute_target_box(
                        subject.bbox,
                        frame_size,
                        self.ratio,
                        self.params,
                        build_anchors(subject.keypoints, subject.bbox),
                    )
                    if subject is not None
                    else None
                )

            if pending_fill:
                first_index = (
                    self._prev_detect_index if self._prev_detect_index is not None else index
                )
                span = max(index - first_index, 1)
                for fill_index, fill_frame in pending_fill:
                    t = (fill_index - first_index) / span
                    self.stats.interpolated_frames += 1
                    outcomes.append(
                        self._render_fill(fill_index, fill_frame, frame_size, t, target)
                    )
                pending_fill.clear()

            outcomes.append(self._render_key(index, frame, frame_size, target, plan, count))

            self._prev_detect_index = index
            self._prev_target = target
            if plan is not None:
                self._prev_bbox = plan.subjects[0] if plan.subjects else None
            else:
                self._prev_bbox = subject.bbox if subject is not None else None
            self._prev_count = count

        # segment 以最后一个关键帧收尾，故此时 pending_fill 必然已清空；
        # 队尾残留的填充帧留在 self._queue 里，等下一个关键帧到了再插值。
        return outcomes

    def _speaker_bonuses(
        self,
        frame: np.ndarray,
        index: int,
        detections: list,
        frame_size: tuple[float, float],
    ) -> dict[tuple[float, float, float, float], float] | None:
        """说话人跟随：给"正在说话的人"一个打分偏置（没人说话 / 只有一个人时返回 ``None``）。

        只在关键帧上评估（本来就要跑推理，顺带把嘴动一起算了），
        且只在**画面里有两个及以上主要人物**时才参与 —— 单人视频无从选择，也就无需计算。
        """
        if self.speaker is None or self.mouth is None or self.speaker_weight <= 0.0:
            return None
        if not self.speaker.enabled:
            return None

        candidates = filter_small(detections, frame_size, self.min_person_height_ratio)
        if len(candidates) < 2:
            # 只有一个人（或人都太小）：不判定说话人，但仍然要把嘴动分析器的
            # "上一帧"基准推进到本帧。否则基准会停留在好几帧之前，等下一个人
            # 出现时算出的帧间差分是跨越多帧的巨大差值 —— 那是噪声，不是嘴动。
            self.mouth.activity(frame, [])
            return None

        motions = self.mouth.activity(
            frame,
            [item.bbox for item in candidates],
            [item.keypoints for item in candidates],
        )
        biases = self.speaker.biases(index, [item.bbox for item in candidates], motions)
        if biases is None:
            return None
        if any(biases):
            self.stats.speaker_frames += 1
        return {
            tuple(item.bbox): self.speaker_weight * bias
            for item, bias in zip(candidates, biases, strict=True)
        }

    def _render_key(
        self,
        index: int,
        frame: np.ndarray,
        frame_size: tuple[float, float],
        target: CropBox | None,
        plan: MultiPlan | None,
        person_count: int,
    ) -> FrameOutcome:
        """关键帧：画面里有多人就用多人分屏，否则走原来的单窗口裁剪。"""
        if plan is not None:
            return self._render_multi(index, frame, plan, person_count)
        return self._render(index, frame, target, person_count)

    def _render_fill(
        self,
        index: int,
        frame: np.ndarray,
        frame_size: tuple[float, float],
        t: float,
        target: CropBox | None,
    ) -> FrameOutcome:
        """填充帧（抽帧检测的中间帧）：多人分屏时每个窗口各自插值、各自平滑。"""
        if self.composer is not None:
            plan = self.composer.interpolate(t, frame_size)
            if plan is not None:
                return self._render_multi(index, frame, plan, self._prev_count)
        interpolated = interpolate_boxes(self._prev_target, target, t, self.ratio)
        return self._render(index, frame, interpolated, self._prev_count)

    def _render_tail(self, index: int, frame: np.ndarray) -> FrameOutcome:
        """队尾残留的填充帧：沿用最后一次关键帧的目标（多人时复用各窗口）。"""
        frame_size = (float(frame.shape[1]), float(frame.shape[0]))
        if self.composer is not None:
            plan = self.composer.interpolate(1.0, frame_size)
            if plan is not None:
                return self._render_multi(index, frame, plan, self._prev_count)
        return self._render(index, frame, self._prev_target, self._prev_count)

    def _render_multi(
        self,
        index: int,
        frame: np.ndarray,
        plan: MultiPlan,
        person_count: int,
    ) -> FrameOutcome:
        """多人分屏帧：多个小窗口拼成一帧（尺寸仍是恒定的 ``out_size``）。"""
        assert self.composer is not None  # plan 只可能来自 composer
        output = self.composer.render(frame, plan)
        self.composer.note_frame(plan)

        self.stats.frames += 1
        self.stats.multi_frames += 1
        self.stats.windows_peak = max(self.stats.windows_peak, plan.count)
        self.stats.modes[MODE_MULTI] += 1
        return FrameOutcome(
            index=index,
            frame=output,
            person_count=person_count,
            has_person=True,
            mode=MODE_MULTI,
            box=plan.boxes[0],
            subject_bbox=plan.subjects[0] if plan.subjects else None,
            windows=self.composer.windows(plan),
        )

    def _render(
        self,
        index: int,
        frame: np.ndarray,
        target: CropBox | None,
        person_count: int,
    ) -> FrameOutcome:
        frame_size = (float(frame.shape[1]), float(frame.shape[0]))
        box = self.smoother.update(target, frame_size)
        output = crop_frame(frame, box, self.out_size) if self.crop else frame

        self.stats.frames += 1
        self.stats.modes[box.mode] += 1
        return FrameOutcome(
            index=index,
            frame=output,
            person_count=person_count,
            has_person=target is not None,
            mode=box.mode,
            box=box,
            subject_bbox=self._prev_bbox,
        )
