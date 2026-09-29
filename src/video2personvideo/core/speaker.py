"""主动说话人判定：把"嘴在动"与"有人在说话"对上号。

YOLO 说得出"画面里有几个人"，说不出"谁在说话"。于是把两路信号合起来：

* 音频侧（:mod:`~video2personvideo.core.audio`）：逐帧语音能量包络 + 语音标记；
* 视觉侧（:mod:`~video2personvideo.core.mouth`）：每个人的嘴部运动强度。

在滑动窗口内对**每个人的嘴动信号与语音能量**做归一化互相关，
再叠加"说话时嘴动得比安静时多"的对比度，得分最高者即当前说话人。

两个工程要点（否则镜头会乱跳）：

* **身份连续性**：候选人按 IoU 关联成轨迹（track），相关性才有时序意义；
* **切换滞回**：挑战者必须"连续多帧 + 明显领先"才能夺位，静音期间不换人。

本模块只做纯计算（``numpy`` / 纯 Python），不读文件、不碰 OpenCV / Qt，
可脱离视频单测。
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

from .audio import SpeechEnvelope
from .subject import iou

#: 人物框：(x1, y1, x2, y2)
BBox = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class SpeakerParams:
    """说话人判定参数（全部可在 ``AppConfig`` / ``configs/default.yaml`` 中调整）。"""

    #: 相关性滑窗长度（帧）：约 1~1.5 秒，太短易误判、太长跟不上换人
    window_frames: int = 45
    #: 计算相关性所需的最少样本数（不足则不给分，宁可不判定）
    min_samples: int = 12
    #: 夺位所需的最低得分（0~1）
    min_score: float = 0.20
    #: 挑战者需要领先在位者多少分
    switch_margin: float = 0.15
    #: 挑战者需要连续领先多少个判定回合（每次判定 = 一个关键帧）
    switch_hold: int = 8
    #: 候选人关联成轨迹的 IoU 门限
    track_iou: float = 0.20
    #: 轨迹多少帧没再出现就丢弃（人物走出画面）
    max_missing_frames: int = 20
    #: 说话时嘴动强度的绝对下限：低于它认为"读不出嘴型"，不给分
    min_mouth_motion: float = 0.004

    def __post_init__(self) -> None:
        if int(self.window_frames) < 2:
            raise ValueError(f"window_frames 必须 ≥ 2，当前为 {self.window_frames}")
        if int(self.min_samples) < 2:
            raise ValueError(f"min_samples 必须 ≥ 2，当前为 {self.min_samples}")
        if int(self.min_samples) > int(self.window_frames):
            raise ValueError("min_samples 不能大于 window_frames")
        if not 0.0 <= float(self.min_score) <= 1.0:
            raise ValueError(f"min_score 必须在 [0, 1] 区间内，当前为 {self.min_score}")
        if float(self.switch_margin) < 0.0:
            raise ValueError(f"switch_margin 不能为负，当前为 {self.switch_margin}")
        if int(self.switch_hold) < 1:
            raise ValueError(f"switch_hold 必须 ≥ 1，当前为 {self.switch_hold}")
        if not 0.0 < float(self.track_iou) <= 1.0:
            raise ValueError(f"track_iou 必须在 (0, 1] 区间内，当前为 {self.track_iou}")
        if int(self.max_missing_frames) < 1:
            raise ValueError(f"max_missing_frames 必须 ≥ 1，当前为 {self.max_missing_frames}")
        if float(self.min_mouth_motion) < 0.0:
            raise ValueError(f"min_mouth_motion 不能为负，当前为 {self.min_mouth_motion}")


#: 默认说话人判定参数
DEFAULT_SPEAKER_PARAMS = SpeakerParams()

#: 相关性与对比度在最终得分中的权重
_CORRELATION_WEIGHT = 0.6
_CONTRAST_WEIGHT = 0.4


def correlation(first: Sequence[float], second: Sequence[float]) -> float:
    """皮尔逊相关系数（−1~1）；样本太少或方差为 0 时返回 0。"""
    if len(first) != len(second) or len(first) < 2:
        return 0.0
    count = len(first)
    mean_first = sum(first) / count
    mean_second = sum(second) / count
    dev_first = [value - mean_first for value in first]
    dev_second = [value - mean_second for value in second]
    numerator = sum(a * b for a, b in zip(dev_first, dev_second, strict=True))
    norm_first = math.sqrt(sum(value * value for value in dev_first))
    norm_second = math.sqrt(sum(value * value for value in dev_second))
    denominator = norm_first * norm_second
    if denominator <= 1e-9:
        return 0.0
    return max(-1.0, min(1.0, numerator / denominator))


def track_score(
    mouth: Sequence[float],
    energy: Sequence[float],
    active: Sequence[bool],
    params: SpeakerParams = DEFAULT_SPEAKER_PARAMS,
) -> float:
    """给一条轨迹打"正在说话"的分（0~1）。

    = 0.6 × max(皮尔逊相关, 0) + 0.4 × 说话时嘴动的相对增量

    只要求"嘴动"与"语音能量"同步：全程都在动嘴（咀嚼、笑）的人相关性不会高，
    而嘴一直不动的人对比度接近 0。
    """
    if len(mouth) < params.min_samples:
        return 0.0
    flags = list(active)
    if not any(flags):
        return 0.0

    speaking = [value for value, flag in zip(mouth, flags, strict=True) if flag]
    quiet = [value for value, flag in zip(mouth, flags, strict=True) if not flag]
    speech_level = sum(speaking) / len(speaking)
    if speech_level < params.min_mouth_motion:
        return 0.0

    quiet_level = sum(quiet) / len(quiet) if quiet else 0.0
    contrast = max(0.0, speech_level - quiet_level) / speech_level
    correlated = max(0.0, correlation(mouth, energy))
    return _CORRELATION_WEIGHT * correlated + _CONTRAST_WEIGHT * min(contrast, 1.0)


@dataclass(slots=True)
class _Track:
    """一个候选人的时间轨迹。

    这里**不**自带 ``score``：打分必须用 :class:`ActiveSpeakerSelector` 的
    ``params``（``min_mouth_motion`` / ``min_samples`` 等用户可调），
    见 :meth:`ActiveSpeakerSelector._track_score`。
    """

    id: int
    bbox: BBox
    last_index: int
    mouth: deque[float]
    energy: deque[float]
    active: deque[bool]


class ActiveSpeakerSelector:
    """多人物场景下判定"现在谁在说话"，并给出主角打分偏置。

    用法（每个关键帧调用一次）::

        biases = selector.biases(index, bboxes, mouth_motions)
        if biases is not None:           # 判定出了说话人
            score += speaker_weight * biases[i]

    没有音频、判定不了、或画面里不足两个人时返回 ``None``，
    调用方据此退回原来的主角打分规则（面积 + 居中 + 时序 + 置信度）。
    """

    def __init__(
        self,
        envelope: SpeechEnvelope | None = None,
        params: SpeakerParams = DEFAULT_SPEAKER_PARAMS,
    ) -> None:
        self.envelope = envelope
        self.params = params
        self.switches = 0
        self._tracks: list[_Track] = []
        self._next_id = 0
        self._current: int | None = None
        self._challenger: int | None = None
        self._challenger_streak = 0

    @property
    def enabled(self) -> bool:
        return self.envelope is not None and self.envelope.frame_count > 0

    @property
    def current_track(self) -> int | None:
        """当前认定的说话人轨迹 id（``None`` = 还没认定）。"""
        return self._current

    def reset(self) -> None:
        self._tracks.clear()
        self._current = None
        self._challenger = None
        self._challenger_streak = 0

    def biases(
        self,
        index: int,
        bboxes: Sequence[BBox],
        mouth_motions: Sequence[float],
    ) -> list[float] | None:
        """返回与 ``bboxes`` 等长的偏置（当前说话人 1.0，其余 0.0）；无法判定时 ``None``。"""
        if not self.enabled or len(bboxes) < 2 or len(mouth_motions) != len(bboxes):
            return None

        tracks = self._assign(index, bboxes, mouth_motions)
        scores = {track.id: self._track_score(track) for track in tracks}
        best = self._best(scores)
        self._update_incumbent(scores, best)

        if self._current is None:
            return None
        return [1.0 if track.id == self._current else 0.0 for track in tracks]

    # ------------------------------------------------------------- 内部逻辑
    def _assign(
        self, index: int, bboxes: Sequence[BBox], mouth_motions: Sequence[float]
    ) -> list[_Track]:
        """把本帧的候选人与已有轨迹按 IoU 关联，返回与输入等长的轨迹列表。"""
        alive = [
            track
            for track in self._tracks
            if index - track.last_index <= self.params.max_missing_frames
        ]
        self._tracks = alive

        assigned: list[_Track] = []
        claimed: set[int] = set()
        energy = self.envelope.energy_at(index) if self.envelope is not None else 0.0
        speech = self.envelope.is_speech(index) if self.envelope is not None else False

        for bbox, motion in zip(bboxes, mouth_motions, strict=True):
            track = self._match(bbox, alive, claimed)
            if track is None:
                track = _Track(
                    id=self._next_id,
                    bbox=bbox,
                    last_index=index,
                    mouth=deque(maxlen=self.params.window_frames),
                    energy=deque(maxlen=self.params.window_frames),
                    active=deque(maxlen=self.params.window_frames),
                )
                self._next_id += 1
                alive.append(track)
                claimed.add(track.id)  # 同一帧里后面的候选不能抢走它
            else:
                claimed.add(track.id)
                track.bbox = bbox
                track.last_index = index
            track.mouth.append(max(float(motion), 0.0))
            track.energy.append(float(energy))
            track.active.append(bool(speech))
            assigned.append(track)

        if self._current is not None and all(track.id != self._current for track in alive):
            # 原说话人已经离开画面（轨迹被丢弃）：放权，等新的一轮认定。
            # 注意只在"轨迹真的消失"时清空 —— 单帧漏检不换人，避免镜头反复横跳。
            self._current = None
            self._challenger = None
            self._challenger_streak = 0
        return assigned

    def _match(self, bbox: BBox, tracks: Sequence[_Track], claimed: set[int]) -> _Track | None:
        """按 IoU 找最像同一个人的轨迹。"""
        best: _Track | None = None
        best_iou = self.params.track_iou
        for track in tracks:
            if track.id in claimed:
                continue
            overlap = iou(track.bbox, bbox)
            if overlap >= best_iou:
                best, best_iou = track, overlap
        return best

    def _track_score(self, track: _Track) -> float:
        """用**本判定器**的参数给轨迹打分。

        必须传 ``self.params``：否则用户调过的 ``min_mouth_motion`` /
        ``min_samples`` 会被默认参数覆盖，"远景小脸"等场景下判定永远不生效。
        """
        return track_score(track.mouth, track.energy, track.active, self.params)

    def _best(self, scores: dict[int, float]) -> int | None:
        if not scores:
            return None
        track_id = max(scores, key=lambda key: (scores[key], -key))
        if scores[track_id] < self.params.min_score:
            return None
        return track_id

    def _update_incumbent(self, scores: dict[int, float], best: int | None) -> None:
        """带滞回的换人判定。

        在位者本帧没被关联上（单帧漏检）时按 0 分处理：连续多帧都读不到它、
        而挑战者一直领先，才会真的换人。
        """
        if self._current is None:
            if best is not None:
                self._current = best
            self._reset_challenger()
            return

        if best is None or best == self._current:
            self._reset_challenger()
            return

        current_score = scores.get(self._current, 0.0)
        if scores[best] - current_score < self.params.switch_margin:
            self._reset_challenger()
            return

        if self._challenger == best:
            self._challenger_streak += 1
        else:
            self._challenger = best
            self._challenger_streak = 1

        if self._challenger_streak >= self.params.switch_hold:
            self._current = best
            self.switches += 1
            self._reset_challenger()

    def _reset_challenger(self) -> None:
        self._challenger = None
        self._challenger_streak = 0
