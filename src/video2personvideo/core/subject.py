"""主角选择：多人物场景下决定"谁是被追踪的主要人物"。

只做数学计算（``math`` / 纯 Python），不依赖 numpy、OpenCV、Qt。

打分 = 面积得分 + 位置得分 + 时序连续得分 + 置信度得分，
权重可在 ``AppConfig`` 中调整。
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from .pose import Keypoints

#: 人物框：(x1, y1, x2, y2)，源画面像素坐标
BBox = tuple[float, float, float, float]

#: 默认主角最小高度占比：低于此值视为"没有人物"（背景里的路人 / 小人）。
#: 只认画面里占比足够大的人，数值调大后路人会被忽略；与
#: :data:`~video2personvideo.core.framing.DEFAULT_MIN_PERSON_HEIGHT_RATIO` 保持一致。
DEFAULT_MIN_HEIGHT_RATIO = 0.73

#: 默认主角最低清晰度（0~1 的比率，算法见 :mod:`~video2personvideo.core.sharpness`）：
#: 低于它的人算"背景人物" —— 被镜头虚化的路人、远处海报 / 屏幕里的人即便"看着很大"，
#: 也不会被当成主要人物。与"够大"是**并列**条件，两者都达标才算主要人物。
DEFAULT_MIN_PERSON_SHARPNESS = 0.30


@dataclass(frozen=True, slots=True)
class Detection:
    """一帧里的一条结构化检测结果。"""

    bbox: BBox
    confidence: float = 1.0
    class_id: int = 0
    #: COCO-17 姿态关键点（仅姿态模型提供；用于精修构图锚点）
    keypoints: Keypoints | None = None
    #: 人物框内的清晰度比率（0~1，算法见 :mod:`~video2personvideo.core.sharpness`）。
    #: ``None`` = 还没算过（视为达标，不参与过滤），由流水线在检测后补上。
    sharpness: float | None = None

    @property
    def width(self) -> float:
        return max(float(self.bbox[2]) - float(self.bbox[0]), 0.0)

    @property
    def height(self) -> float:
        return max(float(self.bbox[3]) - float(self.bbox[1]), 0.0)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (
            (float(self.bbox[0]) + float(self.bbox[2])) / 2.0,
            (float(self.bbox[1]) + float(self.bbox[3])) / 2.0,
        )


@dataclass(frozen=True, slots=True)
class SubjectWeights:
    """主角打分权重。"""

    area: float = 1.0
    center: float = 0.6
    continuity: float = 1.2
    confidence: float = 0.3

    @property
    def total(self) -> float:
        """各权重之和，即基础打分的理论满分（用于把得分归一化到 0~1）。"""
        return float(self.area) + float(self.center) + float(self.continuity) + float(self.confidence)


#: 默认打分权重
DEFAULT_WEIGHTS = SubjectWeights()


def iou(first: BBox, second: BBox) -> float:
    """两个框的交并比（0~1）。"""
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_first = max(float(first[2]) - float(first[0]), 0.0) * max(
        float(first[3]) - float(first[1]), 0.0
    )
    area_second = max(float(second[2]) - float(second[0]), 0.0) * max(
        float(second[3]) - float(second[1]), 0.0
    )
    union = area_first + area_second - inter
    return inter / union if union > 0 else 0.0


def passes_sharpness(detection: Detection, min_sharpness: float = 0.0) -> bool:
    """清晰度是否达标（"够清晰"这一半的判定）。

    ``detection.sharpness`` 为 ``None``（没算过 / 无法判定）时一律**放行**，
    于是没接清晰度判定的调用方（单测、画框标注模式）行为与以前完全一致。
    """
    floor = max(float(min_sharpness), 0.0)
    if floor <= 0.0 or detection.sharpness is None:
        return True
    return float(detection.sharpness) >= floor


def filter_small(
    detections: Iterable[Detection],
    frame_size: tuple[float, float],
    min_height_ratio: float = DEFAULT_MIN_HEIGHT_RATIO,
    min_sharpness: float = 0.0,
) -> list[Detection]:
    """过滤掉"人物过小"或"画面够糊"的检测，剩下的才视为有效人物。

    "够大"与"够清晰"是**并列**条件：高度不达标的人（背景里的路人 / 小人）与
    清晰度不达标的人（被虚化的背景人物）都不算主要人物。
    """
    frame_h = float(frame_size[1]) if frame_size else 0.0
    threshold = max(min_height_ratio, 0.0) * frame_h
    return [
        item
        for item in detections
        if item.width > 0
        and item.height > 0
        and item.height >= threshold
        and passes_sharpness(item, min_sharpness)
    ]


def score_detection(
    detection: Detection,
    frame_size: tuple[float, float],
    *,
    prev_bbox: BBox | None = None,
    weights: SubjectWeights = DEFAULT_WEIGHTS,
    bonus: float = 0.0,
) -> float:
    """给单个检测打分，越大越像"主角"。

    返回值是**归一化到 0~1** 的常规得分（四项加权之和 ÷ 权重之和，除以正数不改变
    候选人之间的排序），``bonus`` 再叠加在这个 0~1 的量纲上。归一化的意义是把
    "外部偏置"和"常规得分"放到同一把尺子上：``bonus = 1.0`` 即"无论对方多大 /
    多居中 / 多连续都会胜出"，``bonus = 0.2`` 则是"除非对方明显更抢镜"。
    不归一化的话两项量纲不可比，偏置会被常规得分吞掉（见 ``select_subject``）。

    ``bonus`` 是外部信号给出的额外偏置（当前用于"正在说话的人"，
    见 :mod:`~video2personvideo.core.speaker`），非负。
    """
    frame_w, frame_h = float(frame_size[0]), float(frame_size[1])
    if frame_w <= 0 or frame_h <= 0:
        return 0.0

    score = weights.area * (detection.area / (frame_w * frame_h))

    center_x, center_y = detection.center
    half_diagonal = max(math.hypot(frame_w, frame_h) / 2.0, 1.0)
    distance = math.hypot(center_x - frame_w / 2.0, center_y - frame_h / 2.0) / half_diagonal
    score += weights.center * max(0.0, 1.0 - distance)

    score += weights.confidence * min(max(detection.confidence, 0.0), 1.0)

    if prev_bbox is not None:
        score += weights.continuity * iou(prev_bbox, detection.bbox)

    scale = weights.total
    normalized = score / scale if scale > 0.0 else score
    return normalized + max(float(bonus), 0.0)


def select_subject(
    detections: Sequence[Detection],
    frame_size: tuple[float, float],
    *,
    prev_bbox: BBox | None = None,
    weights: SubjectWeights = DEFAULT_WEIGHTS,
    min_height_ratio: float = DEFAULT_MIN_HEIGHT_RATIO,
    min_sharpness: float = 0.0,
    bonuses: Mapping[BBox, float] | None = None,
) -> Detection | None:
    """选出"主要人物"；过滤后无人时返回 ``None``（交由兜底策略处理）。

    ``min_height_ratio`` 与 ``min_sharpness`` 是并列的两道门槛：人物既要**够大**，
    也要**够清晰**（清晰度由 :func:`~video2personvideo.core.sharpness.annotate_sharpness`
    预先算进 ``Detection.sharpness``），任一不达标即视为背景人物。

    ``bonuses`` 是以人物框为键的额外偏置（说话人跟随用），量纲与常规得分一致（0~1）：
    命中的检测加上对应分值后，``≥ 1.0`` 的偏置足以让"正在说话的人"压过
    "个子更大 / 更居中 / 时序更连续"的人；比 1.0 小的偏置则是"软优先"。
    """
    candidates = filter_small(detections, frame_size, min_height_ratio, min_sharpness)
    if not candidates:
        return None

    def score(item: Detection) -> float:
        bonus = bonuses.get(tuple(item.bbox), 0.0) if bonuses else 0.0
        return score_detection(
            item, frame_size, prev_bbox=prev_bbox, weights=weights, bonus=bonus
        )

    return max(candidates, key=score)
