"""姿态关键点 → 构图锚点（M1-3 可选增强）。

只在模型自带关键点时启用（如 ``yolo11n-pose.pt``）：用面部关键点更准地定位
**头顶**，用髋部关键点定位**腰部**，让"半身"构图的下边界落在腰/髋，
而不是整条 bbox 的底边——人物抬手、戴帽子时 bbox 顶边会明显偏高，
关键点能纠正这类偏差。

本模块只做纯数学计算（不依赖 numpy / OpenCV / Qt），关键点用
``((x, y, conf), ...)`` 的嵌套元组表示，索引遵循 COCO-17。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

#: 关键点：(x, y, conf)
Keypoint = tuple[float, float, float]
#: 一个目标的关键点序列
Keypoints = Sequence[Keypoint]
#: 人物框：(x1, y1, x2, y2)
BBox = tuple[float, float, float, float]

# ------------------------------------------------------------------ COCO-17
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12

#: 参与"头顶"估计的面部关键点
FACE_POINTS = (NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR)
SHOULDER_POINTS = (LEFT_SHOULDER, RIGHT_SHOULDER)
HIP_POINTS = (LEFT_HIP, RIGHT_HIP)

#: 关键点置信度阈值（低于此值视为不可用）
DEFAULT_KEYPOINT_CONF = 0.5

#: 头高 ≈ 0.9 × 肩宽（人体测量经验值，用于从面部反推头顶）
HEAD_TO_SHOULDER_RATIO = 0.9
#: 拿不到肩宽时的兜底：头高 ≈ 身高 / 7.5
HEAD_TO_BODY_RATIO = 1.0 / 7.5


@dataclass(frozen=True, slots=True)
class BodyAnchors:
    """人体纵向锚点。"""

    #: 头顶（含帽子/抬手纠正后的估计值）
    head_top: float
    #: 腰/髋高度（半身构图的下边界基准）
    waist: float

    @property
    def span(self) -> float:
        return max(self.waist - self.head_top, 1.0)

    @property
    def center(self) -> float:
        return (self.head_top + self.waist) / 2.0

    @property
    def head_height(self) -> float:
        """粗略头高（拿不到肩宽时按 2.2 倍"头顶到鼻子"估算）。"""
        return max(self.span / 3.0, 1.0)


def _confident(keypoints: Keypoints | None, index: int, min_conf: float) -> Keypoint | None:
    if keypoints is None or index >= len(keypoints):
        return None
    point = keypoints[index]
    if len(point) < 3 or float(point[2]) < min_conf:
        return None
    return (float(point[0]), float(point[1]), float(point[2]))


def min_y(
    keypoints: Keypoints | None,
    indices: Sequence[int],
    min_conf: float = DEFAULT_KEYPOINT_CONF,
) -> float | None:
    """给定关键点集合里最靠上（y 最小）的那个 y。"""
    values = [
        point[1]
        for point in (_confident(keypoints, index, min_conf) for index in indices)
        if point is not None
    ]
    return min(values) if values else None


def mean_y(
    keypoints: Keypoints | None,
    indices: Sequence[int],
    min_conf: float = DEFAULT_KEYPOINT_CONF,
) -> float | None:
    """给定关键点集合的 y 均值（全部不可用时返回 ``None``）。"""
    values = [
        point[1]
        for point in (_confident(keypoints, index, min_conf) for index in indices)
        if point is not None
    ]
    return sum(values) / len(values) if values else None


def mean_x(
    keypoints: Keypoints | None,
    indices: Sequence[int],
    min_conf: float = DEFAULT_KEYPOINT_CONF,
) -> float | None:
    """给定关键点集合的 x 均值（全部不可用时返回 ``None``）。"""
    values = [
        point[0]
        for point in (_confident(keypoints, index, min_conf) for index in indices)
        if point is not None
    ]
    return sum(values) / len(values) if values else None


def shoulder_width(
    keypoints: Keypoints | None, min_conf: float = DEFAULT_KEYPOINT_CONF
) -> float | None:
    """左右肩的水平距离（两点都可用时才有值）。"""
    left = _confident(keypoints, LEFT_SHOULDER, min_conf)
    right = _confident(keypoints, RIGHT_SHOULDER, min_conf)
    if left is None or right is None:
        return None
    width = abs(right[0] - left[0])
    return width if width > 1.0 else None


def estimate_head_height(
    keypoints: Keypoints | None,
    bbox: BBox,
    min_conf: float = DEFAULT_KEYPOINT_CONF,
) -> float:
    """估计头高：优先用肩宽，拿不到就按身高比例兜底。"""
    width = shoulder_width(keypoints, min_conf)
    if width is not None:
        return HEAD_TO_SHOULDER_RATIO * width
    body_h = max(float(bbox[3]) - float(bbox[1]), 1.0)
    return max(HEAD_TO_BODY_RATIO * body_h, 1.0)


def build_anchors(
    keypoints: Keypoints | None,
    bbox: BBox,
    *,
    min_conf: float = DEFAULT_KEYPOINT_CONF,
) -> BodyAnchors | None:
    """由关键点得到"头顶 / 腰部"锚点；信息不足时返回 ``None``（退回 bbox 行为）。

    需要**面部 + 髋部**都可用：髋部缺失时不猜（宁可退回原来的 bbox 逻辑），
    避免把"半身"边界错切到肩膀上。
    """
    if not keypoints:
        return None

    face_y = min_y(keypoints, FACE_POINTS, min_conf)
    hip_y = mean_y(keypoints, HIP_POINTS, min_conf)
    if face_y is None or hip_y is None:
        return None

    head_height = estimate_head_height(keypoints, bbox, min_conf)
    # 眼睛大致在头部中点：头顶 ≈ 最上面部关键点 - 半个头高。
    # 取 max(bbox 顶边, 估计头顶)：bbox 顶边通常就是头顶，
    # 但人物抬手时 bbox 会高出一截，此时用关键点把它拉回来。
    head_top = max(float(bbox[1]), face_y - 0.5 * head_height)

    if hip_y <= head_top:
        return None
    return BodyAnchors(head_top=head_top, waist=hip_y)
