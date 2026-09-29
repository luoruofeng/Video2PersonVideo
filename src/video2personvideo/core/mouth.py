"""视觉侧：嘴部 ROI 与"嘴在动"的强度。

YOLO 只能告诉我们"画面上有谁"，告诉不了"谁在说话"（它没有音频输入，
COCO-17 关键点里也没有嘴部关键点）。要看嘴型只能自己动手：

1. **定位**——在人物框里找出嘴部区域，以及一块用来做参照的"上半脸"区域；
2. **测量**——用两个区域的**帧间差分之差**度量"嘴在动"：
   头部整体平移 / 转动、镜头微动会同时影响两处，相减即被抵消，
   剩下的才真正来自嘴部自身的运动（说话时嘴的开合）。

嘴部 ROI 的定位逐级降级，拿不到就返回 0 分（判定不了就不参与投票，
宁可维持原主角，也不让镜头乱跳）：

1. 姿态关键点（把模型换成 ``*-pose.pt``）：鼻 / 眼 / 耳 + 肩宽推算头部框 —— 最准；
2. Haar 人脸检测（OpenCV 自带，无需额外权重）：在人物框上部找脸，取脸的下半部分；
3. 人物框比例兜底：假定 bbox 顶边就是头顶，按人头占身高的比例切一刀（粗，但可用）。

纯几何部分（:func:`head_box_from_keypoints` / :func:`mouth_roi` / :func:`clip_roi`）
不依赖 OpenCV、可脱离视频单测；像素运算只用 numpy。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..utils.logger import get_logger
from .pose import (
    DEFAULT_KEYPOINT_CONF,
    FACE_POINTS,
    Keypoints,
    estimate_head_height,
    mean_x,
    min_y,
)

logger = get_logger(__name__)

#: 人物框：(x1, y1, x2, y2)
BBox = tuple[float, float, float, float]
#: 像素区域：(x, y, 宽, 高)
ROI = tuple[int, int, int, int]

#: 头部框内，嘴部所占的相对位置（相对头部框，上边在下巴附近）
MOUTH_TOP_RATIO = 0.62
MOUTH_BOTTOM_RATIO = 1.0
MOUTH_WIDTH_RATIO = 0.60
#: 用作运动参照的上半脸（眼 / 额）区域
REFERENCE_TOP_RATIO = 0.05
REFERENCE_BOTTOM_RATIO = 0.45
REFERENCE_WIDTH_RATIO = 0.60

#: 由面部关键点反推头顶时，向上多留的比例（眼睛不在头顶，留一点余量）
FACE_TO_HEAD_TOP_RATIO = 0.55
#: 头宽 / 头高（人头略窄于高）
HEAD_WIDTH_RATIO = 0.80

#: 没有关键点、也没检测到人脸时的兜底：假定人物框顶部是头顶
FALLBACK_HEAD_HEIGHT_RATIO = 0.30
FALLBACK_HEAD_WIDTH_RATIO = 0.75
#: 兜底路径要求的最小头部高度（像素）：太小的脸读不出嘴型，直接放弃
FALLBACK_MIN_HEAD_PX = 18.0


def head_box_from_keypoints(
    keypoints: Keypoints | None,
    bbox: BBox,
    *,
    min_conf: float = DEFAULT_KEYPOINT_CONF,
) -> BBox | None:
    """用面部关键点 + 肩宽推算头部框（拿不到面部关键点时返回 ``None``）。"""
    if not keypoints:
        return None
    center_x = mean_x(keypoints, FACE_POINTS, min_conf)
    face_y = min_y(keypoints, FACE_POINTS, min_conf)
    if center_x is None or face_y is None:
        return None

    head_height = estimate_head_height(keypoints, bbox, min_conf)
    head_top = max(float(bbox[1]), face_y - FACE_TO_HEAD_TOP_RATIO * head_height)
    head_width = HEAD_WIDTH_RATIO * head_height
    return (center_x - head_width / 2.0, head_top, center_x + head_width / 2.0, head_top + head_height)


def fallback_head_box(bbox: BBox, frame_size: tuple[float, float]) -> BBox | None:
    """没有任何定位信息时的粗略头部框：取人物框顶部一块。

    只在人像构图（半身 / 特写）下勉强成立；头部太小则返回 ``None``。
    """
    x1, y1, x2, y2 = (float(value) for value in bbox)
    body_h = y2 - y1
    frame_h = float(frame_size[1]) if frame_size else 0.0
    head_h = min(FALLBACK_HEAD_HEIGHT_RATIO * body_h, 0.5 * frame_h if frame_h else body_h)
    if head_h < FALLBACK_MIN_HEAD_PX:
        return None
    head_w = FALLBACK_HEAD_WIDTH_RATIO * head_h
    center_x = (x1 + x2) / 2.0
    return (center_x - head_w / 2.0, y1, center_x + head_w / 2.0, y1 + head_h)


def _relative_roi(
    head: BBox, top_ratio: float, bottom_ratio: float, width_ratio: float
) -> ROI:
    x1, y1, x2, y2 = (float(value) for value in head)
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)
    center_x = (x1 + x2) / 2.0
    roi_w = max(width * width_ratio, 1.0)
    roi_h = max(height * (bottom_ratio - top_ratio), 1.0)
    return (
        int(round(center_x - roi_w / 2.0)),
        int(round(y1 + height * top_ratio)),
        int(round(roi_w)),
        int(round(roi_h)),
    )


def mouth_roi(head: BBox) -> ROI:
    """头部框里的嘴部区域（含下巴）。"""
    return _relative_roi(head, MOUTH_TOP_RATIO, MOUTH_BOTTOM_RATIO, MOUTH_WIDTH_RATIO)


def reference_roi(head: BBox) -> ROI:
    """头部框里的"上半脸"参照区域（眼 / 额），用来抵消头部整体运动。"""
    return _relative_roi(head, REFERENCE_TOP_RATIO, REFERENCE_BOTTOM_RATIO, REFERENCE_WIDTH_RATIO)


def clip_roi(roi: ROI, frame_size: tuple[float, float]) -> ROI | None:
    """把区域夹取到画面内；完全在画面外或过小则返回 ``None``。"""
    x, y, width, height = (int(value) for value in roi)
    frame_w = int(frame_size[0])
    frame_h = int(frame_size[1])
    if frame_w <= 0 or frame_h <= 0:
        return None
    x = max(0, min(x, frame_w - 1))
    y = max(0, min(y, frame_h - 1))
    width = max(0, min(width, frame_w - x))
    height = max(0, min(height, frame_h - y))
    if width < 2 or height < 2:
        return None
    return x, y, width, height


def to_gray(frame: np.ndarray) -> np.ndarray:
    """BGR（或已是灰度 / 单通道）帧 → 灰度图。只用 numpy，不依赖 OpenCV。"""
    array = np.asarray(frame)
    if array.ndim == 2:
        return array.astype(np.uint8, copy=False)
    if array.shape[2] < 3:  # 单通道（或带 alpha 的单通道）：直接用
        return np.clip(array[..., 0], 0, 255).astype(np.uint8)
    weights = np.array([0.114, 0.587, 0.299], dtype=np.float32)  # BGR 顺序
    gray = array[..., :3].astype(np.float32) @ weights
    return np.clip(gray, 0.0, 255.0).astype(np.uint8)


def roi_motion(previous: np.ndarray, current: np.ndarray, roi: ROI, *, scale: float = 255.0) -> float:
    """区域内的帧间差分能量（0~1）：值越大说明这块像素变化越剧烈。"""
    x, y, width, height = (int(value) for value in roi)
    before = previous[y : y + height, x : x + width]
    after = current[y : y + height, x : x + width]
    if before.size == 0 or after.size == 0 or before.shape != after.shape:
        return 0.0
    diff = np.abs(after.astype(np.int16) - before.astype(np.int16))
    return float(diff.mean()) / float(scale)


def mouth_activity(
    previous: np.ndarray,
    current: np.ndarray,
    mouth: ROI,
    reference: ROI,
    *,
    reference_weight: float = 1.0,
) -> float:
    """嘴部运动强度 = 嘴部帧间差异 − 上半脸帧间差异（抵消头部整体运动）。

    结果截断到非负：参照区动得更多时说明变化来自头动而非嘴动。
    """
    mouth_motion = roi_motion(previous, current, mouth)
    reference_motion = roi_motion(previous, current, reference)
    return max(0.0, mouth_motion - float(reference_weight) * reference_motion)


class FaceLocator:
    """用 OpenCV 自带的 Haar 级联在人物框上部找人脸（没有姿态关键点时的兜底）。

    只做"找脸"这一件事，且限制搜索范围与最小尺寸，避免误检与小脸噪声；
    级联文件缺失（某些精简版 OpenCV 构建）时 :attr:`available` 为 ``False``。
    """

    def __init__(
        self,
        *,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        search_top_ratio: float = 0.45,
        min_face_ratio: float = 0.18,
    ) -> None:
        self.scale_factor = float(scale_factor)
        self.min_neighbors = int(min_neighbors)
        self.search_top_ratio = float(search_top_ratio)
        self.min_face_ratio = float(min_face_ratio)
        self._cascade = self._load_cascade()

    @staticmethod
    def _load_cascade():
        try:
            import cv2
        except ImportError:  # pragma: no cover - 运行期依赖已声明
            return None
        try:
            path = str(cv2.data.haarcascades) + "haarcascade_frontalface_default.xml"
            cascade = cv2.CascadeClassifier(path)
        except Exception as exc:  # noqa: BLE001 - 任何加载失败都退化为不可用
            logger.debug("加载 Haar 人脸级联失败：%s", exc)
            return None
        if cascade is None or cascade.empty():  # pragma: no cover - 取决于 OpenCV 构建
            logger.debug("Haar 人脸级联不可用（当前 OpenCV 构建可能未附带）")
            return None
        return cascade

    @property
    def available(self) -> bool:
        return self._cascade is not None

    def locate(self, gray: np.ndarray, bbox: BBox) -> BBox | None:
        """在人物框上部找人脸，返回画面坐标下的人脸框；找不到返回 ``None``。"""
        if self._cascade is None:
            return None

        frame_h, frame_w = gray.shape[:2]
        x1 = max(int(round(min(bbox[0], bbox[2]))), 0)
        x2 = min(int(round(max(bbox[0], bbox[2]))), frame_w)
        y1 = max(int(round(min(bbox[1], bbox[3]))), 0)
        y2 = min(int(round(max(bbox[1], bbox[3]))), frame_h)
        span_h = int((y2 - y1) * self.search_top_ratio)
        if x2 - x1 < 8 or span_h < 8:
            return None

        region = gray[y1 : y1 + span_h, x1:x2]
        min_side = max(16, int(region.shape[0] * self.min_face_ratio))
        try:
            faces = self._cascade.detectMultiScale(
                region,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=(min_side, min_side),
            )
        except Exception as exc:  # noqa: BLE001 - 检测失败按"没找到脸"处理
            logger.debug("人脸检测失败：%s", exc)
            return None
        if len(faces) == 0:
            return None

        # 取最大的一张：通常就是人物框上部的那颗头
        fx, fy, width, height = max(faces, key=lambda item: int(item[2]) * int(item[3]))
        return (
            float(x1 + fx),
            float(y1 + fy),
            float(x1 + fx + width),
            float(y1 + fy + height),
        )


@dataclass(slots=True)
class MouthActivityAnalyzer:
    """逐帧评估每个候选人物的"嘴在动"强度。

    需要上一帧的灰度图做帧间差分，因此**按关键帧顺序调用**即可
    （抽帧检测时只对关键帧调用，正好也是判定说话人的时机）。
    第一次调用没有参照帧，返回全 0（先建立基准，不影响后续判定）。
    """

    reference_weight: float = 1.0
    face_locator: FaceLocator | None = field(default=None)
    fallback_ratio: bool = True
    _previous_gray: np.ndarray | None = field(default=None, init=False, repr=False)

    def reset(self) -> None:
        self._previous_gray = None

    def activity(
        self,
        frame: np.ndarray,
        bboxes: list[BBox],
        keypoints: list[Keypoints | None] | None = None,
    ) -> list[float]:
        """返回与 ``bboxes`` 等长的嘴动强度列表（0 表示"测不出来"，不参与投票）。"""
        gray = to_gray(frame)
        previous = self._previous_gray
        self._previous_gray = gray

        if previous is None or not bboxes:
            return [0.0] * len(bboxes)

        frame_size = (float(gray.shape[1]), float(gray.shape[0]))
        values: list[float] = []
        for position, bbox in enumerate(bboxes):
            points = (
                keypoints[position]
                if keypoints is not None and position < len(keypoints)
                else None
            )
            values.append(self._one(previous, gray, bbox, points, frame_size))
        return values

    # ------------------------------------------------------------- 内部逻辑
    def _one(
        self,
        previous: np.ndarray,
        current: np.ndarray,
        bbox: BBox,
        keypoints: Keypoints | None,
        frame_size: tuple[float, float],
    ) -> float:
        head = head_box_from_keypoints(keypoints, bbox)
        if head is None and self.face_locator is not None:
            head = self.face_locator.locate(current, bbox)
        if head is None and self.fallback_ratio:
            head = fallback_head_box(bbox, frame_size)
        if head is None:
            return 0.0

        mouth = clip_roi(mouth_roi(head), frame_size)
        reference = clip_roi(reference_roi(head), frame_size)
        if mouth is None or reference is None:
            return 0.0
        return mouth_activity(
            previous, current, mouth, reference, reference_weight=self.reference_weight
        )
