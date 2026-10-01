"""清晰度判定：把"人物框里的画面够不够清晰"量成一个 0~1 的比率。

判断"谁才是画面里的主要人物"如果只看大小（bbox 高度占画面的比例），会留一个
漏洞：**被镜头虚化掉的背景人物**、贴在远处的海报 / 屏幕里的人，同样可能"看着
很大"，却并不是站在镜头前的那个人。这里补上第二个参考量 —— **清晰度**：

1. 取人物框内的区域，转灰度；
2. 缩放到统一高度（:data:`SHARPNESS_REFERENCE_HEIGHT`），抵掉源视频分辨率的差异，
   于是同一份阈值在 480p 与 4K 素材上都能用；
3. 用 Laplacian 方差（经典的"对焦实不实"度量）算边缘强度 ``v``；
4. 压成 :data:`LAPLACIAN_REFERENCE` 决定的 0~1 比率 ``v / (v + 参考值)`` ——
   ``≈0`` 糊成一片、``0.5`` 正好等于参考值、``→1`` 非常锐利。

阈值由全局常量 :data:`~video2personvideo.core.subject.DEFAULT_MIN_PERSON_SHARPNESS`
给出，并可通过 ``AppConfig.min_person_sharpness`` / ``--min-sharpness`` 让用户调整。
只有**同时**满足"够大"（``min_person_height_ratio``）与"够清晰"（该项）的检测，
才算主要人物；任一不达标的一律当背景人物处理。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import cv2
import numpy as np

from .subject import BBox, Detection

#: 算清晰度前把人物框缩放到的高度（像素）：抵掉分辨率差异，
#: 于是同一份阈值在低分辨率与 4K 素材上都能用。
SHARPNESS_REFERENCE_HEIGHT = 128

#: Laplacian 方差的参考值：它对应的清晰度正好是 0.5（越大越锐利）。
LAPLACIAN_REFERENCE = 100.0

#: 人物框小于这个边长（像素）时不判清晰度（返回 ``None`` = 未知，照旧放行）
MIN_ROI_SIDE = 8


def _clamp_bbox(bbox: BBox, width: int, height: int) -> tuple[int, int, int, int]:
    """把浮点框夹到画面内并转成整型像素框。"""
    x1, y1, x2, y2 = (float(value) for value in bbox)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    left = min(max(int(round(x1)), 0), width)
    top = min(max(int(round(y1)), 0), height)
    right = min(max(int(round(x2)), 0), width)
    bottom = min(max(int(round(y2)), 0), height)
    return left, top, right, bottom


def _to_gray(image: np.ndarray) -> np.ndarray:
    """三通道转灰度；已经是灰度 / 单通道时原样返回。"""
    if image.ndim == 3 and image.shape[2] >= 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3:
        return image[:, :, 0]
    return image


def _resize_to_height(gray: np.ndarray, height: int) -> np.ndarray:
    """等比缩放到指定高度（缩小时用 AREA、放大时用 CUBIC）。"""
    src_h, src_w = gray.shape[:2]
    if src_h <= 0 or src_w <= 0 or src_h == height:
        return gray
    scale = height / float(src_h)
    target_w = max(int(round(src_w * scale)), 1)
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(gray, (target_w, height), interpolation=interpolation)


def region_sharpness(
    frame: np.ndarray,
    bbox: BBox,
    *,
    reference_height: int = SHARPNESS_REFERENCE_HEIGHT,
    reference: float = LAPLACIAN_REFERENCE,
) -> float | None:
    """人物框内的清晰度比率（0~1）。

    返回 ``None`` 表示**无法判定**（人物框太小 / 越界 / 画面为空），
    调用方应当把它当作"未知、照旧放行"，而不是"不清晰"。
    """
    if frame is None or getattr(frame, "size", 0) == 0:
        return None

    frame_h, frame_w = int(frame.shape[0]), int(frame.shape[1])
    left, top, right, bottom = _clamp_bbox(bbox, frame_w, frame_h)
    if right - left < MIN_ROI_SIDE or bottom - top < MIN_ROI_SIDE:
        return None

    gray = _to_gray(frame[top:bottom, left:right])
    if gray.size == 0:
        return None
    gray = _resize_to_height(gray, max(int(reference_height), MIN_ROI_SIDE))

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = float(laplacian.var())
    if variance <= 0.0:
        return 0.0
    scale = max(float(reference), 1e-6)
    return variance / (variance + scale)


def annotate_sharpness(
    frame: np.ndarray,
    detections: Sequence[Detection],
    *,
    frame_size: tuple[float, float] | None = None,
    min_height_ratio: float = 0.0,
    reference_height: int = SHARPNESS_REFERENCE_HEIGHT,
    reference: float = LAPLACIAN_REFERENCE,
) -> list[Detection]:
    """给检测补上 :attr:`Detection.sharpness`（返回新列表，原对象不变）。

    只给"高度达标、**有可能**成为主要人物"的检测计算清晰度：过小的人物反正会被
    ``min_person_height_ratio`` 过滤掉，不必为它们花时间。剩余检测保持
    ``sharpness=None``（未知 ⇒ 照旧放行），因此结果与旧行为逐帧一致。
    """
    items = list(detections)
    if frame is None or getattr(frame, "size", 0) == 0 or not items:
        return items

    size = (
        (float(frame.shape[1]), float(frame.shape[0]))
        if frame_size is None
        else (float(frame_size[0]), float(frame_size[1]))
    )
    ceiling = max(float(min_height_ratio), 0.0) * size[1]

    annotated: list[Detection] = []
    for item in items:
        if ceiling > 0.0 and item.height < ceiling:
            annotated.append(item)
            continue
        value = region_sharpness(
            frame,
            item.bbox,
            reference_height=reference_height,
            reference=reference,
        )
        annotated.append(item if value is None else replace(item, sharpness=value))
    return annotated


__all__ = [
    "LAPLACIAN_REFERENCE",
    "MIN_ROI_SIDE",
    "SHARPNESS_REFERENCE_HEIGHT",
    "annotate_sharpness",
    "region_sharpness",
]
