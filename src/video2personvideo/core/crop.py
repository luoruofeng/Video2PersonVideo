"""像素级裁剪与缩放：源帧 → 固定输出尺寸。

流水线固定为 ``源帧 → 按取景框切片 → 缩放到固定目标尺寸``（不变量 2），
输出尺寸是常量、取景框是变量。任何情况下输出帧的尺寸都恒等于 ``out_size``，
绝不出现黑边、变形或越界（不变量 4）。

多人分屏时，这里还负责**把若干小窗口拼回一张输出帧**（:func:`compose_multi_frame`）：
每个窗口各自切片缩放到自己的格子，贴到一块底色画布上。格子尺寸由
:mod:`~video2personvideo.core.layout` 保证为偶数、互不重叠、不越界。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import cv2
import numpy as np

from .framing import CropBox
from .ratio import validate_out_size

__all__ = [
    "WindowSlice",
    "compose_multi_frame",
    "crop_frame",
    "validate_out_size",
]

#: 多人分屏时的默认底色（BGR），缝隙与空窗都填它
DEFAULT_BACKGROUND: tuple[int, int, int] = (18, 18, 18)


class WindowSlice(NamedTuple):
    """多人分屏里的一格：从哪切（``box``）、贴到哪（``rect``）、给谁用（``subject``）。"""

    box: CropBox
    #: 输出画布上的目标矩形 ``(x, y, w, h)``，宽高均为偶数
    rect: tuple[int, int, int, int]
    #: 该窗口对应的人物框（源画面坐标），仅画框标注时使用
    subject: tuple[float, float, float, float] | None = None

    @property
    def size(self) -> tuple[int, int]:
        return int(self.rect[2]), int(self.rect[3])


def crop_frame(
    frame: np.ndarray, box: CropBox, out_size: tuple[int, int]
) -> np.ndarray:
    """按取景框切片并缩放到固定输出尺寸。

    :param frame: 源帧（BGR）
    :param box: 源画面坐标系下的取景框
    :param out_size: 目标输出尺寸 ``(宽, 高)``，宽高必须为偶数
    :returns: 尺寸**恒等于** ``out_size`` 的新帧
    :raises ValueError: 输出尺寸非法或源帧为空
    :raises RuntimeError: 切片越界 / 缩放结果尺寸不符（绝不静默继续）
    """
    if frame is None or not isinstance(frame, np.ndarray) or frame.ndim not in (2, 3):
        raise ValueError("源帧必须是 BGR ndarray")

    out_w, out_h = validate_out_size(out_size)
    frame_h, frame_w = int(frame.shape[0]), int(frame.shape[1])
    if frame_w <= 0 or frame_h <= 0:
        raise ValueError(f"源帧尺寸非法：{frame_w}x{frame_h}")

    x, y, width, height = box.fit_to(frame_w, frame_h).as_int_even(frame_w, frame_h)
    if x < 0 or y < 0 or x + width > frame_w or y + height > frame_h:
        raise RuntimeError(
            f"裁剪区域越界：box=({x},{y},{width},{height})，画面={frame_w}x{frame_h}"
        )

    patch = frame[y : y + height, x : x + width]
    if patch.shape[0] != height or patch.shape[1] != width:
        raise RuntimeError(
            f"切片尺寸不符：得到 {patch.shape[1]}x{patch.shape[0]}，期望 {width}x{height}"
        )

    shrinking = width >= out_w and height >= out_h
    interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LANCZOS4
    result = cv2.resize(patch, (out_w, out_h), interpolation=interpolation)

    if result.shape[1] != out_w or result.shape[0] != out_h:
        raise RuntimeError(
            f"缩放后尺寸不符：得到 {result.shape[1]}x{result.shape[0]}，期望 {out_w}x{out_h}"
        )
    return result


def compose_multi_frame(
    frame: np.ndarray,
    windows: Sequence[WindowSlice],
    out_size: tuple[int, int],
    *,
    background: tuple[int, int, int] = DEFAULT_BACKGROUND,
    annotate: bool = False,
) -> np.ndarray:
    """把若干小窗口拼成一帧（多人分屏）。

    每个窗口先按自己的取景框切片、缩放到格子尺寸，再贴到底色画布上。
    输出尺寸**恒等于** ``out_size``；格子之间的缝隙与空窗都填 ``background``。

    :param frame: 源帧（BGR）
    :param windows: 各窗口的取景框 + 目标格子 + （可选）人物框
    :param out_size: 输出尺寸 ``(宽, 高)``，宽高必须为偶数
    :param background: 缝隙 / 空窗的填充色（BGR）
    :param annotate: 是否在每个窗口里画出该人物的检测框
    :returns: 尺寸**恒等于** ``out_size`` 的新帧
    :raises ValueError: 输出尺寸非法、源帧为空或没有任何窗口
    :raises RuntimeError: 格子越界 / 与画布不匹配（绝不静默继续）
    """
    if frame is None or not isinstance(frame, np.ndarray) or frame.ndim not in (2, 3):
        raise ValueError("源帧必须是 BGR ndarray")
    if not windows:
        raise ValueError("多人分屏至少要有一个窗口")

    out_w, out_h = validate_out_size(out_size)
    canvas = np.empty((out_h, out_w, 3), dtype=frame.dtype)
    canvas[:] = tuple(int(max(0, min(255, channel))) for channel in background)

    for window in windows:
        x, y, width, height = (int(value) for value in window.rect)
        if width <= 0 or height <= 0:
            raise RuntimeError(f"窗口格子尺寸非法：{window.rect}")
        if x < 0 or y < 0 or x + width > out_w or y + height > out_h:
            raise RuntimeError(f"窗口格子越界：{window.rect}，画布={out_w}x{out_h}")
        patch = crop_frame(frame, window.box, (width, height))
        if annotate and window.subject is not None:
            _draw_subject_in_patch(patch, window.subject, window.box)
        canvas[y : y + height, x : x + width] = patch

    return canvas


def _draw_subject_in_patch(
    patch: np.ndarray,
    subject_bbox: tuple[float, float, float, float],
    box: CropBox,
) -> None:
    """把源画面坐标下的检测框映射到某个小窗口内并画出来（就地修改）。"""
    if box.w <= 0 or box.h <= 0:
        return
    scale_x = patch.shape[1] / box.w
    scale_y = patch.shape[0] / box.h
    x1 = int(round((float(subject_bbox[0]) - box.x) * scale_x))
    y1 = int(round((float(subject_bbox[1]) - box.y) * scale_y))
    x2 = int(round((float(subject_bbox[2]) - box.x) * scale_x))
    y2 = int(round((float(subject_bbox[3]) - box.y) * scale_y))
    cv2.rectangle(patch, (x1, y1), (x2, y2), (0, 255, 0), 2)
