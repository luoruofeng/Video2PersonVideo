"""像素级裁剪与缩放：源帧 → 固定输出尺寸。

流水线固定为 ``源帧 → 按取景框切片 → 缩放到固定目标尺寸``（不变量 2），
输出尺寸是常量、取景框是变量。任何情况下输出帧的尺寸都恒等于 ``out_size``，
绝不出现黑边、变形或越界（不变量 4）。

拼窗口的场景（多人分屏、无人物时的「全画面适配 / 全景 + 特写」）走
:func:`compose_multi_frame`：每个窗口各自取景、缩放到自己的格子，贴到一块画布上。
格子尺寸由 :mod:`~video2personvideo.core.layout` 保证为偶数、互不重叠、不越界。

窗口有两种贴法：

* ``fit=False``（默认）：**裁满格子** —— 取景框的比例恒等于格子比例，等比放大铺满；
* ``fit=True``：**完整放下** —— 取景内容等比缩小到格子内，四周空出来的部分填底色。
  用于"整幅画面放进某一格"这类不能裁剪的场景。

画布底色可以是一块纯色，也可以是**同一帧的模糊放大版**（``blur=True``）：
后者用于全画面适配 —— 横屏素材放进竖屏时上下会空出大片，用画面自身的模糊版
填充比黑边自然得多，也不会引入任何"画面之外"的信息。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NamedTuple

import cv2
import numpy as np

from .framing import CropBox
from .ratio import validate_out_size

__all__ = [
    "DEFAULT_BACKGROUND",
    "WindowSlice",
    "blurred_cover",
    "compose_multi_frame",
    "crop_frame",
    "validate_out_size",
]

#: 多人分屏时的默认底色（BGR），缝隙与空窗都填它
DEFAULT_BACKGROUND: tuple[int, int, int] = (18, 18, 18)

#: 模糊背景的降采样倍数：先缩小再模糊再放大，比直接糊一张大图快一个数量级
BLUR_DOWNSCALE = 16

#: 模糊强度（高斯核 sigma，相对降采样后短边的比例）：越大越糊
BLUR_SIGMA_RATIO = 0.32


class WindowSlice(NamedTuple):
    """分屏里的一格：从哪切（``box``）、贴到哪（``rect``）、给谁用（``subject``）。"""

    box: CropBox
    #: 输出画布上的目标矩形 ``(x, y, w, h)``，宽高均为偶数
    rect: tuple[int, int, int, int]
    #: 该窗口对应的人物框（源画面坐标），仅画框标注时使用
    subject: tuple[float, float, float, float] | None = None
    #: ``True`` = 把取景内容**完整放下**（等比缩小到格子内，四周填底色）；
    #: ``False`` = 裁满格子（默认，要求取景框比例与格子比例一致）
    fit: bool = False
    #: 贴图不透明度（0~1）：用于"特写窗口淡入"这类过渡，1 = 完全不透明
    alpha: float = 1.0

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
    patch = _crop_patch(frame, box)
    shrinking = patch.shape[1] >= out_w and patch.shape[0] >= out_h
    interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LANCZOS4
    result = cv2.resize(patch, (out_w, out_h), interpolation=interpolation)

    if result.shape[1] != out_w or result.shape[0] != out_h:
        raise RuntimeError(
            f"缩放后尺寸不符：得到 {result.shape[1]}x{result.shape[0]}，期望 {out_w}x{out_h}"
        )
    return result


def _crop_patch(frame: np.ndarray, box: CropBox) -> np.ndarray:
    """按取景框从源帧里切片（只切不缩放），带完整的越界校验。"""
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
    return patch


def blurred_cover(frame: np.ndarray, out_size: tuple[int, int]) -> np.ndarray:
    """源帧"填满画布"后重度模糊的版本（用作全画面适配的背景，代替黑边）。

    先等比缩放到**完全覆盖**画布、居中裁一块，再降采样 → 高斯模糊 → 放大回画布尺寸：
    真正参与模糊运算的只有画布像素数的约 1/256，代价可忽略，观感也更柔和。
    """
    out_w, out_h = validate_out_size(out_size)
    frame_h, frame_w = int(frame.shape[0]), int(frame.shape[1])
    if frame_w <= 0 or frame_h <= 0:
        raise ValueError(f"源帧尺寸非法：{frame_w}x{frame_h}")

    scale = max(out_w / frame_w, out_h / frame_h)
    big_w = max(int(math.ceil(frame_w * scale)), out_w)
    big_h = max(int(math.ceil(frame_h * scale)), out_h)
    big = cv2.resize(frame, (big_w, big_h), interpolation=cv2.INTER_AREA)
    x = (big_w - out_w) // 2
    y = (big_h - out_h) // 2
    cover = big[y : y + out_h, x : x + out_w]

    small_w = max(out_w // BLUR_DOWNSCALE, 2)
    small_h = max(out_h // BLUR_DOWNSCALE, 2)
    small = cv2.resize(cover, (small_w, small_h), interpolation=cv2.INTER_AREA)
    sigma = max(min(small_w, small_h) * BLUR_SIGMA_RATIO, 0.8)
    small = cv2.GaussianBlur(small, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return cv2.resize(small, (out_w, out_h), interpolation=cv2.INTER_LINEAR)


def compose_multi_frame(
    frame: np.ndarray,
    windows: Sequence[WindowSlice],
    out_size: tuple[int, int],
    *,
    background: tuple[int, int, int] = DEFAULT_BACKGROUND,
    annotate: bool = False,
    blur: bool = False,
) -> np.ndarray:
    """把若干小窗口拼成一帧（多人分屏 / 无人物兜底）。

    每个窗口先按自己的取景框切片、缩放到格子尺寸，再贴到画布上。
    输出尺寸**恒等于** ``out_size``；格子之间的缝隙与空窗都填 ``background``。

    :param frame: 源帧（BGR）
    :param windows: 各窗口的取景框 + 目标格子 + （可选）人物框 / 贴法 / 不透明度
    :param out_size: 输出尺寸 ``(宽, 高)``，宽高必须为偶数
    :param background: 缝隙 / 空窗 / 留白的填充色（BGR）
    :param annotate: 是否在每个窗口里画出该人物的检测框
    :param blur: 画布底色是否用"同一帧的模糊放大版"（全画面适配时用，比纯色自然）
    :returns: 尺寸**恒等于** ``out_size`` 的新帧
    :raises ValueError: 输出尺寸非法、源帧为空或没有任何窗口
    :raises RuntimeError: 格子越界 / 与画布不匹配（绝不静默继续）
    """
    if frame is None or not isinstance(frame, np.ndarray) or frame.ndim not in (2, 3):
        raise ValueError("源帧必须是 BGR ndarray")
    if not windows:
        raise ValueError("至少要有一个窗口")

    out_w, out_h = validate_out_size(out_size)
    canvas = _background_canvas(frame, out_size, background, blur)

    for window in windows:
        alpha = min(max(float(window.alpha), 0.0), 1.0)
        if alpha <= 0.0:
            continue
        x, y, width, height = (int(value) for value in window.rect)
        if width <= 0 or height <= 0:
            raise RuntimeError(f"窗口格子尺寸非法：{window.rect}")
        if x < 0 or y < 0 or x + width > out_w or y + height > out_h:
            raise RuntimeError(f"窗口格子越界：{window.rect}，画布={out_w}x{out_h}")

        patch, inner = _window_patch(frame, window, (width, height), background)
        if annotate and window.subject is not None:
            _draw_subject_in_patch(patch, window.subject, window.box, inner)
        if alpha >= 1.0:
            canvas[y : y + height, x : x + width] = patch
        else:
            region = canvas[y : y + height, x : x + width]
            canvas[y : y + height, x : x + width] = cv2.addWeighted(
                region, 1.0 - alpha, patch, alpha, 0.0
            )

    return canvas


def _background_canvas(
    frame: np.ndarray,
    out_size: tuple[int, int],
    background: tuple[int, int, int],
    blur: bool,
) -> np.ndarray:
    """画布底色：模糊放大版（``blur=True``）或一块纯色。"""
    if blur:
        return blurred_cover(frame, out_size)
    out_w, out_h = validate_out_size(out_size)
    canvas = np.empty((out_h, out_w, 3), dtype=frame.dtype)
    canvas[:] = _color_tuple(background)
    return canvas


def _window_patch(
    frame: np.ndarray,
    window: WindowSlice,
    size: tuple[int, int],
    background: tuple[int, int, int],
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """取一格要贴的像素图，返回 ``(图, 图在格子里占的矩形)``。

    ``fit=False`` 时贴满整格（矩形 = 整格）；``fit=True`` 时等比缩小居中，
    返回的矩形用于标注映射（黑边区域内不画框）。
    """
    if not window.fit:
        return crop_frame(frame, window.box, size), (0, 0, int(size[0]), int(size[1]))

    out_w, out_h = int(size[0]), int(size[1])
    patch = _crop_patch(frame, window.box)
    height, width = int(patch.shape[0]), int(patch.shape[1])
    scale = min(out_w / width, out_h / height) if width > 0 and height > 0 else 1.0
    fit_w = min(max(int(round(width * scale)), 2), out_w)
    fit_h = min(max(int(round(height * scale)), 2), out_h)
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4

    canvas = np.empty((out_h, out_w, 3), dtype=frame.dtype)
    canvas[:] = _color_tuple(background)
    x = (out_w - fit_w) // 2
    y = (out_h - fit_h) // 2
    canvas[y : y + fit_h, x : x + fit_w] = cv2.resize(
        patch, (fit_w, fit_h), interpolation=interpolation
    )
    return canvas, (x, y, fit_w, fit_h)


def _color_tuple(background: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(int(max(0, min(255, channel))) for channel in background)


def _draw_subject_in_patch(
    patch: np.ndarray,
    subject_bbox: tuple[float, float, float, float],
    box: CropBox,
    inner: tuple[int, int, int, int] | None = None,
) -> None:
    """把源画面坐标下的检测框映射到某个小窗口内并画出来（就地修改）。

    ``inner`` 是该窗口图像在格子里真正占的矩形（``fit=True`` 时四周有留白），
    画框按它做偏移并夹在图像范围内，留白区域不会出现画到一半的框。
    """
    if box.w <= 0 or box.h <= 0:
        return
    left, top, inner_w, inner_h = inner or (0, 0, patch.shape[1], patch.shape[0])
    scale_x = inner_w / box.w
    scale_y = inner_h / box.h
    x1 = int(round((float(subject_bbox[0]) - box.x) * scale_x)) + left
    y1 = int(round((float(subject_bbox[1]) - box.y) * scale_y)) + top
    x2 = int(round((float(subject_bbox[2]) - box.x) * scale_x)) + left
    y2 = int(round((float(subject_bbox[3]) - box.y) * scale_y)) + top
    x1, x2 = sorted((min(max(x1, left), left + inner_w), min(max(x2, left), left + inner_w)))
    y1, y2 = sorted((min(max(y1, top), top + inner_h), min(max(y2, top), top + inner_h)))
    cv2.rectangle(patch, (x1, y1), (x2, y2), (0, 255, 0), 2)
