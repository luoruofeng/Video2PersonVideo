"""目标长宽比：定义、解析、校验与目标像素推导。

纯数据层，不依赖 OpenCV / Qt，方便被单元测试穷举。

设计不变量 1 要求 ``AspectRatio`` 同时承载两件事：

* **宽高比值**（``w / h``）：决定输出视频每一帧的形状；
* **目标像素尺寸**（``target_width × target_height``）：决定输出的清晰度。

输出尺寸一旦确定就全程不变（不变量 2），因此目标宽高都必须为偶数
（H.264 的硬性要求）。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from fractions import Fraction
from math import gcd

#: 自定义比例推导目标像素时使用的基准短边长度
BASE_SHORT_SIDE = 1080

#: 允许的宽高比范围（宽 / 高），超出即视为非法输入
MIN_RATIO = 0.1
MAX_RATIO = 10.0

#: 预置比例 → 默认目标像素（短边统一取 1080，宽高均为偶数）
_PRESET_TABLE: tuple[tuple[str, int, int], ...] = (
    ("16:9", 1920, 1080),
    ("9:16", 1080, 1920),
    ("1:1", 1080, 1080),
    ("4:3", 1440, 1080),
    ("3:4", 1080, 1440),
    ("4:5", 1080, 1350),
    ("5:4", 1350, 1080),
    ("3:2", 1620, 1080),
    ("2:3", 1080, 1620),
    ("21:9", 2520, 1080),
    ("2:1", 2160, 1080),
)

#: 默认比例（竖屏半身构图最常用）
DEFAULT_RATIO_NAME = "9:16"

_RATIO_RE = re.compile(r"^\s*(\d+)\s*[:/]\s*(\d+)\s*$")
_PIXEL_RE = re.compile(r"^\s*(\d+)\s*[xX*×]\s*(\d+)\s*$")
_NUMBER_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*$")


class RatioError(ValueError):
    """比例非法：格式无法识别、为 0 / 负数，或超出合理区间。"""


def even_ceil(value: float) -> int:
    """向上取整到最近的偶数（最小 2）。"""
    result = max(int(math.ceil(float(value))), 2)
    return result if result % 2 == 0 else result + 1


def even_floor(value: float) -> int:
    """向下取整到最近的偶数（最小 2）。"""
    result = int(math.floor(float(value)))
    return max(result - (result % 2), 2)


def validate_out_size(out_size: tuple[int, int]) -> tuple[int, int]:
    """校验输出尺寸：必须为正、且宽高均为偶数（H.264 要求）。"""
    out_w, out_h = int(out_size[0]), int(out_size[1])
    if out_w <= 0 or out_h <= 0:
        raise ValueError(f"输出尺寸必须为正：{out_size}")
    if out_w % 2 or out_h % 2:
        raise ValueError(f"输出尺寸宽高必须为偶数（H.264 要求）：{out_size}")
    return out_w, out_h


def target_size_for(w: int, h: int, *, base_short_side: int = BASE_SHORT_SIDE) -> tuple[int, int]:
    """按"短边取 ``base_short_side``"推导目标像素尺寸，结果保证为偶数。"""
    if w >= h:
        height = float(base_short_side)
        width = base_short_side * w / h
    else:
        width = float(base_short_side)
        height = base_short_side * h / w
    return even_ceil(width), even_ceil(height)


@dataclass(frozen=True, slots=True)
class AspectRatio:
    """一个具体的输出比例：既是形状（``w:h``），也是尺寸（目标像素）。"""

    name: str
    w: int
    h: int
    target_width: int
    target_height: int

    def __post_init__(self) -> None:
        if self.w <= 0 or self.h <= 0:
            raise RatioError(f"比例必须为正整数：{self.name}")
        if self.target_width <= 0 or self.target_height <= 0:
            raise RatioError(f"目标像素必须为正：{self.target_width}x{self.target_height}")
        if self.target_width % 2 or self.target_height % 2:
            raise RatioError(
                f"目标像素宽高必须为偶数（H.264 要求）：{self.target_width}x{self.target_height}"
            )
        if abs(self.value - self.target_width / self.target_height) > 1e-3:
            raise RatioError(
                f"目标像素 {self.target_width}x{self.target_height} 与比例 {self.name} 不匹配"
            )

    # ------------------------------------------------------------- 只读属性
    @property
    def value(self) -> float:
        """宽高比（宽 / 高）。"""
        return self.w / self.h

    @property
    def target_size(self) -> tuple[int, int]:
        """OpenCV 的 ``(宽, 高)``。"""
        return self.target_width, self.target_height

    def with_target(self, width: int, height: int) -> AspectRatio:
        """换一份目标像素，比例保持不变。"""
        return AspectRatio(self.name, self.w, self.h, int(width), int(height))

    def describe(self) -> str:
        return f"{self.name} · {self.target_width}×{self.target_height}"

    def __str__(self) -> str:  # pragma: no cover - 便于日志
        return self.name


def _build_presets() -> tuple[AspectRatio, ...]:
    result: list[AspectRatio] = []
    for name, width, height in _PRESET_TABLE:
        w_text, h_text = name.split(":")
        result.append(AspectRatio(name, int(w_text), int(h_text), width, height))
    return tuple(result)


#: 预置比例表（顺序即 GUI 展示顺序）
PRESET_RATIOS: tuple[AspectRatio, ...] = _build_presets()

#: 按名称索引的预置比例表
PRESET_BY_NAME: dict[str, AspectRatio] = {item.name: item for item in PRESET_RATIOS}


def preset_names() -> tuple[str, ...]:
    """预置比例名称列表（供 GUI / 文档展示）。"""
    return tuple(item.name for item in PRESET_RATIOS)


def default_ratio() -> AspectRatio:
    """返回默认比例。"""
    return PRESET_BY_NAME[DEFAULT_RATIO_NAME]


def _check_range(value: float, text: str) -> None:
    if not MIN_RATIO <= value <= MAX_RATIO:
        raise RatioError(
            f"比例 {text} 超出合理区间 [{MIN_RATIO:g}, {MAX_RATIO:g}]，宽高比 = {value:.4f}"
        )


def _even_pair(width: int, height: int) -> tuple[int, int]:
    """把像素尺寸规整为偶数（向上取整），满足不变量 2。"""
    return even_ceil(width), even_ceil(height)


def parse_ratio(
    spec: str | AspectRatio,
    *,
    target: tuple[int, int] | None = None,
    base_short_side: int = BASE_SHORT_SIDE,
) -> AspectRatio:
    """解析比例描述，支持四种写法：

    * 预置名称：``"9:16"``、``"16:9"``
    * 自定义比例：``"3:5"``、``"7/4"``（目标像素按短边 1080 推导）
    * 目标像素：``"1080x1920"``、``"1920*1080"``（自动约分出名称为比例并向上取偶数）
    * 纯数字比值：``"1.7778"``

    :param target: 显式指定目标像素 ``(宽, 高)``，覆盖默认推导
    :raises RatioError: 格式非法、为 0 / 负数或超出合理区间
    """
    if isinstance(spec, AspectRatio):
        return spec if target is None else spec.with_target(*target)

    raw = str(spec).strip()
    if not raw:
        raise RatioError("比例不能为空")

    match = _RATIO_RE.match(raw)
    if match:
        w, h = int(match.group(1)), int(match.group(2))
        if w <= 0 or h <= 0:
            raise RatioError(f"比例必须为正整数：{raw}")
        _check_range(w / h, raw)
        name = f"{w}:{h}"
        if target is None:
            preset = PRESET_BY_NAME.get(name)
            if preset is not None:
                return preset
            width, height = target_size_for(w, h, base_short_side=base_short_side)
        else:
            width, height = _even_pair(*target)
        return AspectRatio(name, w, h, width, height)

    match = _PIXEL_RE.match(raw)
    if match:
        raw_width, raw_height = int(match.group(1)), int(match.group(2))
        if raw_width <= 0 or raw_height <= 0:
            raise RatioError(f"目标像素必须为正整数：{raw}")
        width, height = _even_pair(raw_width, raw_height)
        divisor = gcd(width, height)
        w, h = width // divisor, height // divisor
        _check_range(w / h, raw)
        return AspectRatio(f"{w}:{h}", w, h, width, height)

    match = _NUMBER_RE.match(raw)
    if match:
        value = float(match.group(1))
        if value <= 0:
            raise RatioError(f"比例必须为正：{raw}")
        _check_range(value, raw)
        fraction = Fraction(value).limit_denominator(100)
        w, h = int(fraction.numerator), int(fraction.denominator)
        if target is None:
            width, height = target_size_for(w, h, base_short_side=base_short_side)
        else:
            width, height = _even_pair(*target)
        return AspectRatio(f"{w}:{h}", w, h, width, height)

    raise RatioError(f"无法识别的比例格式：{raw!r}（示例：9:16 / 1080x1920 / 1.7778）")


def resolve_ratio(
    spec: str | AspectRatio,
    *,
    target_width: int | None = None,
    target_height: int | None = None,
) -> AspectRatio:
    """按"比例 + 可选目标像素"解析，两者只给一个时忽略另一来源。"""
    target = None
    if target_width and target_height and target_width > 0 and target_height > 0:
        target = (int(target_width), int(target_height))
    return parse_ratio(spec, target=target)


def max_crop_size(
    frame_w: float, frame_h: float, ratio: AspectRatio
) -> tuple[float, float]:
    """在 ``frame_w × frame_h`` 的画幅内，能容纳的最大等比取景框尺寸。"""
    if frame_w <= 0 or frame_h <= 0:
        raise ValueError(f"画面尺寸必须为正：{frame_w}x{frame_h}")
    width = float(frame_w)
    height = width / ratio.value
    if height > frame_h:
        height = float(frame_h)
        width = height * ratio.value
    return width, height


def scale_factor(crop_w: float, crop_h: float, target_size: tuple[int, int]) -> float:
    """取景框 → 目标像素的统一缩放倍率（取景框与目标比例一致时两向相等）。"""
    if crop_w <= 0 or crop_h <= 0:
        return 0.0
    return float(target_size[0]) / float(crop_w)
