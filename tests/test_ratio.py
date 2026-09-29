"""比例模块测试：解析、校验、目标像素推导（M5 / 不变量 1）。"""

from __future__ import annotations

import pytest

from video2personvideo.core.ratio import (
    BASE_SHORT_SIDE,
    PRESET_BY_NAME,
    PRESET_RATIOS,
    AspectRatio,
    RatioError,
    default_ratio,
    max_crop_size,
    parse_ratio,
    preset_names,
    resolve_ratio,
    scale_factor,
    target_size_for,
)

EXPECTED_PRESETS = {
    "16:9": (1920, 1080),
    "9:16": (1080, 1920),
    "1:1": (1080, 1080),
    "4:3": (1440, 1080),
    "3:4": (1080, 1440),
    "4:5": (1080, 1350),
    "5:4": (1350, 1080),
    "3:2": (1620, 1080),
    "2:3": (1080, 1620),
    "21:9": (2520, 1080),
    "2:1": (2160, 1080),
}


@pytest.mark.parametrize(("name", "target"), EXPECTED_PRESETS.items())
def test_presets_carry_target_pixels(name: str, target: tuple[int, int]) -> None:
    ratio = PRESET_BY_NAME[name]
    assert ratio.target_size == target
    assert ratio.target_width % 2 == 0 and ratio.target_height % 2 == 0
    assert abs(ratio.value - target[0] / target[1]) < 1e-6


def test_preset_table_covers_required_ratios() -> None:
    assert set(preset_names()) == set(EXPECTED_PRESETS)
    assert len(PRESET_RATIOS) == len(EXPECTED_PRESETS)


def test_default_ratio_is_portrait() -> None:
    assert default_ratio().name == "9:16"


@pytest.mark.parametrize(
    ("text", "expected_value"),
    [
        ("9:16", 9 / 16),
        ("9 : 16", 9 / 16),
        ("3/5", 3 / 5),
        ("1080x1920", 9 / 16),
        ("1920*1080", 16 / 9),  # 视为目标像素写法
        ("1.7778", 16 / 9),
    ],
)
def test_parse_ratio_formats(text: str, expected_value: float) -> None:
    ratio = parse_ratio(text)
    assert isinstance(ratio, AspectRatio)
    assert ratio.value == pytest.approx(expected_value, rel=1e-3)


def test_parse_ratio_pixel_spec_keeps_pixels() -> None:
    ratio = parse_ratio("1080x1920")
    assert ratio.name == "9:16"
    assert ratio.target_size == (1080, 1920)


def test_parse_ratio_bumps_odd_pixels_to_even() -> None:
    ratio = parse_ratio("1081x1921")
    assert ratio.target_width % 2 == 0
    assert ratio.target_height % 2 == 0
    assert ratio.target_size == (1082, 1922)


def test_parse_ratio_custom_ratio_derives_target() -> None:
    ratio = parse_ratio("3:5")
    assert ratio.target_size == (1080, 1800)
    # 短边固定为 1080，且为偶数
    assert min(ratio.target_size) == BASE_SHORT_SIDE


def test_parse_ratio_with_explicit_target() -> None:
    ratio = parse_ratio("9:16", target=(540, 960))
    assert ratio.target_size == (540, 960)


def test_parse_ratio_accepts_object_identity() -> None:
    ratio = PRESET_BY_NAME["1:1"]
    assert parse_ratio(ratio) is ratio


@pytest.mark.parametrize(
    "bad",
    ["", "   ", "0:16", "9:0", "-1:2", "abc", "0", "0x0", "30:1", "1:30"],
)
def test_parse_ratio_rejects_illegal(bad: str) -> None:
    with pytest.raises(RatioError):
        parse_ratio(bad)


def test_ratio_rejects_odd_target_size() -> None:
    with pytest.raises(RatioError):
        AspectRatio("x", 1, 1, 101, 100)


def test_ratio_rejects_mismatched_target() -> None:
    with pytest.raises(RatioError):
        AspectRatio("16:9", 16, 9, 1000, 1000)


def test_target_size_for_short_side() -> None:
    assert target_size_for(1, 1) == (1080, 1080)
    assert target_size_for(9, 16) == (1080, 1920)
    assert target_size_for(16, 9) == (1920, 1080)


def test_resolve_ratio_prefers_explicit_target() -> None:
    assert resolve_ratio("9:16", target_width=720, target_height=1280).target_size == (720, 1280)
    assert resolve_ratio("9:16", target_width=720).target_size == (1080, 1920)


def test_max_crop_size_fits_inside_frame() -> None:
    ratio = PRESET_BY_NAME["9:16"]
    width, height = max_crop_size(1920, 1080, ratio)
    assert width <= 1920 and height <= 1080
    assert abs(width / height - ratio.value) < 1e-9

    width, height = max_crop_size(1080, 1920, ratio)
    assert (width, height) == (1080.0, 1920.0)


def test_scale_factor() -> None:
    assert scale_factor(540, 960, (1080, 1920)) == pytest.approx(2.0)
    assert scale_factor(0, 0, (1080, 1920)) == 0.0
