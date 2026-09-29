"""裁剪模块测试：输出尺寸恒定、偶数对齐、无黑边（M5 / 不变量 2、4）。"""

from __future__ import annotations

import numpy as np
import pytest

from video2personvideo.core.crop import crop_frame, validate_out_size
from video2personvideo.core.framing import CropBox, center_box
from video2personvideo.core.ratio import PRESET_BY_NAME, parse_ratio


def _frame(width: int = 1920, height: int = 1080, value: int = 180) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


def test_output_size_is_exactly_target() -> None:
    ratio = PRESET_BY_NAME["9:16"]
    frame = _frame()
    box = center_box((frame.shape[1], frame.shape[0]), ratio)
    output = crop_frame(frame, box, ratio.target_size)
    assert output.shape[:2] == (ratio.target_height, ratio.target_width)


@pytest.mark.parametrize("name", ["16:9", "9:16", "1:1", "4:5", "21:9", "2:1", "3:4"])
def test_every_preset_produces_its_own_size(name: str) -> None:
    ratio = parse_ratio(name)
    frame = _frame(1280, 720)
    box = center_box((1280, 720), ratio)
    output = crop_frame(frame, box, ratio.target_size)
    assert output.shape[1] == ratio.target_width
    assert output.shape[0] == ratio.target_height
    assert abs(output.shape[1] / output.shape[0] - ratio.value) < 1e-9


def test_solid_frame_has_no_black_border() -> None:
    """源帧是纯色时，输出也不该出现黑边（不变量 4）。"""
    ratio = PRESET_BY_NAME["9:16"]
    frame = _frame(640, 480, value=200)
    box = center_box((640, 480), ratio)
    output = crop_frame(frame, box, ratio.target_size)
    assert output.min() > 100
    assert int(output.min()) == int(output.max())


def test_upscale_keeps_uniform_color() -> None:
    ratio = PRESET_BY_NAME["16:9"]
    frame = _frame(32, 32, value=90)  # 放大到 1920x1080
    output = crop_frame(frame, CropBox(0, 0, 32, 32), ratio.target_size)
    assert output.shape[:2] == (1080, 1920)
    assert output.min() == 90 and output.max() == 90


def test_out_of_bounds_box_is_clamped_not_black() -> None:
    frame = _frame(640, 480, value=150)
    box = CropBox(-500, -500, 200, 400)
    output = crop_frame(frame, box, (128, 256))
    assert output.shape[:2] == (256, 128)
    assert output.min() > 100


def test_oversized_box_is_shrunk_to_fit() -> None:
    frame = _frame(640, 480, value=150)
    box = CropBox(0, 0, 5000, 5000)
    output = crop_frame(frame, box, (108, 108))
    assert output.shape[:2] == (108, 108)
    assert output.min() > 100


def test_tiny_box_still_produces_target_size() -> None:
    frame = _frame(640, 480, value=150)
    output = crop_frame(frame, CropBox(100.4, 100.6, 3.2, 5.8), (240, 426))
    assert output.shape[:2] == (426, 240)


@pytest.mark.parametrize(("width", "height"), [(0, 100), (100, 0), (-2, 100), (101, 100), (100, 101)])
def test_invalid_out_size_rejected(width: int, height: int) -> None:
    with pytest.raises(ValueError):
        validate_out_size((width, height))


def test_crop_frame_rejects_bad_input() -> None:
    ratio = PRESET_BY_NAME["1:1"]
    box = CropBox(0, 0, 100, 100)
    with pytest.raises(ValueError):
        crop_frame(None, box, ratio.target_size)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        crop_frame(np.zeros((10, 10, 3), np.uint8), box, (101, 100))
    with pytest.raises(ValueError):
        crop_frame(np.zeros((0, 0, 3), np.uint8), box, (100, 100))


def test_crop_frame_does_not_modify_source() -> None:
    frame = _frame(320, 240, value=77)
    before = frame.copy()
    crop_frame(frame, CropBox(20, 20, 100, 100), (64, 64))
    assert np.array_equal(frame, before)
