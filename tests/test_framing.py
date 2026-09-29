"""构图模块测试：取景框几何 + compute_target_box 纯函数（M5 覆盖率重点）。"""

from __future__ import annotations

import pytest

from video2personvideo.core.framing import (
    DEFAULT_FRAMING_PARAMS,
    MODE_CENTER,
    MODE_CLOSEUP,
    MODE_FULLBODY,
    MODE_HALFBODY,
    CropBox,
    FramingParams,
    center_box,
    classify_composition,
    compute_target_box,
    compute_upper_body_box,
    interpolate_boxes,
    is_truncated,
    mode_label,
)
from video2personvideo.core.pose import BodyAnchors
from video2personvideo.core.ratio import PRESET_BY_NAME, parse_ratio

PORTRAIT = PRESET_BY_NAME["9:16"]
LANDSCAPE = PRESET_BY_NAME["16:9"]
SQUARE = PRESET_BY_NAME["1:1"]


# ------------------------------------------------------------------ CropBox
def test_clamp_translates_without_resizing() -> None:
    box = CropBox(-30, -40, 100, 200)
    clamped = box.clamp_to(640, 480)
    assert clamped.w == 100 and clamped.h == 200
    assert clamped.x == 0 and clamped.y == 0


def test_clamp_pushes_box_back_inside() -> None:
    box = CropBox(600, 400, 100, 200)
    clamped = box.clamp_to(640, 480)
    assert clamped.x == pytest.approx(540)
    assert clamped.y == pytest.approx(280)
    assert clamped.w == 100 and clamped.h == 200


def test_fit_shrinks_preserving_ratio() -> None:
    box = CropBox(-10, -10, 400, 800)  # 9:16
    fitted = box.fit_to(200, 200)
    assert fitted.w <= 200 and fitted.h <= 200
    assert fitted.aspect == pytest.approx(0.5)
    assert fitted.x >= 0 and fitted.y >= 0


def test_as_int_even_is_even_and_inside() -> None:
    box = CropBox(3.4, 5.6, 101.7, 203.2)
    x, y, width, height = box.as_int_even(160, 240)
    assert width % 2 == 0 and height % 2 == 0
    assert x >= 0 and y >= 0
    assert x + width <= 160 and y + height <= 240


def test_matches_ratio_helper() -> None:
    assert CropBox(0, 0, 540, 960).matches_ratio(PORTRAIT, tol=1e-9)
    assert not CropBox(0, 0, 540, 961).matches_ratio(PORTRAIT, tol=1e-9)


def test_clamp_rejects_invalid_frame() -> None:
    with pytest.raises(ValueError):
        CropBox(0, 0, 10, 10).clamp_to(0, 100)


# --------------------------------------------------------- 构图分档与阈值
@pytest.mark.parametrize(
    ("bbox_height_ratio", "expected"),
    [(0.95, MODE_CLOSEUP), (0.8, MODE_CLOSEUP), (0.4, MODE_HALFBODY), (0.1, MODE_FULLBODY)],
)
def test_classify_composition(bbox_height_ratio: float, expected: str) -> None:
    frame_h = 1000.0
    bbox = (100.0, 100.0, 300.0, 100.0 + bbox_height_ratio * frame_h)
    assert classify_composition(bbox, frame_h) == expected


def test_is_truncated_detects_edges() -> None:
    assert is_truncated((0, 100, 200, 900), (1000, 1000))
    assert is_truncated((100, 100, 200, 1000), (1000, 1000))
    assert not is_truncated((100, 100, 200, 900), (1000, 1000))


def test_mode_labels_are_chinese() -> None:
    assert mode_label(MODE_HALFBODY) == "半身"
    assert mode_label(MODE_FULLBODY) == "全身"
    assert mode_label("unknown-mode") == "unknown-mode"


# ------------------------------------------------------ compute_target_box
def _assert_box_inside(box: CropBox, frame_w: float, frame_h: float) -> None:
    assert box.x >= -1e-6 and box.y >= -1e-6
    assert box.x + box.w <= frame_w + 1e-6
    assert box.y + box.h <= frame_h + 1e-6


def _assert_ratio(box: CropBox, ratio) -> None:
    assert box.matches_ratio(ratio, tol=1e-6), box.describe()


def test_landscape_source_to_portrait_output() -> None:
    frame = (1920.0, 1080.0)
    bbox = (860.0, 200.0, 1060.0, 900.0)  # 高 700 / 1080 = 0.65 → 半身
    box = compute_target_box(bbox, frame, PORTRAIT)
    _assert_ratio(box, PORTRAIT)
    _assert_box_inside(box, *frame)
    assert box.mode == MODE_HALFBODY
    # 竖屏比例在横屏源里能容纳的最大宽度 = 画面高 × 比例
    assert box.w <= 1080 * PORTRAIT.value + 1e-6


def test_portrait_source_to_landscape_output() -> None:
    frame = (1080.0, 1920.0)
    bbox = (440.0, 320.0, 640.0, 1600.0)  # 高 1280 / 1920 = 0.667 → 半身
    box = compute_target_box(bbox, frame, LANDSCAPE)
    _assert_ratio(box, LANDSCAPE)
    _assert_box_inside(box, *frame)
    assert box.w == pytest.approx(1080.0)


def test_square_output() -> None:
    frame = (1920.0, 1080.0)
    bbox = (900.0, 100.0, 1100.0, 1000.0)
    box = compute_target_box(bbox, frame, SQUARE)
    _assert_ratio(box, SQUARE)
    _assert_box_inside(box, *frame)
    assert box.w == pytest.approx(box.h)


def test_person_filling_frame_is_closeup() -> None:
    frame = (1920.0, 1080.0)
    bbox = (700.0, 10.0, 1200.0, 1075.0)
    box = compute_target_box(bbox, frame, PORTRAIT)
    assert box.mode == MODE_CLOSEUP
    _assert_ratio(box, PORTRAIT)


def test_tiny_person_hits_min_box_protection() -> None:
    params = FramingParams(min_box_px=400)
    frame = (1920.0, 1080.0)
    bbox = (950.0, 500.0, 970.0, 530.0)  # 30px 高，远小于 min_box_px
    box = compute_target_box(bbox, frame, LANDSCAPE, params)
    assert box.h >= 400
    _assert_ratio(box, LANDSCAPE)
    _assert_box_inside(box, *frame)


def test_truncated_person_uses_conservative_framing() -> None:
    frame = (1920.0, 1080.0)
    complete = compute_target_box((860.0, 200.0, 1060.0, 900.0), frame, PORTRAIT)
    truncated = compute_target_box((860.0, 0.0, 1060.0, 900.0), frame, PORTRAIT)
    # 人体被上边缘截断 → 取更大的景（框不低于完整人体时的框）
    assert truncated.h >= complete.h


def test_halfbody_keeps_headroom() -> None:
    params = DEFAULT_FRAMING_PARAMS
    frame = (1920.0, 1080.0)
    bbox = (860.0, 400.0, 1060.0, 900.0)  # 高 500/1080 = 0.46 → 半身
    box = compute_target_box(bbox, frame, PORTRAIT, params)
    assert box.mode == MODE_HALFBODY
    # 头顶应当留在框内，且位于框高 headroom 附近之上
    assert box.y <= bbox[1]
    assert (bbox[1] - box.y) / box.h <= params.headroom + 0.02


def test_compute_target_box_rejects_bad_frame() -> None:
    with pytest.raises(ValueError):
        compute_target_box((0, 0, 10, 10), (0, 0), PORTRAIT)


# ------------------------------------------------ compute_upper_body_box（多人分屏）
def test_upper_body_box_zooms_far_person_into_half_body() -> None:
    """远景小人（整条 bbox 是全身）应被收成上半身，而不是全身 + 四周场景。"""
    params = FramingParams(min_box_px=2)  # 关掉最小边长保护，只比较取景逻辑
    frame = (1920.0, 1080.0)
    bbox = (930.0, 500.0, 970.0, 610.0)  # 宽 40 / 高 110，只占画面高度 10%

    plain = compute_target_box(bbox, frame, PORTRAIT, params)
    tight = compute_upper_body_box(bbox, frame, PORTRAIT, params)

    assert plain.mode == MODE_FULLBODY  # 现状：小人物被判成"全身"，于是连腿一起框
    _assert_ratio(tight, PORTRAIT)
    _assert_box_inside(tight, *frame)
    # 只装上半身：框底停在人物脚之上
    assert tight.y <= bbox[1]
    assert tight.y + tight.h < bbox[3]
    # 左右不再有"一大片非人物场景"：人物宽度占了框宽的大头
    assert (bbox[2] - bbox[0]) / tight.w > 0.8
    # 总面积明显小于"整条 bbox 取景"
    assert tight.area < plain.area * 0.5


def test_upper_body_box_uses_keypoint_anchors() -> None:
    """有关键点时，上半身以"头顶 → 腰"为准，下半身不进框。"""
    frame = (1920.0, 1080.0)
    bbox = (930.0, 500.0, 970.0, 900.0)
    anchors = BodyAnchors(head_top=500.0, waist=620.0)

    box = compute_upper_body_box(bbox, frame, PORTRAIT, anchors=anchors)

    _assert_ratio(box, PORTRAIT)
    _assert_box_inside(box, *frame)
    assert box.y <= anchors.head_top  # 头顶留白
    assert box.y + box.h >= anchors.waist  # 腰在框内
    assert box.y + box.h < bbox[3]  # 腰以下不进框


def test_upper_body_box_keeps_torso_only_box_intact() -> None:
    """近景 / 坐姿（人物框本身就是上半身）不该被再切一刀。"""
    frame = (1920.0, 1080.0)
    bbox = (900.0, 300.0, 1000.0, 430.0)  # 宽 100 / 高 130：矮胖框 → 整条都算上半身

    box = compute_upper_body_box(bbox, frame, PORTRAIT)

    _assert_ratio(box, PORTRAIT)
    assert box.y <= bbox[1]
    assert box.y + box.h >= bbox[3]
    assert (bbox[2] - bbox[0]) / box.w > 0.8


def test_upper_body_box_ignores_degenerate_anchors() -> None:
    """锚点跨度退化（头顶与腰重合）时退回 bbox 估算，不产生畸形框。"""
    frame = (1920.0, 1080.0)
    bbox = (930.0, 500.0, 970.0, 610.0)

    plain = compute_upper_body_box(bbox, frame, PORTRAIT)
    degenerate = compute_upper_body_box(bbox, frame, PORTRAIT, anchors=BodyAnchors(500.0, 500.0))

    assert degenerate.h == pytest.approx(plain.h)
    assert degenerate.y == pytest.approx(plain.y)


def test_upper_body_box_min_size_protection() -> None:
    params = FramingParams(min_box_px=400)
    frame = (1920.0, 1080.0)
    bbox = (950.0, 500.0, 970.0, 530.0)

    box = compute_upper_body_box(bbox, frame, LANDSCAPE, params)

    assert box.h >= 400
    _assert_ratio(box, LANDSCAPE)
    _assert_box_inside(box, *frame)


@pytest.mark.parametrize("name", ["16:9", "9:16", "1:1", "4:5", "2:1"])
def test_upper_body_box_keeps_ratio_and_inside_frame(name: str) -> None:
    ratio = parse_ratio(name)
    frame = (1920.0, 1080.0)
    bbox = (930.0, 500.0, 970.0, 610.0)

    box = compute_upper_body_box(bbox, frame, ratio)

    _assert_ratio(box, ratio)
    _assert_box_inside(box, *frame)


def test_upper_body_box_rejects_bad_frame() -> None:
    with pytest.raises(ValueError):
        compute_upper_body_box((0, 0, 10, 10), (0, 0), PORTRAIT)


def test_framing_params_validation() -> None:
    with pytest.raises(ValueError):
        FramingParams(closeup_fill=0.0)
    with pytest.raises(ValueError):
        FramingParams(closeup_ratio=0.2, halfbody_ratio=0.5)
    with pytest.raises(ValueError):
        FramingParams(headroom=0.8)


# ------------------------------------------------------------ center / 插值
def test_center_box_is_largest_and_centered() -> None:
    box = center_box((1920.0, 1080.0), PORTRAIT)
    assert box.mode == MODE_CENTER
    _assert_ratio(box, PORTRAIT)
    assert box.cx == pytest.approx(960.0)
    assert box.cy == pytest.approx(540.0)


def test_interpolate_boxes_is_linear_and_ratio_safe() -> None:
    first = CropBox(0.0, 0.0, 540.0, 960.0)
    second = CropBox(200.0, 40.0, 270.0, 480.0)
    middle = interpolate_boxes(first, second, 0.5, PORTRAIT)
    assert middle.cx == pytest.approx((first.cx + second.cx) / 2)
    assert middle.cy == pytest.approx((first.cy + second.cy) / 2)
    assert middle.h == pytest.approx((first.h + second.h) / 2)
    _assert_ratio(middle, PORTRAIT)


def test_interpolate_handles_missing_endpoints() -> None:
    box = CropBox(0.0, 0.0, 540.0, 960.0)
    assert interpolate_boxes(None, box, 0.3, PORTRAIT) is box
    assert interpolate_boxes(box, None, 0.3, PORTRAIT) is box


def test_interpolate_clamps_t() -> None:
    first = CropBox(0.0, 0.0, 540.0, 960.0)
    second = CropBox(100.0, 0.0, 540.0, 960.0)
    assert interpolate_boxes(first, second, 5.0, PORTRAIT).cx == pytest.approx(370.0)
    assert interpolate_boxes(first, second, -5.0, PORTRAIT).cx == pytest.approx(270.0)


def test_ratio_variants_all_produce_valid_boxes() -> None:
    frame = (1920.0, 1080.0)
    bbox = (860.0, 200.0, 1060.0, 900.0)
    for name in ("16:9", "9:16", "1:1", "4:5", "21:9", "2:1"):
        ratio = parse_ratio(name)
        box = compute_target_box(bbox, frame, ratio)
        _assert_ratio(box, ratio)
        _assert_box_inside(box, *frame)
