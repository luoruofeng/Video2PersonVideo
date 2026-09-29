"""视觉侧测试：头部 / 嘴部 ROI 定位、帧间差分、嘴动强度。"""

from __future__ import annotations

import numpy as np
import pytest

from video2personvideo.core.mouth import (
    FaceLocator,
    MouthActivityAnalyzer,
    clip_roi,
    fallback_head_box,
    head_box_from_keypoints,
    mouth_activity,
    mouth_roi,
    reference_roi,
    roi_motion,
    to_gray,
)

FRAME_SIZE = (320.0, 240.0)
BBOX = (20.0, 60.0, 110.0, 220.0)


def _blank(value: int = 30) -> np.ndarray:
    return np.full((240, 320, 3), value, dtype=np.uint8)


# ------------------------------------------------------------------ 灰度
def test_to_gray_uses_bgr_weights() -> None:
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    frame[0, 0] = (0, 255, 0)  # 纯绿（BGR）

    gray = to_gray(frame)

    assert gray.shape == (2, 2)
    assert int(gray[0, 0]) == pytest.approx(149, abs=2)


def test_to_gray_passes_through_gray() -> None:
    frame = np.full((4, 4), 7, dtype=np.uint8)
    assert to_gray(frame).shape == (4, 4)


# ------------------------------------------------------------------ ROI 定位
def test_head_box_from_keypoints_contains_face(pose_keypoints) -> None:
    keypoints = pose_keypoints(BBOX)
    head = head_box_from_keypoints(keypoints, BBOX)

    assert head is not None
    x1, y1, x2, y2 = head
    nose_x, nose_y = keypoints[0][0], keypoints[0][1]
    assert x1 < nose_x < x2
    assert y1 <= nose_y < y2
    assert not (x2 - x1) > (y2 - y1)  # 头略窄于高


def test_head_box_without_keypoints() -> None:
    assert head_box_from_keypoints(None, BBOX) is None


def test_head_box_without_face_keypoints() -> None:
    keypoints = tuple((0.0, 0.0, 0.0) for _ in range(17))
    assert head_box_from_keypoints(keypoints, BBOX) is None


def test_mouth_roi_sits_below_reference_roi(pose_keypoints) -> None:
    head = head_box_from_keypoints(pose_keypoints(BBOX), BBOX)
    assert head is not None

    mouth = mouth_roi(head)
    reference = reference_roi(head)

    assert mouth[1] > reference[1]
    assert mouth[0] == reference[0]
    assert reference[1] + reference[3] <= mouth[1] + mouth[3]
    assert all(isinstance(value, int) for value in mouth)


def test_clip_roi_keeps_inside_frame() -> None:
    # 左上角越界：平移回画面内（尺寸不动，避免把 ROI 从嘴上挪开）
    assert clip_roi((-10, -10, 40, 40), FRAME_SIZE) == (0, 0, 40, 40)
    # 右下角越界：就地裁掉超出部分
    assert clip_roi((300, 220, 40, 40), FRAME_SIZE) == (300, 220, 20, 20)


def test_clip_roi_rejects_degenerate_areas() -> None:
    assert clip_roi((10, 10, 1, 5), FRAME_SIZE) is None
    assert clip_roi((10, 10, 5, 5), (0.0, 0.0)) is None


def test_fallback_head_box_needs_enough_pixels() -> None:
    tiny = (10.0, 10.0, 30.0, 30.0)  # 身高 20px → 头高 6px
    assert fallback_head_box(tiny, FRAME_SIZE) is None

    head = fallback_head_box(BBOX, FRAME_SIZE)
    assert head is not None
    assert head[1] == pytest.approx(BBOX[1])
    assert head[3] - head[1] <= 0.5 * FRAME_SIZE[1]


# ------------------------------------------------------------------ 运动测量
def test_roi_motion_detects_change() -> None:
    before = _blank(0)
    after = _blank(0)
    after[100:110, 50:60] = 255  # 只改动了 ROI 里的一小块

    assert roi_motion(before, after, (50, 100, 10, 10)) > 0.0
    assert roi_motion(before, after, (200, 10, 20, 20)) == 0.0


def test_mouth_activity_ignores_whole_head_motion() -> None:
    """整颗头一起平移（嘴与上半脸同时变）时，嘴动强度应为 0。"""
    mouth = (54, 90, 22, 18)
    reference = (54, 62, 22, 19)

    before = _blank(0)
    after = _blank(0)
    before[60:110, 50:80] = 100  # 头部整体灰块
    after[60:110, 50:80] = 200

    assert mouth_activity(before, after, mouth, reference) == 0.0


def test_mouth_activity_detects_mouth_only_motion() -> None:
    mouth = (54, 90, 22, 18)
    reference = (54, 62, 22, 19)

    before = _blank(0)
    after = _blank(0)
    after[90:108, 54:76] = 255  # 只有嘴部区域变了

    assert mouth_activity(before, after, mouth, reference) > 0.0


# ------------------------------------------------------------------ 分析器
def test_analyzer_first_frame_has_no_reference() -> None:
    analyzer = MouthActivityAnalyzer(face_locator=None)

    assert analyzer.activity(_blank(), [BBOX]) == [0.0]


def test_analyzer_flags_the_moving_mouth() -> None:
    analyzer = MouthActivityAnalyzer(face_locator=None)
    speaking = (20.0, 60.0, 110.0, 220.0)
    silent = (200.0, 60.0, 290.0, 220.0)
    mouth = mouth_roi(fallback_head_box(speaking, FRAME_SIZE))

    frames = []
    for index in range(6):
        frame = _blank()
        if index % 2:  # 只有说话的人嘴部在动
            frame[mouth[1] : mouth[1] + mouth[3], mouth[0] : mouth[0] + mouth[2]] = 240
        frames.append(frame)

    analyzer.activity(frames[0], [speaking, silent])
    values = analyzer.activity(frames[1], [speaking, silent])

    assert values[0] > 0.0
    assert values[1] == 0.0


def test_analyzer_returns_zero_without_roi() -> None:
    analyzer = MouthActivityAnalyzer(face_locator=None, fallback_ratio=False)
    analyzer.activity(_blank(), [BBOX])
    tiny = (1.0, 1.0, 20.0, 20.0)  # 既无关键点又禁用兜底 → 测不出来

    assert analyzer.activity(_blank(), [tiny]) == [0.0]


def test_analyzer_reset_clears_reference() -> None:
    analyzer = MouthActivityAnalyzer(face_locator=None)
    analyzer.activity(_blank(), [BBOX])
    analyzer.reset()

    assert analyzer.activity(_blank(), [BBOX]) == [0.0]


# ------------------------------------------------------------- Haar 人脸兜底
class _StubCascade:
    """假装在搜索区域 (5, 5) 处找到一张 20×20 的脸。"""

    def detectMultiScale(self, region, **kwargs):  # noqa: N802 - 模拟 OpenCV 接口
        return [(5, 5, 20, 20)]


def test_face_locator_maps_coordinates() -> None:
    locator = FaceLocator()
    locator._cascade = _StubCascade()

    face = locator.locate(_blank(), (100.0, 40.0, 200.0, 200.0))

    assert face == (105.0, 45.0, 125.0, 65.0)


def test_face_locator_without_cascade_returns_none() -> None:
    locator = FaceLocator()
    locator._cascade = None

    assert locator.available is False
    assert locator.locate(_blank(), BBOX) is None


def test_face_locator_skips_tiny_boxes() -> None:
    locator = FaceLocator()
    locator._cascade = _StubCascade()

    assert locator.locate(_blank(), (10.0, 10.0, 14.0, 14.0)) is None
