"""主要人物的第二道门槛：清晰度。

覆盖三块：
* ``core/sharpness.py`` 的清晰度比率本身（清晰 > 模糊、分辨率无关）；
* ``core/subject.py`` 的并列过滤（够大 **且** 够清晰才算主要人物）；
* ``core/pipeline.py`` 的整条链路（把虚化的大人物降级为背景人物）。
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from video2personvideo.config import AppConfig
from video2personvideo.core.pipeline import CropPipeline
from video2personvideo.core.ratio import resolve_ratio
from video2personvideo.core.sharpness import annotate_sharpness, region_sharpness
from video2personvideo.core.subject import (
    DEFAULT_MIN_PERSON_SHARPNESS,
    Detection,
    filter_small,
    passes_sharpness,
    select_subject,
)

RATIO = resolve_ratio("9:16")
FRAME_W, FRAME_H = 1920.0, 1080.0
FRAME_SIZE = (FRAME_W, FRAME_H)

#: 两块同样高（940px，占画面 0.87）的人物框：
#: 左边清晰，右边被虚化 —— 只有"够清晰"这一半能把它们区分开。
SHARP_BBOX = (200.0, 100.0, 600.0, 1040.0)
BLUR_BBOX = (760.0, 100.0, 1160.0, 1040.0)

#: 后半段整体虚化，再把左框那块换回清晰噪点（对焦在左边那个人身上）
_NOISE = np.random.default_rng(11).integers(0, 255, (1080, 1920, 3), dtype=np.uint8)
FRAME = cv2.GaussianBlur(_NOISE, (0, 0), 10.0)
FRAME[100:1040, 200:600] = _NOISE[100:1040, 200:600]


# ------------------------------------------------------------------ 清晰度比率
def test_sharpness_separates_in_focus_and_blurred_regions() -> None:
    sharp = region_sharpness(FRAME, SHARP_BBOX)
    blurred = region_sharpness(FRAME, BLUR_BBOX)
    assert sharp is not None and blurred is not None
    assert 0.0 <= blurred < DEFAULT_MIN_PERSON_SHARPNESS <= sharp <= 1.0


def test_sharpness_is_resolution_independent() -> None:
    """同一块画面缩小一半，清晰度比率依然接近（阈值不用跟着分辨率改）。"""
    small_frame = cv2.resize(FRAME, (960, 540), interpolation=cv2.INTER_AREA)
    big = region_sharpness(FRAME, SHARP_BBOX)
    small = region_sharpness(small_frame, tuple(value / 2.0 for value in SHARP_BBOX))
    assert big is not None and small is not None
    assert abs(big - small) < 0.1


def test_sharpness_returns_none_when_region_is_too_small() -> None:
    assert region_sharpness(FRAME, (10.0, 10.0, 12.0, 12.0)) is None


# ------------------------------------------------------------------ 并列过滤
def test_passes_sharpness_treats_unknown_as_ok() -> None:
    assert passes_sharpness(Detection(SHARP_BBOX), 0.9)  # 没算过 → 放行
    assert passes_sharpness(Detection(SHARP_BBOX, sharpness=0.8), 0.3)
    assert not passes_sharpness(Detection(SHARP_BBOX, sharpness=0.1), 0.3)
    assert passes_sharpness(Detection(SHARP_BBOX, sharpness=0.1), 0.0)  # 0 = 关闭


def test_filter_small_requires_both_size_and_sharpness() -> None:
    items = [
        Detection(SHARP_BBOX, sharpness=0.9),
        Detection(BLUR_BBOX, sharpness=0.05),
        Detection((10.0, 10.0, 100.0, 100.0), sharpness=0.9),  # 够清晰但太小
    ]
    assert len(filter_small(items, FRAME_SIZE, 0.3, 0.3)) == 1
    assert len(filter_small(items, FRAME_SIZE, 0.3, 0.0)) == 2


def test_select_subject_demotes_blurred_person() -> None:
    """虚化的人更居中、更抢镜，但清晰度不达标 → 仍旧算背景人物。"""
    items = [
        Detection(SHARP_BBOX, sharpness=0.9),
        Detection(BLUR_BBOX, sharpness=0.05),  # 更靠近画面正中
    ]
    without = select_subject(items, FRAME_SIZE, min_height_ratio=0.3)
    assert without is not None and without.bbox == BLUR_BBOX

    with_sharpness = select_subject(
        items, FRAME_SIZE, min_height_ratio=0.3, min_sharpness=0.3
    )
    assert with_sharpness is not None and with_sharpness.bbox == SHARP_BBOX


def test_annotate_sharpness_fills_only_big_detections() -> None:
    small = Detection((10.0, 10.0, 100.0, 100.0))
    big = Detection(SHARP_BBOX)
    annotated = annotate_sharpness(
        FRAME, [small, big], frame_size=FRAME_SIZE, min_height_ratio=0.3
    )
    assert annotated[0].sharpness is None  # 太小：不必花时间算
    assert annotated[1].sharpness is not None
    assert big.sharpness is None  # 原对象不变


# ------------------------------------------------------------------ 整条链路
class _StubDetector:
    """每帧都返回同样两条检测（左：清晰 / 右：虚化）。"""

    def __init__(self, bboxes) -> None:
        self._bboxes = list(bboxes)

    def detect_boxes_batch(self, frames):  # noqa: ANN001, ANN202 - 测试替身
        return [[Detection(bbox=bbox) for bbox in self._bboxes] for _ in frames]


def _run(min_sharpness: float):
    pipeline = CropPipeline(
        RATIO,
        _StubDetector([SHARP_BBOX, BLUR_BBOX]),
        min_person_height_ratio=0.3,
        min_person_sharpness=min_sharpness,
    )
    outcomes = []
    for index in range(2):
        outcomes.extend(pipeline.process(FRAME, index))
    outcomes.extend(pipeline.flush())
    return outcomes


def test_pipeline_skips_blurred_subject() -> None:
    # 不启用清晰度判定：画面正中那块（虚化）更抢镜
    baseline = _run(0.0)
    assert baseline[1].subject_bbox == BLUR_BBOX

    # 启用清晰度判定：虚化的人被降级为背景人物，留下清晰的那位
    guarded = _run(DEFAULT_MIN_PERSON_SHARPNESS)
    assert guarded[1].subject_bbox == SHARP_BBOX


# ------------------------------------------------------------------ 配置
def test_config_exposes_sharpness_threshold() -> None:
    assert AppConfig().min_person_sharpness == DEFAULT_MIN_PERSON_SHARPNESS
    assert AppConfig.from_mapping({"min_person_sharpness": "0.5"}).min_person_sharpness == 0.5
    with pytest.raises(ValueError):
        AppConfig(min_person_sharpness=1.5).validate()
