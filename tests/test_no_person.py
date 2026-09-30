"""画面里没有主要人物时的显示方式：全画面适配 / 全景 + 特写。

覆盖三块：
* ``core/layout.py`` 的纯几何（适配矩形、过渡几何、全景 + 特写网格）；
* ``core/crop.py`` 的合成（完整放下 / 模糊底 / 不透明度 / 旧行为不变）；
* ``core/pipeline.py`` + ``core/smoothing.py`` 的整条流水线（兜底时机、过渡渐进、
  自动退回、以及 ``scan`` / ``center`` 两档的旧行为不变）。
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from video2personvideo.config import AppConfig
from video2personvideo.core.crop import (
    WindowSlice,
    blurred_cover,
    compose_multi_frame,
    crop_frame,
)
from video2personvideo.core.framing import MODE_FIT, MODE_TILES, CropBox, center_box
from video2personvideo.core.layout import (
    DEFAULT_MULTI_PERSON_POLICY,
    compute_cell_rects,
    fit_rect,
    tiles_layout,
    transition_frame,
)
from video2personvideo.core.noperson import NoPersonDisplay, TilesPlanner, fade_alpha
from video2personvideo.core.pipeline import CropPipeline
from video2personvideo.core.ratio import resolve_ratio
from video2personvideo.core.smoothing import BoxSmoother, SmoothingParams
from video2personvideo.core.subject import Detection

RATIO = resolve_ratio("9:16")
OUT = RATIO.target_size
FRAME_W, FRAME_H = 1920.0, 1080.0
FPS = 30.0

#: 固定随机种子：合成结果逐像素比对需要可复现的素材
FRAME = np.random.default_rng(7).integers(0, 255, (1080, 1920, 3), dtype=np.uint8)
WHOLE = CropBox(0.0, 0.0, FRAME_W, FRAME_H)
LETTERBOX = fit_rect((FRAME_W, FRAME_H), OUT)
MAIN_BBOX = (760.0, 180.0, 1160.0, 900.0)
SMALL_BBOXES = ((200.0, 700.0, 350.0, 950.0), (1600.0, 660.0, 1750.0, 920.0))


# ------------------------------------------------------------------ 纯几何
def test_fit_rect_is_letterboxed_and_even() -> None:
    x, y, width, height = LETTERBOX
    assert width == OUT[0]
    assert height == 608  # 1080 / (1920/1080)
    assert width % 2 == 0 and height % 2 == 0
    assert x == 0 and y == (OUT[1] - height) // 2


def test_fit_rect_matches_canvas_when_ratios_agree() -> None:
    assert fit_rect((1080.0, 1920.0), OUT) == (0, 0, OUT[0], OUT[1])


def _transition(progress: float):  # noqa: ANN202 - 测试内部小工具
    box = center_box((FRAME_W, FRAME_H), RATIO)
    return transition_frame(
        (box.x, box.y, box.w, box.h),
        (0.0, 0.0, FRAME_W, FRAME_H),
        (0.0, 0.0, float(OUT[0]), float(OUT[1])),
        tuple(float(value) for value in LETTERBOX),
        progress,
    )


def test_transition_endpoints() -> None:
    start = _transition(0.0)
    assert start.dst == (0, 0, OUT[0], OUT[1])
    assert start.src[2] < FRAME_W  # 起点还是"人物取景框"

    end = _transition(1.0)
    assert end.src == (0.0, 0.0, FRAME_W, FRAME_H)
    assert end.dst == LETTERBOX


def test_transition_never_stretches_or_leaves_frame() -> None:
    for progress in (0.1, 0.25, 0.5, 0.75, 0.9):
        mid = _transition(progress)
        src_box = CropBox(*mid.src)
        assert src_box.matches_ratio(RATIO, tol=1e9)  # 比例会变，但要与目标框一致
        assert mid.src[0] >= 0.0 and mid.src[1] >= 0.0
        assert mid.src[0] + mid.src[2] <= FRAME_W + 0.5
        assert mid.src[1] + mid.src[3] <= FRAME_H + 0.5
        assert abs((mid.src[2] / mid.src[3]) - (mid.dst[2] / mid.dst[3])) < 0.02


def test_tiles_layout_puts_panorama_on_top() -> None:
    gap = DEFAULT_MULTI_PERSON_POLICY.gap_px(OUT)
    layout = tiles_layout(OUT, (FRAME_W, FRAME_H), 2, gap=gap)
    assert layout is not None and layout.capacity == 3
    rects = compute_cell_rects(layout, OUT, gap)
    assert rects[0][2] == OUT[0]  # 通栏
    assert abs(rects[0][3] - LETTERBOX[3]) <= 4  # 高度 = 通栏宽 / 源比例
    assert all(rect[2] % 2 == 0 and rect[3] % 2 == 0 for rect in rects)
    assert all(
        rect[0] + rect[2] <= OUT[0] and rect[1] + rect[3] <= OUT[1] for rect in rects
    )
    # 互不重叠
    assert rects[1][1] + rects[1][3] <= rects[2][1]


def test_tiles_layout_gives_up_when_no_room() -> None:
    assert tiles_layout((256, 256), (FRAME_W, FRAME_H), 3, gap=0) is None
    assert tiles_layout(OUT, (FRAME_W, FRAME_H), 0, gap=0) is None


def test_fade_alpha() -> None:
    assert fade_alpha(0.0) == 0.0
    assert fade_alpha(0.4) == 0.0
    assert fade_alpha(1.0) == 1.0
    assert 0.0 < fade_alpha(0.8) < 1.0


# ------------------------------------------------------------------ 合成
def test_compose_fit_frame_fills_band_and_background() -> None:
    composed = compose_multi_frame(
        FRAME, [WindowSlice(WHOLE, LETTERBOX, fit=True)], OUT, blur=True
    )
    assert composed.shape[:2] == (OUT[1], OUT[0])

    expected = blurred_cover(FRAME, OUT)
    patch = cv2.resize(FRAME, (LETTERBOX[2], LETTERBOX[3]), interpolation=cv2.INTER_AREA)
    expected[
        LETTERBOX[1] : LETTERBOX[1] + LETTERBOX[3], LETTERBOX[0] : LETTERBOX[0] + LETTERBOX[2]
    ] = patch
    assert np.array_equal(composed, expected)
    # 留白不是黑边（用同一帧的模糊放大版填的）
    assert float(composed[4].mean()) > 1.0


def test_compose_alpha_zero_skips_window_and_alpha_blends() -> None:
    skipped = compose_multi_frame(
        FRAME,
        [WindowSlice(center_box((FRAME_W, FRAME_H), RATIO), (0, 0, 540, 960), alpha=0.0)],
        OUT,
        background=(0, 0, 0),
    )
    assert not skipped.any()

    blended = compose_multi_frame(
        FRAME,
        [WindowSlice(center_box((FRAME_W, FRAME_H), RATIO), (0, 0, 540, 960), alpha=0.5)],
        OUT,
        background=(0, 0, 0),
    )
    opaque = compose_multi_frame(
        FRAME,
        [WindowSlice(center_box((FRAME_W, FRAME_H), RATIO), (0, 0, 540, 960))],
        OUT,
        background=(0, 0, 0),
    )
    assert 0.0 < float(blended[:, :540].mean()) < float(opaque[:, :540].mean())


def test_compose_without_fit_is_unchanged() -> None:
    """``fit=False`` + 纯色底 = 裁剪后铺满格子（多人分屏的旧行为）。"""
    composed = compose_multi_frame(
        FRAME, [WindowSlice(center_box((FRAME_W, FRAME_H), RATIO), (0, 0, 540, 960))], OUT
    )
    assert np.array_equal(
        composed[:960, :540], crop_frame(FRAME, center_box((FRAME_W, FRAME_H), RATIO), (540, 960))
    )
    assert np.array_equal(composed[960, 0], np.array([18, 18, 18], dtype=np.uint8))


# ------------------------------------------------------------------ 流水线
class StubDetector:
    """按帧号给出预置检测：前 ``主人物帧数`` 帧有一个主角，之后可选若干远景小人物。"""

    def __init__(self, main_frames: int = 60, small_persons: bool = False) -> None:
        self.index = -1
        self.main_frames = main_frames
        self.small_persons = small_persons

    def detect_boxes(self, frame):  # noqa: ANN001, ANN201
        self.index += 1
        if self.index < self.main_frames:
            return [Detection(MAIN_BBOX, 0.9)]
        if not self.small_persons:
            return []
        return [Detection(bbox, 0.6) for bbox in SMALL_BBOXES]


def run_pipeline(
    mode: str,
    *,
    small_persons: bool = False,
    total: int = 220,
    main_frames: int = 60,
    hold_frames: int = 30,
    tiles_max: int = 2,
) -> tuple[CropPipeline, list]:
    smoother = BoxSmoother(
        RATIO,
        params=SmoothingParams(
            hold_frames=hold_frames, no_person_mode=mode, no_person_seconds=0.8, fps=FPS
        ),
    )
    pipeline = CropPipeline(
        RATIO,
        StubDetector(main_frames=main_frames, small_persons=small_persons),
        smoother=smoother,
        min_person_height_ratio=0.33,
        no_person=NoPersonDisplay(blur=True, tiles_max=tiles_max, secondary_ratio=0.12),
    )
    outcomes = []
    for index in range(total):
        outcomes.extend(pipeline.process(FRAME, index))
    outcomes.extend(pipeline.flush())
    return pipeline, outcomes


def test_fit_fallback_keeps_output_size_and_transitions() -> None:
    pipeline, outcomes = run_pipeline("fit")
    assert len(outcomes) == 220
    assert all(item.frame.shape[:2] == (OUT[1], OUT[0]) for item in outcomes)

    modes = [item.mode for item in outcomes]
    assert modes[0] != MODE_FIT  # 有人时不受影响
    assert all(item.windows == [] for item in outcomes[:60])
    assert modes[60] == "hold" and modes[89] == "hold"
    assert modes[90] == MODE_FIT  # 保持 30 帧后进入兜底

    assert pipeline.hold_frames == 30
    assert pipeline.fit_frames == 220 - 90
    # 过渡是渐进的：起步那几帧还在往最终画面走，稳定后逐帧一致
    assert not np.array_equal(outcomes[90].frame, outcomes[95].frame)
    assert np.array_equal(outcomes[120].frame, outcomes[130].frame)
    assert float(outcomes[120].frame[3].mean()) > 1.0  # 不是黑边


def test_tiles_fallback_uses_selected_small_persons() -> None:
    pipeline, outcomes = run_pipeline("tiles", small_persons=True)
    assert pipeline.tiles_frames > 50
    assert outcomes[-1].mode == MODE_TILES
    windows = outcomes[-1].windows
    assert len(windows) == 3  # 1 个全景 + 2 个特写
    assert windows[-1].fit is True and windows[-1].alpha == 1.0

    details = [window for window in windows if not window.fit]
    assert len(details) == 2
    assert all(window.box.h < FRAME_H for window in details)  # 只有上半身
    # 过渡早期：特写还是淡入状态（不是硬切）
    assert any(0.0 <= window.alpha < 1.0 for window in outcomes[95].windows)


def test_tiles_falls_back_to_fit_without_candidates() -> None:
    pipeline, outcomes = run_pipeline("tiles", small_persons=False)
    assert pipeline.tiles_frames == 0
    assert pipeline.fit_frames > 0
    assert outcomes[-1].mode == MODE_FIT


def test_legacy_modes_unchanged() -> None:
    center_pipeline, center_outcomes = run_pipeline("center")
    assert center_outcomes[-1].mode == "center"
    assert center_pipeline.fit_frames == 0

    scan_pipeline, scan_outcomes = run_pipeline("scan")
    assert scan_outcomes[-1].mode == "scan"
    assert scan_pipeline.fit_frames == 0


def test_tiles_planner_uses_upper_body_boxes() -> None:
    planner = TilesPlanner(
        RATIO,
        min_person_height_ratio=0.33,
        secondary_ratio=0.12,
        max_details=2,
    )
    detections = [Detection(bbox, 0.6) for bbox in SMALL_BBOXES]
    assert planner.update(detections, (FRAME_W, FRAME_H)) == 2
    assert planner.update([], (FRAME_W, FRAME_H)) == 0
    assert planner.dst_rect((FRAME_W, FRAME_H)) is None  # 没有候选人就没有全景 + 特写


# ------------------------------------------------------------------ 配置
def test_config_mapping_normalizes_no_person_options() -> None:
    cfg = AppConfig.from_mapping(
        {"no_person_mode": " Tiles ", "no_person_blur": "否", "no_person_tiles_max": 3}
    )
    assert cfg.no_person_mode == "tiles"
    assert cfg.no_person_blur is False
    assert cfg.no_person_tiles_max == 3
    assert cfg.smoothing_params().no_person_mode == "tiles"
    assert cfg.no_person_display().tiles_max == 3


def test_config_rejects_bad_no_person_values() -> None:
    with pytest.raises(ValueError, match="no_person_mode"):
        AppConfig.from_mapping({"no_person_mode": "zoom"}).validate()
    with pytest.raises(ValueError, match="no_person_tiles_max"):
        AppConfig(no_person_tiles_max=9).validate()
    with pytest.raises(ValueError, match="no_person_secondary_ratio"):
        AppConfig(no_person_secondary_ratio=1.5).validate()


def test_process_result_reports_no_person_display(tmp_path) -> None:  # noqa: ANN001
    """端到端：合成视频跑一遍，汇总里如实给出「无人物显示」。"""
    from video2personvideo.core.processor import process_video
    from video2personvideo.core.video_io import VideoReader

    source = tmp_path / "src.mp4"
    writer = cv2.VideoWriter(
        str(source), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (int(FRAME_W), int(FRAME_H))
    )
    for _ in range(40):
        writer.write(FRAME)
    writer.release()

    output = tmp_path / "out.mp4"
    cfg = AppConfig(
        source=source,
        output=output,
        aspect_ratio="9:16",
        overwrite=True,
        no_person_mode="fit",
        no_person_seconds=0.2,
        save_audio=False,
        size_guard=False,
        normalize_vfr=False,
        hold_frames=3,
    )
    result = process_video(
        cfg, show_progress=False, detector=StubDetector(main_frames=10, small_persons=False)
    )
    with VideoReader(output) as reader:
        assert reader.meta.size == OUT

    assert result.no_person_mode == "fit"
    assert result.fit_frames > 0
    summary = result.summary()
    assert "无人物显示" in summary and "全画面适配" in summary
