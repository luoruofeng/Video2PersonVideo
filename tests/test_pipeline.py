"""流水线测试：抽帧插值、批量推理等价性、帧序与不变量（M1-6 / M1-7）。"""

from __future__ import annotations

import numpy as np
import pytest

from video2personvideo.core.pipeline import CropPipeline
from video2personvideo.core.ratio import parse_ratio
from video2personvideo.core.smoothing import BoxSmoother
from video2personvideo.core.video_io import VideoReader

RATIO = parse_ratio("9:16")
FRAME_SIZE = (320, 240)
KEYS = 20


def _frames(count: int = KEYS, size: tuple[int, int] = FRAME_SIZE) -> list[np.ndarray]:
    width, height = size
    frames = []
    for index in range(count):
        frame = np.full((height, width, 3), 60 + index, dtype=np.uint8)
        frames.append(frame)
    return frames


def _run(pipeline: CropPipeline, frames: list[np.ndarray]):
    outcomes = []
    for index, frame in enumerate(frames):
        outcomes.extend(pipeline.process(frame, index))
    outcomes.extend(pipeline.flush())
    return outcomes


def _wave(frame: np.ndarray, index: int):
    height, width = frame.shape[:2]
    center = width * 0.25 + width * 0.5 * (index % 10) / 9.0
    half = width * 0.06
    return [(center - half, height * 0.15, center + half, height * 0.95)]


def _build(detector, *, detect_interval: int = 1, infer_batch: int = 1) -> CropPipeline:
    return CropPipeline(
        RATIO,
        detector,
        smoother=BoxSmoother(RATIO, alpha=0.35),
        detect_interval=detect_interval,
        infer_batch=infer_batch,
    )


# --------------------------------------------------------------- 基础行为
@pytest.mark.parametrize("detect_interval", [1, 2, 3, 5])
def test_every_frame_is_emitted_exactly_once(fake_detector, detect_interval: int) -> None:
    frames = _frames()
    pipeline = _build(fake_detector(_wave), detect_interval=detect_interval)
    outcomes = _run(pipeline, frames)

    assert [item.index for item in outcomes] == list(range(KEYS))
    assert pipeline.stats.frames == KEYS
    for item in outcomes:
        assert item.frame.shape[:2] == (RATIO.target_height, RATIO.target_width)
        assert item.box is not None and item.box.matches_ratio(RATIO, tol=1e-6)


def test_interpolation_is_used_between_keyframes(fake_detector) -> None:
    pipeline = _build(fake_detector(_wave), detect_interval=4)
    _run(pipeline, _frames())

    assert pipeline.stats.detected_frames == 5  # 0/4/8/12/16
    # 末尾 3 帧（17/18/19）由 flush 沿用最后一次目标定稿，不计入插值
    assert pipeline.stats.interpolated_frames == KEYS - 5 - 3
    assert pipeline.stats.inference_calls == 5


def test_flush_detects_remaining_keyframes(fake_detector) -> None:
    """视频在关键帧处提前结束也不能漏检（尾部关键帧补一次推理）。"""
    detector = fake_detector(_wave)
    pipeline = _build(detector, detect_interval=4, infer_batch=2)
    frames = _frames(9)  # 0,4,8 → 最后 8 帧必须也被推理到

    outcomes = _run(pipeline, frames)

    assert len(outcomes) == 9
    assert pipeline.stats.detected_frames == 3


# ----------------------------------------------------------- 批量推理等价性
def test_batch_matches_per_frame_results(fake_detector) -> None:
    """infer_batch>1 与逐帧推理的输出必须逐帧完全一致。"""
    frames = _frames()
    single = _run(_build(fake_detector(_wave), infer_batch=1), frames)
    batched_detector = fake_detector(_wave)
    batched = _run(_build(batched_detector, infer_batch=4), frames)

    assert batched_detector.batch_sizes == [4, 4, 4, 4, 4]
    assert [item.index for item in batched] == [item.index for item in single]
    assert [item.mode for item in batched] == [item.mode for item in single]
    assert [item.has_person for item in batched] == [item.has_person for item in single]
    for left, right in zip(single, batched, strict=True):
        assert left.box.cx == pytest.approx(right.box.cx)
        assert left.box.h == pytest.approx(right.box.h)


def test_batch_with_sparse_detection_matches(fake_detector) -> None:
    frames = _frames()
    single = _run(_build(fake_detector(_wave), detect_interval=3, infer_batch=1), frames)
    detector = fake_detector(_wave)
    batched = _run(_build(detector, detect_interval=3, infer_batch=3), frames)

    assert [item.index for item in batched] == [item.index for item in single]
    for left, right in zip(single, batched, strict=True):
        assert left.box.cx == pytest.approx(right.box.cx)
        assert left.box.h == pytest.approx(right.box.h)
    # 7 个关键帧（0..18 每 3 帧）分两批：3 + 3，最后 1 个在 flush 时补
    assert detector.batch_sizes == [3, 3, 1]


def test_batch_keeps_output_resolution_constant(fake_detector, project_logs) -> None:
    pipeline = _build(fake_detector(_wave), detect_interval=2, infer_batch=4)
    outcomes = _run(pipeline, _frames())

    shapes = {item.frame.shape[:2] for item in outcomes}
    assert shapes == {(RATIO.target_height, RATIO.target_width)}
    assert pipeline.stats.inference_calls == 3  # 10 个关键帧 → 4 + 4 + 2
    assert pipeline.stats.max_batch == 4
    assert not [record for record in project_logs if "不一致" in record.getMessage()]


def test_detector_without_batch_api_falls_back(fake_detector) -> None:
    """自定义检测器只实现 detect_boxes 时，批量模式自动退化为逐帧调用。"""

    class LegacyDetector:
        def __init__(self) -> None:
            self.calls = 0

        def detect_boxes(self, frame: np.ndarray):
            self.calls += 1
            return []

    detector = LegacyDetector()
    pipeline = _build(detector, infer_batch=3)
    outcomes = _run(pipeline, _frames())

    assert len(outcomes) == KEYS
    assert detector.calls == KEYS
    # 20 帧按每批 3 帧切：3×6 + 尾部 2 帧 = 7 次调用
    assert pipeline.stats.inference_calls == 7


# ----------------------------------------------------------- 帧率校准（镜头舒适度）
def test_set_fps_reaches_smoother(fake_detector) -> None:
    """构造流水线时还不知道帧率，随后补告知：镜头参数按真实帧率换算。"""
    pipeline = _build(fake_detector(_wave))
    assert pipeline.smoother.fps is None

    pipeline.set_fps(60.0)
    assert pipeline.smoother.fps == pytest.approx(60.0)
    assert pipeline.smoother.effective_max_speed == pytest.approx(
        pipeline.smoother.max_speed / 2.0
    )

    pipeline.set_fps(0.0)  # 非法帧率：保持原样，不改坏镜头参数
    assert pipeline.smoother.fps == pytest.approx(60.0)


# --------------------------------------------------------------- 无人兜底
def test_all_frames_without_person_fallback(fake_detector) -> None:
    pipeline = _build(fake_detector(lambda frame, index: []), detect_interval=2, infer_batch=3)
    outcomes = _run(pipeline, _frames())

    assert len(outcomes) == KEYS
    assert pipeline.center_frames == KEYS
    assert pipeline.hold_frames == 0
    assert all(item.mode == "center" for item in outcomes)


# ----------------------------------------------------- 与 processor 的联动
def test_infer_batch_reaches_processor(tmp_path, make_video, fake_detector, moving_person) -> None:
    from video2personvideo.config import AppConfig
    from video2personvideo.core.processor import process_video

    source = make_video("batch.avi", size=(160, 120), frames=12)
    output = tmp_path / "batch_person.avi"
    detector = fake_detector(moving_person)

    result = process_video(
        AppConfig(
            source=source,
            output=output,
            save_audio=False,
            aspect_ratio="9:16",
            infer_batch=5,
        ),
        show_progress=False,
        detector=detector,
    )

    ratio = parse_ratio("9:16")
    assert detector.batch_sizes == [5, 5, 2]
    assert result.frames_processed == 12
    with VideoReader(output) as reader:
        assert reader.meta.size == ratio.target_size
        assert len(list(reader.frames())) == 12


def test_processor_calibrates_camera_to_source_frame_rate(
    tmp_path, make_video, fake_detector, moving_person, monkeypatch
) -> None:
    """采集任务开跑前，会用源视频的真实帧率校准镜头跟随（端到端护栏）。"""
    from video2personvideo.config import AppConfig
    from video2personvideo.core import smoothing
    from video2personvideo.core.processor import process_video

    seen: list[float | None] = []
    original = smoothing.BoxSmoother.set_fps

    def spy(self, fps):  # noqa: ANN001 - 只为记录调用参数
        seen.append(fps)
        return original(self, fps)

    monkeypatch.setattr(smoothing.BoxSmoother, "set_fps", spy)

    source = make_video("fps.avi", size=(160, 120), frames=10, fps=10.0)
    process_video(
        AppConfig(
            source=source,
            output=tmp_path / "fps_person.avi",
            save_audio=False,
            aspect_ratio="9:16",
        ),
        show_progress=False,
        detector=fake_detector(moving_person),
    )

    assert seen, "处理视频时必须把源帧率告知镜头平滑器"
    assert seen[0] is not None and seen[0] == pytest.approx(10.0, rel=0.5)
