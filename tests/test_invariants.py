"""不变量测试（独立一组，作为验收门槛）+ 端到端回归。

不变量 1：输出文件每一帧的宽/高 恒等于用户所选比例
不变量 2：输出分辨率全程恒定（每帧尺寸完全一致）、宽高均为偶数、无丢帧告警
不变量 3：取景框本身比例恒定
不变量 4：切片永远落在源画面内，输出无黑边

全部用例使用注入的假检测器，不下载模型、不依赖 GPU。
"""

from __future__ import annotations

import numpy as np
import pytest

from video2personvideo.config import AppConfig
from video2personvideo.core.framing import center_box
from video2personvideo.core.pipeline import CropPipeline
from video2personvideo.core.processor import process_video
from video2personvideo.core.ratio import parse_ratio
from video2personvideo.core.smoothing import BoxSmoother
from video2personvideo.core.video_io import VideoReader

#: 遍历所有预置比例时使用的用例（也覆盖横屏 / 竖屏 / 正方形源）
PRESET_NAMES = ["16:9", "9:16", "1:1", "4:3", "3:4", "4:5", "21:9", "2:1"]

#: 源视频统一尺寸（横屏 4:3）
SOURCE_SIZE = (160, 120)
SOURCE_FRAMES = 12


def _static_person(frame: np.ndarray, index: int) -> list[tuple[float, float, float, float]]:
    return [(60.0, 18.0, 100.0, 114.0)]


def _read_output(path) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    with VideoReader(path) as reader:
        meta = reader.meta
        shapes = [(frame.shape[1], frame.shape[0]) for _, frame in reader.frames()]
    return meta.size, shapes


# --------------------------------------------------------------- 不变量 1
@pytest.mark.parametrize("name", PRESET_NAMES)
def test_invariant_1_output_file_matches_requested_ratio(
    name: str, tmp_path, make_video, fake_detector
) -> None:
    ratio = parse_ratio(name)
    source = make_video("src.avi", size=SOURCE_SIZE, frames=SOURCE_FRAMES)
    output = tmp_path / f"out_{name.replace(':', '_')}.avi"

    process_video(
        AppConfig(source=source, output=output, save_audio=False, aspect_ratio=name),
        show_progress=False,
        detector=fake_detector(_static_person),
    )

    (width, height), _ = _read_output(output)
    assert abs(width / height - ratio.value) < 1e-3
    assert (width, height) == ratio.target_size


# --------------------------------------------------------------- 不变量 2
@pytest.mark.parametrize("name", ["16:9", "9:16", "1:1"])
def test_invariant_2_every_frame_has_identical_even_size(
    name: str, tmp_path, make_video, fake_detector, project_logs
) -> None:
    source = make_video("src.avi", size=SOURCE_SIZE, frames=SOURCE_FRAMES)
    output = tmp_path / f"out_{name.replace(':', '_')}.avi"

    result = process_video(
        AppConfig(
            source=source,
            output=output,
            save_audio=False,
            aspect_ratio=name,
            detect_interval=3,
        ),
        show_progress=False,
        detector=fake_detector(_static_person),
    )

    (width, height), shapes = _read_output(output)
    assert width % 2 == 0 and height % 2 == 0
    assert len(shapes) == SOURCE_FRAMES == result.frames_processed
    assert set(shapes) == {(width, height)}

    # VideoWriter 全程不应产生"尺寸不一致，已跳过该帧"告警
    assert not [record for record in project_logs if "不一致" in record.getMessage()]


# --------------------------------------------------------------- 不变量 3
def test_invariant_3_crop_box_ratio_is_constant(make_video, fake_detector) -> None:
    ratio = parse_ratio("9:16")
    source = make_video("src.avi", size=SOURCE_SIZE, frames=SOURCE_FRAMES)
    pipeline = CropPipeline(
        ratio,
        fake_detector(_static_person),
        smoother=BoxSmoother(ratio, alpha=0.4),
        detect_interval=4,
    )

    outcomes = []
    with VideoReader(source) as reader:
        for index, frame in reader.frames():
            outcomes.extend(pipeline.process(frame, index))
        outcomes.extend(pipeline.flush())

    assert len(outcomes) == SOURCE_FRAMES
    for outcome in outcomes:
        assert outcome.box is not None
        assert outcome.box.matches_ratio(ratio, tol=1e-6)
        assert outcome.frame.shape[:2] == (ratio.target_height, ratio.target_width)

    # 抽帧 + 插值路径确实被走到
    assert pipeline.stats.interpolated_frames > 0


# --------------------------------------------------------------- 不变量 4
def test_invariant_4_slices_stay_inside_and_never_letterbox(
    tmp_path, make_video, fake_detector
) -> None:
    """源画面是纯色 → 输出不该出现任何黑边；同时逐帧校验切片落在源画面内。"""
    ratio = parse_ratio("9:16")
    source = make_video("solid.avi", size=SOURCE_SIZE, frames=SOURCE_FRAMES, fill=180)

    def provider(frame: np.ndarray, index: int):
        shift = (index % 6) * 10.0
        return [(40.0 + shift, 12.0, 90.0 + shift, 114.0)]

    pipeline = CropPipeline(ratio, fake_detector(provider))
    outcomes = []
    with VideoReader(source) as reader:
        for index, frame in reader.frames():
            outcomes.extend(pipeline.process(frame, index))
        outcomes.extend(pipeline.flush())

    frame_w, frame_h = SOURCE_SIZE
    for outcome in outcomes:
        x, y, width, height = outcome.box.as_int_even(frame_w, frame_h)
        assert x >= 0 and y >= 0
        assert x + width <= frame_w and y + height <= frame_h
        # 纯色源经裁剪缩放后仍应是纯色（无黑边 / 无 letterbox）
        assert int(outcome.frame.min()) > 120


def test_invariant_4_various_boxes_never_go_out_of_bounds() -> None:
    from video2personvideo.core.crop import crop_frame
    from video2personvideo.core.framing import CropBox, compute_target_box

    ratio = parse_ratio("9:16")
    frame = np.full((120, 160, 3), 200, np.uint8)
    boxes = [
        CropBox(-999, -999, 50, 89),
        CropBox(999, 999, 50, 89),
        CropBox(0, 0, 1000, 8000),
        compute_target_box((0, 0, 20, 120), (160.0, 120.0), ratio),
        center_box((160.0, 120.0), ratio),
    ]
    for box in boxes:
        output = crop_frame(frame, box, ratio.target_size)
        assert output.shape[:2] == (ratio.target_height, ratio.target_width)
        assert int(output.min()) == 200


# ----------------------------------------------------------- 端到端场景
def test_e2e_moving_person_keeps_resolution_and_tracks(
    tmp_path, make_video, fake_detector, moving_person
) -> None:
    ratio = parse_ratio("9:16")
    source = make_video("moving.avi", size=(320, 240), frames=20)
    output = tmp_path / "moving_person.avi"

    result = process_video(
        AppConfig(source=source, output=output, save_audio=False, aspect_ratio="9:16"),
        show_progress=False,
        detector=fake_detector(moving_person),
    )

    (width, height), shapes = _read_output(output)
    assert (width, height) == ratio.target_size
    assert set(shapes) == {(width, height)}
    assert result.frames_processed == 20
    assert result.frames_with_person == 20
    assert result.aspect_ratio == "9:16"
    assert result.crop_mode is True


def test_e2e_person_then_empty_then_person_keeps_size(
    tmp_path, make_video, fake_detector, project_logs
) -> None:
    """前段有人、中段无人、后段有人：全程输出尺寸一致、画面不跳变到非法框。"""

    def provider(frame: np.ndarray, index: int):
        if index < 6 or index >= 14:
            return [(60.0, 18.0, 100.0, 114.0)]
        return []

    source = make_video("gaps.avi", size=SOURCE_SIZE, frames=20)
    output = tmp_path / "gaps_person.avi"

    result = process_video(
        AppConfig(
            source=source,
            output=output,
            save_audio=False,
            aspect_ratio="9:16",
            hold_frames=3,
        ),
        show_progress=False,
        detector=fake_detector(provider),
    )

    (width, height), shapes = _read_output(output)
    assert len(shapes) == 20
    assert set(shapes) == {(width, height)}

    # 无人帧应当走"保持上一帧"、"回中"兜底，并被计入统计
    assert result.hold_frames > 0
    assert result.center_frames > 0
    assert result.frames_with_person == 12
    assert not [record for record in project_logs if "不一致" in record.getMessage()]


def test_e2e_no_person_at_all_still_produces_valid_ratio(
    tmp_path, make_video, fake_detector
) -> None:
    source = make_video("nobody.avi", size=SOURCE_SIZE, frames=8, fill=120)
    output = tmp_path / "nobody_person.avi"

    result = process_video(
        AppConfig(source=source, output=output, save_audio=False, aspect_ratio="1:1"),
        show_progress=False,
        detector=fake_detector(lambda frame, index: []),
    )

    ratio = parse_ratio("1:1")
    (width, height), shapes = _read_output(output)
    assert (width, height) == ratio.target_size
    assert len(shapes) == 8
    assert result.center_frames == 8
    assert result.frames_with_person == 0


def test_e2e_annotate_mode_regression(tmp_path, make_video, fake_detector) -> None:
    """回归：关闭裁剪时保留旧的"逐帧画框标注"行为（输出为原分辨率）。"""
    source = make_video("plain.avi", size=(96, 64), frames=5)
    output = tmp_path / "plain_annotated.avi"

    result = process_video(
        AppConfig(source=source, output=output, save_audio=False, crop=False),
        show_progress=False,
        detector=fake_detector(lambda frame, index: [(10.0, 10.0, 40.0, 50.0)]),
    )

    (width, height), shapes = _read_output(output)
    assert (width, height) == (96, 64)
    assert len(shapes) == 5
    assert result.crop_mode is False
    assert result.mode_counts == {}
    assert "逐帧画框标注" in result.summary()


def test_annotate_inside_crop_mode_draws_box(tmp_path, make_video, fake_detector) -> None:
    source = make_video("ann.avi", size=SOURCE_SIZE, frames=4, fill=120)
    output = tmp_path / "ann_person.avi"

    process_video(
        AppConfig(
            source=source,
            output=output,
            save_audio=False,
            aspect_ratio="9:16",
            annotate=True,
        ),
        show_progress=False,
        detector=fake_detector(_static_person),
    )

    with VideoReader(output) as reader:
        frames = [frame for _, frame in reader.frames()]
    # 画上去的绿框会让绿色通道出现明显高值
    assert any(int(frame[:, :, 1].max()) > 200 for frame in frames)


# ----------------------------------------------------------- 中途取消
def test_stop_callback_truncates_output_but_keeps_size(tmp_path, make_video, fake_detector) -> None:
    source = make_video("long.avi", size=SOURCE_SIZE, frames=20)
    output = tmp_path / "long_person.avi"
    counter = {"calls": 0}

    def stop_cb() -> bool:
        counter["calls"] += 1
        return counter["calls"] > 5

    result = process_video(
        AppConfig(source=source, output=output, save_audio=False, aspect_ratio="9:16"),
        stop_cb=stop_cb,
        show_progress=False,
        detector=fake_detector(_static_person),
    )

    assert result.stopped_early is True
    assert 0 < result.frames_processed < 20
    (width, height), shapes = _read_output(output)
    assert set(shapes) == {(width, height)}
    assert abs(width / height - parse_ratio("9:16").value) < 1e-3
