"""输出与文件管理测试（M4）：命名、临时文件清理、音轨保留、元数据、预览图。"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from video2personvideo.config import AppConfig
from video2personvideo.core.processor import process_video
from video2personvideo.utils import ffmpeg_tools

#: 只有音轨 / 元数据相关的用例才需要 ffmpeg
requires_ffmpeg = pytest.mark.skipif(
    ffmpeg_tools.find_ffmpeg() is None, reason="需要 ffmpeg 才能验证音轨 / 元数据"
)


@pytest.fixture()
def video_with_audio(tmp_path: Path) -> Path:
    """用 ffmpeg 生成一段带音轨的测试视频。"""
    target = tmp_path / "with_audio.mp4"
    args = [
        ffmpeg_tools.find_ffmpeg() or "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=160x120:rate=10:duration=1.2",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=1.2",
        "-c:v",
        "mpeg4",
        "-q:v",
        "6",
        "-c:a",
        "aac",
        "-shortest",
        str(target),
    ]
    completed = subprocess.run(args, capture_output=True, text=True, check=False)
    if completed.returncode != 0 or not target.exists():
        pytest.skip(f"无法生成带音轨的测试视频：{completed.stderr.strip()[-200:]}")
    return target


def _tempoaries(output: Path) -> list[Path]:
    return sorted(output.parent.glob(f"{output.stem}.noaudio.*")) + sorted(
        output.parent.glob(f"{output.stem}.encoded.*")
    )


def test_temp_files_are_cleaned_and_final_output_kept(
    tmp_path, make_video, fake_detector, moving_person
) -> None:
    source = make_video("clip.avi", size=(160, 120), frames=6)
    output = tmp_path / "clip_person.avi"

    result = process_video(
        AppConfig(source=source, output=output, save_audio=False, aspect_ratio="9:16"),
        show_progress=False,
        detector=fake_detector(moving_person),
    )

    assert output.exists() and output.stat().st_size > 0
    assert result.output == output
    assert _tempoaries(output) == []


@requires_ffmpeg
def test_audio_is_kept_on_cropped_output(tmp_path, video_with_audio, fake_detector) -> None:
    output = tmp_path / "with_audio_person.mp4"
    result = process_video(
        AppConfig(
            source=video_with_audio,
            output=output,
            save_audio=True,
            aspect_ratio="9:16",
        ),
        show_progress=False,
        detector=fake_detector(lambda frame, index: [(60.0, 20.0, 100.0, 112.0)]),
    )

    assert result.audio_muxed is True
    assert output.exists()
    assert ffmpeg_tools.has_audio_stream(output) is True
    assert _tempoaries(output) == []


@requires_ffmpeg
def test_output_timeline_follows_audio_duration(tmp_path: Path, fake_detector) -> None:
    """画面时长与音轨时长不等时，输出以音轨时长为准（根治音画不同步）。"""
    source = tmp_path / "drift.mp4"
    args = [
        ffmpeg_tools.find_ffmpeg() or "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=160x120:rate=10:duration=2.0",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=3.0",
        "-c:v",
        "mpeg4",
        "-q:v",
        "6",
        "-c:a",
        "aac",
        str(source),
    ]
    completed = subprocess.run(args, capture_output=True, text=True, check=False)
    if completed.returncode != 0 or not source.exists():
        pytest.skip(f"无法生成测试视频：{completed.stderr.strip()[-200:]}")

    output = tmp_path / "drift_person.mp4"
    result = process_video(
        AppConfig(source=source, output=output, save_audio=True, aspect_ratio="9:16"),
        show_progress=False,
        detector=fake_detector(lambda frame, index: [(60.0, 20.0, 100.0, 112.0)]),
    )

    assert result.audio_muxed is True
    # 画面原本只有 2.0s，音轨 3.0s → 输出画面应被拉伸到 3.0s
    assert ffmpeg_tools.stream_duration(output, kind="video") == pytest.approx(3.0, abs=0.3)


def test_preview_image_is_written_when_enabled(
    tmp_path, make_video, fake_detector, moving_person
) -> None:
    source = make_video("clip.avi", size=(160, 120), frames=6)
    output = tmp_path / "clip_person.avi"

    process_video(
        AppConfig(
            source=source,
            output=output,
            save_audio=False,
            aspect_ratio="1:1",
            write_preview=True,
        ),
        show_progress=False,
        detector=fake_detector(moving_person),
    )

    preview = output.with_name(f"{output.stem}_preview.jpg")
    assert preview.exists() and preview.stat().st_size > 0


@requires_ffmpeg
def test_metadata_is_injected_when_enabled(tmp_path, video_with_audio, fake_detector) -> None:
    output = tmp_path / "with_audio_person.mp4"
    process_video(
        AppConfig(
            source=video_with_audio,
            output=output,
            save_audio=True,
            aspect_ratio="9:16",
            write_metadata=True,
        ),
        show_progress=False,
        detector=fake_detector(lambda frame, index: [(60.0, 20.0, 100.0, 112.0)]),
    )

    info = ffmpeg_tools.probe_media(output)
    assert info is not None
    tags = info.get("format", {}).get("tags", {})
    comment = str(tags.get("comment", ""))
    assert "Video2PersonVideo" in comment
    assert "ratio=9:16" in comment


def test_inject_metadata_returns_false_without_input(tmp_path: Path) -> None:
    assert ffmpeg_tools.inject_metadata(tmp_path / "nope.mp4", {"comment": "x"}) is False
    assert ffmpeg_tools.inject_metadata(tmp_path / "nope.mp4", {}) is False


@requires_ffmpeg
def test_variable_fps_source_is_normalized_before_processing(
    tmp_path: Path, make_vfr_video, fake_detector
) -> None:
    """可变帧率源先归一时间轴：输出恒定帧率、时长对齐原音轨，且不留中间文件。"""
    source = make_vfr_video()
    timing = ffmpeg_tools.analyze_video_timing(source)
    if timing is None or not timing.variable:
        pytest.skip("当前环境生成的样本未被判定为可变帧率")

    output = tmp_path / "vfr_person.mp4"
    result = process_video(
        AppConfig(source=source, output=output, save_audio=True, aspect_ratio="9:16"),
        show_progress=False,
        detector=fake_detector(lambda frame, index: [(60.0, 20.0, 100.0, 112.0)]),
    )

    assert result.source_variable_fps is True
    assert result.timeline_normalized is True
    assert "已归一化为恒定帧率" in result.summary()

    # 输出必须是恒定帧率，且画面时长对齐原音轨（音画同步的验收点）
    info = ffmpeg_tools.stream_info(output, kind="video")
    nominal = ffmpeg_tools._parse_rate(info.get("r_frame_rate"))
    average = ffmpeg_tools._parse_rate(info.get("avg_frame_rate"))
    assert nominal > 0.0 and average > 0.0
    assert nominal == pytest.approx(average, rel=0.05)

    audio_duration = ffmpeg_tools.stream_duration(source, kind="audio")
    assert audio_duration > 0.0
    assert ffmpeg_tools.stream_duration(output, kind="video") == pytest.approx(
        audio_duration, abs=0.3
    )

    # 归一化中间产物必须清掉，不能留在输出目录
    assert not (output.parent / f"{output.stem}.normalized.mp4").exists()


@requires_ffmpeg
def test_constant_frame_rate_source_is_not_normalized(
    tmp_path: Path, video_with_audio, fake_detector
) -> None:
    """恒定帧率源不做多余的归一化转码，也不产生中间文件。"""
    output = tmp_path / "cfr_person.mp4"
    result = process_video(
        AppConfig(source=video_with_audio, output=output, save_audio=True, aspect_ratio="9:16"),
        show_progress=False,
        detector=fake_detector(lambda frame, index: [(60.0, 20.0, 100.0, 112.0)]),
    )

    assert result.source_variable_fps is False
    assert result.timeline_normalized is False
    assert not (output.parent / f"{output.stem}.normalized.mp4").exists()
