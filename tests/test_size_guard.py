"""体积守护测试：输出比原视频大时自动压缩到体积上限以内。

背景：OpenCV 只能写 mp4v（MPEG-4 Part 2），同一画质需要的码率是 H.264 的
好几倍；低分辨率源视频被放大到目标像素后，成品体积常常是原视频的 3~10 倍。
这里验证「超标 → 自动压缩」这条链路的纯函数、压缩函数与流水线接线。
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import pytest

from video2personvideo.config import AppConfig
from video2personvideo.core.processor import format_bytes, process_video
from video2personvideo.utils import ffmpeg_tools

requires_ffmpeg = pytest.mark.skipif(
    ffmpeg_tools.find_ffmpeg() is None, reason="需要 ffmpeg 才能验证体积压缩"
)


def _make_heavy_video(path: Path, *, seconds: float = 2.0) -> Path:
    """生成一段高码率测试视频（无损近似），供压缩用。"""
    args = [
        ffmpeg_tools.find_ffmpeg() or "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size=640x480:rate=25:duration={seconds}",
        "-c:v",
        "libx264",
        "-crf",
        "12",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    completed = subprocess.run(args, capture_output=True, text=True, check=False)
    if completed.returncode != 0 or not path.exists() or path.stat().st_size == 0:
        pytest.skip(f"无法生成测试视频：{completed.stderr.strip()[-200:]}")
    return path


# --------------------------------------------------------------- 纯函数
def test_target_bitrate_derived_from_size_limit() -> None:
    # 1 MB / 10s → 总码率 800 kbps；扣掉 128k 音频即视频码率
    assert ffmpeg_tools.target_video_bitrate_kbps(1_000_000, 10.0, audio_kbps=128) == 672
    assert ffmpeg_tools.target_video_bitrate_kbps(1_000_000, 10.0, audio_kbps=0) == 800
    # 上限过小 → 落到最低码率，而不是无限逼近 0
    assert ffmpeg_tools.target_video_bitrate_kbps(1_000, 100.0, min_kbps=250) == 250
    # 非法输入 → 0，调用方据此跳过压缩
    assert ffmpeg_tools.target_video_bitrate_kbps(0, 10.0) == 0
    assert ffmpeg_tools.target_video_bitrate_kbps(1_000_000, 0.0) == 0


def test_format_bytes() -> None:
    assert format_bytes(0) == "0 B"
    assert format_bytes(512) == "512 B"
    assert format_bytes(1024) == "1.0 KB"
    assert format_bytes(5 * 1024 * 1024) == "5.0 MB"


def test_size_guard_config_validation() -> None:
    assert AppConfig().size_guard is True
    with pytest.raises(ValueError):
        AppConfig(max_size_ratio=0).validate()
    with pytest.raises(ValueError):
        AppConfig(min_video_bitrate_kbps=-1).validate()


def test_size_guard_parsed_from_mapping() -> None:
    cfg = AppConfig.from_mapping(
        {"size_guard": "false", "max_size_ratio": "1.5", "min_video_bitrate_kbps": 400}
    )
    assert cfg.size_guard is False
    assert cfg.max_size_ratio == 1.5
    assert cfg.min_video_bitrate_kbps == 400


# --------------------------------------------------------------- 压缩函数
@requires_ffmpeg
def test_shrink_to_limit_in_place(tmp_path: Path) -> None:
    source = _make_heavy_video(tmp_path / "heavy.mp4")
    before = source.stat().st_size
    limit = before // 3

    reached, kbps = ffmpeg_tools.shrink_to_limit(source, source, limit, min_kbps=100)

    assert reached is True
    assert kbps > 0
    assert source.stat().st_size <= limit < before

    cap = cv2.VideoCapture(str(source))
    try:
        assert cap.isOpened()
        ok, frame = cap.read()
        assert ok and frame is not None
    finally:
        cap.release()
    assert list(tmp_path.glob("*.sizing.*")) == []


@requires_ffmpeg
def test_shrink_to_limit_writes_separate_output(tmp_path: Path) -> None:
    source = _make_heavy_video(tmp_path / "heavy.mp4")
    output = tmp_path / "small.mp4"
    limit = source.stat().st_size // 3
    original_size = source.stat().st_size

    reached, _ = ffmpeg_tools.shrink_to_limit(source, output, limit, min_kbps=100)

    assert reached is True
    assert output.stat().st_size <= limit
    assert source.stat().st_size == original_size  # 源文件原封不动
    assert list(tmp_path.glob("*.sizing.*")) == []


def test_shrink_to_limit_without_ffmpeg_is_noop(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "clip.mp4"
    target.write_bytes(b"x" * 1024)
    monkeypatch.setattr(ffmpeg_tools, "find_ffmpeg", lambda: None)

    assert ffmpeg_tools.shrink_to_limit(target, target, 100) == (False, 0)
    assert target.stat().st_size == 1024  # 文件没被动过


# --------------------------------------------------------------- 流水线
def test_guard_skipped_when_limit_is_generous(
    tmp_path: Path, make_video, fake_detector, moving_person
) -> None:
    source = make_video("clip.avi", size=(160, 120), frames=8)
    output = tmp_path / "clip_person.avi"

    result = process_video(
        AppConfig(
            source=source,
            output=output,
            save_audio=False,
            aspect_ratio="1:1",
            target_width=160,
            target_height=160,
            max_size_ratio=10.0,
        ),
        show_progress=False,
        detector=fake_detector(moving_person),
    )

    assert result.size_guard is False
    assert result.source_bytes > 0 and result.output_bytes > 0
    assert result.size_ratio == pytest.approx(result.output_bytes / result.source_bytes)
    assert "输出体积" in result.summary()


def test_guard_can_be_disabled(
    tmp_path: Path, make_video, fake_detector, moving_person
) -> None:
    source = make_video("clip.avi", size=(160, 120), frames=8)
    output = tmp_path / "clip_person.avi"

    result = process_video(
        AppConfig(
            source=source,
            output=output,
            save_audio=False,
            aspect_ratio="1:1",
            target_width=160,
            target_height=160,
            size_guard=False,
            max_size_ratio=0.01,  # 一定会超标，但守护被关掉了
        ),
        show_progress=False,
        detector=fake_detector(moving_person),
    )

    assert result.size_guard is False
    assert output.exists()
    assert list(tmp_path.glob("*.sizing.*")) == []


@requires_ffmpeg
def test_guard_triggers_on_oversized_output(
    tmp_path: Path, make_video, fake_detector, moving_person
) -> None:
    """输出体积超标时必须尝试压缩，且不能破坏最终产物。"""
    source = make_video("clip.avi", size=(160, 120), frames=8)
    output = tmp_path / "clip_person.avi"

    result = process_video(
        AppConfig(
            source=source,
            output=output,
            save_audio=False,
            aspect_ratio="1:1",
            target_width=160,
            target_height=160,
            max_size_ratio=0.2,  # 输出只允许是原视频的 20%，必然触发
        ),
        show_progress=False,
        detector=fake_detector(moving_person),
    )

    assert result.size_guard is True
    assert result.size_guard_kbps > 0
    assert result.raw_output_bytes > 0  # 报告压缩前的体积，便于说明"压小了多少"
    assert "体积守护" in result.summary()
    assert output.exists() and output.stat().st_size > 0
    assert list(tmp_path.glob("*.sizing.*")) == []

    cap = cv2.VideoCapture(str(output))
    try:
        assert cap.isOpened()
        assert cap.read()[0] is True
    finally:
        cap.release()
