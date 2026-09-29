"""ffmpeg 工具测试：缺少 ffmpeg 时的优雅降级 + 音画时长对齐。"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from video2personvideo.utils import ffmpeg_tools as ft

#: 只有真正调用 ffmpeg / ffprobe 的用例才需要它
requires_ffmpeg = pytest.mark.skipif(
    ft.find_ffmpeg() is None or ft.find_ffprobe() is None,
    reason="需要 ffmpeg / ffprobe 才能验证音画时长",
)


def test_probe_missing_file(tmp_path: Path) -> None:
    assert ft.probe_media(tmp_path / "nope.mp4") is None


def test_has_audio_stream_without_probe(monkeypatch) -> None:
    monkeypatch.setattr(ft, "probe_media", lambda _path: None)
    assert ft.has_audio_stream("whatever.mp4") is False


def test_has_audio_stream_detects_audio(monkeypatch) -> None:
    monkeypatch.setattr(
        ft,
        "probe_media",
        lambda _path: {"streams": [{"codec_type": "video"}, {"codec_type": "audio"}]},
    )
    assert ft.has_audio_stream("video.mp4") is True


def test_mux_audio_without_ffmpeg(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ft, "find_ffmpeg", lambda: None)
    assert ft.mux_audio(tmp_path / "a.mp4", tmp_path / "b.mp4", tmp_path / "c.mp4") is False


def test_mux_audio_without_source_audio(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ft, "find_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(ft, "has_audio_stream", lambda _path: False)
    assert ft.mux_audio(tmp_path / "a.mp4", tmp_path / "b.mp4", tmp_path / "c.mp4") is False


def test_transcode_without_ffmpeg(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ft, "find_ffmpeg", lambda: None)
    assert ft.transcode_h264(tmp_path / "a.mp4", tmp_path / "b.mp4") is False


def test_copy_file_moves_and_overwrites(tmp_path: Path) -> None:
    source = tmp_path / "temp.mp4"
    source.write_bytes(b"data")
    target = tmp_path / "final.mp4"
    target.write_bytes(b"old")

    ft.copy_file(source, target)

    assert target.read_bytes() == b"data"
    assert not source.exists()


# --------------------------------------------------------------- 时长探测
def test_parse_duration_text_handles_clock_format() -> None:
    assert ft._parse_duration_text("00:01:02.500000000") == pytest.approx(62.5)
    assert ft._parse_duration_text("01:00") == pytest.approx(60.0)
    assert ft._parse_duration_text("") == 0.0
    assert ft._parse_duration_text("bogus") == 0.0


def test_parse_rate_handles_fraction_and_plain() -> None:
    assert ft._parse_rate("30000/1001") == pytest.approx(29.97, abs=1e-3)
    assert ft._parse_rate("25") == pytest.approx(25.0)
    assert ft._parse_rate("0/0") == 0.0
    assert ft._parse_rate(None) == 0.0


def test_stream_duration_without_probe(monkeypatch) -> None:
    monkeypatch.setattr(ft, "probe_media", lambda _path: None)

    assert ft.stream_duration("x.mp4", kind="audio") == 0.0
    assert ft.stream_info("x.mp4", kind="video") == {}


def test_stream_duration_falls_back_to_container_duration(monkeypatch) -> None:
    monkeypatch.setattr(
        ft,
        "probe_media",
        lambda _path: {
            "streams": [{"codec_type": "audio", "tags": {}}],
            "format": {"duration": "3.5"},
        },
    )

    assert ft.stream_duration("x.mkv", kind="audio") == pytest.approx(3.5)


@requires_ffmpeg
def test_stream_duration_reads_both_streams(tmp_path: Path) -> None:
    media = _make_media(tmp_path / "mixed.mp4", video=0.5, audio=2.0)

    assert ft.stream_duration(media, kind="video") == pytest.approx(0.5, abs=0.15)
    assert ft.stream_duration(media, kind="audio") == pytest.approx(2.0, abs=0.15)
    assert ft.stream_info(media, kind="video").get("codec_type") == "video"


# ----------------------------------------------------------- 对齐决策（纯逻辑）
def _stub_durations(monkeypatch, *, audio: float, video: float, fps: float = 25.0) -> None:
    monkeypatch.setattr(
        ft,
        "stream_duration",
        lambda _path, kind="audio": audio if kind == "audio" else video,
    )
    monkeypatch.setattr(
        ft,
        "stream_info",
        lambda _path, kind="audio": {"r_frame_rate": f"{fps}/1"},
    )


def test_alignment_plan_skips_when_close_enough(monkeypatch) -> None:
    _stub_durations(monkeypatch, audio=10.02, video=10.0)

    assert ft._alignment_plan(Path("v.mp4"), Path("a.mp4")) is None


def test_alignment_plan_skips_when_duration_unknown(monkeypatch) -> None:
    _stub_durations(monkeypatch, audio=0.0, video=10.0)

    assert ft._alignment_plan(Path("v.mp4"), Path("a.mp4")) is None


def test_alignment_plan_slows_video_that_runs_ahead(monkeypatch) -> None:
    """画面 8s / 音频 10s：画面跑得快，需要把时间轴拉长。"""
    _stub_durations(monkeypatch, audio=10.0, video=8.0, fps=25.0)

    plan = ft._alignment_plan(Path("v.mp4"), Path("a.mp4"))

    assert plan is not None
    assert plan["scale"] == pytest.approx(1.25)
    assert plan["fps"] == pytest.approx(20.0)


def test_alignment_plan_speeds_up_video_that_lags_audio(monkeypatch) -> None:
    """画面 10s / 音频 8s：声音跑到前面，需要把时间轴压缩。"""
    _stub_durations(monkeypatch, audio=8.0, video=10.0, fps=25.0)

    plan = ft._alignment_plan(Path("v.mp4"), Path("a.mp4"))

    assert plan is not None
    assert plan["scale"] == pytest.approx(0.8)
    assert plan["fps"] == pytest.approx(31.25)


def test_alignment_plan_refuses_extreme_drift(monkeypatch) -> None:
    """音频只有 1s、画面却有 10s：更像音轨不完整，不做灾难性拉伸。"""
    _stub_durations(monkeypatch, audio=1.0, video=10.0)

    assert ft._alignment_plan(Path("v.mp4"), Path("a.mp4")) is None


# ----------------------------------------------------------------- 端到端
@requires_ffmpeg
def test_mux_audio_stretches_video_to_match_longer_audio(tmp_path: Path) -> None:
    """画面比声音快（无声视频过短）→ 拉长画面到音频时长。"""
    silent = _make_media(tmp_path / "silent.mp4", video=1.0)
    source = _make_media(tmp_path / "source.mp4", video=0.5, audio=2.0)
    output = tmp_path / "out.mp4"

    assert ft.mux_audio(silent, source, output) is True
    assert ft.has_audio_stream(output) is True
    assert ft.stream_duration(output, kind="video") == pytest.approx(2.0, abs=0.2)


@requires_ffmpeg
def test_mux_audio_compresses_video_to_match_shorter_audio(tmp_path: Path) -> None:
    """声音比画面快（音轨过短）→ 压缩画面到音频时长。"""
    silent = _make_media(tmp_path / "silent.mp4", video=2.0)
    source = _make_media(tmp_path / "source.mp4", video=0.5, audio=1.0)
    output = tmp_path / "out.mp4"

    assert ft.mux_audio(silent, source, output) is True
    assert ft.has_audio_stream(output) is True
    assert ft.stream_duration(output, kind="video") == pytest.approx(1.0, abs=0.2)


@requires_ffmpeg
def test_mux_audio_keeps_copy_path_when_already_synced(tmp_path: Path) -> None:
    """时长本来一致时不做校正重编码（视频流仍是原始 mpeg4）。"""
    silent = _make_media(tmp_path / "silent.mp4", video=1.0)
    source = _make_media(tmp_path / "source.mp4", video=1.0, audio=1.0)
    output = tmp_path / "out.mp4"

    assert ft.mux_audio(silent, source, output) is True
    assert ft.stream_info(output, kind="video").get("codec_name") == "mpeg4"
    assert ft.stream_duration(output, kind="video") == pytest.approx(1.0, abs=0.15)


def _make_media(
    path: Path,
    *,
    video: float,
    audio: float | None = None,
    rate: int = 10,
) -> Path:
    """用 lavfi 生成测试媒体：可只带画面，也可画面 + 音轨且两者时长不同。"""
    args = [
        ft.find_ffmpeg() or "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=size=160x120:rate={rate}:duration={video}",
    ]
    if audio is None:
        args += ["-c:v", "mpeg4", "-q:v", "6", "-an"]
    else:
        args += [
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={audio}",
            "-c:v",
            "mpeg4",
            "-q:v",
            "6",
            "-c:a",
            "aac",
        ]
    args.append(str(path))

    completed = subprocess.run(args, capture_output=True, text=True, check=False)
    if completed.returncode != 0 or not path.exists() or path.stat().st_size == 0:
        pytest.skip(f"无法生成测试媒体：{completed.stderr.strip()[-200:]}")
    return path


# ------------------------------------------------- 时间轴探测（VFR 判定，纯逻辑）
def _stub_video_stream(monkeypatch, **fields: object) -> None:
    """把 ``probe_media`` 换成一个只返回指定视频流的假实现。"""
    stream = {"codec_type": "video", **fields}
    monkeypatch.setattr(ft, "probe_media", lambda _path: {"streams": [stream]})


def test_analyze_video_timing_without_probe(monkeypatch) -> None:
    monkeypatch.setattr(ft, "probe_media", lambda _path: None)

    assert ft.analyze_video_timing("x.mp4") is None


def test_analyze_video_timing_without_video_stream(monkeypatch) -> None:
    monkeypatch.setattr(ft, "probe_media", lambda _path: {"streams": [{"codec_type": "audio"}]})

    assert ft.analyze_video_timing("x.mp4") is None


def test_analyze_video_timing_accepts_constant_frame_rate(monkeypatch) -> None:
    _stub_video_stream(
        monkeypatch,
        r_frame_rate="30/1",
        avg_frame_rate="30/1",
        nb_frames="300",
        duration="10.0",
    )

    timing = ft.analyze_video_timing("x.mp4")

    assert timing is not None
    assert timing.variable is False
    assert timing.fps == pytest.approx(30.0)
    assert timing.frame_count == 300


def test_analyze_video_timing_flags_mismatched_rates(monkeypatch) -> None:
    """标称 30fps / 平均 20fps：典型可变帧率，归一到 30fps 以保住原帧。"""
    _stub_video_stream(
        monkeypatch,
        r_frame_rate="30/1",
        avg_frame_rate="20/1",
        nb_frames="40",
        duration="2.0",
    )

    timing = ft.analyze_video_timing("x.mp4")

    assert timing is not None
    assert timing.variable is True
    assert timing.fps == pytest.approx(30.0)
    assert timing.reason


def test_analyze_video_timing_flags_inconsistent_timeline(monkeypatch) -> None:
    """帧率看着一致，但「帧数 ÷ 帧率」与流时长对不上，同样算时间轴异常。"""
    _stub_video_stream(
        monkeypatch,
        r_frame_rate="30/1",
        avg_frame_rate="30/1",
        nb_frames="90",
        duration="2.0",
    )

    timing = ft.analyze_video_timing("x.mp4")

    assert timing is not None
    assert timing.variable is True


def test_choose_cfr_fps_snaps_to_standard_rate() -> None:
    assert ft._choose_cfr_fps(20.0, 30.0) == pytest.approx(30.0)
    assert ft._choose_cfr_fps(29.97, 29.97) == pytest.approx(29.97)
    assert ft._choose_cfr_fps(0.0, 25.0) == pytest.approx(25.0)


def test_choose_cfr_fps_ignores_absurd_nominal_rate() -> None:
    """标称帧率虚高到平均值的几十倍（如 1000/1）时退回平均值，避免帧数爆炸。"""
    assert ft._choose_cfr_fps(25.0, 1000.0) == pytest.approx(25.0)


def test_detect_variable_timing_without_any_rate() -> None:
    """连帧率都探测不到时不做判定（给不出可靠的归一化目标帧率）。"""
    assert ft._detect_variable_timing(0.0, 0.0, 0, 0.0) == (False, "")


def test_normalize_to_cfr_rejects_bad_fps(tmp_path: Path) -> None:
    assert ft.normalize_to_cfr(tmp_path / "a.mp4", tmp_path / "b.mp4", fps=0.0) is False


def test_normalize_to_cfr_without_ffmpeg(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ft, "find_ffmpeg", lambda: None)

    assert ft.normalize_to_cfr(tmp_path / "a.mp4", tmp_path / "b.mp4", fps=30.0) is False


# --------------------------------------------------------------- 真实 VFR 素材
@requires_ffmpeg
def test_analyze_video_timing_detects_real_vfr(make_vfr_video) -> None:
    timing = ft.analyze_video_timing(make_vfr_video())

    assert timing is not None
    assert timing.variable is True
    assert timing.fps > 0.0
    assert timing.nominal_fps > timing.average_fps


@requires_ffmpeg
def test_normalize_to_cfr_makes_frame_intervals_constant(make_vfr_video, tmp_path) -> None:
    """归一化后每帧 PTS 间隔必须严格相等——这正是"开头慢、后面快"的解药。"""
    source = make_vfr_video()
    timing = ft.analyze_video_timing(source)
    assert timing is not None and timing.variable is True

    target = tmp_path / "cfr.mp4"
    assert ft.normalize_to_cfr(source, target, fps=timing.fps) is True

    info = ft.stream_info(target, kind="video")
    nominal = ft._parse_rate(info.get("r_frame_rate"))
    average = ft._parse_rate(info.get("avg_frame_rate"))
    assert nominal > 0.0
    assert nominal == pytest.approx(average, rel=0.01)

    intervals = _frame_intervals(target)
    assert len(intervals) > 10
    assert max(intervals) - min(intervals) < 1e-3
    assert sum(intervals) / len(intervals) == pytest.approx(1.0 / nominal, rel=0.01)


def _frame_intervals(path: Path) -> list[float]:
    """用 ffprobe 读出每帧 PTS，返回相邻两帧的时间间隔（秒）。"""
    ffprobe = ft.find_ffprobe()
    if ffprobe is None:  # pragma: no cover - 取决于环境
        pytest.skip("需要 ffprobe 才能读取帧时间戳")
    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "frame=pts_time",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    stamps = [
        float(line.strip().rstrip(","))
        for line in completed.stdout.splitlines()
        if line.strip().rstrip(",")
    ]
    return [later - earlier for earlier, later in zip(stamps, stamps[1:], strict=False)]
