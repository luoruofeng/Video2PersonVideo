"""ffmpeg 相关能力：音轨探测、音轨合并、H.264 重编码。

ultralytics/OpenCV 写出的视频不带音频，这里用 ffmpeg-python（探测）
和 ffmpeg 命令行（合并/转码）把原视频音轨补回来。

注意：本模块依赖系统里的 ``ffmpeg`` / ``ffprobe`` 可执行文件，
缺失时所有函数都会优雅降级并返回 ``False``，不会抛异常打断主流程。
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .logger import get_logger

logger = get_logger(__name__)

_FASTSTART_SUFFIXES = {".mp4", ".m4v", ".mov"}
_MP4_FRIENDLY_SUFFIXES = {".mp4", ".m4v", ".mov"}

#: VFR 判定：标称帧率（r_frame_rate）与平均帧率（avg_frame_rate）的相对差异阈值
_VFR_RATE_RATIO = 0.02
#: VFR 判定：「帧数 ÷ 平均帧率」与流时长的相对差异阈值（时间轴不自洽）
_VFR_DURATION_RATIO = 0.02
#: 归一化目标帧率的候选档位：对齐常见标准帧率，播放器兼容性更好
_STANDARD_FPS = (
    8.0,
    10.0,
    12.0,
    15.0,
    16.0,
    20.0,
    23.976,
    24.0,
    25.0,
    29.97,
    30.0,
    48.0,
    50.0,
    59.94,
    60.0,
    120.0,
)
#: 归一化目标帧率的合理区间
_MIN_CFR_FPS = 1.0
_MAX_CFR_FPS = 240.0
#: 标称帧率最多允许是平均帧率的多少倍（超出说明 r_frame_rate 虚高，改用平均值）
_MAX_FPS_UPSAMPLE = 3.0


def find_ffmpeg() -> str | None:
    """返回 ffmpeg 可执行文件路径，找不到返回 ``None``。"""
    return shutil.which("ffmpeg")


def find_ffprobe() -> str | None:
    """返回 ffprobe 可执行文件路径，找不到返回 ``None``。"""
    return shutil.which("ffprobe")


def ffmpeg_available() -> bool:
    return find_ffmpeg() is not None


def probe_media(path: str | Path) -> dict | None:
    """探测媒体流信息，失败返回 ``None``。

    优先用 ffmpeg-python 的 ``probe``，不可用时退回直接调用 ffprobe。
    """
    target = Path(path)
    if not target.exists():
        return None

    try:
        import ffmpeg  # ffmpeg-python

        return ffmpeg.probe(str(target))
    except ImportError:
        logger.debug("未安装 ffmpeg-python，改用 ffprobe 探测")
    except Exception as exc:  # ffmpeg.Error / ffprobe 不存在等
        logger.debug("ffmpeg-python 探测失败：%s", exc)

    ffprobe = find_ffprobe()
    if ffprobe is None:
        return None
    try:
        completed = subprocess.run(
            [ffprobe, "-v", "error", "-print_format", "json", "-show_streams", str(target)],
            capture_output=True,
            check=True,
            text=True,
        )
        return json.loads(completed.stdout)
    except Exception as exc:
        logger.debug("ffprobe 探测失败：%s", exc)
        return None


def has_audio_stream(path: str | Path) -> bool:
    """判断媒体文件是否含音频流。无法探测时返回 ``False``（按无音轨处理）。"""
    info = probe_media(path)
    if not info:
        return False
    return any(stream.get("codec_type") == "audio" for stream in info.get("streams", []))


def stream_info(path: str | Path, *, kind: str = "audio") -> dict:
    """返回第一条指定类型流（``audio`` / ``video``）的探测信息，找不到返回 ``{}``。"""
    info = probe_media(path)
    if not info:
        return {}
    for stream in info.get("streams", []):
        if stream.get("codec_type") == kind:
            return stream
    return {}


def stream_duration(path: str | Path, *, kind: str = "audio") -> float:
    """返回指定类型流的时长（秒），探测不到返回 ``0.0``。"""
    info = probe_media(path)
    return _duration_from_info(info, kind) if info else 0.0


def container_duration(path: str | Path) -> float:
    """返回整段媒体的时长（秒），探测不到返回 ``0.0``。

    优先取容器时长，取不到再退回第一条视频流——体积换算（码率 = 体积 ÷ 时长）
    依赖它，所以这里尽量给出可信值。
    """
    info = probe_media(path)
    if not info:
        return 0.0
    container = info.get("format", {})
    seconds = _positive_float(container.get("duration")) or _parse_duration_text(
        container.get("tags", {}).get("DURATION")
    )
    if seconds > 0.0:
        return seconds
    return max(_duration_from_info(info, "video"), _duration_from_info(info, "audio"))


def stream_bitrate(path: str | Path, *, kind: str = "video") -> int:
    """返回指定类型流的码率（bps），探测不到返回 ``0``。

    优先取流的 ``bit_rate``（精确到流），取不到再退回容器总码率
    （含音频，用于估算音量级时已经够用）。
    """
    info = probe_media(path)
    if not info:
        return 0
    for stream in info.get("streams", []):
        if stream.get("codec_type") == kind:
            rate = _positive_float(stream.get("bit_rate"))
            if rate > 0.0:
                return int(rate)
    return int(_positive_float(info.get("format", {}).get("bit_rate")))


@dataclass(slots=True)
class VideoTiming:
    """源视频的时间轴信息，用来决定「是否需要先归一到恒定帧率」。

    ``variable`` 为 ``True`` 表示源是可变帧率（VFR，手机录像 / 录屏 / 剪辑导出
    很常见）或时间轴不自洽：每帧的真实显示时长并不相等。此时若照旧逐帧按单一
    fps 写出，画面会被"均匀摊平"——帧率高的片段被拉长（慢放）、帧率低的片段
    被压缩（快进），而音频仍按真实时间轴走，于是表现为"开头慢、后面突然变快"。
    """

    #: 归一化时使用的目标恒定帧率（0 = 无法确定）
    fps: float
    #: 标称帧率 ``r_frame_rate``
    nominal_fps: float
    #: 平均帧率 ``avg_frame_rate``
    average_fps: float
    #: 视频流时长（秒），探测不到为 0
    duration: float
    #: 视频流总帧数，探测不到为 0
    frame_count: int
    #: 是否可变帧率 / 时间轴不自洽
    variable: bool = False
    #: 判定原因（写入日志，便于排查）
    reason: str = ""

    def describe(self) -> str:
        rate = f"{self.average_fps:.3f} FPS" if self.average_fps > 0.0 else "帧率未知"
        frames = f"{self.frame_count} 帧" if self.frame_count > 0 else "帧数未知"
        text = f"{rate} / {frames} / 约 {self.duration:.2f}s"
        if self.variable:
            text += f"（可变帧率：{self.reason}）"
        return text


def analyze_video_timing(path: str | Path) -> VideoTiming | None:
    """探测视频的帧率 / 时长信息，并判断它是不是可变帧率（VFR）。

    命中以下任一条即视为"需要先归一化"：

    * ``r_frame_rate``（标称帧率）与 ``avg_frame_rate``（平均帧率）相差超过 2%；
    * ``帧数 ÷ 平均帧率`` 与流时长相差超过 2%（时间轴不自洽）。

    探测失败或文件里没有视频流时返回 ``None``（调用方按"不需要归一化"处理）。
    """
    info = probe_media(path)
    if not info:
        return None
    stream = next(
        (item for item in info.get("streams", []) if item.get("codec_type") == "video"),
        None,
    )
    if stream is None:
        return None

    nominal = _parse_rate(stream.get("r_frame_rate"))
    average = _parse_rate(stream.get("avg_frame_rate"))
    frame_count = int(_positive_float(stream.get("nb_frames")))
    duration = _duration_from_info(info, "video")
    if duration <= 0.0 and frame_count > 0 and average > 0.0:
        duration = frame_count / average

    variable, reason = _detect_variable_timing(nominal, average, frame_count, duration)
    return VideoTiming(
        fps=_choose_cfr_fps(average, nominal),
        nominal_fps=nominal,
        average_fps=average,
        duration=duration,
        frame_count=frame_count,
        variable=variable,
        reason=reason,
    )


def _detect_variable_timing(
    nominal: float, average: float, frame_count: int, duration: float
) -> tuple[bool, str]:
    """判断时间轴是否为可变帧率，返回 ``(是否 VFR, 原因)``。"""
    if nominal <= 0.0 and average <= 0.0:
        # 一个可信帧率都拿不到：给不出可靠的归一化目标帧率，交给旧流程兜底
        return False, ""

    if nominal > 0.0 and average > 0.0:
        ratio = abs(nominal - average) / max(nominal, average)
        if ratio > _VFR_RATE_RATIO:
            return True, (
                f"标称 {nominal:.3f}fps 与平均 {average:.3f}fps 不一致"
                f"（相差 {ratio * 100:.1f}%）"
            )

    if frame_count > 0 and duration > 0.0 and average > 0.0:
        expected = frame_count / average
        ratio = abs(expected - duration) / duration
        if ratio > _VFR_DURATION_RATIO:
            return True, (
                f"{frame_count} 帧 ÷ {average:.3f}fps = {expected:.2f}s，"
                f"与流时长 {duration:.2f}s 不符"
            )

    return False, ""


def _choose_cfr_fps(average: float, nominal: float) -> float:
    """挑选归一化目标帧率：优先保帧（取标称帧率），虚高时退回平均帧率。

    标称帧率是"能容纳所有帧间隔的最小帧率"，按它重采样不会丢掉原帧；但它
    常被写成理论上限（如 1000/1），因此限制为不超过平均帧率的 3 倍。最终再
    对齐到最接近的标准帧率档位，播放器兼容性更好。
    """
    if nominal > 0.0 and (average <= 0.0 or nominal <= average * _MAX_FPS_UPSAMPLE):
        base = nominal
    else:
        base = average
    if base <= 0.0:
        return 0.0

    base = min(max(base, _MIN_CFR_FPS), _MAX_CFR_FPS)
    for standard in _STANDARD_FPS:
        if standard >= base - 1e-6:
            return standard
    return base


def _duration_from_info(info: dict, kind: str) -> float:
    """从 ffprobe 结果里取指定类型流的时长（秒）。

    ffprobe 对流时长的表述因容器而异（``duration`` / ``tags.DURATION`` /
    ``nb_frames``），这里逐级兜底，尽量拿到可信值——音画时长对齐完全依赖它。
    """
    streams = info.get("streams", [])
    for stream in streams:
        if stream.get("codec_type") != kind:
            continue
        seconds = _positive_float(stream.get("duration"))
        if seconds <= 0.0:
            seconds = _parse_duration_text(stream.get("tags", {}).get("DURATION"))
        if seconds <= 0.0:
            frames = _positive_float(stream.get("nb_frames"))
            rate = _positive_float(stream.get("sample_rate")) or _parse_rate(
                stream.get("avg_frame_rate") or stream.get("r_frame_rate")
            )
            if frames > 0.0 and rate > 0.0:
                seconds = frames / rate
        if seconds > 0.0:
            return seconds

    # 容器里只有这一条流时，容器时长即该流时长（mkv/webm 常见）
    if len(streams) == 1:
        container = info.get("format", {})
        return _positive_float(container.get("duration")) or _parse_duration_text(
            container.get("tags", {}).get("DURATION")
        )
    return 0.0


def _positive_float(value: object) -> float:
    """把探测到的字符串/数字转成正浮点数，非法值返回 ``0.0``。"""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    return number if number > 0.0 else 0.0


def _parse_duration_text(text: object) -> float:
    """解析 ffprobe 的时间串，如 ``"00:01:02.500000000"`` → ``62.5``。"""
    if not text:
        return 0.0
    try:
        parts = [float(part) for part in str(text).split(":")]
    except ValueError:
        return 0.0
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60.0 + part
    return seconds if seconds > 0.0 else 0.0


def _parse_rate(text: object) -> float:
    """解析 ffprobe 的帧率，如 ``"30000/1001"`` → ``29.97``。"""
    if not text:
        return 0.0
    raw = str(text)
    if "/" in raw:
        numerator, _, denominator = raw.partition("/")
        divisor = _positive_float(denominator)
        return _positive_float(numerator) / divisor if divisor > 0.0 else 0.0
    return _positive_float(raw)


def _mux_with_ffmpeg_python(
    silent_video: Path, audio_source: Path, output: Path, *, ffmpeg_bin: str, overwrite: bool
) -> bool:
    """用 ffmpeg-python 合并视频流与音轨（默认流选择即可完成 mux）。"""
    try:
        import ffmpeg  # ffmpeg-python
    except ImportError:
        return False

    try:
        video = ffmpeg.input(str(silent_video))
        audio = ffmpeg.input(str(audio_source)).audio
        node = ffmpeg.output(
            video,
            audio,
            str(output),
            vcodec="copy",
            acodec="copy",
            shortest=None,
            loglevel="error",
        )
        node = node.overwrite_output() if overwrite else node
        node.run(quiet=True, cmd=ffmpeg_bin)
    except Exception as exc:
        logger.warning("ffmpeg-python 合并音轨失败（%s），改用命令行方式重试", exc)
        return False

    return output.exists() and output.stat().st_size > 0


def _mux_with_cli(
    silent_video: Path, audio_source: Path, output: Path, *, ffmpeg_bin: str, overwrite: bool
) -> bool:
    """命令行兜底合并：显式 map 0:v / 1:a，音频统一转 aac 保证兼容性。"""
    args = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        str(silent_video),
        "-i",
        str(audio_source),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
    ]
    if output.suffix.lower() in _FASTSTART_SUFFIXES:
        args += ["-movflags", "+faststart"]
    args.append(str(output))

    try:
        completed = subprocess.run(args, capture_output=True, text=True, check=False)
    except OSError as exc:
        logger.warning("调用 ffmpeg 失败：%s", exc)
        return False

    if completed.returncode != 0:
        logger.warning("ffmpeg 合并音轨失败：%s", completed.stderr.strip()[-500:])
        return False
    return output.exists() and output.stat().st_size > 0


#: 音画时长校正允许的时间轴缩放范围：超出说明源文件本身不同步，宁可不改
_MIN_TIME_SCALE = 0.5
_MAX_TIME_SCALE = 2.0
#: 时长相对容差与绝对下限：小于该偏差不值得为对齐重编码一遍
_DURATION_TOLERANCE_RATIO = 0.005
_DURATION_TOLERANCE_SECONDS = 0.08


def _alignment_plan(silent_video: Path, audio_source: Path) -> dict | None:
    """判断无声视频是否需要向音频时长对齐。

    这正是"画面快于声音 / 声音快于画面"的病灶：OpenCV 写出的视频时长是
    ``帧数 ÷ 源 fps``，源视频一旦是可变帧率（手机录像 / 录屏）或 fps 读偏，
    它与原音轨时长就不再相等，直接 mux 必然错位。这里返回把画面时间轴
    整体缩放到音频时长的参数，``None`` 表示偏差在容差内、无需校正。
    """
    audio_dur = stream_duration(audio_source, kind="audio")
    video_dur = stream_duration(silent_video, kind="video")
    if audio_dur <= 0.0 or video_dur <= 0.0:
        return None

    drift = audio_dur - video_dur
    tolerance = max(_DURATION_TOLERANCE_SECONDS, _DURATION_TOLERANCE_RATIO * audio_dur)
    if abs(drift) <= tolerance:
        return None

    scale = audio_dur / video_dur
    if not _MIN_TIME_SCALE <= scale <= _MAX_TIME_SCALE:
        logger.warning(
            "音画时长相差 %.0f%%（音频 %.2fs / 画面 %.2fs），疑似源文件音轨本身不完整，"
            "保持画面原速不做拉伸",
            abs(scale - 1.0) * 100.0,
            audio_dur,
            video_dur,
        )
        return None

    fps = _parse_rate(stream_info(silent_video, kind="video").get("r_frame_rate"))
    return {
        "scale": scale,
        # 时间轴被缩放后帧率随之变化，取目标恒定帧率 = 原帧率 ÷ 缩放系数
        "fps": fps / scale if fps > 0.0 else 0.0,
        "describe": f"画面 {video_dur:.2f}s → {audio_dur:.2f}s（{scale:.3f}×）",
    }


def _mux_aligned(
    silent_video: Path,
    audio_source: Path,
    output: Path,
    *,
    ffmpeg_bin: str,
    overwrite: bool,
    scale: float,
    fps: float,
    crf: int,
) -> bool:
    """按音频时长重定时画面后再合并（帧数不变，只缩放时间戳）。"""
    args = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        str(silent_video),
        "-i",
        str(audio_source),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        # setpts 缩放时间戳 → 输出时长与音频一致（画面逐帧保留，不丢帧）
        "-vf",
        f"setpts=PTS*{scale:.9f}",
        "-c:v",
        "libx264",
        "-crf",
        str(int(crf)),
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
    ]
    if fps > 0.0:
        # setpts 之后时间轴是变频的，按目标帧率规整回恒定帧率便于播放
        args += ["-r", f"{fps:.6f}"]
    args += ["-c:a", "aac", "-b:a", "192k", "-shortest"]
    if output.suffix.lower() in _FASTSTART_SUFFIXES:
        args += ["-movflags", "+faststart"]
    args.append(str(output))

    try:
        completed = subprocess.run(args, capture_output=True, text=True, check=False)
    except OSError as exc:
        logger.warning("调用 ffmpeg 做音画对齐失败：%s", exc)
        return False

    if completed.returncode != 0:
        logger.warning("音画对齐合并失败：%s", completed.stderr.strip()[-500:])
        return False
    return output.exists() and output.stat().st_size > 0


def mux_audio(
    silent_video: str | Path,
    audio_source: str | Path,
    output: str | Path,
    *,
    overwrite: bool = True,
    crf: int = 23,
) -> bool:
    """把 ``audio_source`` 的音轨合并到 ``silent_video``，写出到 ``output``。

    合并前先比对画面与音轨的真实时长：偏差超过容差时按音频时长重定时画面，
    从根上避免"画面比声音快 / 声音比画面快"的音画不同步；偏差在容差内则
    直接复制视频流，不额外重编码。

    成功返回 ``True``；缺少 ffmpeg 或合并失败返回 ``False``（调用方应保留无声版本）。
    """
    silent_video, audio_source, output = map(Path, (silent_video, audio_source, output))

    ffmpeg_bin = find_ffmpeg()
    if ffmpeg_bin is None:
        logger.warning("未找到 ffmpeg，无法保留音轨（结果视频将没有声音）")
        return False
    if not has_audio_stream(audio_source):
        logger.info("原视频没有音频流，跳过音轨合并")
        return False

    output.parent.mkdir(parents=True, exist_ok=True)

    plan = _alignment_plan(silent_video, audio_source)
    if plan is not None:
        if _mux_aligned(
            silent_video,
            audio_source,
            output,
            ffmpeg_bin=ffmpeg_bin,
            overwrite=overwrite,
            crf=crf,
            scale=plan["scale"],
            fps=plan["fps"],
        ):
            logger.info("音轨合并完成（已校正音画时长：%s）", plan["describe"])
            return True
        logger.warning("音画时长校正失败，退回按原时长直接合并")

    if _mux_with_ffmpeg_python(
        silent_video, audio_source, output, ffmpeg_bin=ffmpeg_bin, overwrite=overwrite
    ):
        logger.info("音轨合并完成：%s", output.name)
        return True
    if _mux_with_cli(
        silent_video, audio_source, output, ffmpeg_bin=ffmpeg_bin, overwrite=overwrite
    ):
        logger.info("音轨合并完成（命令行方式）：%s", output.name)
        return True

    logger.warning("音轨合并失败，将输出无声视频")
    return False


def transcode_h264(
    source: str | Path,
    output: str | Path,
    *,
    crf: int = 23,
    preset: str = "medium",
    overwrite: bool = True,
) -> bool:
    """用 libx264 重编码为 H.264（体积更小、播放器/分享兼容性更好）。

    成功返回 ``True``；缺少 ffmpeg 或失败返回 ``False``。
    """
    source, output = Path(source), Path(output)
    ffmpeg_bin = find_ffmpeg()
    if ffmpeg_bin is None:
        logger.warning("未找到 ffmpeg，跳过重编码")
        return False

    pix_fmt = "yuv420p" if output.suffix.lower() in _MP4_FRIENDLY_SUFFIXES else None
    try:
        import ffmpeg  # ffmpeg-python

        stream = ffmpeg.input(str(source)).video
        kwargs: dict[str, object] = {"vcodec": "libx264", "crf": crf, "preset": preset}
        if pix_fmt:
            kwargs["pix_fmt"] = pix_fmt
        node = ffmpeg.output(stream, str(output), **kwargs)
        node = node.overwrite_output() if overwrite else node
        node.run(quiet=True, cmd=ffmpeg_bin)
        return output.exists() and output.stat().st_size > 0
    except ImportError:
        logger.debug("未安装 ffmpeg-python，改用命令行重编码")
    except Exception as exc:
        logger.warning("ffmpeg-python 重编码失败（%s），改用命令行重试", exc)

    args = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        str(source),
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-preset",
        preset,
    ]
    if pix_fmt:
        args += ["-pix_fmt", pix_fmt]
        args += ["-movflags", "+faststart"]
    else:
        args += ["-pix_fmt", "yuv420p"]
    args.append(str(output))

    completed = subprocess.run(args, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        logger.warning("重编码失败：%s", completed.stderr.strip()[-500:])
        return False
    return output.exists() and output.stat().st_size > 0


def normalize_to_cfr(
    source: str | Path,
    output: str | Path,
    *,
    fps: float,
    crf: int = 18,
    preset: str = "veryfast",
    overwrite: bool = True,
) -> bool:
    """把源视频的**视频流**重采样成恒定帧率（丢弃音轨），写出到 ``output``。

    用 ffmpeg 的 ``fps`` 滤镜按每一帧的**真实时间戳**重采样：帧率高的片段丢掉
    冗余帧、帧率低的片段复制帧，最终每帧间隔严格相等，同时整条时间轴仍对齐原
    时间轴。这样后续用 OpenCV 逐帧处理时输出的画面节奏就与原视频一致，不会再
    出现"开头慢、后面突然变快"。

    只处理画面、不碰音频：音轨的时间轴本来就是真实的，调用方仍从原视频取音轨。

    成功返回 ``True``；缺少 ffmpeg、``fps`` 非法或转码失败返回 ``False``
    （调用方应退化为按原始帧率处理）。
    """
    source, output = Path(source), Path(output)
    if fps <= 0.0:
        logger.warning("归一化目标帧率非法（%s），跳过时间轴归一化", fps)
        return False

    ffmpeg_bin = find_ffmpeg()
    if ffmpeg_bin is None:
        logger.warning("未找到 ffmpeg，无法归一化时间轴")
        return False

    output.parent.mkdir(parents=True, exist_ok=True)
    args = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        str(source),
        "-an",
        # fps 滤镜按真实时间戳重采样，输出时间戳天然是恒定帧率
        "-vf",
        f"fps={fps:.6f}",
        "-c:v",
        "libx264",
        "-crf",
        str(int(crf)),
        "-preset",
        preset,
        "-pix_fmt",
        "yuv420p",
    ]
    if output.suffix.lower() in _FASTSTART_SUFFIXES:
        args += ["-movflags", "+faststart"]
    args.append(str(output))

    try:
        completed = subprocess.run(args, capture_output=True, text=True, check=False)
    except OSError as exc:
        logger.warning("调用 ffmpeg 归一化时间轴失败：%s", exc)
        return False

    if completed.returncode != 0 or not output.exists() or output.stat().st_size == 0:
        logger.warning("时间轴归一化失败：%s", completed.stderr.strip()[-500:])
        output.unlink(missing_ok=True)
        return False

    logger.debug("时间轴已归一化为恒定 %.3f FPS：%s", fps, output.name)
    return True


def target_video_bitrate_kbps(
    max_bytes: int,
    duration: float,
    *,
    audio_kbps: int = 128,
    min_kbps: int = 250,
) -> int:
    """按"整段不超过 ``max_bytes``"反推视频码率（kbps）。

    纯函数：``max_bytes × 8 ÷ 时长`` 得到总码率，扣掉音频预留后即视频码率；
    结果不会低于 ``min_kbps``（再低就只剩马赛克了，不如老实告知压不到）。
    参数非法时返回 ``0``，调用方据此跳过压缩。
    """
    if max_bytes <= 0 or duration <= 0:
        return 0
    total_kbps = max_bytes * 8.0 / duration / 1000.0
    video_kbps = total_kbps - max(audio_kbps, 0)
    return int(max(video_kbps, min_kbps))


def shrink_to_limit(
    source: str | Path,
    output: str | Path,
    max_bytes: int,
    *,
    audio_kbps: int = 128,
    min_kbps: int = 250,
    preset: str = "veryfast",
    attempts: int = 3,
    overwrite: bool = True,
) -> tuple[bool, int]:
    """把 ``source`` 压到不超过 ``max_bytes``，写出 ``output``。

    做法：按体积上限反推码率 → ABR 编码 → 复查体积，超标就按 0.8 系数再压一轮
    （最多 ``attempts`` 轮）。ABR 对复杂画面会略微超发，复查是必要的兜底。

    返回 ``(是否压到限制内, 实际使用的视频码率 kbps)``。任何一步不可行
    （无 ffmpeg、探测不到时长、码率为 0）都返回 ``(False, 0)`` 且**不动** ``output``；
    若多轮之后仍超标但确实变小了，会用当前最小的结果覆盖 ``output`` 并返回
    ``(False, 实际码率)``——宁可小一点，也不白白浪费一次重编码。
    """
    source, output = Path(source), Path(output)
    in_place = source.resolve() == output.resolve()
    ffmpeg_bin = find_ffmpeg()
    if ffmpeg_bin is None:
        logger.warning("未找到 ffmpeg，跳过体积压缩")
        return False, 0

    duration = container_duration(source)
    if duration <= 0.0:
        logger.warning("无法探测视频时长，跳过体积压缩")
        return False, 0

    keeps_audio = has_audio_stream(source)
    reserve_kbps = audio_kbps if keeps_audio else 0
    rate = target_video_bitrate_kbps(
        max_bytes, duration, audio_kbps=reserve_kbps, min_kbps=min_kbps
    )
    if rate <= 0:  # pragma: no cover - duration 已校验过，此处仅防御
        return False, 0

    best_size = output.stat().st_size if output.exists() else 2**63 - 1
    used_rate = rate

    for attempt in range(max(1, attempts)):
        temp = output.with_name(f"{output.stem}.sizing{output.suffix}")
        args = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y" if overwrite else "-n",
            "-i",
            str(source),
            "-c:v",
            "libx264",
            "-b:v",
            f"{rate}k",
            "-maxrate",
            f"{int(rate * 1.25)}k",
            "-bufsize",
            f"{int(rate * 2)}k",
            "-preset",
            preset,
            "-pix_fmt",
            "yuv420p",
        ]
        args += ["-c:a", "aac", "-b:a", f"{audio_kbps}k"] if keeps_audio else ["-an"]
        if output.suffix.lower() in _FASTSTART_SUFFIXES:
            args += ["-movflags", "+faststart"]
        args.append(str(temp))

        try:
            completed = subprocess.run(args, capture_output=True, text=True, check=False)
        except OSError as exc:
            logger.warning("调用 ffmpeg 压缩体积失败：%s", exc)
            temp.unlink(missing_ok=True)
            return False, 0

        if completed.returncode != 0 or not temp.exists() or temp.stat().st_size == 0:
            logger.warning("体积压缩失败：%s", completed.stderr.strip()[-500:])
            temp.unlink(missing_ok=True)
            return False, 0

        size = temp.stat().st_size
        used_rate = rate
        if size < best_size:  # 只保留更小的结果，越压越大直接丢掉
            if output.exists() and not in_place:
                output.unlink()
            temp.replace(output)
            best_size = size
        else:
            temp.unlink(missing_ok=True)

        if size <= max_bytes:
            return True, used_rate

        logger.debug(
            "第 %d 轮压缩后体积仍超过上限（%.2f MB > %.2f MB），降低码率重试",
            attempt + 1,
            size / 1024 / 1024,
            max_bytes / 1024 / 1024,
        )
        rate = max(int(rate * 0.8), min_kbps)

    logger.warning("多轮压缩后仍超过体积上限（当前 %.2f MB）", best_size / 1024 / 1024)
    return False, used_rate


def inject_metadata(path: str | Path, metadata: dict[str, str]) -> bool:
    """原地写入容器元数据（重封装，不重新编码）。

    成功返回 ``True``；缺少 ffmpeg、元数据为空或写入失败返回 ``False``。
    """
    target = Path(path)
    if not metadata or not target.exists():
        return False

    ffmpeg_bin = find_ffmpeg()
    if ffmpeg_bin is None:
        logger.debug("未找到 ffmpeg，跳过元数据写入")
        return False

    temp = target.with_name(f"{target.stem}.meta{target.suffix}")
    args = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(target),
        "-map",
        "0",
        "-c",
        "copy",
    ]
    for key, value in metadata.items():
        args += ["-metadata", f"{key}={value}"]
    if target.suffix.lower() in _FASTSTART_SUFFIXES:
        args += ["-movflags", "+faststart"]
    args.append(str(temp))

    try:
        completed = subprocess.run(args, capture_output=True, text=True, check=False)
    except OSError as exc:  # pragma: no cover - 缺少可执行文件
        logger.debug("写入元数据失败：%s", exc)
        return False

    if completed.returncode != 0 or not temp.exists() or temp.stat().st_size == 0:
        logger.debug("写入元数据失败：%s", completed.stderr.strip()[-300:])
        temp.unlink(missing_ok=True)
        return False

    try:
        temp.replace(target)
    except OSError as exc:  # pragma: no cover
        logger.debug("元数据文件替换失败：%s", exc)
        temp.unlink(missing_ok=True)
        return False
    logger.debug("已写入元数据：%s", ", ".join(metadata))
    return True


def copy_file(source: str | Path, output: str | Path) -> None:
    """把临时文件移动到最终路径（跨分区时退化为复制+删除）。"""
    source, output = Path(source), Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    try:
        source.replace(output)
    except OSError:
        shutil.copy2(source, output)
        source.unlink(missing_ok=True)
