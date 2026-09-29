"""视频处理流水线：逐帧检测 → 选主角 → 构图裁剪 → 写视频 → 补回音轨。"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import cv2
from tqdm import tqdm

from ..config import AppConfig
from ..utils import ffmpeg_tools
from ..utils.logger import get_logger
from . import audio
from .detector import PersonDetector
from .framing import MODE_ANNOTATE, CropBox, mode_label
from .mouth import FaceLocator, MouthActivityAnalyzer
from .pipeline import CropPipeline, FrameOutcome
from .smoothing import BoxSmoother
from .speaker import ActiveSpeakerSelector
from .video_io import VideoReader, VideoWriter, save_frame_image

logger = get_logger(__name__)

#: 截图功能最多保存的张数，避免上万张图片塞满磁盘
MAX_SNAPSHOTS = 300

#: 实时预览窗口标题
PREVIEW_WINDOW = "Video2PersonVideo - 实时预览"


@dataclass(slots=True)
class ProgressInfo:
    """一次进度上报：既有总体百分比，也有帧级 / 模式 / ETA 细节。"""

    fraction: float
    message: str
    frame_index: int = 0
    frame_total: int = 0
    mode: str = ""
    elapsed: float = 0.0
    eta: float = 0.0
    fps: float = 0.0
    phase: str = ""

    @property
    def mode_label(self) -> str:
        return mode_label(self.mode) if self.mode else ""


#: 进度回调：接收 :class:`ProgressInfo`
ProgressCallback = Callable[[ProgressInfo], None]
#: 停止回调：返回 True 表示请求中止处理
StopCallback = Callable[[], bool]


@dataclass(slots=True)
class ProcessResult:
    """一次处理任务的统计结果。"""

    source: Path
    output: Path
    frames_processed: int
    frames_with_person: int
    person_count_total: int
    person_count_peak: int
    audio_muxed: bool
    reencoded: bool
    elapsed: float
    snapshots: int = 0
    stopped_early: bool = False
    skipped: bool = False
    skip_reason: str = ""
    output_size: tuple[int, int] = (0, 0)
    aspect_ratio: str = ""
    crop_mode: bool = False
    hold_frames: int = 0
    center_frames: int = 0
    mode_counts: dict[str, int] = field(default_factory=dict)
    #: 原视频 / 输出视频的体积（字节），取不到时分别为 0
    source_bytes: int = 0
    output_bytes: int = 0
    #: 是否因输出体积超标触发过自动压缩
    size_guard: bool = False
    #: 自动压缩后是否真的压到了「原视频 × max_size_ratio」以内
    size_guard_reached: bool = False
    #: 自动压缩实际使用的视频码率（kbps），未压缩时为 0
    size_guard_kbps: int = 0
    #: 自动压缩前的输出体积（字节），未压缩时为 0
    raw_output_bytes: int = 0
    #: 源视频是否为可变帧率（VFR）或时间轴不自洽
    source_variable_fps: bool = False
    #: 是否已把源视频时间轴归一化为恒定帧率后再处理
    timeline_normalized: bool = False
    #: 源视频时间轴判定说明（探测失败时为空串）
    timeline_note: str = ""
    #: "跟随正在说话的人"是否真的生效（未启用 / 无音轨 / 未检测到语音时为 False）
    speaker_tracking: bool = False
    #: 说话人跟随的状态说明（无论是否生效都会给出原因）
    speaker_note: str = ""
    #: 判定出说话人并据此选主角的关键帧数
    speaker_frames: int = 0
    #: 说话人切换次数
    speaker_switches: int = 0
    #: 多人分屏的状态说明（未启用 / 单人画面 / 已启用，无论是否生效都会给出原因）
    multi_note: str = ""
    #: 真正以多人分屏输出的帧数
    multi_frames: int = 0
    #: 单帧最多同时显示了几个窗口
    windows_peak: int = 0
    #: 各布局各用了多少帧，如 ``{"portrait·3 人（2×2，3 窗）": 120}``
    multi_layouts: dict[str, int] = field(default_factory=dict)

    @property
    def speed_fps(self) -> float:
        return self.frames_processed / self.elapsed if self.elapsed > 0 else 0.0

    @property
    def size_ratio(self) -> float:
        """输出体积 / 原视频体积；缺任一数据时返回 0。"""
        if self.source_bytes <= 0 or self.output_bytes <= 0:
            return 0.0
        return self.output_bytes / self.source_bytes

    def summary(self) -> str:
        if self.skipped:
            return "\n".join(
                [
                    f"输入视频     : {self.source}",
                    f"输出视频     : {self.output}",
                    f"状态         : 已跳过（{self.skip_reason}）",
                ]
            )

        lines = [
            f"输入视频     : {self.source}",
            f"输出视频     : {self.output}",
        ]
        if self.output_size[0]:
            mode = "裁剪构图" if self.crop_mode else "逐帧画框标注"
            lines.append(
                f"输出规格     : {self.output_size[0]}×{self.output_size[1]}"
                f"（{self.aspect_ratio}，{mode}）"
            )
        lines += [
            f"处理帧数     : {self.frames_processed} 帧（耗时 {self.elapsed:.1f}s，约 {self.speed_fps:.1f} FPS）",
            f"含人帧数     : {self.frames_with_person}",
            f"累计检出人次 : {self.person_count_total}（单帧最多 {self.person_count_peak} 人）",
            f"保留原音轨   : {'是' if self.audio_muxed else '否'}",
        ]
        if self.source_variable_fps:
            detail = f"（{self.timeline_note}）" if self.timeline_note else ""
            if self.timeline_normalized:
                lines.append(f"时间轴处理   : 源为可变帧率，已归一化为恒定帧率{detail}")
            else:
                lines.append(f"时间轴处理   : 源为可变帧率，本轮未能归一化{detail}")
        if self.output_bytes:
            volume = format_bytes(self.output_bytes)
            if self.source_bytes:
                volume += (
                    f"（原视频 {format_bytes(self.source_bytes)}，{self.size_ratio:.2f}×）"
                )
            lines.append(f"输出体积     : {volume}")
        if self.size_guard:
            before = f"{format_bytes(self.raw_output_bytes)} → " if self.raw_output_bytes else ""
            if self.size_guard_reached:
                lines.append(
                    "体积守护     : 已自动压缩到原视频以内"
                    f"（{before}{format_bytes(self.output_bytes)}，"
                    f"视频码率 {self.size_guard_kbps} kbps）"
                )
            else:
                lines.append(
                    f"体积守护     : 已自动压缩但仍大于原视频（{before}"
                    f"{format_bytes(self.output_bytes)}，已到最低码率无法再小）"
                )
        if self.crop_mode and (self.hold_frames or self.center_frames):
            lines.append(
                f"兜底帧数     : 保持上一帧 {self.hold_frames} 帧 / 回中 {self.center_frames} 帧"
            )
        if self.crop_mode and self.speaker_note:
            detail = ""
            if self.speaker_tracking:
                detail = f"（判定 {self.speaker_frames} 帧，切换说话人 {self.speaker_switches} 次）"
            lines.append(f"说话人跟随   : {self.speaker_note}{detail}")
        if self.crop_mode and self.multi_note:
            detail = ""
            if self.multi_frames:
                detail = f"（分屏 {self.multi_frames} 帧，单帧最多 {self.windows_peak} 个窗口）"
            lines.append(f"多人分屏     : {self.multi_note}{detail}")
            if self.multi_layouts:
                used = "、".join(
                    f"{name} {count} 帧"
                    for name, count in sorted(
                        self.multi_layouts.items(), key=lambda item: -item[1]
                    )
                )
                lines.append(f"分屏布局     : {used}")
        if self.mode_counts:
            detail = "、".join(
                f"{mode_label(name)} {count} 帧" for name, count in self.mode_counts.items()
            )
            lines.append(f"构图模式     : {detail}")
        if self.reencoded:
            lines.append("视频编码     : 已重编码为 H.264")
        if self.snapshots:
            lines.append(f"保存截图     : {self.snapshots} 张")
        if self.stopped_early:
            lines.append("提示         : 处理被提前终止，输出视频为已处理部分的截断结果")
        return "\n".join(lines)


def process_video(
    cfg: AppConfig,
    *,
    progress_cb: ProgressCallback | None = None,
    stop_cb: StopCallback | None = None,
    show_progress: bool = True,
    detector: object | None = None,
) -> ProcessResult:
    """跑完一次完整流水线。

    :param cfg: 处理参数（见 :class:`~video2personvideo.config.AppConfig`）
    :param progress_cb: 进度回调，GUI 用它刷新进度条
    :param stop_cb: 停止回调，返回 ``True`` 时提前结束
    :param show_progress: 是否在控制台显示 tqdm 进度条
    :param detector: 可注入的检测器（测试用；``None`` 时自行构造 :class:`PersonDetector`）
    """
    cfg.validate()
    if cfg.source is None:
        raise ValueError("未指定输入视频（-i/--source）")

    source = Path(cfg.source)
    if not source.exists():
        raise FileNotFoundError(f"输入视频不存在：{source}")

    output = Path(cfg.output) if cfg.output else cfg.output_path_for(source)
    if not output.suffix:
        output = output.with_suffix(".mp4")

    started = time.perf_counter()

    if output.exists() and not cfg.overwrite:
        reason = "输出文件已存在（overwrite=false）"
        logger.warning("跳过处理：%s -> %s", reason, output)
        return ProcessResult(
            source=source,
            output=output,
            frames_processed=0,
            frames_with_person=0,
            person_count_total=0,
            person_count_peak=0,
            audio_muxed=False,
            reencoded=False,
            elapsed=0.0,
            skipped=True,
            skip_reason=reason,
            aspect_ratio=cfg.aspect_ratio if cfg.crop else "原始比例",
            crop_mode=bool(cfg.crop),
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    temp_video = output.with_name(f"{output.stem}.noaudio{output.suffix}")
    final_video = output.with_name(f"{output.stem}.encoded{output.suffix}")

    if detector is None:
        detector = PersonDetector(
            model=cfg.model,
            conf=cfg.conf,
            iou=cfg.iou,
            imgsz=cfg.imgsz,
            classes=cfg.classes or None,
            max_det=cfg.max_det,
            device=cfg.device,
            use_keypoints=cfg.use_keypoints,
        )

    ratio = cfg.resolve_ratio() if cfg.crop else None
    pipeline: CropPipeline | None = None
    if ratio is not None:
        multi_policy = cfg.multi_person_policy()
        logger.info("多人分屏：%s", multi_policy.describe())
        pipeline = CropPipeline(
            ratio,
            detector,
            params=cfg.framing_params(),
            smoother=BoxSmoother(ratio, params=cfg.smoothing_params()),
            weights=cfg.subject_weights(),
            min_person_height_ratio=cfg.min_person_height_ratio,
            detect_interval=cfg.detect_interval,
            infer_batch=cfg.infer_batch,
            crop=True,
            annotate=cfg.annotate,
            multi=multi_policy,
        )

    # 先探测源视频的音频 / 视频信息：可变帧率（VFR）时把画面归一到恒定帧率，
    # 否则 OpenCV 逐帧按单一 fps 写回会把画面"均匀摊平"（开头慢放、后面快进）。
    work_source, timing, normalized_path = _prepare_source_timeline(
        cfg, source, output, progress_cb=progress_cb
    )

    processed = 0
    frames_with_person = 0
    person_total = 0
    person_peak = 0
    snapshots = 0
    stopped_early = False
    preview_saved = False
    last_frame = None

    with VideoReader(work_source) as reader:
        meta = reader.meta
        logger.info("开始处理：%s", meta.describe())
        _notify(progress_cb, ProgressInfo(0.0, "正在加载模型…", phase="init"))

        # 镜头跟随的"每秒感受"按源视频真实帧率校准（构造流水线时还不知道帧率）：
        # 60fps 素材不再比 30fps 快一倍，保持时长也不会因为帧率不同而变长变短。
        if pipeline is not None:
            pipeline.set_fps(meta.fps)

        out_size = ratio.target_size if ratio is not None else meta.size
        total_frames = meta.frame_count or 0
        writer = VideoWriter(temp_video, fps=meta.fps, size=out_size)

        # "跟随正在说话的人"需要知道帧率与总帧数（音频包络按帧对齐），故放在这里挂载
        speaker_note = _attach_speaker_tracking(cfg, pipeline, source, meta, progress_cb)

        def emit(outcome: FrameOutcome) -> None:
            nonlocal processed, frames_with_person, person_total, person_peak
            nonlocal snapshots, preview_saved, last_frame

            frame = outcome.frame
            if (
                cfg.annotate
                and not outcome.is_multi  # 多人分屏的框已经在拼接时画进各自的窗口了
                and outcome.subject_bbox is not None
                and outcome.box is not None
            ):
                frame = _draw_subject(frame, outcome.subject_bbox, outcome.box)
            writer.write(frame)

            processed += 1
            if outcome.has_person:
                frames_with_person += 1
            person_total += outcome.person_count
            person_peak = max(person_peak, outcome.person_count)

            if cfg.save_frames_dir and outcome.has_person and snapshots < MAX_SNAPSHOTS:
                save_frame_image(frame, cfg.save_frames_dir, outcome.index)
                snapshots += 1

            if cfg.write_preview and not preview_saved:
                _save_preview(frame, output)
                preview_saved = True

            last_frame = frame

            done = outcome.index + 1
            elapsed = time.perf_counter() - started
            fraction = min(done / total_frames, 1.0) if total_frames else 1.0
            eta = (elapsed / fraction - elapsed) if 0.0 < fraction < 1.0 else 0.0
            _notify(
                progress_cb,
                ProgressInfo(
                    fraction=fraction,
                    message=(
                        f"已处理 {done}/{total_frames or '?'} 帧 · {outcome.mode_label}"
                        f" · 本帧 {outcome.person_count} 人"
                    ),
                    frame_index=done,
                    frame_total=total_frames,
                    mode=outcome.mode,
                    elapsed=elapsed,
                    eta=eta,
                    fps=(processed / elapsed if elapsed > 0 else 0.0),
                ),
            )

        try:
            bar = tqdm(
                total=total_frames or None,
                desc="人像裁剪" if pipeline is not None else "人像检测",
                unit="帧",
                disable=not show_progress,
                dynamic_ncols=True,
            )
            with bar:
                for index, frame in reader.frames():
                    if stop_cb is not None and stop_cb():
                        stopped_early = True
                        logger.warning("收到停止请求，已处理 %d 帧", processed)
                        break

                    if pipeline is not None:
                        outcomes = pipeline.process(frame, index)
                    else:
                        annotated, count = detector.detect(frame)
                        outcomes = [
                            FrameOutcome(index, annotated, count, count > 0, MODE_ANNOTATE, None)
                        ]

                    for outcome in outcomes:
                        emit(outcome)
                        bar.update(1)
                    bar.set_postfix_str(
                        f"本帧 {outcomes[-1].person_count if outcomes else 0} 人"
                        f" · 峰值 {person_peak}",
                        refresh=False,
                    )

                    if cfg.show and last_frame is not None:
                        cv2.imshow(PREVIEW_WINDOW, last_frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            stopped_early = True
                            logger.warning("预览窗口收到 q 键，提前结束")
                            break

                if pipeline is not None and not stopped_early:
                    for outcome in pipeline.flush():
                        emit(outcome)
                        bar.update(1)
        finally:
            writer.close()
            if cfg.show:
                cv2.destroyAllWindows()

    _notify(progress_cb, ProgressInfo(1.0, "正在合成音轨…", phase="finalize"))
    audio_muxed, reencoded = _finalize(
        cfg, source, temp_video, final_video, output, progress_cb=progress_cb
    )
    # 只清理中间产物，绝不能把最终输出一起删掉
    _cleanup(temp_video, final_video, normalized_path)

    # 体积守护：输出比原视频还大是 OpenCV 的 mp4v 编码器造成的，
    # 这里按用户设定自动压回去（无 ffmpeg 时只告警，绝不打断流程）。
    guard = _enforce_size_guard(cfg, source, output, progress_cb=progress_cb)

    elapsed = time.perf_counter() - started
    if cfg.write_metadata and output.exists():
        _write_metadata(cfg, ratio, out_size, output)

    multi_frames = pipeline.stats.multi_frames if pipeline is not None else 0
    multi_note = _describe_multi(pipeline, multi_frames)

    result = ProcessResult(
        source=source,
        output=output,
        frames_processed=processed,
        frames_with_person=frames_with_person,
        person_count_total=person_total,
        person_count_peak=person_peak,
        audio_muxed=audio_muxed,
        reencoded=reencoded,
        elapsed=elapsed,
        snapshots=snapshots,
        stopped_early=stopped_early,
        output_size=(int(out_size[0]), int(out_size[1])),
        aspect_ratio=ratio.name if ratio is not None else "原始比例",
        crop_mode=pipeline is not None,
        hold_frames=pipeline.hold_frames if pipeline is not None else 0,
        center_frames=pipeline.center_frames if pipeline is not None else 0,
        mode_counts=pipeline.stats.as_dict() if pipeline is not None else {},
        source_bytes=_file_size(source),
        output_bytes=_file_size(output),
        size_guard=guard.applied,
        size_guard_reached=guard.reached,
        size_guard_kbps=guard.kbps,
        raw_output_bytes=guard.raw_bytes,
        source_variable_fps=bool(timing is not None and timing.variable),
        timeline_normalized=normalized_path is not None,
        timeline_note=timing.reason if timing is not None else "",
        speaker_tracking=bool(pipeline is not None and pipeline.speaker is not None),
        speaker_note=speaker_note,
        speaker_frames=pipeline.stats.speaker_frames if pipeline is not None else 0,
        speaker_switches=(
            pipeline.speaker.switches if pipeline is not None and pipeline.speaker else 0
        ),
        multi_note=multi_note,
        multi_frames=multi_frames,
        windows_peak=pipeline.stats.windows_peak if pipeline is not None else 0,
        multi_layouts=(
            dict(pipeline.composer.layout_counts)
            if pipeline is not None and pipeline.composer is not None
            else {}
        ),
    )
    logger.info("处理完成：%s", output)
    logger.debug("\n%s", result.summary())
    return result


def _attach_speaker_tracking(
    cfg: AppConfig,
    pipeline: CropPipeline | None,
    source: Path,
    meta,
    progress_cb: ProgressCallback | None = None,
) -> str:
    """给流水线挂上"跟随正在说话的人"（可选增强），返回状态说明。

    YOLO 本身判断不了"谁在说话"（纯图像模型、关键点里也没有嘴部），
    所以这里补上音频这一半：用 ffmpeg 抽音轨算出**逐帧语音能量包络**，
    处理时再与每个人的嘴部运动做相关（``core/speaker.py``）。

    任何一步不可用（关闭开关、没有 ffmpeg、源视频没有音轨、检测不到语音）
    都**安静地退回**原来的主角打分规则，绝不因为这项增强而中断处理。
    """
    if pipeline is None:
        return "未启用（画框标注模式）"
    if not cfg.speaker_tracking:
        return "未启用"

    if not ffmpeg_tools.ffmpeg_available():
        logger.info("未找到 ffmpeg，无法分析音轨：「跟随正在说话的人」已跳过")
        return "未启用（缺少 ffmpeg）"

    frame_count = int(meta.frame_count) if meta.frame_count else _estimate_frame_count(source, meta)
    if frame_count <= 0:
        logger.info("无法确定视频帧数，「跟随正在说话的人」已跳过")
        return "未启用（无法确定帧数）"

    envelope = audio.extract_envelope(source, fps=meta.fps, frame_count=frame_count)
    if envelope is None:
        logger.info("源视频没有可用音轨（或解码失败）：「跟随正在说话的人」已跳过")
        return "未启用（源无可用音轨）"
    if envelope.active_frames == 0:
        logger.info("音轨里没有检测到语音：「跟随正在说话的人」已跳过")
        return "未启用（未检测到语音）"

    pipeline.speaker = ActiveSpeakerSelector(envelope=envelope, params=cfg.speaker_params())
    if pipeline.mouth is None:
        pipeline.mouth = MouthActivityAnalyzer(face_locator=FaceLocator())
    pipeline.speaker_weight = float(cfg.speaker_weight)
    logger.info(
        "已启用「跟随正在说话的人」：多人物时优先对准正在说话的人（%s）",
        envelope.describe(),
    )
    _notify(progress_cb, ProgressInfo(0.0, "已启用「跟随正在说话的人」…", phase="init"))
    return "已启用（多人物时优先对准正在说话的人）"


def _describe_multi(pipeline: CropPipeline | None, frames: int) -> str:
    """多人分屏的状态说明（无论是否真的用上，都给出原因）。

    和说话人跟随一样，这项功能只影响"画面里有两个及以上主要人物"的帧：
    单人视频的输出与关闭它时逐帧一致，所以说清楚"为什么没生效"比一句"已启用"更有用。
    """
    if pipeline is None:
        return "未启用（画框标注模式）"
    if pipeline.composer is None:
        return "未启用（配置关闭，整段视频同时只显示一个人）"
    if frames > 0:
        return "已启用（多个主要人物各占一个上半身小窗口）"
    return "已启用（本片未出现多个主要人物，画面与单人模式一致）"


def _estimate_frame_count(source: Path, meta) -> int:
    """OpenCV 报不出总帧数时，用 ffprobe 的时长 × 帧率兜底估算。"""
    duration = ffmpeg_tools.container_duration(source)
    if duration <= 0.0 or meta.fps <= 0:
        return 0
    return int(duration * float(meta.fps))


def _prepare_source_timeline(
    cfg: AppConfig,
    source: Path,
    output: Path,
    *,
    progress_cb: ProgressCallback | None = None,
) -> tuple[Path, ffmpeg_tools.VideoTiming | None, Path | None]:
    """探测源视频的帧率 / 时长信息，必要时先把画面归一到恒定帧率。

    返回 ``(供逐帧读取的视频路径, 时间轴信息, 需要清理的临时文件)``。

    任何一步不可用（开关关闭、没有 ffmpeg、探测失败、转码失败）都原样返回
    ``source``，绝不因为这项优化而中断处理——只是退化回"按原始帧率处理"的旧行为。
    """
    timing = ffmpeg_tools.analyze_video_timing(source)
    if timing is not None:
        logger.info("源视频时间轴：%s", timing.describe())

    if not cfg.normalize_vfr or timing is None or not timing.variable:
        return source, timing, None

    if not ffmpeg_tools.ffmpeg_available():
        logger.warning(
            "源视频为可变帧率（%s），但没有 ffmpeg 可用于归一时间轴；"
            "输出画面可能出现「忽快忽慢」，安装 ffmpeg 后即可自动修复",
            timing.reason,
        )
        return source, timing, None

    if timing.fps <= 0.0:
        logger.warning("无法确定归一化目标帧率，按原始帧率继续")
        return source, timing, None

    normalized = output.with_name(f"{output.stem}.normalized.mp4")
    _notify(
        progress_cb,
        ProgressInfo(
            0.0,
            f"源视频为可变帧率，正在归一时间轴（{timing.fps:.3f} FPS）…",
            phase="normalize",
        ),
    )
    # 归一化产物还要被逐帧解码再编码，画质按"至少 CRF 18"处理，避免二次损失
    if ffmpeg_tools.normalize_to_cfr(
        source, normalized, fps=timing.fps, crf=min(int(cfg.crf), 18)
    ):
        logger.info(
            "已把画面归一化为恒定 %.3f FPS（%s）；音轨仍取自原视频",
            timing.fps,
            timing.reason,
        )
        return normalized, timing, normalized

    logger.warning("时间轴归一化失败，按原始帧率继续（画面可能出现「忽快忽慢」）")
    return source, timing, None


def _draw_subject(
    frame, subject_bbox: tuple[float, float, float, float], box: CropBox
):
    """把源画面坐标系下的检测框映射到裁剪后的输出帧上并画出来。"""
    scale = frame.shape[1] / box.w if box.w else 1.0
    x1 = int(round((subject_bbox[0] - box.x) * scale))
    y1 = int(round((subject_bbox[1] - box.y) * scale))
    x2 = int(round((subject_bbox[2] - box.x) * scale))
    y2 = int(round((subject_bbox[3] - box.y) * scale))
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, "person", (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame


def _save_preview(frame, output: Path) -> None:
    """保存首帧构图预览图 ``<名称>_preview.jpg``。"""
    target = output.with_name(f"{output.stem}_preview.jpg")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(target), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        logger.debug("已保存构图预览图：%s", target)
    except Exception as exc:  # noqa: BLE001 - 预览图失败不应中断处理
        logger.debug("保存预览图失败：%s", exc)


def _write_metadata(cfg: AppConfig, ratio, out_size, output: Path) -> None:
    """给最终输出写入可追溯的元数据标记（缺少 ffmpeg 时静默跳过）。"""
    from .. import __version__

    comment = " | ".join(
        [
            f"Video2PersonVideo {__version__}",
            f"ratio={ratio.name if ratio is not None else 'original'}",
            f"size={int(out_size[0])}x{int(out_size[1])}",
            f"model={cfg.model}",
            f"processed={time.strftime('%Y-%m-%d %H:%M:%S')}",
        ]
    )
    if not ffmpeg_tools.inject_metadata(output, {"comment": comment}):
        logger.debug("未能写入元数据标记（缺少 ffmpeg 或格式不支持）")


def _finalize(
    cfg: AppConfig,
    source: Path,
    temp_video: Path,
    final_video: Path,
    output: Path,
    *,
    progress_cb: ProgressCallback | None = None,
) -> tuple[bool, bool]:
    """把无声临时视频加工成最终输出，返回 (是否保留音轨, 是否重编码)。"""
    reencoded = False
    stage = temp_video

    if cfg.reencode:
        _notify(progress_cb, ProgressInfo(1.0, "正在重编码为 H.264…", phase="finalize"))
        if ffmpeg_tools.transcode_h264(temp_video, final_video, crf=cfg.crf):
            stage = final_video
            reencoded = True
        else:
            logger.warning("重编码未生效，直接使用 OpenCV 原始编码输出")

    if cfg.save_audio:
        # mux_audio 内部会比对画面与音轨的真实时长，发现偏差时按音频时长重定时画面，
        # 避免输出"画面快于声音 / 声音快于画面"。
        if ffmpeg_tools.mux_audio(stage, source, output, crf=cfg.crf):
            return True, reencoded
        logger.info("未保留音轨（缺少 ffmpeg 或原视频无音频）")

    ffmpeg_tools.copy_file(stage, output)
    return False, reencoded


class SizeGuardOutcome(NamedTuple):
    """体积守护的执行结果。

    * ``applied``：是否真的重新压缩过（缺少 ffmpeg / 探测失败时为 ``False``）
    * ``reached``：是否压到了「原视频 × max_size_ratio」以内
    * ``kbps``：实际使用的视频码率（kbps）
    * ``raw_bytes``：压缩前的输出体积（字节），用于如实报告「压小了多少」
    """

    applied: bool = False
    reached: bool = False
    kbps: int = 0
    raw_bytes: int = 0


def _enforce_size_guard(
    cfg: AppConfig,
    source: Path,
    output: Path,
    *,
    progress_cb: ProgressCallback | None = None,
) -> SizeGuardOutcome:
    """体积守护：输出体积超过「原视频 × ``max_size_ratio``」时自动重压缩。

    OpenCV 只能写 ``mp4v``（MPEG-4 Part 2），同一个画面它需要的码率是 H.264 的
    好几倍；再叠加"低分辨率源视频被放大到目标像素"，输出比原视频大几倍很常见。
    这里按原视频体积反推码率，用 H.264 重压回去，压不动则如实告警。

    关闭守护、体积未超标、取不到体积或缺少 ffmpeg 时返回全 False 的
    :class:`SizeGuardOutcome`（``raw_bytes`` 只在真的压过时才有值）。
    """
    if not cfg.size_guard or not output.exists():
        return SizeGuardOutcome()

    source_bytes = _file_size(source)
    if source_bytes <= 0:
        return SizeGuardOutcome()

    limit = int(source_bytes * float(cfg.max_size_ratio))
    output_bytes = _file_size(output)
    if output_bytes <= limit:
        return SizeGuardOutcome()

    logger.info(
        "输出体积 %s 超过原视频的 %.2f 倍（上限 %s），启动体积守护",
        format_bytes(output_bytes),
        float(cfg.max_size_ratio),
        format_bytes(limit),
    )
    if not ffmpeg_tools.ffmpeg_available():
        logger.warning(
            "未找到 ffmpeg，无法自动压缩输出体积（可安装 ffmpeg，或用 --no-size-guard 关掉守护）"
        )
        return SizeGuardOutcome()

    _notify(progress_cb, ProgressInfo(1.0, "输出体积超标，正在自动压缩…", phase="finalize"))
    reached, kbps = ffmpeg_tools.shrink_to_limit(
        output,
        output,
        limit,
        min_kbps=int(cfg.min_video_bitrate_kbps),
    )
    if kbps <= 0:  # 连一次有效编码都没跑起来，保持 output 原样
        return SizeGuardOutcome()

    if reached:
        logger.info(
            "体积守护完成：%s → %s（视频码率 %d kbps）",
            format_bytes(output_bytes),
            format_bytes(_file_size(output)),
            kbps,
        )
    else:
        logger.warning(
            "体积守护已尽力压缩，仍为 %s（最低码率 %d kbps 限制）",
            format_bytes(_file_size(output)),
            int(cfg.min_video_bitrate_kbps),
        )
    return SizeGuardOutcome(True, reached, kbps, output_bytes)


def _file_size(path: Path) -> int:
    """文件字节数；取不到返回 0。"""
    try:
        return path.stat().st_size
    except OSError:
        return 0


def format_bytes(size: int) -> str:
    """把字节数格式化成 ``12.3 MB`` 这样的可读文本（跨模块复用，故为公开函数）。"""
    value = float(max(int(size), 0))
    unit = "B"
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0 or unit == "GB":
            break
        value /= 1024.0
    return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"


def _cleanup(*paths: Path | None) -> None:
    """删除中间产物（``None`` 会被跳过）。"""
    for path in paths:
        if path is None:
            continue
        try:
            if path.exists():
                path.unlink()
        except OSError as exc:  # pragma: no cover
            logger.debug("清理临时文件失败 %s：%s", path, exc)


def _notify(callback: ProgressCallback | None, info: ProgressInfo) -> None:
    if callback is None:
        return
    try:
        callback(info)
    except Exception as exc:  # noqa: BLE001 - 回调异常不应中断处理
        logger.debug("进度回调异常：%s", exc)
