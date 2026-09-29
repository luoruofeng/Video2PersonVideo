"""流水线接入测试：多人物时主角是否真的跟着"正在说话的人"走。"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence

import numpy as np
import pytest

from video2personvideo.config import AppConfig
from video2personvideo.core.audio import SpeechEnvelope
from video2personvideo.core.mouth import (
    MouthActivityAnalyzer,
    fallback_head_box,
    mouth_roi,
    to_gray,
)
from video2personvideo.core.pipeline import CropPipeline
from video2personvideo.core.ratio import parse_ratio
from video2personvideo.core.smoothing import BoxSmoother
from video2personvideo.core.speaker import ActiveSpeakerSelector, SpeakerParams

RATIO = parse_ratio("9:16").with_target(180, 320)
FRAME_SIZE = (320, 240)
FRAMES = 30

#: 左边的人（略小），右边的人（更大更"抢镜"，默认规则会选中他）
LEFT = (20.0, 60.0, 110.0, 220.0)
RIGHT = (200.0, 40.0, 300.0, 230.0)

PARAMS = SpeakerParams(window_frames=20, min_samples=6, switch_hold=3, min_mouth_motion=0.004)


def _envelope(count: int = FRAMES, block: int = 5) -> SpeechEnvelope:
    energy = tuple(1.0 if (index // block) % 2 == 0 else 0.0 for index in range(count))
    return SpeechEnvelope(
        fps=10.0, energy=energy, active=tuple(value >= 0.5 for value in energy)
    )


ENVELOPE = _envelope()


def _detector(fake_detector):
    """两个固定人物框的假检测器（左小右大）。"""

    def provider(frame: np.ndarray, index: int):
        return [LEFT, RIGHT]

    return fake_detector(provider)


def _build(detector, *, speaker=None, mouth=None, speaker_weight: float = 2.0) -> CropPipeline:
    return CropPipeline(
        RATIO,
        detector,
        smoother=BoxSmoother(RATIO, alpha=0.4),
        speaker=speaker,
        mouth=mouth,
        speaker_weight=speaker_weight,
    )


def _run(pipeline: CropPipeline, frames: Sequence[np.ndarray]):
    outcomes = []
    for index, frame in enumerate(frames):
        outcomes.extend(pipeline.process(frame, index))
    outcomes.extend(pipeline.flush())
    return outcomes


def _subject_box(outcomes) -> tuple[float, float, float, float] | None:
    return outcomes[-1].subject_bbox


# ------------------------------------------------------------------ 基线行为
def test_without_speaker_tracking_picks_the_bigger_person(fake_detector) -> None:
    """没开说话人跟随时，按面积 / 居中打分：右边更大的人胜出。"""
    frames = [np.full((240, 320, 3), 40, dtype=np.uint8) for _ in range(FRAMES)]
    outcomes = _run(_build(_detector(fake_detector)), frames)

    assert _subject_box(outcomes) == RIGHT
    assert outcomes[-1].mode in {"halfbody", "fullbody", "closeup"}


def test_speaker_tracking_follows_the_moving_mouth(fake_detector) -> None:
    """左边的人在说话（嘴在动）时，即使他更小，镜头也应该跟左边。"""
    frames = [np.full((240, 320, 3), 40, dtype=np.uint8) for _ in range(FRAMES)]
    mouth_roi_box = mouth_roi(fallback_head_box(LEFT, FRAME_SIZE))
    for index, frame in enumerate(frames):
        if ENVELOPE.is_speech(index):
            # 说话的每一帧嘴部内容都在变（帧间差异大）；静音帧保持不动
            value = 240 if index % 2 else 90
            x, y, w, h = mouth_roi_box
            frame[y : y + h, x : x + w] = value

    pipeline = _build(
        _detector(fake_detector),
        speaker=ActiveSpeakerSelector(ENVELOPE, PARAMS),
        mouth=MouthActivityAnalyzer(face_locator=None),
        speaker_weight=2.0,
    )
    outcomes = _run(pipeline, frames)

    assert pipeline.stats.speaker_frames > 0
    assert _subject_box(outcomes) == LEFT


def test_speaker_outranks_bigger_more_centered_incumbent(fake_detector) -> None:
    """说话人必须真的成为主角，哪怕对方更大、更居中、而且是上一帧的主角。

    回归用例：偏置曾经直接叠加在**未归一化**的主角打分上（常规得分可以到 2.4 以上，
    而默认偏置只有 1.8），于是"正在说话的人"经常压不过更抢镜的在位者，
    镜头就一直不对准说话人 —— 判定环节其实是对的，坏在融合环节的量纲。
    """
    big = (120.0, 20.0, 280.0, 235.0)   # 更抢镜：常规规则下的主角
    talker = (5.0, 80.0, 60.0, 220.0)   # 说话人：更小、贴画面左边
    frames_count = 60
    envelope = _envelope(frames_count, block=10)

    def provider(frame: np.ndarray, index: int):
        return [big, talker]

    mouth_box = mouth_roi(fallback_head_box(talker, FRAME_SIZE))
    frames = []
    for index in range(frames_count):
        frame = np.full((240, 320, 3), 40, dtype=np.uint8)
        if envelope.is_speech(index):  # 只有说话的人嘴在动
            x, y, w, h = mouth_box
            frame[y : y + h, x : x + w] = 240 if index % 2 else 90
        frames.append(frame)

    pipeline = _build(
        fake_detector(provider),
        speaker=ActiveSpeakerSelector(envelope, PARAMS),
        mouth=MouthActivityAnalyzer(face_locator=None),
        speaker_weight=1.8,  # 配置里的默认值
    )
    outcomes = _run(pipeline, frames)

    assert pipeline.stats.speaker_frames > 0
    assert _subject_box(outcomes) == talker


def test_single_person_frames_still_advance_mouth_baseline(fake_detector) -> None:
    """画面里不足两个主要人物时，也要把嘴动分析器的"上一帧"基准推进到本帧。

    回归用例：基准曾经只在"≥2 个主要人物"的帧上更新，于是人物数量变化时，
    算出的帧间差分是跨了好几帧的巨大差值（噪声），把判定带偏。
    """

    def provider(frame: np.ndarray, index: int):
        return [LEFT, RIGHT] if index != 1 else [LEFT]

    frames = [np.full((240, 320, 3), 40 + index * 10, dtype=np.uint8) for index in range(3)]
    pipeline = _build(
        fake_detector(provider),
        speaker=ActiveSpeakerSelector(ENVELOPE, PARAMS),
        mouth=MouthActivityAnalyzer(face_locator=None),
        speaker_weight=2.0,
    )

    pipeline.process(frames[0], 0)
    pipeline.process(frames[1], 1)  # 只有一个人：不判定说话人，但基准要跟上

    assert pipeline.mouth is not None
    assert pipeline.mouth._previous_gray is not None
    assert np.array_equal(pipeline.mouth._previous_gray, to_gray(frames[1]))


def test_speaker_bonus_ignores_single_person(fake_detector) -> None:
    """画面里只有一个主要人物时不必判定，也不会因此改变行为。"""

    def provider(frame: np.ndarray, index: int):
        return [LEFT]

    frames = [np.full((240, 320, 3), 40, dtype=np.uint8) for _ in range(FRAMES)]
    pipeline = _build(
        fake_detector(provider),
        speaker=ActiveSpeakerSelector(ENVELOPE, PARAMS),
        mouth=MouthActivityAnalyzer(face_locator=None),
        speaker_weight=2.0,
    )
    outcomes = _run(pipeline, frames)

    assert pipeline.stats.speaker_frames == 0
    assert _subject_box(outcomes) == LEFT


def test_pipeline_without_speaker_behaves_as_before(fake_detector) -> None:
    """没挂判定器时，嘴动分析器不会被创建，也不会碰像素。"""
    pipeline = _build(_detector(fake_detector))

    assert pipeline.speaker is None
    assert pipeline.mouth is None


# ------------------------------------------------------- 处理器接线（配置 → 流水线）
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="需要 ffmpeg 才能生成/读取音轨")
def test_processor_enables_speaker_tracking_with_audio(
    tmp_path, fake_detector, moving_person
) -> None:
    """有音轨的真实文件：处理器应把说话人判定挂到流水线上，并在汇总里说明。"""
    from video2personvideo.core.processor import process_video

    ffmpeg = shutil.which("ffmpeg")
    source = tmp_path / "talk.mp4"
    completed = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=160x120:rate=10:duration=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2",
            "-af",
            "volume='if(lt(t,1),0,1)':eval=frame",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(source),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0 or not source.exists():  # pragma: no cover - 取决于环境
        pytest.skip(f"无法生成测试视频：{completed.stderr.strip()[-200:]}")

    cfg = AppConfig(
        source=source,
        output=tmp_path / "talk_person.mp4",
        save_audio=False,
        aspect_ratio="9:16",
        speaker_tracking=True,
    )

    result = process_video(cfg, show_progress=False, detector=fake_detector(moving_person))

    assert result.frames_processed == 20
    assert result.speaker_tracking is True
    assert result.speaker_note.startswith("已启用")
    assert "说话人跟随" in result.summary()


def test_processor_skips_speaker_tracking_without_audio(tmp_path, make_video, fake_detector, moving_person) -> None:
    """无音轨的视频：安静降级，处理结果与以前一致，并在汇总里说明原因。"""
    from video2personvideo.core.processor import process_video

    source = make_video("silent.avi", size=(160, 120), frames=12)
    cfg = AppConfig(
        source=source,
        output=tmp_path / "silent_person.avi",
        save_audio=False,
        aspect_ratio="9:16",
        speaker_tracking=True,
    )

    result = process_video(cfg, show_progress=False, detector=fake_detector(moving_person))

    assert result.frames_processed == 12
    assert result.speaker_tracking is False
    assert result.speaker_note  # 无论是否生效都要给出原因
    assert "说话人跟随" in result.summary()


def test_processor_respects_disabled_speaker_tracking(tmp_path, make_video, fake_detector, moving_person) -> None:
    from video2personvideo.core.processor import process_video

    source = make_video("off.avi", size=(160, 120), frames=12)
    cfg = AppConfig(
        source=source,
        output=tmp_path / "off_person.avi",
        save_audio=False,
        aspect_ratio="9:16",
        speaker_tracking=False,
    )

    result = process_video(cfg, show_progress=False, detector=fake_detector(moving_person))

    assert result.speaker_tracking is False
    assert result.speaker_note == "未启用"


# ------------------------------------------------------------------ 配置联动
def test_config_builds_speaker_params() -> None:
    cfg = AppConfig(speaker_window_frames=30, speaker_switch_hold=5, speaker_switch_margin=0.3)
    params = cfg.speaker_params()

    assert params.window_frames == 30
    assert params.switch_hold == 5
    assert params.switch_margin == pytest.approx(0.3)


def test_config_validation_rejects_bad_speaker_params() -> None:
    with pytest.raises(ValueError):
        AppConfig(speaker_weight=-1.0).validate()
    with pytest.raises(ValueError):
        AppConfig(speaker_switch_hold=0).validate()
    with pytest.raises(ValueError):
        AppConfig(speaker_window_frames=1).validate()
