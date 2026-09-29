"""音频侧测试：逐帧能量包络、语音活动判定、ffmpeg 抽音轨的降级行为。"""

from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest

from video2personvideo.core import audio
from video2personvideo.core.audio import (
    SpeechEnvelope,
    VadParams,
    frame_energy,
    normalize_energy,
    speech_mask,
)

SAMPLE_RATE = 16000


def _tone(seconds: float, *, amplitude: float = 0.5, frequency: float = 220.0) -> np.ndarray:
    time = np.arange(int(seconds * SAMPLE_RATE), dtype=np.float32) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * frequency * time)).astype(np.float32)


# --------------------------------------------------------------- 逐帧能量
def test_frame_energy_splits_audio_by_frame() -> None:
    energy = frame_energy(_tone(1.0), SAMPLE_RATE, fps=10.0, frame_count=10)

    assert energy.shape == (10,)
    assert np.all(energy > 0.0)
    # 纯正弦各帧能量应当基本一致
    assert float(energy.std()) < 0.05 * float(energy.mean())


def test_frame_energy_matches_manual_rms() -> None:
    samples = _tone(0.5, amplitude=0.25)
    energy = frame_energy(samples, SAMPLE_RATE, fps=2.0, frame_count=1)

    expected = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    assert energy[0] == pytest.approx(expected, rel=1e-3)


def test_frame_energy_is_zero_for_missing_tail() -> None:
    """音频比画面短时，多出来的帧按静音处理（0），不会抛错。"""
    energy = frame_energy(_tone(0.25), SAMPLE_RATE, fps=10.0, frame_count=10)

    assert energy.shape == (10,)
    assert np.all(energy[:2] > 0.0)  # 0~0.25s 覆盖前两帧
    assert np.all(energy[3:] == 0.0)


def test_frame_energy_handles_degenerate_inputs() -> None:
    assert frame_energy([], SAMPLE_RATE, fps=10.0, frame_count=3).tolist() == [0.0, 0.0, 0.0]
    assert frame_energy(_tone(0.1), SAMPLE_RATE, fps=0.0, frame_count=3).size == 3
    assert frame_energy(_tone(0.1), SAMPLE_RATE, fps=10.0, frame_count=0).size == 0


# --------------------------------------------------------------- 归一化 / VAD
def test_normalize_energy_uses_percentile() -> None:
    """单个爆音不该把其余帧压扁（用 95 分位而不是最大值做参考）。"""
    values = normalize_energy([1.0] * 20 + [100.0])

    assert float(values[20]) == pytest.approx(1.0)  # 爆音被截断
    assert all(float(value) == pytest.approx(1.0) for value in values[:20])


def test_normalize_energy_all_zero() -> None:
    assert normalize_energy([0.0, 0.0]).tolist() == [0.0, 0.0]


def test_speech_mask_marks_loud_half() -> None:
    quiet = np.zeros(50, dtype=np.float32)
    loud = np.full(50, 0.8, dtype=np.float32)
    mask = speech_mask(np.concatenate([quiet, loud]))

    assert not mask[:50].any()
    assert mask[50:].all()


def test_speech_mask_rejects_flat_noise() -> None:
    """全是底噪（无起伏）时不应判成语音。"""
    mask = speech_mask(np.full(30, 0.5, dtype=np.float32))

    assert not mask.any()


def test_speech_mask_ignores_isolated_spike() -> None:
    """几帧爆音不会把周围的底噪一起判成语音。"""
    values = np.full(100, 0.05, dtype=np.float32)
    values[10:13] = 1.0
    mask = speech_mask(values)

    assert mask.sum() == 3
    assert mask[10:13].all()


def test_speech_mask_handles_talking_almost_all_the_time() -> None:
    """全片几乎都是说话（停顿零散地穿插其中）时仍要判得出语音。"""
    values = np.full(200, 0.9, dtype=np.float32)
    for start in range(0, 200, 20):
        values[start : start + 3] = 0.02  # 每 20 帧喘一口气

    mask = speech_mask(values)

    assert mask.sum() > 150


# --------------------------------------------------------------- 包络对象
def test_envelope_from_samples_aligns_with_frames() -> None:
    samples = np.concatenate([np.zeros(SAMPLE_RATE, dtype=np.float32), _tone(1.0)])
    envelope = SpeechEnvelope.from_samples(samples, SAMPLE_RATE, fps=10.0, frame_count=20)

    assert envelope.frame_count == 20
    assert envelope.energy_at(0) == 0.0
    assert not envelope.is_speech(0)
    assert envelope.is_speech(19)
    assert 0.4 < envelope.active_ratio < 0.6


def test_envelope_out_of_range_is_silent() -> None:
    envelope = SpeechEnvelope.from_samples(_tone(0.5), SAMPLE_RATE, fps=10.0, frame_count=5)

    assert envelope.energy_at(99) == 0.0
    assert envelope.energy_at(-1) == 0.0
    assert envelope.is_speech(99) is False
    assert "帧" in envelope.describe()


# ------------------------------------------------------- ffmpeg 抽取（降级路径）
def test_extract_envelope_without_ffmpeg(monkeypatch) -> None:
    monkeypatch.setattr(audio.ffmpeg_tools, "find_ffmpeg", lambda: None)

    assert audio.extract_envelope("whatever.mp4", fps=30.0, frame_count=10) is None


def test_extract_envelope_without_audio_stream(monkeypatch) -> None:
    monkeypatch.setattr(audio.ffmpeg_tools, "find_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(audio.ffmpeg_tools, "has_audio_stream", lambda path: False)

    assert audio.extract_envelope("whatever.mp4", fps=30.0, frame_count=10) is None


def test_extract_envelope_rejects_bad_frame_count(monkeypatch) -> None:
    monkeypatch.setattr(audio.ffmpeg_tools, "find_ffmpeg", lambda: "ffmpeg")

    assert audio.extract_envelope("whatever.mp4", fps=0.0, frame_count=10) is None
    assert audio.extract_envelope("whatever.mp4", fps=30.0, frame_count=0) is None


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="需要 ffmpeg 才能抽取音轨")
def test_extract_envelope_from_real_video(tmp_path) -> None:
    """真实视频上跑一遍：前 1 秒静音、后 1 秒有声，应只在后半段判出语音。"""
    ffmpeg = shutil.which("ffmpeg")
    source = tmp_path / "talking.mp4"
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
            "testsrc=size=64x48:rate=30:duration=2",
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

    envelope = audio.extract_envelope(source, fps=30.0, frame_count=60)

    assert envelope is not None
    assert envelope.frame_count == 60
    assert envelope.active_frames > 0
    assert not envelope.is_speech(5)  # 开头是静音
    assert envelope.is_speech(45)  # 后半段有声


def test_vad_params_validation() -> None:
    with pytest.raises(ValueError):
        VadParams(threshold_ratio=0.0)
    with pytest.raises(ValueError):
        VadParams(noise_multiple=0.5)
    with pytest.raises(ValueError):
        VadParams(min_energy=1.5)
