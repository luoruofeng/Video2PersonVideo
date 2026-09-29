"""音频侧：逐帧语音能量包络与语音活动（VAD）。

"多个主要人物时对准正在说话的那个人"，光靠 YOLO 做不到：它是纯图像模型，
既没有音频输入，COCO-17 关键点里也没有嘴部关键点（只有鼻、眼、耳、肩、髋…）。
所以这件事必须自己补齐，音频这一半由本模块负责：

* 把音轨压成**与视频帧一一对应的语音能量包络**（每帧一个 RMS 值）；
* 给出**语音 / 静音**标记，让判定环节知道"现在有没有人在说话"。

另外两半分别在 :mod:`~video2personvideo.core.mouth`（谁在动嘴）与
:mod:`~video2personvideo.core.speaker`（把两路信号对上号）。

纯计算部分（:func:`frame_energy` / :class:`SpeechEnvelope`）不依赖 ffmpeg，
可脱离视频单测；只有 :func:`extract_envelope` 需要系统里有 ffmpeg。
"""

from __future__ import annotations

import contextlib
import math
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..utils import ffmpeg_tools
from ..utils.logger import get_logger

logger = get_logger(__name__)

#: 语音分析采样率：16 kHz 足够覆盖人声主要频段，且体积小
DEFAULT_SAMPLE_RATE = 16000

#: 从 ffmpeg 管道读取 PCM 的块大小（字节），控制内存占用
_READ_CHUNK_BYTES = 1 << 16

#: 判定语音活动时用的能量分位（抗个别爆音）
_ENERGY_PERCENTILE = 95.0
#: 估计底噪的分位：取偏低的分位而不是中位数 —— 说话占满全片时中位数就是语音本身，
#: 拿它当底噪会把所有帧都判成"不够响"
_NOISE_PERCENTILE = 10.0


@dataclass(frozen=True, slots=True)
class VadParams:
    """语音活动判定参数。"""

    #: 相对门限：能量低于「参考能量 × 该值」视为静音
    threshold_ratio: float = 0.12
    #: 同时要求高于底噪（中位数）的多少倍，避免安静环境里的底噪被判成语音
    noise_multiple: float = 3.0
    #: 绝对下限（归一化能量）：低于它一律视为静音
    min_energy: float = 0.02

    def __post_init__(self) -> None:
        if not 0.0 < float(self.threshold_ratio) <= 1.0:
            raise ValueError(f"threshold_ratio 必须在 (0, 1] 区间内，当前为 {self.threshold_ratio}")
        if float(self.noise_multiple) < 1.0:
            raise ValueError(f"noise_multiple 必须 ≥ 1，当前为 {self.noise_multiple}")
        if not 0.0 <= float(self.min_energy) <= 1.0:
            raise ValueError(f"min_energy 必须在 [0, 1] 区间内，当前为 {self.min_energy}")


#: 默认语音活动参数
DEFAULT_VAD_PARAMS = VadParams()


@dataclass(frozen=True, slots=True)
class SpeechEnvelope:
    """逐帧的语音能量包络。

    ``energy[i]`` 是第 ``i`` 帧对应时间区间内的 RMS 能量（已归一化到 0~1），
    ``active[i]`` 是该帧是否判定为"有人在说话"。
    """

    fps: float
    energy: tuple[float, ...]
    active: tuple[bool, ...]

    @property
    def frame_count(self) -> int:
        return len(self.energy)

    @property
    def active_frames(self) -> int:
        return sum(1 for flag in self.active if flag)

    @property
    def active_ratio(self) -> float:
        return self.active_frames / self.frame_count if self.energy else 0.0

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0

    def energy_at(self, index: int) -> float:
        """第 ``index`` 帧的能量；越界（音频比画面短）返回 0。"""
        if 0 <= index < len(self.energy):
            return float(self.energy[index])
        return 0.0

    def is_speech(self, index: int) -> bool:
        """第 ``index`` 帧是否有人在说话；越界返回 ``False``。"""
        if 0 <= index < len(self.active):
            return bool(self.active[index])
        return False

    def describe(self) -> str:
        return (
            f"{self.frame_count} 帧 / {self.duration:.1f}s，"
            f"其中 {(self.active_ratio * 100):.0f}% 的帧检测到语音"
        )

    @classmethod
    def from_samples(
        cls,
        samples: Sequence[float] | np.ndarray,
        sample_rate: int,
        fps: float,
        frame_count: int,
        params: VadParams = DEFAULT_VAD_PARAMS,
    ) -> SpeechEnvelope:
        """由单声道 PCM 采样构造包络（纯函数，可直接喂数组做单测）。"""
        energy = frame_energy(samples, sample_rate, fps, frame_count)
        normalized = normalize_energy(energy)
        return cls(
            fps=float(fps),
            energy=tuple(float(value) for value in normalized),
            active=tuple(bool(flag) for flag in speech_mask(normalized, params)),
        )


def frame_energy(
    samples: Sequence[float] | np.ndarray,
    sample_rate: int,
    fps: float,
    frame_count: int,
) -> np.ndarray:
    """把音频采样折算成"每视频帧一个 RMS 值"（未归一化）。

    音频比画面短时，多出来的帧能量记 0（按静音处理）。
    """
    count = max(int(frame_count), 0)
    energies = np.zeros(count, dtype=np.float32)
    if count == 0 or float(fps) <= 0.0 or int(sample_rate) <= 0:
        return energies

    block = np.asarray(samples, dtype=np.float32).reshape(-1)
    if block.size == 0:
        return energies

    samples_per_frame = float(sample_rate) / float(fps)
    _fill_energies(block, energies, 0, 0, samples_per_frame)
    return energies


def _fill_energies(
    block: np.ndarray,
    energies: np.ndarray,
    next_frame: int,
    base_sample: int,
    samples_per_frame: float,
) -> tuple[int, int]:
    """把 ``block`` 里的样本按帧边界累计进 ``energies``。

    :param block: 一段连续的单声道采样（``block[0]`` 的全局下标是 ``base_sample``）
    :param next_frame: 下一个待填充的帧下标
    :return: ``(下一个待填充帧, 本块被消费的样本数)``
    """
    frame_count = energies.size
    if block.size == 0:
        return next_frame, 0

    squared = np.square(block, dtype=np.float64)
    cumulative = np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(squared)))
    while next_frame < frame_count:
        start = int(round(next_frame * samples_per_frame)) - base_sample
        end = int(round((next_frame + 1) * samples_per_frame)) - base_sample
        if end > block.size:
            break
        start = max(start, 0)
        if end > start:
            energies[next_frame] = math.sqrt((cumulative[end] - cumulative[start]) / (end - start))
        next_frame += 1

    if next_frame >= frame_count:
        return next_frame, block.size
    consumed = int(round(next_frame * samples_per_frame)) - base_sample
    return next_frame, max(0, min(block.size, consumed))


def normalize_energy(energy: np.ndarray | Sequence[float]) -> np.ndarray:
    """按 95 分位归一化到 0~1（用分位而不是最大值，避免单个爆音把其余帧压扁）。"""
    values = np.asarray(energy, dtype=np.float32)
    if values.size == 0:
        return values
    reference = float(np.percentile(values, _ENERGY_PERCENTILE))
    if reference <= 0.0:
        return np.zeros_like(values)
    return np.clip(values / reference, 0.0, 1.0)


def speech_mask(
    energy: np.ndarray | Sequence[float], params: VadParams = DEFAULT_VAD_PARAMS
) -> np.ndarray:
    """按能量包络判定语音帧（相对门限 + 底噪倍数 + 绝对下限）。"""
    values = np.asarray(energy, dtype=np.float32)
    if values.size == 0:
        return np.zeros(0, dtype=bool)

    reference = float(np.percentile(values, _ENERGY_PERCENTILE))
    if reference <= 0.0:
        return np.zeros(values.size, dtype=bool)

    noise = float(np.percentile(values, _NOISE_PERCENTILE))
    threshold = max(
        float(params.min_energy),
        float(params.threshold_ratio) * reference,
        float(params.noise_multiple) * noise,
    )
    return values >= threshold


def extract_envelope(
    source: str | Path,
    *,
    fps: float,
    frame_count: int,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    params: VadParams = DEFAULT_VAD_PARAMS,
    timeout: float = 600.0,
) -> SpeechEnvelope | None:
    """用 ffmpeg 抽音轨并构造语音能量包络。

    任何一步不可用（没有 ffmpeg、没有音轨、解码失败）都返回 ``None``，
    由调用方安静地退回"按画面打分选主角"，绝不因为这项增强而中断处理。
    """
    if float(fps) <= 0.0 or int(frame_count) <= 0:
        return None
    ffmpeg = ffmpeg_tools.find_ffmpeg()
    if ffmpeg is None:
        return None
    if not ffmpeg_tools.has_audio_stream(source):
        return None

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(int(sample_rate)),
        "-f",
        "s16le",
        "-",
    ]
    energies = np.zeros(int(frame_count), dtype=np.float32)
    samples_per_frame = float(sample_rate) / float(fps)

    try:
        process = subprocess.Popen(  # noqa: S603 - 参数为本地常量 + 用户自己的输入路径
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError as exc:  # pragma: no cover - 极少见（ffmpeg 无法启动）
        logger.debug("启动 ffmpeg 失败，跳过语音分析：%s", exc)
        return None

    pending = np.zeros(0, dtype=np.float32)
    base_sample = 0
    next_frame = 0
    byte_tail = b""
    stdout = process.stdout
    timed_out = False
    try:
        while stdout is not None:
            chunk = stdout.read(_READ_CHUNK_BYTES)
            if not chunk:
                break
            chunk = byte_tail + chunk
            if len(chunk) % 2:
                byte_tail, chunk = chunk[-1:], chunk[:-1]
            else:
                byte_tail = b""
            if not chunk:
                continue
            block = np.frombuffer(chunk, dtype="<i2").astype(np.float32) / 32768.0
            if pending.size:
                block = np.concatenate((pending, block))
            next_frame, consumed = _fill_energies(
                block, energies, next_frame, base_sample, samples_per_frame
            )
            pending = block[consumed:]
            base_sample += consumed
    except Exception as exc:  # noqa: BLE001 - 读流失败不应中断处理
        logger.debug("读取音轨失败，跳过语音分析：%s", exc)
        return None
    finally:
        if stdout is not None:
            with contextlib.suppress(OSError):
                stdout.close()
        try:
            process.wait(timeout=float(timeout))
        except subprocess.TimeoutExpired:  # pragma: no cover - 极端异常输入
            process.kill()
            timed_out = True

    if timed_out:
        logger.debug("ffmpeg 抽取音轨超时，跳过语音分析")
        return None
    if next_frame <= 0:
        logger.debug("未能从音轨中取到音频采样，跳过语音分析")
        return None

    normalized = normalize_energy(energies)
    envelope = SpeechEnvelope(
        fps=float(fps),
        energy=tuple(float(value) for value in normalized),
        active=tuple(bool(flag) for flag in speech_mask(normalized, params)),
    )
    logger.debug("语音包络：%s", envelope.describe())
    return envelope
