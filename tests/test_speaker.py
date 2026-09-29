"""主动说话人判定测试：相关性、得分、轨迹关联与切换滞回。"""

from __future__ import annotations

import pytest

from video2personvideo.core.audio import SpeechEnvelope
from video2personvideo.core.speaker import (
    ActiveSpeakerSelector,
    SpeakerParams,
    correlation,
    track_score,
)

BOXES = [(50.0, 50.0, 150.0, 230.0), (200.0, 50.0, 300.0, 230.0)]

#: 测试用参数：窗口更短，换人滞回更短，便于构造用例
PARAMS = SpeakerParams(
    window_frames=20,
    min_samples=6,
    min_score=0.20,
    switch_margin=0.15,
    switch_hold=3,
    min_mouth_motion=0.004,
)


def _envelope(count: int = 400, block: int = 5) -> SpeechEnvelope:
    """语音能量：每 5 帧交替"说话 / 停顿"的方波。"""
    energy = tuple(1.0 if (index // block) % 2 == 0 else 0.0 for index in range(count))
    return SpeechEnvelope(
        fps=30.0, energy=energy, active=tuple(value >= 0.5 for value in energy)
    )


ENVELOPE = _envelope()


def _speaking_mouths(index: int, *, first: bool) -> list[float]:
    active = ENVELOPE.is_speech(index)
    value = 1.0 if active else 0.0
    return [value, 0.0] if first else [0.0, value]


# ------------------------------------------------------------------ 相关性
def test_correlation_basic() -> None:
    assert correlation([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)
    assert correlation([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)
    assert correlation([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) == 0.0
    assert correlation([1.0, 2.0], [1.0, 2.0, 3.0]) == 0.0
    assert correlation([1.0], [1.0]) == 0.0


# ------------------------------------------------------------------ 打分
def test_track_score_rewards_synchronised_mouth() -> None:
    mouth = [1.0 if flag else 0.0 for flag in ENVELOPE.active[:20]]
    energy = [ENVELOPE.energy[index] for index in range(20)]
    active = [ENVELOPE.active[index] for index in range(20)]

    assert track_score(mouth, energy, active, PARAMS) == pytest.approx(1.0)


def test_track_score_is_zero_for_still_mouth() -> None:
    energy = list(ENVELOPE.energy[:20])
    active = list(ENVELOPE.active[:20])

    assert track_score([0.5] * 20, energy, active, PARAMS) == 0.0
    assert track_score([0.0] * 20, energy, active, PARAMS) == 0.0


def test_track_score_needs_speech_in_window() -> None:
    energy = [0.0] * 20
    active = [False] * 20
    mouth = [0.8] * 20

    assert track_score(mouth, energy, active, PARAMS) == 0.0


def test_track_score_needs_enough_samples() -> None:
    assert track_score([1.0] * 3, [1.0] * 3, [True] * 3, PARAMS) == 0.0


def test_track_score_rejects_unreadable_mouth() -> None:
    """嘴动幅度低于下限（远景小脸）时不给分，避免拿噪声当依据。"""
    mouth = [PARAMS.min_mouth_motion / 10.0] * 20
    energy = list(ENVELOPE.energy[:20])
    active = list(ENVELOPE.active[:20])

    assert track_score(mouth, energy, active, PARAMS) == 0.0


# ------------------------------------------------------------------ 选择器
def test_selector_without_envelope_never_judges() -> None:
    selector = ActiveSpeakerSelector(params=PARAMS)

    assert selector.enabled is False
    assert selector.biases(0, BOXES, [1.0, 0.0]) is None


def test_selector_ignores_single_person() -> None:
    selector = ActiveSpeakerSelector(ENVELOPE, PARAMS)

    assert selector.biases(0, [BOXES[0]], [1.0]) is None


def test_selector_picks_the_person_whose_mouth_matches_audio() -> None:
    selector = ActiveSpeakerSelector(ENVELOPE, PARAMS)

    results = [
        selector.biases(index, BOXES, _speaking_mouths(index, first=True))
        for index in range(PARAMS.min_samples * 2)
    ]

    assert results[: PARAMS.min_samples - 1] == [None] * (PARAMS.min_samples - 1)
    assert results[-1] == [1.0, 0.0]
    assert selector.current_track == 0
    assert selector.switches == 0


def test_selector_switches_only_after_hold() -> None:
    """换人需要"连续多帧 + 明显领先"：验证切换发生在滞回之后，且全程只切一次。"""
    selector = ActiveSpeakerSelector(ENVELOPE, PARAMS)
    established = 30
    for index in range(established):
        selector.biases(index, BOXES, _speaking_mouths(index, first=True))
    assert selector.current_track == 0

    switched_at: int | None = None
    for offset in range(60):
        index = established + offset
        biases = selector.biases(index, BOXES, _speaking_mouths(index, first=False))
        if biases == [0.0, 1.0] and switched_at is None:
            switched_at = index

    assert switched_at is not None
    assert selector.switches == 1
    # 挑战者要攒够 min_samples 才能拿到分，再连续领先 switch_hold 个回合才夺位
    first_possible = established + PARAMS.min_samples - 1
    assert switched_at >= first_possible
    assert switched_at - first_possible >= PARAMS.switch_hold - 1
    assert selector.current_track == 1


def test_selector_scoring_honours_configured_params() -> None:
    """轨迹打分必须用判定器自己的参数，而不是内置默认值。

    回归用例：``_Track.score`` 曾经固定调用 ``track_score(...)`` 且不传参数，
    于是用户在配置里调的 ``speaker_min_mouth_motion`` / ``min_samples``
    对打分完全没有影响（README 里"调低它让远景小脸也能判定"因此失效）。
    """
    tiny = 0.002  # 高于宽松下限 0.001、低于默认下限 0.004
    base = {"window_frames": 20, "min_samples": 6, "switch_hold": 3}

    def mouths(index: int) -> list[float]:
        return [tiny if ENVELOPE.is_speech(index) else 0.0, 0.0]

    strict = ActiveSpeakerSelector(ENVELOPE, SpeakerParams(min_mouth_motion=0.01, **base))
    assert [strict.biases(index, BOXES, mouths(index)) for index in range(20)] == [None] * 20
    assert strict.current_track is None

    loose = ActiveSpeakerSelector(ENVELOPE, SpeakerParams(min_mouth_motion=0.001, **base))
    results = [loose.biases(index, BOXES, mouths(index)) for index in range(8)]

    # 攒够 min_samples 就应判定成功（若打分用默认的 min_samples=12 / 下限 0.004 则一直不判定）
    assert results[base["min_samples"] - 1] == [1.0, 0.0]
    assert loose.current_track == 0


def test_selector_keeps_incumbent_through_short_silence() -> None:
    """静音期间不换人（镜头不该因为没人说话就乱动）。"""
    selector = ActiveSpeakerSelector(ENVELOPE, PARAMS)
    for index in range(30):
        selector.biases(index, BOXES, _speaking_mouths(index, first=True))

    for index in range(30, 40):
        biases = selector.biases(index, BOXES, [0.0, 0.0])

    assert biases == [1.0, 0.0]
    assert selector.switches == 0


def test_selector_releases_when_speaker_leaves_frame() -> None:
    """说话的人走出画面（轨迹被丢弃）后，不再硬跟着一个不存在的人。"""
    stranger = (400.0, 50.0, 500.0, 230.0)
    params = SpeakerParams(
        window_frames=20,
        min_samples=6,
        switch_hold=3,
        switch_margin=0.15,
        max_missing_frames=5,
        min_mouth_motion=0.004,
    )
    selector = ActiveSpeakerSelector(ENVELOPE, params)
    for index in range(30):
        selector.biases(index, BOXES, _speaking_mouths(index, first=True))
    assert selector.current_track == 0

    # 说话的人离开画面，画面里换成另外两个人（也都没动嘴）
    for index in range(30, 45):
        biases = selector.biases(index, [BOXES[1], stranger], [0.0, 0.0])

    assert selector.current_track is None
    assert biases is None


def test_speaker_params_validation() -> None:
    with pytest.raises(ValueError):
        SpeakerParams(window_frames=1)
    with pytest.raises(ValueError):
        SpeakerParams(min_samples=1)
    with pytest.raises(ValueError):
        SpeakerParams(window_frames=10, min_samples=20)
    with pytest.raises(ValueError):
        SpeakerParams(min_score=1.5)
    with pytest.raises(ValueError):
        SpeakerParams(switch_margin=-0.1)
    with pytest.raises(ValueError):
        SpeakerParams(switch_hold=0)
    with pytest.raises(ValueError):
        SpeakerParams(track_iou=0.0)
    with pytest.raises(ValueError):
        SpeakerParams(max_missing_frames=0)
    with pytest.raises(ValueError):
        SpeakerParams(min_mouth_motion=-0.1)
