"""时序平滑测试：无跳变、比例恒定、死区与三级兜底（M5 / 不变量 3）。

额外覆盖"镜头稳不稳"这一层体验：自由活动区（人物走动时镜头不动）、平移速度
限幅（不甩镜）、缩放比平移更慢（不呼吸式变焦）、不冲过目标（不回摆）。
"""

from __future__ import annotations

from itertools import pairwise

import pytest

from video2personvideo.core.framing import MODE_CENTER, MODE_HOLD, CropBox, center_box
from video2personvideo.core.ratio import PRESET_BY_NAME
from video2personvideo.core.smoothing import (
    CAMERA_PRESETS,
    MAX_SPEED_FACTOR,
    BoxSmoother,
    SmoothingParams,
    match_preset,
    preset_values,
)

RATIO = PRESET_BY_NAME["9:16"]
FRAME = (1920.0, 1080.0)


def _box(cx: float, height: float = 600.0) -> CropBox:
    width = height * RATIO.value
    return CropBox(cx - width / 2, 200.0, width, height, mode="halfbody")


def test_first_update_snaps_to_target() -> None:
    smoother = BoxSmoother(RATIO, alpha=0.3)
    target = _box(900.0)
    box = smoother.update(target, FRAME)
    assert box.cx == pytest.approx(target.cx)
    assert box.h == pytest.approx(target.h)


def test_ratio_is_constant_across_updates() -> None:
    smoother = BoxSmoother(RATIO, alpha=0.4)
    for index in range(30):
        box = smoother.update(_box(300 + index * 40), FRAME)
        assert box.matches_ratio(RATIO, tol=1e-6)


def test_output_never_jumps_more_than_alpha() -> None:
    smoother = BoxSmoother(RATIO, alpha=0.25, deadzone=0.0)
    previous = smoother.update(_box(300.0), FRAME)
    for index in range(1, 40):
        target = _box(300.0 + index * 30.0)
        current = smoother.update(target, FRAME)
        assert abs(current.cx - previous.cx) <= 0.25 * abs(target.cx - previous.cx) + 1e-6
        previous = current


def test_deadzone_ignores_micro_movement() -> None:
    smoother = BoxSmoother(RATIO, alpha=1.0, deadzone=0.05)
    first = smoother.update(_box(900.0, height=600.0), FRAME)
    # 位移 1px，远小于 5% 的死区（600 * 0.05 = 30px）
    second = smoother.update(_box(901.0, height=600.0), FRAME)
    assert second.cx == pytest.approx(first.cx)
    assert second.h == pytest.approx(first.h)


def test_hold_frames_keep_last_box() -> None:
    smoother = BoxSmoother(RATIO, alpha=0.5, hold_frames=3)
    last = smoother.update(_box(900.0), FRAME)
    for _ in range(3):
        held = smoother.update(None, FRAME)
        assert held.mode == MODE_HOLD
        assert held.cx == pytest.approx(last.cx)
        assert held.h == pytest.approx(last.h)
    assert smoother.holds == 3
    assert smoother.missing == 3


def test_center_fallback_after_hold_timeout() -> None:
    """超时回中：先切到居中模式，再用若干帧缓慢平移过去（不再"啪"地跳一下）。"""
    smoother = BoxSmoother(RATIO, alpha=1.0, hold_frames=2)
    last = smoother.update(_box(300.0), FRAME)
    smoother.update(None, FRAME)
    smoother.update(None, FRAME)
    fallen_back = smoother.update(None, FRAME)
    expected = center_box(FRAME, RATIO)

    assert fallen_back.mode == MODE_CENTER
    assert fallen_back.matches_ratio(RATIO, tol=1e-6)
    assert smoother.centers == 1
    # 一帧之内不会走完到画面中心的一半路程（速度受限）
    assert abs(fallen_back.cx - last.cx) < 0.5 * abs(expected.cx - last.cx)

    current = fallen_back
    for _ in range(180):
        current = smoother.update(None, FRAME)
        assert current.matches_ratio(RATIO, tol=1e-6)
    assert current.cx == pytest.approx(expected.cx, abs=2.0)
    assert current.h == pytest.approx(expected.h, abs=2.0)
    assert smoother.centers > 1


def test_no_target_at_all_uses_center() -> None:
    smoother = BoxSmoother(RATIO)
    box = smoother.update(None, FRAME)
    assert box.mode == MODE_CENTER
    assert box.matches_ratio(RATIO, tol=1e-6)
    assert smoother.centers == 1


def test_recovering_target_returns_to_tracking() -> None:
    smoother = BoxSmoother(RATIO, alpha=0.5, deadzone=0.0, hold_frames=5)
    smoother.update(_box(300.0), FRAME)
    smoother.update(None, FRAME)
    recovered = smoother.update(_box(1200.0), FRAME)
    assert recovered.mode == "halfbody"
    assert smoother.missing == 0
    # 恢复追踪后继续朝目标收敛；速度上限是平滑饱和的（最快 4 倍基准），
    # 因此横跨 900px 的追赶要 30 帧出头才贴到目标 —— 这是刻意的"慢一点、不甩镜"。
    for _ in range(40):
        recovered = smoother.update(_box(1200.0), FRAME)
    assert recovered.cx == pytest.approx(1200.0, abs=1.0)


def test_reset_clears_state() -> None:
    smoother = BoxSmoother(RATIO)
    smoother.update(_box(900.0), FRAME)
    smoother.reset()
    assert smoother.box is None
    assert smoother.frames == 0
    assert smoother.holds == 0
    assert smoother.centers == 0


# --------------------------------------------------- 镜头稳定（缓解晕动感）
def test_free_zone_keeps_camera_still() -> None:
    """核心体验：人物在自由活动区内走动时镜头一动不动（背景完全静止）。"""
    smoother = BoxSmoother(RATIO, alpha=0.5, deadzone=0.15)
    first = smoother.update(_box(900.0), FRAME)
    zone = 0.15 * max(first.w, first.h)

    for offset in (0.9, -0.9, 0.5, -0.5, 0.2):
        moved = smoother.update(_box(900.0 + zone * offset), FRAME)
        assert moved.cx == pytest.approx(first.cx)
        assert moved.h == pytest.approx(first.h)


def test_camera_pans_only_after_leaving_free_zone() -> None:
    """超出自由活动区后镜头只推"多出来的那一部分"，稳定输出连续、无跳变。"""
    smoother = BoxSmoother(RATIO, alpha=0.5, deadzone=0.15, max_speed=0.0, accel=1.0)
    first = smoother.update(_box(900.0), FRAME)
    zone = 0.15 * max(first.w, first.h)

    just_inside = smoother.update(_box(900.0 + zone), FRAME)
    assert just_inside.cx == pytest.approx(first.cx)

    outside = smoother.update(_box(900.0 + zone + 20.0), FRAME)
    assert outside.cx > just_inside.cx
    assert outside.cx - just_inside.cx <= 0.5 * 20.0 + 1e-6


def test_pan_speed_is_limited() -> None:
    """目标突然大跳（遮挡后重新出现）也不会一帧甩过去。"""
    smoother = BoxSmoother(RATIO, params=SmoothingParams(max_speed=0.02, accel=1.0))
    first = smoother.update(_box(200.0), FRAME)
    jumped = smoother.update(_box(1700.0), FRAME)

    limit = 0.02 * first.w * MAX_SPEED_FACTOR
    assert 0.0 < jumped.cx - first.cx <= limit + 1e-6
    assert jumped.cx < 1700.0


def test_zoom_is_slower_than_pan() -> None:
    """推拉比平移更迟钝：同样的变化幅度，高度反应明显小于位置反应。"""
    smoother = BoxSmoother(RATIO, alpha=0.25, deadzone=0.15, zoom_alpha=0.08, zoom_deadzone=0.10)
    first = smoother.update(_box(700.0, height=600.0), FRAME)
    # 同时横移 300px 并被放大到 900px 高
    moved = smoother.update(_box(1000.0, height=900.0), FRAME)

    pan_ratio = abs(moved.cx - first.cx) / first.w
    zoom_ratio = abs(moved.h - first.h) / first.h
    assert zoom_ratio < pan_ratio


def test_camera_never_overshoots_target() -> None:
    """镜头不会冲过目标后反复回摆（回摆本身就是眩晕来源）。"""
    smoother = BoxSmoother(RATIO, alpha=0.6, deadzone=0.0)
    smoother.update(_box(400.0), FRAME)

    previous = 400.0
    box = None
    for _ in range(60):
        box = smoother.update(_box(1100.0), FRAME)
        assert box.cx <= 1100.0 + 1e-6
        assert box.cx >= previous - 1e-6
        previous = box.cx
    assert box is not None
    assert box.cx == pytest.approx(1100.0, abs=1.0)


def test_fast_moving_person_stays_inside_frame() -> None:
    """人物持续快速单向走动时镜头跟得上，人不会被甩出画面、框也不会越界。"""
    smoother = BoxSmoother(RATIO, alpha=0.25, deadzone=0.15)
    frame_w = FRAME[0]
    person_x = 400.0
    smoother.update(_box(person_x), FRAME)

    for _ in range(60):
        person_x += 15.0
        box = smoother.update(_box(person_x), FRAME)
        left = box.cx - box.w / 2.0
        right = box.cx + box.w / 2.0
        assert left - 1e-6 <= person_x <= right + 1e-6
        assert left >= 0.0 and right <= frame_w + 1e-6


# ------------------------------------------------------------ 档位与参数
def test_camera_presets_go_from_stable_to_responsive() -> None:
    keys = list(CAMERA_PRESETS)
    assert keys == ["lock", "calm", "standard", "active"]

    params = [CAMERA_PRESETS[key].params for key in keys]
    for stable, responsive in pairwise(params):
        assert stable.deadzone > responsive.deadzone
        assert stable.max_speed < responsive.max_speed
        assert stable.zoom_alpha < responsive.zoom_alpha
        assert stable.zoom_deadzone > responsive.zoom_deadzone


def test_preset_values_round_trip() -> None:
    for key in CAMERA_PRESETS:
        values = preset_values(key)
        assert values["smoothing_alpha"] == pytest.approx(CAMERA_PRESETS[key].params.alpha)
        assert match_preset(values["smoothing_alpha"]) == key
    assert match_preset(0.6) is None
    assert match_preset(None) is None


def test_default_params_are_used_when_not_given() -> None:
    smoother = BoxSmoother(RATIO)
    defaults = SmoothingParams()
    assert smoother.alpha == pytest.approx(defaults.alpha)
    assert smoother.deadzone == pytest.approx(defaults.deadzone)
    assert smoother.zoom_alpha == pytest.approx(defaults.zoom_alpha)
    assert smoother.max_speed == pytest.approx(defaults.max_speed)
    assert smoother.hold_frames == defaults.hold_frames


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha": 0.0},
        {"alpha": 1.5},
        {"deadzone": -0.1},
        {"hold_frames": -1},
        {"zoom_alpha": 0.0},
        {"zoom_deadzone": -0.01},
        {"max_speed": -0.1},
        {"accel": 0.0},
        {"accel": 1.5},
        {"fps": 0.0},
        {"fps": -30.0},
    ],
)
def test_invalid_parameters_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        BoxSmoother(RATIO, **kwargs)


# ------------------------------------------- 帧率归一化：镜头"每秒感受"与帧率无关
def _track(
    smoother: BoxSmoother,
    *,
    fps: float,
    seconds: float,
    speed_px_per_s: float,
    start: float = 400.0,
) -> list[float]:
    """让目标以固定的**真实速度**走一段，返回每一帧的镜头中心（像素）。"""
    step = speed_px_per_s / fps
    x = start
    smoother.update(_box(x), FRAME)
    centers = [x]
    for _ in range(int(round(seconds * fps))):
        x += step
        centers.append(smoother.update(_box(x), FRAME).cx)
    return centers


def _pan_peak_px_per_s(fps: float, *, normalized: bool) -> float:
    """目标瞬间跳到远处时，镜头平移的**真实速度**峰值（像素/秒）。"""
    smoother = BoxSmoother(RATIO, fps=fps) if normalized else BoxSmoother(RATIO)
    smoother.update(_box(200.0), FRAME)
    peak = 0.0
    for _ in range(int(round(fps))):
        before = smoother.box.cx
        smoother.update(_box(1700.0), FRAME)
        peak = max(peak, abs(smoother.box.cx - before) * fps)
    return peak


def test_peak_pan_speed_does_not_scale_with_frame_rate() -> None:
    """爆发式追赶时，24 / 30 / 60fps 的镜头**真实速度**峰值必须一致（不甩镜）。"""
    slow, standard, fast = (
        _pan_peak_px_per_s(fps, normalized=True) for fps in (24.0, 30.0, 60.0)
    )
    assert standard == pytest.approx(slow, rel=0.02)
    assert fast == pytest.approx(standard, rel=0.02)
    # 且受"基准速度 × 最大放宽倍数"封顶，与参考帧率换算后一致
    cap = 0.025 * _box(0.0).w * MAX_SPEED_FACTOR * 30.0
    assert standard <= cap * 1.05


def test_without_frame_rate_the_camera_speed_doubles_at_60fps() -> None:
    """回归护栏：不告知帧率时 60fps 会按两倍真实速度甩镜 —— 这正是要修掉的不适来源。"""
    slow = _pan_peak_px_per_s(30.0, normalized=False)
    fast = _pan_peak_px_per_s(60.0, normalized=False)
    assert fast == pytest.approx(2.0 * slow, rel=0.05)


def test_same_real_motion_keeps_same_camera_lag_across_frame_rates() -> None:
    """目标以同样真实速度走动：30fps 与 60fps 的镜头位置几乎重合（滞后只差几像素）。"""
    slow = BoxSmoother(RATIO, alpha=0.25, deadzone=0.15, fps=30.0)
    fast = BoxSmoother(RATIO, alpha=0.25, deadzone=0.15, fps=60.0)
    slow_centers = _track(slow, fps=30.0, seconds=2.0, speed_px_per_s=300.0)
    fast_centers = _track(fast, fps=60.0, seconds=2.0, speed_px_per_s=300.0)
    assert slow_centers[-1] == pytest.approx(fast_centers[-1], abs=12.0)


def test_effective_params_scale_with_frame_rate() -> None:
    """换算规则：速度 / 平滑系数按比例变小，保持帧数按比例变大。"""
    params = SmoothingParams(alpha=0.25, zoom_alpha=0.08, max_speed=0.025, hold_frames=30)
    fast = BoxSmoother(RATIO, params=params, fps=60.0)
    assert fast.effective_max_speed == pytest.approx(params.max_speed / 2.0)
    assert fast.effective_alpha == pytest.approx(1.0 - (1.0 - params.alpha) ** 0.5)
    assert fast.effective_alpha < params.alpha
    assert fast.effective_zoom_alpha < params.zoom_alpha
    assert fast.effective_hold_frames == 60

    legacy = BoxSmoother(RATIO, params=params)  # 不知道帧率 = 旧行为，原样使用
    assert legacy.fps is None
    assert legacy.effective_alpha == pytest.approx(params.alpha)
    assert legacy.effective_max_speed == pytest.approx(params.max_speed)
    assert legacy.effective_hold_frames == params.hold_frames

    # hold_frames=0（明确"不保持"）不能被换算成 1
    assert BoxSmoother(RATIO, hold_frames=0, fps=60.0).effective_hold_frames == 0


def test_hold_duration_is_time_based_across_frame_rates() -> None:
    """保持目标时长的语义是"秒"：60fps 保持的帧数翻倍，实际时长才一样。"""
    slow = BoxSmoother(RATIO, hold_frames=30, fps=30.0)
    fast = BoxSmoother(RATIO, hold_frames=30, fps=60.0)
    slow.update(_box(900.0), FRAME)
    fast.update(_box(900.0), FRAME)
    for _ in range(60):
        slow.update(None, FRAME)
        fast.update(None, FRAME)
    assert slow.holds == 30
    assert fast.holds == 60


def test_set_fps_updates_and_ignores_invalid() -> None:
    smoother = BoxSmoother(RATIO)
    assert smoother.fps is None

    smoother.set_fps(0.0)  # 探测失败：保持原样，不能把镜头参数改坏
    assert smoother.fps is None
    assert smoother.effective_max_speed == pytest.approx(smoother.max_speed)

    smoother.set_fps(60.0)
    assert smoother.fps == pytest.approx(60.0)
    assert smoother.effective_max_speed == pytest.approx(smoother.max_speed / 2.0)

    smoother.set_fps(None)
    assert smoother.fps is None
    assert smoother.effective_alpha == pytest.approx(smoother.alpha)
