"""多人分屏的窗口镜头：每个小窗口都必须"正常速度播放"。

覆盖三件事（对应"小窗口里的画面动得太快 / 像快进"这个症状）：

* ``core/smoothing.window_smoothing_params``：窗口镜头按秒封顶（平移 / 缩放），
  且不再因为人物偏离而放大到单窗口那样的倍数；
* ``core/multi.MultiPersonComposer`` 的座位策略：窗口"认人"（不互换内容）、
  换人时**直接就位**（切镜头）而不是从旧人物身上横扫；
* 单窗口模式不受影响：``SmoothingParams`` 的默认值仍然是单窗口那套参数。
"""

from __future__ import annotations

import numpy as np
import pytest

from video2personvideo.core.framing import DEFAULT_FRAMING_PARAMS, CropBox
from video2personvideo.core.layout import DEFAULT_MULTI_PERSON_POLICY
from video2personvideo.core.multi import MultiPersonComposer, WindowCamera
from video2personvideo.core.ratio import resolve_ratio
from video2personvideo.core.smoothing import (
    CAMERA_PRESETS,
    CATCHUP,
    DEFAULT_SMOOTHING_PARAMS,
    DEFAULT_WINDOW_PAN_SPEED,
    MAX_SPEED_FACTOR,
    REFERENCE_FPS,
    WINDOW_MAX_SPEED_FACTOR,
    BoxSmoother,
    SmoothingParams,
    window_smoothing_params,
)
from video2personvideo.core.subject import Detection

RATIO = resolve_ratio("9:16")
FRAME_W, FRAME_H = 1920.0, 1080.0
FRAME = (FRAME_W, FRAME_H)
FPS = 30.0

#: 两个人：身高占画面 900/1080 ≈ 0.83，稳过默认的 0.73 阈值
LEFT = (300.0, 100.0, 700.0, 1000.0)
RIGHT = (1200.0, 100.0, 1600.0, 1000.0)
#: 第三个人（"新来的"），位置离左边那个人很远，横扫与切镜头一眼能分辨
NEWCOMER = (760.0, 100.0, 1160.0, 1000.0)


def _detection(bbox: tuple[float, float, float, float]) -> Detection:
    return Detection(bbox=bbox, confidence=0.9)


def _speed_cap_per_second(params: SmoothingParams) -> float:
    """这套参数的**每秒**平移上限（单位：窗口自身的宽度）。

    ``_apply_fps`` 会把 ``max_speed`` 按 ``REFERENCE_FPS / fps`` 折算成每帧值，
    因此"每秒"上限 = ``max_speed × max_speed_factor × REFERENCE_FPS``，与帧率无关。
    """
    return params.max_speed * params.max_speed_factor * REFERENCE_FPS


# ------------------------------------------------------------------ 窗口参数
def test_window_params_cap_pan_speed_per_second() -> None:
    """分屏窗口的平移基准上限 = 「每秒 pan_speed 个窗口宽度」（默认半格 / 秒）。"""
    params = window_smoothing_params(DEFAULT_SMOOTHING_PARAMS)
    assert params.max_speed * REFERENCE_FPS == pytest.approx(DEFAULT_WINDOW_PAN_SPEED)
    # 人物快要走出窗口时才会放宽，且只放宽到 1.6 倍（单窗口是 4 倍）
    assert _speed_cap_per_second(params) == pytest.approx(
        DEFAULT_WINDOW_PAN_SPEED * params.max_speed_factor
    )
    assert params.max_speed_factor < MAX_SPEED_FACTOR / 2
    # 单窗口的跟手档是 1.5 个框宽/秒、越界还放大 4 倍 —— 差距必须有量级上的差别
    assert _speed_cap_per_second(DEFAULT_SMOOTHING_PARAMS) >= 3 * _speed_cap_per_second(params)


def test_window_params_are_lazier_and_softer() -> None:
    """"更懒"（死区更大）+"更柔和"（平滑 / 加速度更小），但构图参数一个都不动。"""
    for preset in CAMERA_PRESETS.values():
        window = window_smoothing_params(preset.params)
        assert window.deadzone > preset.params.deadzone
        assert window.zoom_deadzone > preset.params.zoom_deadzone
        assert window.alpha < preset.params.alpha
        assert window.zoom_alpha < preset.params.zoom_alpha
        assert window.accel < preset.params.accel
        assert window.max_speed <= preset.params.max_speed
    window = window_smoothing_params(DEFAULT_SMOOTHING_PARAMS)
    assert window.no_person_mode == DEFAULT_SMOOTHING_PARAMS.no_person_mode
    assert window.hold_frames == DEFAULT_SMOOTHING_PARAMS.hold_frames


def test_window_pan_speed_is_configurable() -> None:
    slow = window_smoothing_params(DEFAULT_SMOOTHING_PARAMS, pan_speed=0.1)
    fast = window_smoothing_params(DEFAULT_SMOOTHING_PARAMS, pan_speed=2.0)
    assert _speed_cap_per_second(slow) < _speed_cap_per_second(fast)
    # 非法值退回默认，绝不出现"零速度"把镜头焊死
    assert window_smoothing_params(DEFAULT_SMOOTHING_PARAMS, pan_speed=0.0).max_speed > 0


def test_default_single_window_params_unchanged() -> None:
    """单窗口仍然用原来的参数（这次改动不许影响单人画面）。"""
    assert SmoothingParams().catchup == CATCHUP
    assert SmoothingParams().max_speed_factor == MAX_SPEED_FACTOR
    assert SmoothingParams(
        alpha=0.45,
        deadzone=0.06,
        zoom_alpha=0.15,
        zoom_deadzone=0.05,
        max_speed=0.05,
    ) == DEFAULT_SMOOTHING_PARAMS


def test_window_smoother_never_exceeds_screen_speed_budget() -> None:
    """真跑一遍：人物一直在窗口外时，每帧位移也不超过"每秒 pan_speed"的预算。"""
    params = window_smoothing_params(DEFAULT_SMOOTHING_PARAMS, pan_speed=DEFAULT_WINDOW_PAN_SPEED)
    smoother = BoxSmoother(RATIO, params=params, fps=FPS)
    start = CropBox(0.0, 0.0, 400.0, 400.0 / RATIO.value)
    box = smoother.snap_to(start, FRAME)

    # 目标固定在很远的地方：模拟"人物快走出窗口"的极端情况
    far = CropBox(FRAME_W - 400.0, 0.0, 400.0, 400.0 / RATIO.value)
    budget_per_frame = params.max_speed * params.max_speed_factor * start.w
    for _ in range(30):
        previous = box
        box = smoother.update(far, FRAME)
        moved = ((box.cx - previous.cx) ** 2 + (box.cy - previous.cy) ** 2) ** 0.5
        assert moved <= budget_per_frame + 1e-6

    # 一秒的极限位移 = 每秒 pan_speed × 放宽倍数（而不是单窗口那样的"好几个框宽")
    seconds = 30 / FPS
    assert abs(box.cx - start.cx) <= start.w * _speed_cap_per_second(params) * seconds + 1e-6
    assert abs(box.cx - start.cx) <= start.w * 1.0


def test_window_camera_is_far_slower_than_the_raw_preset() -> None:
    """同一个"人物快出窗口"的场景：稳镜头的位移只有旧行为的一小部分。"""
    start = CropBox(0.0, 0.0, 400.0, 400.0 / RATIO.value)
    far = CropBox(FRAME_W - 400.0, 0.0, 400.0, 400.0 / RATIO.value)

    def travelled(params: SmoothingParams) -> float:
        smoother = BoxSmoother(RATIO, params=params, fps=FPS)
        box = smoother.snap_to(start, FRAME)
        travelled = 0.0
        for _ in range(30):  # 一秒
            previous = box
            box = smoother.update(far, FRAME)
            travelled += abs(box.cx - previous.cx)
        return travelled

    window = travelled(window_smoothing_params(DEFAULT_SMOOTHING_PARAMS))
    raw = travelled(DEFAULT_SMOOTHING_PARAMS)
    assert window > 0.0  # 仍然跟得上（不是"焊死"）
    assert window < raw / 3.0


def test_snap_to_places_box_immediately() -> None:
    """``snap_to`` 一帧到位，不做平滑（换人时"切镜头"而不是扫过去）。"""
    smoother = BoxSmoother(RATIO, params=DEFAULT_SMOOTHING_PARAMS, fps=FPS)
    smoother.snap_to(CropBox(0.0, 0.0, 400.0, 400.0 / RATIO.value), FRAME)
    target = CropBox(1200.0, 300.0, 600.0, 600.0 / RATIO.value).fit_to(FRAME_W, FRAME_H)
    box = smoother.snap_to(target, FRAME)
    assert (box.x, box.y, box.w, box.h) == pytest.approx((target.x, target.y, target.w, target.h))
    assert (box.cx - target.cx) == pytest.approx(0.0)


# ------------------------------------------------------------------ 座位策略
def _composer(*, seat_lock: bool = True, stable: bool = True) -> MultiPersonComposer:
    composer = MultiPersonComposer(
        RATIO,
        policy=DEFAULT_MULTI_PERSON_POLICY,
        smoother_params=SmoothingParams(),
        camera=WindowCamera(stable=stable, seat_lock=seat_lock),
    )
    composer.set_fps(FPS)
    return composer


def _speaker_key(detection: Detection) -> tuple[float, float, float, float]:
    return tuple(detection.bbox)


def test_seat_lock_keeps_each_window_on_the_same_person() -> None:
    """说话人换人时，窗口内容**不互换**：谁坐哪个窗口就一直坐那个窗口。"""
    composer = _composer()
    composer.update(FRAME, [_detection(LEFT), _detection(RIGHT)])
    seats = {tuple(slot.bbox): slot.seat for slot in composer._slots}
    assert sorted(seats.values()) == [1, 2]

    # 左右两个人的说话人偏置来回切换，座位都不该变
    for speaker in ([LEFT, RIGHT] * 3):
        bonuses = {bbox: (1.8 if bbox == speaker else 0.0) for bbox in (LEFT, RIGHT)}
        plan = composer.update(FRAME, [_detection(LEFT), _detection(RIGHT)], bonuses)
        assert plan is not None
        after = {tuple(slot.bbox): slot.seat for slot in composer._slots}
        assert after == seats

    # 窗口 1 的内容始终是同一个人（左上角那个人）
    assert plan.boxes[0].cx < plan.boxes[1].cx


def test_speaker_switch_never_moves_the_windows() -> None:
    """说话人每帧都换一次也不动窗口：内容是"同一个人一直在同一个窗口"。"""
    composer = _composer()
    boxes = []

    def run(bonus_bbox: tuple[float, float, float, float]) -> None:
        bonuses = {bbox: (1.8 if bbox == bonus_bbox else 0.0) for bbox in (LEFT, RIGHT)}
        plan = composer.update(FRAME, [_detection(LEFT), _detection(RIGHT)], bonuses)
        assert plan is not None
        boxes.append([(box.cx, box.cy) for box in plan.boxes])

    for speaker in (LEFT, RIGHT) * 5:
        run(speaker)

    # 位置完全不动（人物没动、镜头更不会因为"谁在说话"而移动）
    first = boxes[0]
    assert all(centers == first for centers in boxes[1:])


def test_seat_lock_off_falls_back_to_speaker_order() -> None:
    """关掉「窗口认人」= 旧行为：每个关键帧按说话人重新排座（窗口内容会互换）。"""
    composer = _composer(seat_lock=False)
    composer.update(FRAME, [_detection(LEFT), _detection(RIGHT)])
    first = {tuple(slot.bbox): slot.seat for slot in composer._slots}
    assert first[LEFT] == 1

    bonuses = {LEFT: 0.0, RIGHT: 1.8}
    plan = composer.update(FRAME, [_detection(LEFT), _detection(RIGHT)], bonuses)
    assert plan is not None
    after = {tuple(slot.bbox): slot.seat for slot in composer._slots}
    assert after[RIGHT] == 1 and after[LEFT] == 2
    assert plan.boxes[0].cx > plan.boxes[1].cx  # 说话人换到了 1 号窗口


def test_new_occupant_cuts_instead_of_sweeping() -> None:
    """窗口换人时**直接就位**：新来的人第一帧就在自己该在的位置，不会从旧人扫过来。"""
    composer = _composer()
    composer.update(FRAME, [_detection(LEFT), _detection(RIGHT)])
    left_seat = next(slot.seat for slot in composer._slots if tuple(slot.bbox) == LEFT)
    assert left_seat == 1

    # 左边的人离开：座位先"保持"住（这几帧依旧是分屏，不会闪一下），
    # 保持期满（hold_frames 帧）之后只剩一个人 → 本轮不做分屏；随后来了一个新人顶替他
    held = 0
    while composer.update(FRAME, [_detection(RIGHT)]) is not None:
        held += 1
        assert held <= int(composer.smoother_params.hold_frames)
    assert held >= 2  # 漏检一两帧绝不会让分屏消失（那就成了"画面在闪"）

    plan = composer.update(FRAME, [_detection(RIGHT), _detection(NEWCOMER)])
    assert plan is not None

    boxes = composer.windows(plan)
    newcomer_box = next(
        window.box for window in boxes if abs(window.subject[0] - NEWCOMER[0]) < 1e-6
    )
    # 新人（画面中部）的窗口应当已经落在中部，而不是还在左边那个人的位置上
    assert abs(newcomer_box.cx - (NEWCOMER[0] + NEWCOMER[2]) / 2.0) < 400.0


def test_multi_frames_use_the_window_camera_params() -> None:
    """分屏窗口真的用上了"稳镜头"参数；关掉开关则原样使用用户档位。"""
    stable = _composer()
    stable.update(FRAME, [_detection(LEFT), _detection(RIGHT)])
    for slot in stable._slots:
        assert slot.smoother.params.deadzone > DEFAULT_SMOOTHING_PARAMS.deadzone
        assert _speed_cap_per_second(slot.smoother.params) < 1.0

    raw = _composer(stable=False)
    raw.update(FRAME, [_detection(LEFT), _detection(RIGHT)])
    for slot in raw._slots:
        assert slot.smoother.params.deadzone == DEFAULT_SMOOTHING_PARAMS.deadzone
        assert slot.smoother.params.max_speed == DEFAULT_SMOOTHING_PARAMS.max_speed


# ------------------------------------------------------------------ 整条流水线
class _MovingDetector:
    """两个人一起横向走动（抽帧检测的流水线上跑分屏）。"""

    def __init__(self, frames: int = 45) -> None:
        self.frames = frames
        self.index = -1

    def detect_boxes(self, frame):  # noqa: ANN001, ANN201
        self.index += 1
        step = 6.0 * self.index
        return [
            Detection(bbox=(200.0 + step, 100.0, 600.0 + step, 1000.0), confidence=0.9),
            Detection(bbox=(1200.0 + step, 100.0, 1600.0 + step, 1000.0), confidence=0.9),
        ]


def test_pipeline_keeps_window_speed_normal_and_never_drops_frames() -> None:
    """整条流水线：帧数不丢、输出尺寸恒定、每个窗口的屏幕位移不超预算。"""
    from video2personvideo.core.pipeline import CropPipeline

    frames = 45
    frame = np.zeros((int(FRAME_H), int(FRAME_W), 3), dtype=np.uint8)
    pipeline = CropPipeline(
        RATIO,
        _MovingDetector(frames),
        params=DEFAULT_FRAMING_PARAMS,
        smoother=BoxSmoother(RATIO, params=SmoothingParams()),
        multi=DEFAULT_MULTI_PERSON_POLICY,
        detect_interval=3,  # 走"关键帧 + 插值"这条最容易出问题的路径
    )
    pipeline.set_fps(FPS)
    outcomes = []
    for index in range(frames):
        outcomes.extend(pipeline.process(frame, index))
    outcomes.extend(pipeline.flush())

    assert len(outcomes) == frames
    assert all(item.frame.shape[:2] == (1920, 1080) for item in outcomes)
    assert pipeline.stats.multi_frames == frames

    budget_per_frame = DEFAULT_WINDOW_PAN_SPEED * WINDOW_MAX_SPEED_FACTOR / FPS
    previous = None
    for outcome in outcomes:
        windows = outcome.windows
        if previous is not None and len(windows) == len(previous):
            for window, old_window in zip(windows, previous, strict=True):
                box, rect = window.box, window.rect
                old_box = old_window.box
                moved = abs(box.cx - old_box.cx) * (rect[2] / max(box.w, 1.0))
                moved += abs(box.cy - old_box.cy) * (rect[3] / max(box.h, 1.0))
                assert moved <= (budget_per_frame * rect[2] + 2.0) * 2.0, (
                    f"窗口位移 {moved:.1f} px 超过预算"
                )
        previous = windows
