"""多人分屏：画面里有多个主要人物时，每人一个上半身窗口（流水线级）。

只用人造帧 + 假检测器，不依赖模型；颜色用来分辨"这个窗口里是谁"。
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from video2personvideo.core.framing import DEFAULT_FRAMING_PARAMS, compute_target_box
from video2personvideo.core.layout import DEFAULT_MULTI_PERSON_POLICY, cell_ratio
from video2personvideo.core.pipeline import CropPipeline
from video2personvideo.core.ratio import parse_ratio
from video2personvideo.core.smoothing import BoxSmoother

#: 输出 360×640（9:16），短边小一点让测试跑得快
RATIO = parse_ratio("9:16", target=(360, 640))
#: 测试里把缝隙关掉，格子尺寸才好写断言（缝隙本身在 test_layout 里覆盖）
POLICY = replace(DEFAULT_MULTI_PERSON_POLICY, gap_ratio=0.0)
FRAME_SIZE = (320, 240)  # (宽, 高)

HALF_HEIGHT = RATIO.target_height // 2  # 上下两格的分界

LEFT = (20.0, 40.0, 100.0, 220.0)
MIDDLE = (120.0, 40.0, 200.0, 220.0)
RIGHT = (220.0, 40.0, 300.0, 220.0)
#: 高度只占画面 7.5%，低于 min_person_height_ratio（8%）→ 属于"背景里的路人"
TINY = (150.0, 100.0, 170.0, 118.0)

COLOR_LEFT = (0, 0, 220)  # BGR：红
COLOR_MIDDLE = (0, 140, 220)  # 橙
COLOR_RIGHT = (0, 220, 0)  # 绿

PEOPLE = (
    (LEFT, COLOR_LEFT),
    (MIDDLE, COLOR_MIDDLE),
    (RIGHT, COLOR_RIGHT),
)


def _frame(people=PEOPLE, size=FRAME_SIZE) -> np.ndarray:
    """按人物框涂上各自的颜色，方便断言"哪个窗口里是谁"。"""
    frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    for bbox, color in people:
        x1, y1, x2, y2 = (int(value) for value in bbox)
        frame[y1:y2, x1:x2] = color
    return frame


def _provider(boxes_per_frame):
    """按帧序吐出检测框；序列走完后重复最后一帧。"""

    def _provide(frame, index):
        boxes = boxes_per_frame[min(index, len(boxes_per_frame) - 1)]
        return list(boxes)

    return _provide


def _build(detector, *, multi=POLICY, detect_interval: int = 1):
    return CropPipeline(
        RATIO,
        detector,
        smoother=BoxSmoother(RATIO, alpha=0.6),
        multi=multi,
        detect_interval=detect_interval,
    )


def _run(pipeline: CropPipeline, frames: list[np.ndarray]):
    outcomes = []
    for index, frame in enumerate(frames):
        outcomes.extend(pipeline.process(frame, index))
    outcomes.extend(pipeline.flush())
    return outcomes


def _count(image: np.ndarray, color, tol: int = 60) -> int:
    diff = np.abs(image.astype(np.int16) - np.array(color, dtype=np.int16))
    return int((diff.max(axis=2) <= tol).sum())


# ------------------------------------------------------------------ 基本行为
def test_two_people_become_two_windows(fake_detector) -> None:
    frames = [_frame() for _ in range(4)]
    pipeline = _build(fake_detector(_provider([[LEFT, RIGHT]])))
    outcomes = _run(pipeline, frames)

    assert len(outcomes) == 4
    for item in outcomes:
        assert item.frame.shape[:2] == (640, 360)  # 输出尺寸恒定（不变量 2）
        assert item.mode == "multi"
        assert item.is_multi
        assert item.window_count == 2
        # 每个窗口的取景框严格贴着"该窗口自己的比例"（不变量 3 的推广）
        for window in item.windows:
            width, height = window.size
            assert abs(window.box.aspect - width / height) < 1e-6

    # 竖屏 2 人 = 上下两格（["1", "2"]）
    assert outcomes[-1].windows[0].rect == (0, 0, 360, 320)
    assert outcomes[-1].windows[1].rect == (0, 320, 360, 320)

    assert pipeline.stats.multi_frames == 4
    assert pipeline.stats.windows_peak == 2
    assert pipeline.stats.modes["multi"] == 4


def test_each_window_shows_its_own_person(fake_detector) -> None:
    frames = [_frame() for _ in range(3)]
    outcomes = _run(_build(fake_detector(_provider([[LEFT, RIGHT]]))), frames)

    top, bottom = outcomes[-1].frame[:HALF_HEIGHT], outcomes[-1].frame[HALF_HEIGHT:]
    assert _count(top, COLOR_LEFT) > _count(top, COLOR_RIGHT)
    assert _count(bottom, COLOR_RIGHT) > _count(bottom, COLOR_LEFT)


def test_three_people_use_portrait_layout(fake_detector) -> None:
    frames = [_frame() for _ in range(3)]
    outcomes = _run(_build(fake_detector(_provider([[LEFT, MIDDLE, RIGHT]]))), frames)

    last = outcomes[-1]
    assert last.window_count == 3
    # 竖屏 3 人 = ["11", "23"]：上通栏大窗 + 下排两个
    assert [window.rect for window in last.windows] == [
        (0, 0, 360, 320),
        (0, 320, 180, 320),
        (180, 320, 180, 320),
    ]


#: 远景小人物：整条框是站姿全身，高度只占画面 ~10%
FAR_LEFT = (30.0, 120.0, 70.0, 230.0)
FAR_RIGHT = (250.0, 120.0, 290.0, 230.0)


def test_small_far_people_get_tight_upper_body_windows(fake_detector) -> None:
    """人物在画面里占比很小时，每个小窗口只装上半身，不再是全身 + 左右大片场景。"""
    people = ((FAR_LEFT, COLOR_LEFT), (FAR_RIGHT, COLOR_RIGHT))
    frames = [_frame(people=people) for _ in range(3)]
    outcomes = _run(_build(fake_detector(_provider([[FAR_LEFT, FAR_RIGHT]]))), frames)

    last = outcomes[-1]
    assert last.window_count == 2
    for window, bbox in zip(last.windows, (FAR_LEFT, FAR_RIGHT), strict=True):
        box = window.box
        rect = cell_ratio(window.rect)
        assert box.matches_ratio(rect, tol=1e-6)
        # 头顶在框内、小腿及以下不在框内 → 只切了上半身
        assert box.y <= bbox[1] + 1e-6
        assert box.y + box.h < bbox[3]
        # 比"整条人物框 + 四周留白"的老取景明显收紧
        loose = compute_target_box(bbox, FRAME_SIZE, rect, DEFAULT_FRAMING_PARAMS)
        assert box.area < loose.area * 0.5


def test_tiny_person_is_treated_as_background(fake_detector) -> None:
    """画面里只有"一个主要人物 + 一个路人"时不该开分屏。"""
    frames = [_frame() for _ in range(3)]
    outcomes = _run(_build(fake_detector(_provider([[LEFT, TINY]]))), frames)

    assert all(item.mode != "multi" for item in outcomes)
    assert outcomes[-1].windows == []


# ------------------------------------------------------------------ 开关与回退
def test_disabled_multi_never_opens_windows(fake_detector) -> None:
    frames = [_frame() for _ in range(3)]
    pipeline = _build(fake_detector(_provider([[LEFT, RIGHT]])), multi=None)
    outcomes = _run(pipeline, frames)

    assert all(item.mode != "multi" for item in outcomes)
    assert all(item.windows == [] for item in outcomes)
    assert pipeline.stats.multi_frames == 0
    assert pipeline.composer is None


def test_single_person_output_is_identical_with_multi_enabled(fake_detector) -> None:
    """默认开启多人分屏，也不该改变"画面里只有一个人"时的输出。"""
    frames = [_frame(people=((LEFT, COLOR_LEFT),)) for _ in range(4)]
    off = _run(_build(fake_detector(_provider([[LEFT]])), multi=None), frames)
    on = _run(_build(fake_detector(_provider([[LEFT]]))), frames)

    assert all(item.mode != "multi" for item in on)
    for before, after in zip(off, on, strict=True):
        assert np.array_equal(before.frame, after.frame)


def test_person_count_drop_falls_back_and_recovers(fake_detector) -> None:
    sequence = [[LEFT, RIGHT]] * 4 + [[LEFT]] * 3 + [[LEFT, RIGHT]] * 4
    pipeline = _build(fake_detector(_provider(sequence)))
    outcomes = _run(pipeline, [_frame() for _ in range(len(sequence))])

    modes = [item.mode for item in outcomes]
    assert modes[:4] == ["multi"] * 4
    assert all(mode != "multi" for mode in modes[4:7])
    assert modes[7:] == ["multi"] * 4
    # 座位轨迹保留着：中途单人时不清空，人回来自动接着跟
    assert pipeline.composer is not None and pipeline.composer.slots >= 2


def test_layout_ignores_single_frame_flicker(fake_detector) -> None:
    """人数偶尔多出一个（一帧）不该立刻重排布局。"""
    flicker: list[list] = []
    for _ in range(4):
        flicker.append([LEFT, RIGHT])
        flicker.append([LEFT, RIGHT, MIDDLE])
    settled = [[LEFT, RIGHT, MIDDLE]] * 4

    pipeline = _build(fake_detector(_provider(flicker + settled)))
    outcomes = _run(pipeline, [_frame() for _ in range(len(flicker) + len(settled))])

    assert [item.window_count for item in outcomes[: len(flicker)]] == [2] * len(flicker)
    assert outcomes[-1].window_count == 3


def test_tiny_windows_reduce_person_count(fake_detector) -> None:
    """输出分辨率很低时宁可少显示几个人，也不把窗口切到看不清。"""
    tiny_ratio = parse_ratio("9:16", target=(180, 320))  # 4 人时每格只有 90×160
    pipeline = CropPipeline(
        tiny_ratio,
        fake_detector(_provider([[LEFT, MIDDLE, RIGHT]])),
        smoother=BoxSmoother(tiny_ratio, alpha=0.6),
        multi=POLICY,
    )
    outcomes = _run(pipeline, [_frame() for _ in range(3)])

    # 3 人布局里最窄的格子只有 90 像素宽 → 降到 2 人
    assert outcomes[-1].window_count == 2
    assert outcomes[-1].mode == "multi"


# ------------------------------------------------------------------ 与其他特性
def test_detection_interval_and_interpolation(fake_detector) -> None:
    frames = [_frame() for _ in range(9)]
    pipeline = _build(fake_detector(_provider([[LEFT, RIGHT]])), detect_interval=3)
    outcomes = _run(pipeline, frames)

    assert [item.index for item in outcomes] == list(range(9))
    assert all(item.frame.shape[:2] == (640, 360) for item in outcomes)
    assert all(item.mode == "multi" for item in outcomes)
    assert pipeline.stats.interpolated_frames > 0


def test_annotate_draws_inside_each_window(fake_detector) -> None:
    """标注模式下，每个窗口里画的是"自己这个人"的框。"""
    gray = (128, 128, 128)  # 用灰度当人物，免得和绿色框线混淆
    people = ((LEFT, gray), (RIGHT, gray))
    frames = [_frame(people=people) for _ in range(2)]
    plain = _run(_build(fake_detector(_provider([[LEFT, RIGHT]]))), frames)

    pipeline = CropPipeline(
        RATIO,
        fake_detector(_provider([[LEFT, RIGHT]])),
        smoother=BoxSmoother(RATIO, alpha=0.6),
        multi=POLICY,
        annotate=True,
    )
    painted = _run(pipeline, frames)

    assert _count(plain[-1].frame, (0, 255, 0), tol=60) == 0  # 默认不画框
    assert _count(painted[-1].frame, (0, 255, 0), tol=60) > 0  # 开启后每个窗口都有框
    assert painted[-1].frame.shape[:2] == (640, 360)


def test_speaker_gets_the_first_window(fake_detector) -> None:
    """判定出说话人后，1 号窗口留给正在说话的人。"""

    class _Speaker:
        enabled = True
        switches = 0

        def biases(self, index, boxes, motions):
            return [1.0 if tuple(box) == RIGHT else 0.0 for box in boxes]

    class _Mouth:
        def activity(self, frame, boxes, keypoints=None):
            return [0.02] * len(boxes)

    pipeline = _build(fake_detector(_provider([[LEFT, RIGHT]])))
    pipeline.speaker = _Speaker()
    pipeline.mouth = _Mouth()
    pipeline.speaker_weight = 1.8

    outcomes = _run(pipeline, [_frame() for _ in range(3)])
    top = outcomes[-1].frame[:HALF_HEIGHT]

    assert _count(top, COLOR_RIGHT) > _count(top, COLOR_LEFT)


def test_order_score_follows_subject_score(fake_detector) -> None:
    """默认按空间位置排窗口；``order=score`` 则完全按主角打分（谁大谁坐 1 号窗）。"""
    big_right = (180.0, 20.0, 310.0, 235.0)
    small_left = (10.0, 60.0, 80.0, 200.0)
    people = ((small_left, COLOR_LEFT), (big_right, COLOR_RIGHT))
    frames = [_frame(people=people) for _ in range(3)]
    boxes = [[small_left, big_right]]

    spatial = _run(_build(fake_detector(_provider(boxes))), frames)
    scored = _run(
        _build(fake_detector(_provider(boxes)), multi=replace(POLICY, order="score")),
        frames,
    )

    assert _count(spatial[-1].frame[:HALF_HEIGHT], COLOR_LEFT) > _count(
        spatial[-1].frame[:HALF_HEIGHT], COLOR_RIGHT
    )
    assert _count(scored[-1].frame[:HALF_HEIGHT], COLOR_RIGHT) > _count(
        scored[-1].frame[:HALF_HEIGHT], COLOR_LEFT
    )


@pytest.mark.parametrize(
    ("ratio_name", "target"),
    [("9:16", (180, 320)), ("1:1", (240, 240)), ("16:9", (320, 180))],
)
def test_output_size_is_constant_for_any_ratio(fake_detector, ratio_name: str, target) -> None:
    ratio = parse_ratio(ratio_name, target=target)
    pipeline = CropPipeline(
        ratio,
        fake_detector(_provider([[LEFT, MIDDLE, RIGHT]])),
        smoother=BoxSmoother(ratio, alpha=0.6),
        multi=POLICY,
    )
    outcomes = _run(pipeline, [_frame() for _ in range(3)])

    expected = (ratio.target_height, ratio.target_width)
    assert {item.frame.shape[:2] for item in outcomes} == {expected}
    assert all(item.mode == "multi" for item in outcomes)


def test_set_fps_is_forwarded_to_every_window(fake_detector) -> None:
    """分屏时每个窗口的镜头也要按真实帧率校准，否则小窗同样会甩镜。"""
    pipeline = _build(fake_detector(_provider([[LEFT, RIGHT]])))
    pipeline.set_fps(60.0)
    _run(pipeline, [_frame() for _ in range(4)])

    composer = pipeline.composer
    assert composer is not None
    assert composer.smoother_params.fps == pytest.approx(60.0)
    slots = composer._slots
    assert slots  # 已经排过座，每个座位都有自己的平滑器
    assert all(slot.smoother.fps == pytest.approx(60.0) for slot in slots)
    assert all(
        slot.smoother.effective_max_speed == pytest.approx(slot.smoother.max_speed / 2.0)
        for slot in slots
    )
