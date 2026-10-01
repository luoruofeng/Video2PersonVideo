"""多人分屏"画面不停闪烁"的回归测试。

症状：分屏显示主要人物时画面一直闪 —— 每帧都在「两格分屏 ↔ 三格分屏」或
「分屏 ↔ 整屏单人」之间来回跳。

原因（两条叠加）：

1. 某一帧漏检一个人（漏检、转身、短暂遮挡），本帧的"合格人数"就少一个，
   布局**立刻**跟着换（窗口数量 / 位置 / 大小整块重排）；
2. 滞回值又被本帧人数"夹"了一下（``min(stabilize(ceiling), ceiling)``），
   于是人数一抖，滞回形同虚设。

修好之后：漏检的人由**还在跟踪的座位**顶上（人物框保持不动、画面继续播放），
布局要连续 ``switch_hold`` 帧（撤窗口则要 ``hold_frames`` 帧）确认才真的改变。
"""

from __future__ import annotations

from collections.abc import Sequence

from video2personvideo.core.layout import DEFAULT_MULTI_PERSON_POLICY
from video2personvideo.core.multi import MultiPersonComposer
from video2personvideo.core.ratio import resolve_ratio
from video2personvideo.core.smoothing import SmoothingParams
from video2personvideo.core.subject import Detection

RATIO = resolve_ratio("9:16")
FRAME = (1920.0, 1080.0)
FPS = 30.0

#: 三个人，身高都占画面 900/1080 ≈ 0.83，稳过默认的 0.73 阈值
LEFT = (300.0, 100.0, 700.0, 1000.0)
MIDDLE = (760.0, 100.0, 1160.0, 1000.0)
RIGHT = (1200.0, 100.0, 1600.0, 1000.0)


def _detection(bbox: tuple[float, float, float, float]) -> Detection:
    return Detection(bbox=bbox, confidence=0.9)


def _composer() -> MultiPersonComposer:
    composer = MultiPersonComposer(
        RATIO,
        policy=DEFAULT_MULTI_PERSON_POLICY,
        smoother_params=SmoothingParams(),
    )
    composer.set_fps(FPS)
    return composer


def _run(
    composer: MultiPersonComposer,
    per_round: Sequence[Sequence[Detection]],
    rounds: int,
) -> list[tuple[tuple[int, int, int, int], ...] | None]:
    """跑 ``rounds`` 轮（每轮取 ``per_round`` 里对应的检测），返回每轮的窗口矩形。"""
    seen: list[tuple[tuple[int, int, int, int], ...] | None] = []
    for index in range(rounds):
        plan = composer.update(FRAME, per_round[index % len(per_round)])
        seen.append(None if plan is None else plan.rects)
    return seen


# ------------------------------------------------------------------ 人数抖动
def test_single_frame_dropout_never_changes_the_layout() -> None:
    """三人画面里第 3 人每隔一帧漏检：布局必须**逐帧完全一致**（不闪）。"""
    composer = _composer()
    seen = _run(
        composer,
        [
            [_detection(LEFT), _detection(MIDDLE), _detection(RIGHT)],
            [_detection(LEFT), _detection(MIDDLE)],
        ],
        8,
    )

    assert all(rects is not None and len(rects) == 3 for rects in seen)
    assert len(set(seen)) == 1  # 每一帧的窗口矩形都一样 —— 画面不会闪


def test_dropout_never_falls_back_to_single_window() -> None:
    """两人画面里有一人漏检：绝不能出现「整屏单人 ↔ 两格分屏」的来回跳。"""
    composer = _composer()
    seen = _run(
        composer,
        [[_detection(LEFT), _detection(RIGHT)], [_detection(LEFT)]],
        8,
    )

    assert all(rects is not None and len(rects) == 2 for rects in seen)
    assert len(set(seen)) == 1


def test_missing_person_window_keeps_playing() -> None:
    """漏检的人在窗口里"保持不动"：镜头不跳、人物框沿用座位上的上一帧结果。"""
    composer = _composer()
    live = composer.update(FRAME, [_detection(LEFT), _detection(RIGHT)])
    assert live is not None
    before = [window.box for window in composer.windows(live)]

    plan = composer.update(FRAME, [_detection(LEFT)])  # 右边的人这一帧漏检
    assert plan is not None and plan.count == 2
    after = [window.box for window in composer.windows(plan)]
    # 漏检那个窗口的取景框仍然锁在原来的位置上（没有横扫、也没有换成别人的内容）
    assert (after[1].cx, after[1].cy) == (before[1].cx, before[1].cy)
    # 窗口名单里仍然是它的主人（画框标注 / 统计用）
    assert plan.subjects[1] == RIGHT


# ------------------------------------------------------------------ 滞回时序
def test_layout_downgrade_waits_for_hold_frames() -> None:
    """撤窗口要等 ``hold_frames`` 帧：漏检几十帧之前布局一直不变。"""
    composer = _composer()
    hold = int(composer.smoother_params.hold_frames)
    assert composer.update(FRAME, [_detection(LEFT), _detection(RIGHT)]) is not None

    held = 0
    while composer.update(FRAME, [_detection(LEFT)]) is not None:
        held += 1
        assert held <= hold  # 保持期一到就退出分屏，不会无限拖
    assert held >= 2  # 一帧漏检不足以撤掉窗口


def test_layout_upgrade_waits_for_switch_hold() -> None:
    """加窗口要等 ``switch_hold`` 帧：路人一闪而过不会立刻占一个格子。"""
    composer = _composer()
    hold = int(DEFAULT_MULTI_PERSON_POLICY.switch_hold)
    assert composer.update(FRAME, [_detection(LEFT), _detection(RIGHT)]).count == 2

    three = [_detection(LEFT), _detection(MIDDLE), _detection(RIGHT)]
    counts = []
    for _ in range(hold + 2):
        plan = composer.update(FRAME, three)
        assert plan is not None
        counts.append(plan.count)

    assert counts[: hold - 1] == [2] * (hold - 1)  # 先稳住两格
    assert counts[hold - 1] == 3  # 连续确认之后才加第三个窗口
    assert counts[-1] == 3


def test_layout_returns_to_single_window_after_a_long_dropout() -> None:
    """长时间只剩一个人之后确实回到单窗口（保持期不是"永不换布局"）。"""
    composer = _composer()
    assert composer.update(FRAME, [_detection(LEFT), _detection(RIGHT)]) is not None

    hold = int(composer.smoother_params.hold_frames)
    for _ in range(hold + 1):
        plan = composer.update(FRAME, [_detection(LEFT)])
    assert plan is None  # 人真的走了 → 交回单窗口逻辑

    # 回到单窗口之后再出现第二个人：立刻恢复分屏（不必重新等保持期）
    assert composer.update(FRAME, [_detection(LEFT), _detection(RIGHT)]) is not None


def test_single_person_never_waits_for_a_split() -> None:
    """画面里本来就只有一个主要人物（或一个人都没有）时立刻走单窗口，不做无谓等待。"""
    composer = _composer()
    assert composer.update(FRAME, []) is None
    assert composer.update(FRAME, [_detection(LEFT)]) is None
    assert composer.update(FRAME, [_detection(LEFT)]) is None
