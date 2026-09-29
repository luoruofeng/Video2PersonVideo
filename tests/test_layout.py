"""多人分屏布局：网格解析、像素划分、查表与配置加载（全部纯函数）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from video2personvideo.core.layout import (
    DEFAULT_MULTI_PERSON_POLICY,
    MAX_WINDOWS,
    LayoutError,
    MultiPersonPolicy,
    auto_grid,
    cell_ratio,
    compute_cell_rects,
    fit_capacity,
    load_multi_person_policy,
    orientation_group,
    parse_grid,
    parse_layout_entry,
    policy_from_mapping,
)
from video2personvideo.core.ratio import parse_ratio

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


# --------------------------------------------------------------------- 网格解析
def test_parse_grid_two_by_two() -> None:
    layout = parse_grid(["12", "34"])

    assert (layout.rows, layout.cols) == (2, 2)
    assert layout.capacity == 4
    assert [(cell.index, cell.row, cell.col) for cell in layout.cells] == [
        (1, 0, 0),
        (2, 0, 1),
        (3, 1, 0),
        (4, 1, 1),
    ]
    assert all(cell.is_single for cell in layout.cells)


def test_parse_grid_merged_cell() -> None:
    layout = parse_grid(["11", "23"])

    assert layout.capacity == 3
    main = layout.cells[0]
    assert (main.row, main.col, main.row_span, main.col_span) == (0, 0, 1, 2)
    assert main.area == 2
    assert layout.main_index == 1


def test_parse_grid_pads_short_rows() -> None:
    layout = parse_grid(["1", "23"])

    assert (layout.rows, layout.cols) == (2, 2)
    assert [(cell.index, cell.row, cell.col) for cell in layout.cells] == [
        (1, 0, 0),
        (2, 1, 0),
        (3, 1, 1),
    ]


def test_parse_grid_blank_cells_are_ignored() -> None:
    layout = parse_grid(["1.2", "..."])  # 只用来占位，第二行全是留白
    assert layout.capacity == 2
    assert (layout.cells[0].row, layout.cells[0].col) == (0, 0)
    assert (layout.cells[1].row, layout.cells[1].col) == (0, 2)


def test_parse_grid_rejects_irregular_shape() -> None:
    with pytest.raises(LayoutError, match="不是规则矩形"):
        parse_grid(["11", "12"])


def test_parse_grid_rejects_empty() -> None:
    with pytest.raises(LayoutError):
        parse_grid([])
    with pytest.raises(LayoutError):
        parse_grid(["..."])


def test_parse_grid_validates_weights() -> None:
    layout = parse_grid(["12", "34"], col_weights=[1.0, 2.0], row_weights=[1.0, 1.0])
    assert layout.col_weights == (1.0, 2.0)

    with pytest.raises(LayoutError):
        parse_grid(["12", "34"], col_weights=[1.0])
    with pytest.raises(LayoutError):
        parse_grid(["12", "34"], col_weights=[1.0, 0.0])


# --------------------------------------------------------------------- 自动网格
@pytest.mark.parametrize("count", range(1, MAX_WINDOWS + 1))
@pytest.mark.parametrize("aspect", [0.5625, 1.0, 1.7778, 2.3333])
def test_auto_grid_covers_every_person_without_holes(count: int, aspect: float) -> None:
    layout = auto_grid(count, aspect)

    assert layout.capacity == count
    # 网格里没有空洞：所有格子面积之和 = 行 × 列
    assert sum(cell.area for cell in layout.cells) == layout.rows * layout.cols


def test_auto_grid_prefers_square_cells_for_portrait() -> None:
    layout = auto_grid(4, 0.5625)  # 竖屏 4 人 → 2 行 1 列？不，应为 2×2 或 4×1
    # 竖屏下单元格宽高比 = aspect * rows / cols，应尽量接近 0.8
    cell_aspect = 0.5625 * layout.rows / layout.cols
    assert 0.4 <= cell_aspect <= 1.6


# ------------------------------------------------------------------- 像素划分
def test_compute_cell_rects_without_gap() -> None:
    layout = parse_grid(["1", "2"])
    rects = compute_cell_rects(layout, (1080, 1920), 0)

    assert rects == ((0, 0, 1080, 960), (0, 960, 1080, 960))


def _assert_tiles(rects, out_size) -> None:
    out_w, out_h = out_size
    canvas = np.zeros((out_h // 2, out_w // 2), dtype=np.int32)
    for order, (x, y, width, height) in enumerate(rects, start=1):
        assert width % 2 == 0 and height % 2 == 0, "格子宽高必须是偶数"
        assert x >= 0 and y >= 0 and x + width <= out_w and y + height <= out_h
        block = canvas[y // 2 : (y + height) // 2, x // 2 : (x + width) // 2]
        assert not block.any(), "格子之间不能重叠"
        block[:] = order
    # 除缝隙外应被完全覆盖：未覆盖的像素只能是缝隙（不超过格子边界外一圈）
    assert (canvas == 0).sum() <= (out_w // 2) * (out_h // 2) // 4


@pytest.mark.parametrize("ratio_name", ["9:16", "1:1", "16:9", "4:5", "21:9"])
@pytest.mark.parametrize("count", [2, 3, 4, 5, 6])
def test_compute_cell_rects_stay_inside_and_even(ratio_name: str, count: int) -> None:
    ratio = parse_ratio(ratio_name)
    policy = DEFAULT_MULTI_PERSON_POLICY
    layout = policy.layout_for(ratio, count)
    rects = compute_cell_rects(layout, ratio.target_size, policy.gap_px(ratio.target_size))

    assert len(rects) == count
    _assert_tiles(rects, ratio.target_size)


def test_compute_cell_rects_honours_column_weights() -> None:
    layout = parse_grid(["12"], col_weights=[3.0, 1.0])
    left, right = compute_cell_rects(layout, (1080, 1080), 0)

    assert left[2] == 810
    assert right[2] == 270
    assert left[2] + right[2] == 1080


def test_compute_cell_rects_clamps_gap_to_fit() -> None:
    layout = parse_grid(["123"])
    rects = compute_cell_rects(layout, (60, 60), 999)  # 缝隙大到放不下
    for _, _, width, height in rects:
        assert width >= 2 and height >= 2


# ---------------------------------------------------------------------- 比例
def test_cell_ratio_matches_rect_size() -> None:
    ratio = cell_ratio((0, 0, 540, 960))

    assert ratio.value == pytest.approx(540 / 960)
    assert ratio.target_size == (540, 960)


def test_cell_ratio_rejects_odd_size() -> None:
    with pytest.raises(ValueError):
        cell_ratio((0, 0, 541, 960))


def test_orientation_group() -> None:
    assert orientation_group(0.5625) == "portrait"
    assert orientation_group(1.0) == "square"
    assert orientation_group(1.7778) == "landscape"


# --------------------------------------------------------------------- 查表
@pytest.mark.parametrize("ratio_name", ["9:16", "1:1", "16:9", "4:5", "21:9", "3:5"])
@pytest.mark.parametrize("count", [2, 3, 4, 5, 6, 7])
def test_layout_for_always_returns_enough_windows(ratio_name: str, count: int) -> None:
    ratio = parse_ratio(ratio_name)
    layout = DEFAULT_MULTI_PERSON_POLICY.layout_for(ratio, count)

    assert layout.capacity == count
    assert len(compute_cell_rects(layout, ratio.target_size, 4)) == count


def test_layout_for_uses_exact_ratio_before_group() -> None:
    custom = parse_layout_entry(["1", "2"], 2, "custom")
    policy = MultiPersonPolicy(
        groups={},
        layouts={"square": {2: parse_layout_entry(["12"], 2, "square")}, "1:1": {2: custom}},
    )
    ratio = parse_ratio("1:1")

    layout = policy.layout_for(ratio, 2)

    assert layout.cells[0].col_span == 1  # 用到了 "1:1" 的覆盖，而不是族的 ["12"]
    assert layout.capacity == 2


def test_policy_limits_and_gap() -> None:
    policy = MultiPersonPolicy(max_persons=99, gap_ratio=0.01)

    assert policy.limit() == MAX_WINDOWS
    assert MultiPersonPolicy(max_persons=1).limit() == 2
    assert policy.gap_px((1080, 1920)) == 10  # 0.01 × 1080，偶数
    assert policy.gap_px((1081, 1920)) % 2 == 0


def test_fit_capacity_removes_holes() -> None:
    layout = parse_grid(["12", "34"])
    squeezed = fit_capacity(layout, 3)

    assert squeezed.capacity == 3
    assert squeezed.cells[-1].col_span == 2  # 4 号的地盘并给了同行邻居
    rects = compute_cell_rects(squeezed, (1080, 1080), 0)
    assert rects[2] == (0, 540, 1080, 540)


def test_policy_describe_mentions_state() -> None:
    assert "开启" in DEFAULT_MULTI_PERSON_POLICY.describe()


# ------------------------------------------------------------------- 配置加载
def test_repo_layout_config_loads_and_matches_builtin() -> None:
    policy = load_multi_person_policy(CONFIGS_DIR / "multi_person_layout.yaml")

    assert policy.source.endswith("multi_person_layout.yaml")
    for group, table in DEFAULT_MULTI_PERSON_POLICY.layouts.items():
        for count, layout in table.items():
            assert policy.layouts[group][count].cells == layout.cells


def test_layout_config_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_multi_person_policy(CONFIGS_DIR / "not-exist.yaml")


def test_policy_from_mapping_overrides_subset() -> None:
    policy = policy_from_mapping(
        {
            "gap_ratio": 0.02,
            "background": "#101010",
            "layouts": {"portrait": {2: ["1", "2"], 3: ["1", "2", "3"]}},
        }
    )

    assert policy.gap_ratio == pytest.approx(0.02)
    assert policy.background == (16, 16, 16)
    # 只覆盖了 2 / 3 人，其余档位仍沿用内置默认
    assert policy.layouts["portrait"][4].capacity == 4
    assert policy.layouts["landscape"][2].capacity == 2


def test_policy_from_mapping_rejects_bad_count() -> None:
    with pytest.raises(LayoutError, match="窗口数"):
        policy_from_mapping({"layouts": {"portrait": {3: ["12", "34"]}}})


def test_policy_from_mapping_rejects_bad_color() -> None:
    with pytest.raises(LayoutError):
        policy_from_mapping({"background": "12"})
    with pytest.raises(LayoutError):
        policy_from_mapping({"background": [1, 2]})
