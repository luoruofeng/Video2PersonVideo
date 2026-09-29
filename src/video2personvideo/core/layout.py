"""多人分屏布局：让一帧里同时显示多个人的上半身小窗口。

画面里出现**两个及以上主要人物**时，每人给一个"上半身"小窗口；小窗口怎么排，
由「输出长宽比 + 人数」在配置文件（``configs/multi_person_layout.yaml``）里查表决定：

* ``layouts.<族名或比例名>.<人数>``：窗口网格。用字符网格描述，例如竖屏 3 人
  ``["11", "23"]`` 表示"上面一个通栏大窗（1 号），下面左右两个小窗（2、3 号）"，
  同一个数字占多格即合并单元格；``.`` / ``_`` 表示留白（填底色）。
* 数字编号就是**窗口优先级**：1 号优先给"正在说话的人 / 主角"，其余按空间顺序填。
* 表里查不到的（人数超出、自定义比例）会按画面宽高比自动生成一个尽量方的网格，
  所以任何输入都不会崩，只是排得不如查表精致。

本模块是纯数据 / 纯函数层：只做网格解析、查表、像素划分与比例推导，
不依赖 OpenCV / numpy / Qt，可脱离视频单测。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import gcd
from pathlib import Path
from typing import Any

import yaml

from ..utils.logger import get_logger
from .ratio import AspectRatio, even_floor, validate_out_size

logger = get_logger(__name__)

#: 仓库自带的多人布局配置文件（不存在时用内置默认布局）
DEFAULT_LAYOUT_CONFIG_PATH = Path("configs") / "multi_person_layout.yaml"

#: 半身小窗口的理想宽高比（宽 / 高）：人像本身竖长，0.8 附近既容得下头肩、
#: 两侧又不会空得太多；自动生成网格时以它为目标
IDEAL_CELL_ASPECT = 0.8

#: 单格最小边长（像素）：再小就只剩马赛克，宁可不显示那么多人
MIN_CELL_PX = 96

#: 自动生成网格时，"浪费一格"折算成的代价（相对"格子比例不理想"的代价）。
#: 给得足够大，才会在"5 人用 3×2（浪费 1 格但格子比例好）"和
#: "5 人用 5×1（不浪费但格子扁得没法看）"之间选前者。
WASTE_PENALTY = 0.35

#: 支持的窗口上限（网格编号只用 1~9）
MAX_WINDOWS = 9

#: 网格里的留白字符
_BLANK_CHARS = frozenset("._- \t")

#: 内置默认比例族：比例名 → 族名
DEFAULT_GROUPS: dict[str, str] = {
    "16:9": "landscape",
    "4:3": "landscape",
    "3:2": "landscape",
    "5:4": "landscape",
    "21:9": "landscape",
    "2:1": "landscape",
    "1:1": "square",
    "9:16": "portrait",
    "4:5": "portrait",
    "3:4": "portrait",
    "2:3": "portrait",
}

#: 内置默认布局表（与 ``configs/multi_person_layout.yaml`` 保持一致）
_DEFAULT_TABLES: dict[str, dict[int, tuple[str, ...]]] = {
    # 竖屏：上下堆叠为主，半身窗口天然竖长
    "portrait": {
        2: ("1", "2"),
        3: ("11", "23"),
        4: ("12", "34"),
        5: ("11", "23", "45"),
        6: ("12", "34", "56"),
    },
    # 方形：左右并排为主，4 人正好 2×2
    "square": {
        2: ("12",),
        3: ("11", "23"),
        4: ("12", "34"),
        5: ("112", "345"),
        6: ("123", "456"),
    },
    # 横屏：左大右小（3 人）/ 上下两排（4 人及以上）
    "landscape": {
        2: ("12",),
        3: ("12", "13"),
        4: ("12", "34"),
        5: ("112", "345"),
        6: ("123", "456"),
    },
}


class LayoutError(ValueError):
    """布局配置非法：网格不是规则矩形、权重长度不符、人数与格子数不匹配等。"""


# --------------------------------------------------------------------------- 网格


@dataclass(frozen=True, slots=True)
class CellSpec:
    """一个窗口在网格里的位置（行列从 0 开始，``span`` 至少为 1）。"""

    index: int
    row: int
    col: int
    row_span: int = 1
    col_span: int = 1

    @property
    def area(self) -> int:
        """占了几格（用于判断是不是"大窗"）。"""
        return self.row_span * self.col_span

    @property
    def is_single(self) -> bool:
        return self.row_span == 1 and self.col_span == 1


@dataclass(frozen=True, slots=True)
class GridLayout:
    """一块窗口网格：行列数、每个窗口占哪几格、（可选的）行列权重。"""

    rows: int
    cols: int
    cells: tuple[CellSpec, ...]
    label: str = ""
    row_weights: tuple[float, ...] = ()
    col_weights: tuple[float, ...] = ()

    @property
    def capacity(self) -> int:
        """能放几个窗口。"""
        return len(self.cells)

    @property
    def main_index(self) -> int:
        """最大窗口的编号（并列时取编号最小的）。"""
        if not self.cells:
            return 1
        smallest = min(cell.index for cell in self.cells)
        biggest = max(cell.area for cell in self.cells)
        candidates = [cell.index for cell in self.cells if cell.area == biggest]
        return min(candidates, default=smallest)

    def describe(self) -> str:
        return f"{self.label or '自动布局'}（{self.rows}×{self.cols}，{self.capacity} 窗）"


def parse_grid(
    lines: Sequence[str],
    *,
    row_weights: Sequence[float] | None = None,
    col_weights: Sequence[float] | None = None,
    label: str = "",
) -> GridLayout:
    """把字符网格解析成 :class:`GridLayout`（纯函数）。

    ``lines`` 每行一个字符串，字符为窗口编号（``1``~``9``），相同编号的若干格
    必须构成一个矩形；``.`` / ``_`` / ``-`` / 空格表示留白。行长度允许不一致
    （短的按留白补齐）。

    :raises LayoutError: 网格为空、窗口不是规则矩形或权重个数不符
    """
    rows_text = [str(line) for line in lines if str(line).strip()]
    if not rows_text:
        raise LayoutError("布局网格不能为空")

    cols = max(len(line) for line in rows_text)
    grid = [line.ljust(cols, ".") for line in rows_text]

    positions: dict[str, list[tuple[int, int]]] = {}
    order: list[str] = []
    for row, line in enumerate(grid):
        for col, char in enumerate(line):
            if char in _BLANK_CHARS:
                continue
            if char not in positions:
                positions[char] = []
                order.append(char)
            positions[char].append((row, col))

    if not positions:
        raise LayoutError("布局网格里没有任何窗口")
    if len(positions) > MAX_WINDOWS:
        raise LayoutError(f"窗口数（{len(positions)}）超过上限 {MAX_WINDOWS}")

    cells: list[CellSpec] = []
    for index, char in enumerate(order, start=1):
        points = positions[char]
        top = min(row for row, _ in points)
        bottom = max(row for row, _ in points)
        left = min(col for _, col in points)
        right = max(col for _, col in points)
        expected = (bottom - top + 1) * (right - left + 1)
        if expected != len(points):
            raise LayoutError(
                f"窗口 {char!r} 不是规则矩形（占了 {len(points)} 格，外接矩形却是 {expected} 格）"
            )
        cells.append(
            CellSpec(index, top, left, bottom - top + 1, right - left + 1)
        )

    return GridLayout(
        rows=len(grid),
        cols=cols,
        cells=tuple(cells),
        label=label,
        row_weights=_normalize_weights(row_weights, len(grid), "行"),
        col_weights=_normalize_weights(col_weights, cols, "列"),
    )


def auto_grid(count: int, aspect_value: float = 1.0, *, label: str = "") -> GridLayout:
    """按画面宽高比自动生成一个"尽量方"的网格（查不到表时的兜底）。

    目标：让每个小窗口的宽高比尽量接近 :data:`IDEAL_CELL_ASPECT`，
    同时尽量少浪费格子（多余的格子由最后一个窗口吃掉，不留空洞）。
    """
    count = max(min(int(count), MAX_WINDOWS), 1)
    if count == 1:
        return parse_grid(["1"], label=label or "单窗口")

    best: tuple[float, int, int] | None = None
    for rows in range(1, count + 1):
        cols = math.ceil(count / rows)
        last_row = count - (rows - 1) * cols
        if rows * cols < count or rows * cols > MAX_WINDOWS or last_row <= 0:
            # 最后一行必须至少有一个窗口，否则多余的空格无法被"同一行合并"吃掉
            continue
        cell_aspect = max(aspect_value, 1e-3) * rows / cols
        cost = abs(math.log(cell_aspect / IDEAL_CELL_ASPECT)) + WASTE_PENALTY * (
            rows * cols - count
        )
        if best is None or cost < best[0]:
            best = (cost, rows, cols)
    assert best is not None  # count ≤ 9 时必然有解（1×count / count×1 都在范围内）
    _, rows, cols = best

    grid = [["." for _ in range(cols)] for _ in range(rows)]
    placed = 0
    for row in range(rows):
        for col in range(cols):
            if grid[row][col] != "." or placed >= count:
                continue
            placed += 1
            char = str(placed)
            grid[row][col] = char
            if placed == count:  # 最后一个窗口吃掉本行剩余格子，避免空洞
                for extra in range(col + 1, cols):
                    if grid[row][extra] == ".":
                        grid[row][extra] = char
    return parse_grid(["".join(row) for row in grid], label=label or f"自动 {rows}×{cols}")


# ------------------------------------------------------------------ 像素划分


def compute_cell_rects(
    layout: GridLayout,
    out_size: tuple[int, int],
    gap: int = 0,
) -> tuple[tuple[int, int, int, int], ...]:
    """把网格划分成输出画布上的像素矩形（顺序与 ``layout.cells`` 一致）。

    保证：每个矩形的宽高都是**偶数**、互不重叠、完全落在画布内，
    所有矩形加缝隙之和恰好等于 ``out_size``（不变量：输出尺寸恒定）。
    """
    out_w, out_h = validate_out_size(out_size)
    gap = _clamp_gap(gap, out_w, out_h, layout.rows, layout.cols)

    col_px = _distribute(out_w, layout.col_weights, layout.cols, gap)
    row_px = _distribute(out_h, layout.row_weights, layout.rows, gap)
    col_start = _prefix(col_px, gap)
    row_start = _prefix(row_px, gap)

    rects: list[tuple[int, int, int, int]] = []
    for cell in layout.cells:
        x = col_start[cell.col]
        y = row_start[cell.row]
        width = sum(col_px[cell.col : cell.col + cell.col_span]) + gap * (cell.col_span - 1)
        height = sum(row_px[cell.row : cell.row + cell.row_span]) + gap * (cell.row_span - 1)
        rects.append((x, y, width, height))
    return tuple(rects)


def cell_ratio(rect: tuple[int, int, int, int], name: str = "") -> AspectRatio:
    """把某个格子的像素矩形变成"该窗口自己的输出比例"。

    多人分屏里每个格子形状不同，取景框必须按**格子比例**来算，
    这样切片后才是等比缩放（不拉伸变形）。
    """
    _, _, width, height = (int(value) for value in rect)
    width, height = validate_out_size((width, height))
    divisor = gcd(width, height) or 1
    return AspectRatio(
        name or f"{width // divisor}:{height // divisor}",
        width // divisor,
        height // divisor,
        width,
        height,
    )


def fit_capacity(layout: GridLayout, count: int) -> GridLayout:
    """把窗口数多于 ``count`` 的布局收成"只用前 ``count`` 个窗口"。

    多余窗口的格子**留给相邻窗口**（同一行/列内并入编号更小的那个），
    因此不会出现空洞，也不会因为配置写多了一格就露底色。
    """
    count = max(int(count), 1)
    if layout.capacity <= count:
        return layout

    kept = [cell for cell in layout.cells if cell.index <= count]
    dropped = [cell for cell in layout.cells if cell.index > count]
    cells = list(kept)
    for extra in dropped:
        # 先找同一行、再有同一列的邻居来"吸收"这块地
        host = _pick_host(cells, extra, same_row=True) or _pick_host(cells, extra, same_row=False)
        if host is None:  # pragma: no cover - 网格必然有邻居
            continue
        position = cells.index(host)
        cells[position] = _merge_cells(host, extra)
    return GridLayout(
        rows=layout.rows,
        cols=layout.cols,
        cells=tuple(sorted(cells, key=lambda item: item.index)),
        label=layout.label,
        row_weights=layout.row_weights,
        col_weights=layout.col_weights,
    )


def _pick_host(cells: Sequence[CellSpec], extra: CellSpec, *, same_row: bool) -> CellSpec | None:
    best: CellSpec | None = None
    best_gap = None
    for cell in cells:
        if same_row and cell.row != extra.row:
            continue
        if not same_row and cell.col != extra.col:
            continue
        gap = abs(cell.col - extra.col) + abs(cell.row - extra.row)
        if best_gap is None or gap < best_gap:
            best, best_gap = cell, gap
    return best


def _merge_cells(first: CellSpec, second: CellSpec) -> CellSpec:
    top = min(first.row, second.row)
    bottom = max(first.row + first.row_span, second.row + second.row_span)
    left = min(first.col, second.col)
    right = max(first.col + first.col_span, second.col + second.col_span)
    return CellSpec(
        first.index,
        top,
        left,
        bottom - top,
        right - left,
    )


def _prefix(sizes: Sequence[int], gap: int) -> list[int]:
    starts: list[int] = []
    cursor = 0
    for size in sizes:
        starts.append(cursor)
        cursor += int(size) + gap
    return starts


def _clamp_gap(gap: int, out_w: int, out_h: int, rows: int, cols: int) -> int:
    """把缝隙夹到"每格还剩至少 :data:`MIN_CELL_PX`"的安全范围内（并保持偶数）。"""
    gap = max(int(gap), 0)
    gap -= gap % 2
    limit = out_w * out_h
    if cols > 1:
        limit = min(limit, out_w // cols - 2)
    if rows > 1:
        limit = min(limit, out_h // rows - 2)
    limit = max(min(limit, out_w, out_h), 0)
    limit -= limit % 2
    return min(gap, limit)


def _distribute(
    total: int, weights: Sequence[float], count: int, gap: int
) -> tuple[int, ...]:
    """把 ``total`` 像素按权重分给 ``count`` 份（含 ``count - 1`` 个缝隙）。

    每份都是偶数且至少 2 像素，加起来**恰好**等于 ``total - gap × (count - 1)``。
    """
    if count <= 1:
        return (total,)

    usable = total - gap * (count - 1)
    usable = max(usable, 2 * count)
    usable -= usable % 2

    if weights and len(weights) == count:
        weight_sum = float(sum(weights))
    else:
        weights = (1.0,) * count
        weight_sum = float(count)

    sizes = [max(even_floor(usable * float(weight) / weight_sum), 2) for weight in weights]

    diff = usable - sum(sizes)  # 必然是偶数
    while diff > 0:
        for position in range(count):
            if diff <= 0:
                break
            sizes[position] += 2
            diff -= 2
    while diff < 0:
        for position in range(count):
            if diff >= 0:
                break
            if sizes[position] > 2:
                sizes[position] -= 2
                diff += 2
    return tuple(sizes)


def _normalize_weights(
    weights: Sequence[float] | None, count: int, kind: str
) -> tuple[float, ...]:
    if weights is None or len(weights) == 0:
        return ()
    values = tuple(float(item) for item in weights)
    if len(values) != count:
        raise LayoutError(f"{kind}权重个数（{len(values)}）必须等于{kind}数（{count}）")
    if any(value <= 0.0 for value in values):
        raise LayoutError(f"{kind}权重必须为正数：{values}")
    return values


def orientation_group(aspect_value: float) -> str:
    """按宽高比归类比例族：竖屏 / 方形 / 横屏。"""
    if aspect_value < 0.95:
        return "portrait"
    if aspect_value > 1.05:
        return "landscape"
    return "square"


# ------------------------------------------------------------------ 布局策略


@dataclass(frozen=True, eq=False, slots=True)
class MultiPersonPolicy:
    """多人分屏的完整策略：开关、上限、外观与布局表。"""

    #: 是否启用多人分屏（关掉则永远只显示一个人）
    enabled: bool = True
    #: 最多同时显示几个主要人物（超出的按主角打分取舍）
    max_persons: int = 4
    #: 窗口缝隙（相对输出短边的比例）
    gap_ratio: float = 0.006
    #: 缝隙与留白的底色（BGR）
    background: tuple[int, int, int] = (18, 18, 18)
    #: 人物排列方式：``spatial`` = 先说话人、再按空间位置；``score`` = 完全按主角打分
    order: str = "spatial"
    #: 人数变化需要连续多少帧才换布局（防止检测抖动导致窗口闪烁）
    switch_hold: int = 3
    #: 比例名 → 族名
    groups: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_GROUPS))
    #: 族名 / 比例名 → {人数: 网格}
    layouts: Mapping[str, Mapping[int, GridLayout]] = field(default_factory=dict)
    #: 供日志追溯的配置来源
    source: str = ""

    # ------------------------------------------------------------- 只读派生
    def group_of(self, ratio: AspectRatio) -> str:
        """这个比例属于哪个族（没有显式归属时按宽高比判断）。"""
        return self.groups.get(ratio.name) or orientation_group(ratio.value)

    def table_for(self, ratio: AspectRatio) -> Mapping[int, GridLayout]:
        """取该比例对应的布局表（先精确匹配比例名，再退到族名）。"""
        return self.layouts.get(ratio.name) or self.layouts.get(self.group_of(ratio)) or {}

    def gap_px(self, out_size: tuple[int, int]) -> int:
        """缝隙的像素值（按输出短边换算，且保持偶数）。"""
        short_side = min(int(out_size[0]), int(out_size[1]))
        gap = int(round(self.gap_ratio * short_side))
        gap -= gap % 2
        return max(gap, 0)

    def limit(self) -> int:
        """生效的人数上限（至少 2 人，最多 :data:`MAX_WINDOWS`）。"""
        return max(min(int(self.max_persons), MAX_WINDOWS), 2)

    # ------------------------------------------------------------- 查表
    def layout_for(self, ratio: AspectRatio, count: int) -> GridLayout:
        """「比例 + 人数」→ 布局；查不到就按画面宽高比自动生成。"""
        count = max(min(int(count), MAX_WINDOWS), 1)
        layout = self.table_for(ratio).get(count)
        if layout is None:
            layout = auto_grid(count, ratio.value, label=f"自动 {count} 人")
        return fit_capacity(layout, count)

    def describe(self) -> str:
        origin = f"，来自 {self.source}" if self.source else ""
        return (
            f"多人分屏：{'开启' if self.enabled else '关闭'}"
            f"（最多 {self.limit()} 人，排列 {self.order}，切换确认 {self.switch_hold} 帧{origin}）"
        )


def _compile_tables(
    tables: Mapping[str, Mapping[int, Sequence[str]]]
) -> dict[str, dict[int, GridLayout]]:
    compiled: dict[str, dict[int, GridLayout]] = {}
    for name, table in tables.items():
        compiled[name] = {
            int(count): parse_grid(grid, label=f"{name}·{count} 人")
            for count, grid in table.items()
        }
    return compiled


#: 内置默认策略（不依赖任何文件，保证开箱可用 / 打包后也能跑）
DEFAULT_MULTI_PERSON_POLICY = MultiPersonPolicy(
    groups=dict(DEFAULT_GROUPS),
    layouts=_compile_tables(_DEFAULT_TABLES),
    source="内置默认",
)


def parse_layout_entry(spec: Any, count: int, name: str = "") -> GridLayout:
    """解析配置里的一条布局：字符串列表，或 ``{grid: [...], rows: [...], cols: [...]}``。"""
    label = f"{name}·{count} 人" if name else f"{count} 人"
    if isinstance(spec, Mapping):
        grid = spec.get("grid")
        if grid is None:
            raise LayoutError(f"{label} 的布局缺少 grid 字段")
        layout = parse_grid(
            grid, row_weights=spec.get("rows"), col_weights=spec.get("cols"), label=label
        )
    elif isinstance(spec, (list, tuple)):
        layout = parse_grid(spec, label=label)
    else:
        raise LayoutError(f"{label} 的布局格式无法识别（应为字符串列表或含 grid 的映射）")

    if layout.capacity != count:
        raise LayoutError(f"{label} 的窗口数（{layout.capacity}）与人数（{count}）不一致")
    return layout


def policy_from_mapping(data: Mapping[str, Any], *, source: str = "") -> MultiPersonPolicy:
    """从 YAML 内容构造策略：文件里没写的项沿用内置默认。"""
    groups = dict(DEFAULT_GROUPS)
    raw_groups = data.get("groups") or {}
    if not isinstance(raw_groups, Mapping):
        raise LayoutError("groups 必须是「比例名: 族名」的键值对")
    for name, group in raw_groups.items():
        groups[str(name)] = str(group)

    layouts: dict[str, dict[int, GridLayout]] = {
        name: dict(table) for name, table in DEFAULT_MULTI_PERSON_POLICY.layouts.items()
    }
    raw_layouts = data.get("layouts") or {}
    if not isinstance(raw_layouts, Mapping):
        raise LayoutError("layouts 必须是「族名 / 比例名: {人数: 网格}」的键值对")
    for name, table in raw_layouts.items():
        if not isinstance(table, Mapping):
            raise LayoutError(f"layouts.{name} 必须是「人数: 网格」的键值对")
        entries = layouts.setdefault(str(name), {})
        for count_text, spec in table.items():
            entries[int(count_text)] = parse_layout_entry(spec, int(count_text), str(name))

    return MultiPersonPolicy(
        enabled=_to_bool(data.get("enabled"), DEFAULT_MULTI_PERSON_POLICY.enabled),
        max_persons=int(data.get("max_persons", DEFAULT_MULTI_PERSON_POLICY.max_persons)),
        gap_ratio=float(data.get("gap_ratio", DEFAULT_MULTI_PERSON_POLICY.gap_ratio)),
        background=_parse_color(data.get("background"), DEFAULT_MULTI_PERSON_POLICY.background),
        order=str(data.get("order", DEFAULT_MULTI_PERSON_POLICY.order)),
        switch_hold=int(data.get("switch_hold", DEFAULT_MULTI_PERSON_POLICY.switch_hold)),
        groups=groups,
        layouts=layouts,
        source=source,
    )


def load_multi_person_policy(path: str | Path | None = None) -> MultiPersonPolicy:
    """加载多人布局策略。

    ``path`` 为 ``None`` 时用仓库默认路径（不存在就用内置默认，保证开箱可用）；
    显式给了路径但文件不存在时直接报错（用户明确指定了却找不到，不该静默降级）。
    """
    target = Path(path) if path is not None else DEFAULT_LAYOUT_CONFIG_PATH
    if not target.exists():
        if path is not None:
            raise FileNotFoundError(f"布局配置文件不存在：{target}")
        logger.debug("未找到多人布局配置文件，使用内置默认布局")
        return DEFAULT_MULTI_PERSON_POLICY

    with target.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    if not isinstance(data, Mapping):
        raise LayoutError(f"布局配置文件格式错误，应为键值对：{target}")
    policy = policy_from_mapping(data, source=str(target))
    logger.debug("已加载多人布局配置 %s（%s）", target, policy.describe())
    return policy


def _to_bool(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on", "是", "开"}:
        return True
    if text in {"0", "false", "no", "n", "off", "否", "关"}:
        return False
    logger.warning("布局配置里的布尔值 %r 无法识别，按默认值处理", value)
    return bool(default)


def _parse_color(value: Any, default: tuple[int, int, int]) -> tuple[int, int, int]:
    """解析底色：``[B, G, R]`` 列表，或 ``#RRGGBB`` / ``#RGB`` 十六进制。"""
    if value is None:
        return default
    if isinstance(value, str):
        return _parse_hex_color(value)
    try:
        items = [int(max(0, min(255, float(item)))) for item in value]
    except (TypeError, ValueError) as exc:
        raise LayoutError(f"底色必须是 3 个 0~255 的整数（BGR）或 #RRGGBB：{value!r}") from exc
    if len(items) != 3:
        raise LayoutError(f"底色必须是 3 个 0~255 的整数（BGR）：{value!r}")
    return (items[0], items[1], items[2])


def _parse_hex_color(text: str) -> tuple[int, int, int]:
    raw = text.strip().lstrip("#")
    if len(raw) == 3:
        raw = "".join(char * 2 for char in raw)
    if len(raw) != 6 or not _is_hex(raw):
        raise LayoutError(f"底色无法识别：{text!r}（示例：#121212 或 [18, 18, 18]）")
    number = int(raw, 16)
    return ((number & 0xFF), ((number >> 8) & 0xFF), ((number >> 16) & 0xFF))


def _is_hex(text: str) -> bool:
    return all(char in "0123456789abcdefABCDEF" for char in text)


__all__ = [
    "DEFAULT_GROUPS",
    "DEFAULT_LAYOUT_CONFIG_PATH",
    "DEFAULT_MULTI_PERSON_POLICY",
    "IDEAL_CELL_ASPECT",
    "MAX_WINDOWS",
    "MIN_CELL_PX",
    "CellSpec",
    "GridLayout",
    "LayoutError",
    "MultiPersonPolicy",
    "auto_grid",
    "cell_ratio",
    "compute_cell_rects",
    "fit_capacity",
    "load_multi_person_policy",
    "orientation_group",
    "parse_grid",
    "parse_layout_entry",
    "policy_from_mapping",
]
