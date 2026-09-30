"""取景框数据结构与构图策略。

``CropBox`` 描述"从源画面里切哪一块"，``compute_target_box`` 描述
"给定人物框该怎么切"。全部是纯函数：不依赖 OpenCV / YOLO / Qt，
可脱离视频做单元测试。

三条硬性约定（见 TODO.md 的不变量）：

* 取景框宽高比恒等于目标比例（不变量 3）；
* 只平移不缩放地夹取到画面内，绝不产生黑边（不变量 4）；
* 取景框尺寸有上下限保护，避免极端放大或裁出畸形画面（M1-3）。

**唯一例外**：画面里没有主要人物且显示方式为「全画面适配 / 全景+特写」
（:data:`MODE_FIT` / :data:`MODE_TILES`）时不存在"取景框"这个概念 —— 此时
:class:`CropBox` 被借用来表示"从源画面里取出的那一块"（比例会从目标比例连续
过渡到源比例），真正的画面合成由 :mod:`~video2personvideo.core.crop` 完成。
输出尺寸恒定（不变量 2）与"不出现黑边 / 变形 / 越界"（不变量 4）依然成立。
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .pose import BodyAnchors
from .ratio import AspectRatio, even_floor, max_crop_size

#: 构图分档
MODE_CLOSEUP = "closeup"
MODE_HALFBODY = "halfbody"
MODE_FULLBODY = "fullbody"
MODE_CENTER = "center"
MODE_HOLD = "hold"
#: 空镜巡视：画面里没有主要人物时，取景框沿"画面比取景框多出来的那一侧"缓慢往返，
#: 把横屏素材的全部内容都带到竖屏输出里（不丢左右 / 上下两边）
MODE_SCAN = "scan"
#: 全画面适配：画面里没有主要人物时，把**整幅画面**等比缩小完整放进输出画布
#: （上下 / 左右留出的部分用同一帧的模糊放大版填满），一帧看全、镜头完全不移动。
#: 注意：这是唯一一种"取景框比例不等于目标比例"的模式（此时不存在取景框，
#: 直接由裁剪层合成），文件末尾的不变量说明里有专门标注。
MODE_FIT = "fit"
#: 全景 + 特写：没有主要人物时，上方通栏放整幅画面，下方纵向排列若干"次要小人物"的
#: 半身特写 —— 既保住环境全貌，又能看清被主角阈值过滤掉的远景人物
MODE_TILES = "tiles"
MODE_ANNOTATE = "annotate"
MODE_AUTO = "auto"
#: 多人分屏：一帧里同时显示多个人的上半身小窗口
MODE_MULTI = "multi"

#: 模式 → 中文展示名（GUI / 日志用）
MODE_LABELS: dict[str, str] = {
    MODE_CLOSEUP: "头像/特写",
    MODE_HALFBODY: "半身",
    MODE_FULLBODY: "全身",
    MODE_CENTER: "画面居中兜底",
    MODE_HOLD: "保持上一帧",
    MODE_SCAN: "空镜巡视",
    MODE_FIT: "全画面适配",
    MODE_TILES: "全景+特写",
    MODE_ANNOTATE: "画框标注",
    MODE_MULTI: "多人分屏",
    MODE_AUTO: "自动",
}


def mode_label(mode: str) -> str:
    """把内部模式名翻译成中文展示名。"""
    return MODE_LABELS.get(mode, mode)


@dataclass(slots=True)
class CropBox:
    """源画面坐标系下的取景框（浮点像素，左上角为原点）。"""

    x: float
    y: float
    w: float
    h: float
    mode: str = MODE_AUTO

    # ------------------------------------------------------------- 只读属性
    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def aspect(self) -> float:
        return self.w / self.h if self.h else 0.0

    @property
    def center(self) -> tuple[float, float]:
        return self.cx, self.cy

    @property
    def size(self) -> tuple[float, float]:
        return self.w, self.h

    def with_mode(self, mode: str) -> CropBox:
        return replace(self, mode=mode)

    def matches_ratio(self, ratio: AspectRatio, tol: float = 1e-6) -> bool:
        """比例自检：调试期或测试中开启断言用。"""
        return self.h > 0 and abs(self.aspect - ratio.value) <= tol

    # ------------------------------------------------------------- 坐标修正
    def clamp_to(self, frame_w: float, frame_h: float) -> CropBox:
        """越界时**平移**（不缩放），保证整个框落在画面内。

        :raises ValueError: 画面尺寸非法
        """
        frame_w, frame_h = float(frame_w), float(frame_h)
        if frame_w <= 0 or frame_h <= 0:
            raise ValueError(f"画面尺寸必须为正：{frame_w}x{frame_h}")
        x = min(max(self.x, 0.0), max(frame_w - self.w, 0.0))
        y = min(max(self.y, 0.0), max(frame_h - self.h, 0.0))
        if x == self.x and y == self.y:
            return self
        return replace(self, x=x, y=y)

    def fit_to(self, frame_w: float, frame_h: float) -> CropBox:
        """先保证框不超过画面，再平移夹取。缩放时严格保持宽高比。"""
        frame_w, frame_h = float(frame_w), float(frame_h)
        scale = 1.0
        if self.w > frame_w:
            scale = min(scale, frame_w / self.w)
        if self.h > frame_h:
            scale = min(scale, frame_h / self.h)
        box = self if scale >= 1.0 else replace(self, w=self.w * scale, h=self.h * scale)
        return box.clamp_to(frame_w, frame_h)

    def as_int_even(
        self, frame_w: float | None = None, frame_h: float | None = None
    ) -> tuple[int, int, int, int]:
        """转成 ``(x, y, w, h)`` 整型像素框，宽高向下取整到偶数（供 OpenCV 切片）。

        传入画面尺寸时会自动保证 ``x + w <= frame_w``、``y + h <= frame_h``。
        """
        width = even_floor(self.w)
        height = even_floor(self.h)
        if frame_w is not None and width > frame_w:
            width = even_floor(frame_w)
            height = even_floor(width / self.aspect) if self.aspect else height
        if frame_h is not None and height > frame_h:
            height = even_floor(frame_h)
            width = even_floor(height * self.aspect) if self.aspect else width

        x = int(round(self.x))
        y = int(round(self.y))
        x = max(0, min(x, int(frame_w) - width)) if frame_w is not None else max(0, x)
        y = max(0, min(y, int(frame_h) - height)) if frame_h is not None else max(0, y)
        return x, y, width, height

    def describe(self) -> str:
        return f"x={self.x:.0f} y={self.y:.0f} {self.w:.0f}×{self.h:.0f} ({mode_label(self.mode)})"


@dataclass(frozen=True, slots=True)
class FramingParams:
    """构图分档阈值与填充系数（全部写入 ``AppConfig`` / ``configs/default.yaml``）。"""

    #: bbox 高占画面比例 ≥ 此值 → 判定为头像/大特写
    closeup_ratio: float = 0.75
    #: bbox 高占画面比例 ≥ 此值（且低于 closeup_ratio）→ 判定为半身
    halfbody_ratio: float = 0.25
    #: 各档位下"人物高度占取景框高度的比例"，越大人物越大
    closeup_fill: float = 0.90
    halfbody_fill: float = 0.70
    fullbody_fill: float = 0.55
    #: 人体被画面边缘截断时的保守填充（取更小的值 = 取更大的景）
    incomplete_fill: float = 0.50
    #: 半身构图时头顶留白占取景框高度的比例
    headroom: float = 0.08
    #: 判定"贴边 / 被截断"的像素阈值
    edge_margin: float = 2.0
    #: 取景框最小边长（像素），避免人物过小时极端放大
    min_box_px: int = 96

    def __post_init__(self) -> None:
        for name in ("closeup_fill", "halfbody_fill", "fullbody_fill", "incomplete_fill"):
            value = getattr(self, name)
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} 必须在 (0, 1] 区间内，当前为 {value}")
        if self.closeup_ratio <= self.halfbody_ratio:
            raise ValueError("closeup_ratio 必须大于 halfbody_ratio")
        if not 0.0 <= self.headroom < 0.5:
            raise ValueError(f"headroom 必须在 [0, 0.5) 区间内，当前为 {self.headroom}")

    def fill_for(self, mode: str) -> float:
        """返回该档位下"人物高度 / 取景框高度"的目标比例。"""
        if mode == MODE_CLOSEUP:
            return self.closeup_fill
        if mode == MODE_FULLBODY:
            return self.fullbody_fill
        return self.halfbody_fill


#: 默认构图参数
DEFAULT_FRAMING_PARAMS = FramingParams()

#: 默认主角最小高度占比（低于此值视为"没有人物"）：
#: 只有画面里占比足够大的人才算主要人物，背景中的路人 / 小人一律忽略。
DEFAULT_MIN_PERSON_HEIGHT_RATIO = 0.73

BBox = tuple[float, float, float, float]


def classify_composition(
    bbox: BBox, frame_h: float, params: FramingParams = DEFAULT_FRAMING_PARAMS
) -> str:
    """按 bbox 占画面高度的比例做构图分档：头像 / 半身 / 全身。"""
    height = abs(float(bbox[3]) - float(bbox[1]))
    fill = height / float(frame_h) if frame_h else 0.0
    if fill >= params.closeup_ratio:
        return MODE_CLOSEUP
    if fill >= params.halfbody_ratio:
        return MODE_HALFBODY
    return MODE_FULLBODY


def is_truncated(
    bbox: BBox, frame_size: tuple[float, float], edge_margin: float = 2.0
) -> bool:
    """人物框是否贴住画面边缘（判定为不完整人体，需要更保守的取景）。"""
    frame_w, frame_h = float(frame_size[0]), float(frame_size[1])
    x1, y1, x2, y2 = (float(v) for v in bbox)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (
        x1 <= edge_margin
        or y1 <= edge_margin
        or x2 >= frame_w - edge_margin
        or y2 >= frame_h - edge_margin
    )


def center_box(
    frame_size: tuple[float, float], ratio: AspectRatio, mode: str = MODE_CENTER
) -> CropBox:
    """画面正中的最大等比取景框（兜底策略用）。"""
    frame_w, frame_h = float(frame_size[0]), float(frame_size[1])
    width, height = max_crop_size(frame_w, frame_h, ratio)
    return CropBox((frame_w - width) / 2.0, (frame_h - height) / 2.0, width, height, mode=mode)


def compute_target_box(
    bbox: BBox,
    frame_size: tuple[float, float],
    ratio: AspectRatio,
    params: FramingParams = DEFAULT_FRAMING_PARAMS,
    anchors: BodyAnchors | None = None,
) -> CropBox:
    """由人物框推导理想取景框（纯函数）。

    步骤：分档 → 定框高 → 按比例推框宽 → 上下限保护 → 定位 → 夹取到画面内。

    ``anchors`` 可选：姿态关键点给出的"头顶 / 腰部"锚点。半身档位下用它替代
    bbox 的上下边（人物抬手时 bbox 顶边会偏高，关键点能纠正）。
    """
    frame_w, frame_h = float(frame_size[0]), float(frame_size[1])
    if frame_w <= 0 or frame_h <= 0:
        raise ValueError(f"画面尺寸必须为正：{frame_w}x{frame_h}")

    x1, y1, x2, y2 = (float(v) for v in bbox)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    mode = classify_composition((x1, y1, x2, y2), frame_h, params)
    fill = params.fill_for(mode)
    if is_truncated((x1, y1, x2, y2), (frame_w, frame_h), params.edge_margin):
        # 人体不完整：取更保守的景，避免裁到只剩半张脸
        fill = min(fill, params.incomplete_fill)

    person_top, person_bottom = y1, y2
    if anchors is not None and mode == MODE_HALFBODY and anchors.span > 1.0:
        # 半身：下边界落在腰/髋，头顶以关键点为准
        person_top, person_bottom = anchors.head_top, anchors.waist

    person_top = min(max(person_top, 0.0), frame_h)
    person_bottom = min(max(person_bottom, 0.0), frame_h)
    box_h_person = max(person_bottom - person_top, 1.0)

    box_h = box_h_person / fill
    box_w = box_h * ratio.value

    max_w, max_h = max_crop_size(frame_w, frame_h, ratio)
    if box_w > max_w or box_h > max_h:
        shrink = min(max_w / box_w, max_h / box_h)
        box_w *= shrink
        box_h *= shrink

    if box_h < params.min_box_px:
        grow = min(params.min_box_px / box_h, max_h / box_h, max_w / box_w)
        box_h *= grow
        box_w *= grow

    center_x = (x1 + x2) / 2.0
    if mode == MODE_HALFBODY:
        # 以"头顶留白"定位：框顶 = 头顶之上 headroom 个框高
        center_y = person_top - params.headroom * box_h + box_h / 2.0
    else:
        center_y = (person_top + person_bottom) / 2.0

    box = CropBox(center_x - box_w / 2.0, center_y - box_h / 2.0, box_w, box_h, mode=mode)
    return box.fit_to(frame_w, frame_h)


#: 多人分屏"上半身"窗口：人物上半身四周额外留白的比例（0.15 ≈ 每侧留 7%）
UPPER_BODY_MARGIN = 0.15

#: 没有姿态关键点时，用"人物框宽度"估算上半身高度（头顶 → 腰）的倍数。
#: 站姿全身高宽比约 3.5，头到腰约占身高 46%，于是 ≈ 1.6 个框宽；
#: 近景 / 坐姿这类本来就"矮胖"的框（高宽比 ≤ 1.6）不会超过它，自然整条框都算上半身。
UPPER_BODY_SPAN_TO_WIDTH = 1.6


def upper_body_span(
    bbox: BBox, frame_h: float, anchors: BodyAnchors | None = None
) -> tuple[float, float]:
    """人物"上半身"的纵向范围 ``(头顶, 腰)``（纯函数）。

    优先用姿态锚点（头顶 ← 面部关键点，腰 ← 髋关键点）；没有关键点时退回估算：
    头顶取 bbox 顶边，腰取"头顶往下 :data:`UPPER_BODY_SPAN_TO_WIDTH` 个框宽"与
    bbox 底边中更靠近头顶的那个 —— 远景站姿（又高又窄的整条全身框）会被收成
    上半身，近景 / 坐姿（框本来就矮）则保持整条框不动。
    """
    x1, y1, x2, y2 = (float(value) for value in bbox)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    width = max(x2 - x1, 1.0)

    if anchors is not None and anchors.span > 1.0:
        top, bottom = anchors.head_top, anchors.waist
    else:
        top = y1
        bottom = y1 + min(max(y2 - y1, 0.0), UPPER_BODY_SPAN_TO_WIDTH * width)

    top = min(max(top, 0.0), frame_h)
    bottom = min(max(bottom, top + 1.0), frame_h)
    return top, bottom


def compute_upper_body_box(
    bbox: BBox,
    frame_size: tuple[float, float],
    ratio: AspectRatio,
    params: FramingParams = DEFAULT_FRAMING_PARAMS,
    anchors: BodyAnchors | None = None,
) -> CropBox:
    """由人物框推导"只装上半身"的紧致取景框（多人分屏的每个小窗口用）。

    与 :func:`compute_target_box` 的两点关键区别：

    1. **不看人物在画面里的大小分档，一律按上半身取景** —— 人物在画面里占比很小时
       （整条 bbox 是全身）也会被放大成半身，而不是把腿部与四周场景一起框进来；
    2. **取"恰好装下上半身"的最小框** —— 分别按"上半身高度 + 留白"与"人物宽度 +
       留白"算出两个候选框高，取**较大者**，于是两个方向都刚好留出
       :data:`UPPER_BODY_MARGIN`，人物左右不会再剩下大片非人物场景。

    仍然满足 :func:`compute_target_box` 的全部不变量：框的宽高比恒等于 ``ratio``、
    只平移不越界（不出现黑边）、有最小 / 最大尺寸保护。
    """
    frame_w, frame_h = float(frame_size[0]), float(frame_size[1])
    if frame_w <= 0 or frame_h <= 0:
        raise ValueError(f"画面尺寸必须为正：{frame_w}x{frame_h}")

    x1, y1, x2, y2 = (float(value) for value in bbox)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    head_top, waist = upper_body_span((x1, y1, x2, y2), frame_h, anchors)
    span_h = max(waist - head_top, 1.0)
    span_w = max(x2 - x1, 1.0)

    # 候选框高取大者：① 纵向刚好容下上半身；② 横向刚好容下人物宽度
    box_h = max(span_h, span_w / ratio.value) * (1.0 + UPPER_BODY_MARGIN)
    box_w = box_h * ratio.value

    max_w, max_h = max_crop_size(frame_w, frame_h, ratio)
    if box_w > max_w or box_h > max_h:
        shrink = min(max_w / box_w, max_h / box_h)
        box_w *= shrink
        box_h *= shrink

    if box_h < params.min_box_px:
        grow = min(params.min_box_px / box_h, max_h / box_h, max_w / box_w)
        box_h *= grow
        box_w *= grow

    # 纵向定位：框顶落在头顶之上 headroom 个框高处；
    # 但若框高被"横向"撑大 / 被上限压缩，也要保证腰（含下侧留白）仍在框内。
    center_x = (x1 + x2) / 2.0
    top = max(
        head_top - params.headroom * box_h,
        waist + UPPER_BODY_MARGIN * span_h / 2.0 - box_h,
    )
    box = CropBox(center_x - box_w / 2.0, top, box_w, box_h, mode=MODE_HALFBODY)
    return box.fit_to(frame_w, frame_h)


def interpolate_boxes(
    first: CropBox | None, second: CropBox | None, t: float, ratio: AspectRatio
) -> CropBox | None:
    """在两个取景框之间线性插值（抽帧检测的中间帧用），比例恒等于目标比例。"""
    if first is None:
        return second
    if second is None:
        return first
    t = min(max(float(t), 0.0), 1.0)
    cx = first.cx + (second.cx - first.cx) * t
    cy = first.cy + (second.cy - first.cy) * t
    height = first.h + (second.h - first.h) * t
    width = height * ratio.value
    mode = second.mode if t >= 0.5 else first.mode
    return CropBox(cx - width / 2.0, cy - height / 2.0, width, height, mode=mode)
