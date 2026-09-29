"""统一视觉规范：配色、字体、控件样式、高 DPI 适配、明暗主题。

所有界面颜色 / 间距 / 圆角集中在这里，页面只引用 ``theme`` 里的常量或
:func:`active` 返回的调色板，不写魔法值。

主题策略：默认**跟随系统**（Qt 6.5+ 的 ``QStyleHints.colorScheme``），
也可以用环境变量 ``V2PV_THEME=light|dark`` 强制指定。
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass

#: 应用与组织名（``QSettings`` 用它记住上次选择）
APP_NAME = "Video2PersonVideo"
ORG_NAME = "Video2PersonVideo"

#: 窗口「理想」尺寸：屏幕够大时用它
PREFERRED_WINDOW_SIZE = (1000, 760)
#: 窗口尺寸下限：屏幕放不下时收到这里，页面内容交给滚动区承载（不压扁控件）
MIN_WINDOW_SIZE = (760, 560)
#: 窗口最多占屏幕可用区域的比例，给任务栏 / 其它窗口留余量
WINDOW_SCREEN_RATIO = 0.94
#: 拿不到屏幕信息时假定的逻辑分辨率（保守值）
FALLBACK_SCREEN_SIZE = (1280, 800)
#: 兼容旧名字：默认窗口尺寸
DEFAULT_WINDOW_SIZE = PREFERRED_WINDOW_SIZE

#: 主题模式
MODE_AUTO = "auto"
MODE_LIGHT = "light"
MODE_DARK = "dark"

#: 环境变量：强制主题（``light`` / ``dark`` / ``auto``）
THEME_ENV_VAR = "V2PV_THEME"

#: 统一间距
SPACING_SMALL = 6
SPACING = 10
SPACING_LARGE = 16

#: 等宽字体（日志区）
MONO_FONT = "Consolas, 'Courier New', monospace"

STEP_TITLES = ("1 · 选择输入", "2 · 选择比例", "3 · 执行处理", "4 · 结果汇总")

#: 界面基准字号（设备无关像素）。Qt 会按屏幕缩放比例把设备无关像素等比放大，
#: 所以 150% / 200% 缩放下看到的就是等比放大的界面；这里再保证「需要固定高度」
#: 的控件（进度条、滚动条…）都按字体行高推导，字号变大也不会把文字裁掉。
BASE_FONT_PX = 13


@dataclass(frozen=True, slots=True)
class Metrics:
    """由字体实际度量推导出来的控件尺寸（单位同样是设备无关像素）。"""

    font_px: int
    line_height: int
    headline_px: int
    subhead_px: int
    progress_height: int
    thin_progress_height: int
    scrollbar_min_length: int


#: 无法访问 Qt 时的保守度量（数值与 13px 字号下的实测结果一致）
_FALLBACK_METRICS = Metrics(
    font_px=BASE_FONT_PX,
    line_height=BASE_FONT_PX + 4,
    headline_px=BASE_FONT_PX + 3,
    subhead_px=BASE_FONT_PX + 2,
    progress_height=BASE_FONT_PX + 12,
    thin_progress_height=BASE_FONT_PX + 8,
    scrollbar_min_length=BASE_FONT_PX + 11,
)


@dataclass(frozen=True, slots=True)
class Palette:
    """一套完整的配色方案。"""

    name: str
    bg: str
    surface: str
    border: str
    border_soft: str
    text: str
    text_muted: str
    text_disabled: str
    primary: str
    primary_hover: str
    primary_disabled: str
    success: str
    warning: str
    danger: str
    danger_soft: str
    card_selected: str
    #: 进度条 / 卡片选中时的描边（深色下光晕更强）
    card_border: str

    @property
    def is_dark(self) -> bool:
        return self.name == MODE_DARK


LIGHT = Palette(
    name=MODE_LIGHT,
    bg="#ffffff",
    surface="#f8fafc",
    border="#d0d7de",
    border_soft="#e5e7eb",
    text="#1f2328",
    text_muted="#57606a",
    text_disabled="#b1b7bd",
    primary="#2563eb",
    primary_hover="#1d4ed8",
    primary_disabled="#9db8ef",
    success="#16a34a",
    warning="#d97706",
    danger="#dc2626",
    danger_soft="#fef2f2",
    card_selected="#dbeafe",
    card_border="#2563eb",
)

DARK = Palette(
    name=MODE_DARK,
    bg="#0d1117",
    surface="#161b22",
    border="#30363d",
    border_soft="#21262d",
    text="#e6edf3",
    text_muted="#8b949e",
    text_disabled="#5a6572",
    primary="#3b82f6",
    primary_hover="#60a5fa",
    primary_disabled="#1e3a8a",
    success="#3fb950",
    warning="#d29922",
    danger="#f85149",
    danger_soft="#3d1d1d",
    card_selected="#1c3a63",
    card_border="#60a5fa",
)

PALETTES: dict[str, Palette] = {MODE_LIGHT: LIGHT, MODE_DARK: DARK}

_ACTIVE: Palette = LIGHT


def active() -> Palette:
    """当前生效的调色板。"""
    return _ACTIVE


# ------------------------------------------------ 兼容旧写法的颜色常量（浅色）
COLOR_BG = LIGHT.bg
COLOR_SURFACE = LIGHT.surface
COLOR_BORDER = LIGHT.border
COLOR_BORDER_SOFT = LIGHT.border_soft
COLOR_TEXT = LIGHT.text
COLOR_TEXT_MUTED = LIGHT.text_muted
COLOR_TEXT_DISABLED = LIGHT.text_disabled
COLOR_PRIMARY = LIGHT.primary
COLOR_PRIMARY_HOVER = LIGHT.primary_hover
COLOR_PRIMARY_DISABLED = LIGHT.primary_disabled
COLOR_SUCCESS = LIGHT.success
COLOR_WARNING = LIGHT.warning
COLOR_DANGER = LIGHT.danger
COLOR_CARD_SELECTED = LIGHT.card_selected


# ------------------------------------------------------------------ 主题解析
def normalize_mode(mode: str | None = None) -> str:
    """把外部输入规整为 ``auto`` / ``light`` / ``dark``。"""
    if mode is None:
        mode = os.environ.get(THEME_ENV_VAR, MODE_AUTO)
    text = str(mode).strip().lower()
    if text in (MODE_LIGHT, "day", "1", "true"):
        return MODE_LIGHT
    if text in (MODE_DARK, "night", "0", "false"):
        return MODE_DARK
    return MODE_AUTO


def system_scheme(app=None) -> str:
    """读取系统当前的颜色方案，读不到时返回 ``light``。

    Qt 6.5+ 提供 ``QStyleHints.colorScheme()``；老版本或异常情况一律按浅色。
    """
    try:
        from PySide6.QtCore import Qt
    except ImportError:  # pragma: no cover - 未装 PySide6
        return MODE_LIGHT

    hints = None
    if app is not None:
        hints = getattr(app, "styleHints", lambda: None)()
    if hints is None:
        with contextlib.suppress(Exception):
            from PySide6.QtGui import QGuiApplication

            hints = QGuiApplication.styleHints()
    if hints is None:  # pragma: no cover - 无 GUI 环境
        return MODE_LIGHT

    scheme = getattr(hints, "colorScheme", lambda: None)()
    with contextlib.suppress(Exception):
        if scheme == Qt.ColorScheme.Dark:
            return MODE_DARK
        if scheme == Qt.ColorScheme.Light:
            return MODE_LIGHT
    return MODE_LIGHT


def resolve_palette(mode: str | None = None, app=None) -> Palette:
    """把主题模式解析成具体调色板（``auto`` 时读系统设置）。"""
    normalized = normalize_mode(mode)
    if normalized == MODE_AUTO:
        normalized = system_scheme(app)
    return PALETTES.get(normalized, LIGHT)


# ------------------------------------------------------------------ 度量与自适应
def _has_gui() -> bool:
    """是否已经有可用的 GUI 应用实例。

    没有 ``QGuiApplication`` 就去问字体度量会**直接搞崩进程**（不是抛异常、
    也不是返回空），所以任何字体度量的入口都必须先过这一关。
    """
    try:
        from PySide6.QtGui import QGuiApplication

        return QGuiApplication.instance() is not None
    except Exception:  # pragma: no cover - 未装 PySide6
        return False


def base_font():
    """QSS 里 ``font-size`` 对应的字体；没有 GUI 环境时返回 ``None``。"""
    if not _has_gui():
        return None
    try:
        from PySide6.QtGui import QFont, QGuiApplication

        font = QFont(QGuiApplication.font())
        font.setPixelSize(BASE_FONT_PX)
        return font
    except Exception:  # pragma: no cover - 度量失败不该影响启动
        return None


def ui_metrics() -> Metrics:
    """当前系统的界面度量。

    关键在于：**不假设用户的字号 / 缩放比例**，而是量出这套字号真正需要多高，
    再让进度条之类的固定高度控件跟着走，避免文字被裁掉或挤在一起。
    """
    font = base_font()
    if font is None:  # pragma: no cover - 无 GUI 环境
        return _FALLBACK_METRICS
    try:
        from PySide6.QtGui import QFontMetrics

        line = int(QFontMetrics(font).lineSpacing())  # 需要已有 QGuiApplication
    except Exception:  # pragma: no cover - 度量失败不该影响启动
        return _FALLBACK_METRICS
    line = max(line, BASE_FONT_PX + 2)
    return Metrics(
        font_px=BASE_FONT_PX,
        line_height=line,
        headline_px=BASE_FONT_PX + 3,
        subhead_px=BASE_FONT_PX + 2,
        progress_height=line + 8,
        thin_progress_height=line + 4,
        scrollbar_min_length=line + 10,
    )


def text_width_px(text: str, *, padding: int = 8) -> int:
    """按界面基准字号实测文本宽度（替代写死的 ``setFixedWidth``）。"""
    font = base_font()
    if font is None:  # pragma: no cover - 无 GUI 环境
        return len(text) * BASE_FONT_PX + padding
    try:
        from PySide6.QtGui import QFontMetrics

        return int(QFontMetrics(font).horizontalAdvance(text)) + padding
    except Exception:  # pragma: no cover
        return len(text) * BASE_FONT_PX + padding


def screen_available_size(screen=None) -> tuple[int, int]:
    """参考屏幕的可用区域（不含任务栏），拿不到时退回保守值。

    **必须用逻辑像素**：Windows 150% 缩放下，1920×1080 的桌面只有 1280×720 的
    逻辑空间，窗口尺寸写死成 1000×760 就会撑出屏幕、内容被压扁。
    """
    try:
        from PySide6.QtGui import QGuiApplication

        target = screen if screen is not None else QGuiApplication.primaryScreen()
        if target is None:
            return FALLBACK_SCREEN_SIZE
        rect = target.availableGeometry()
        return (max(int(rect.width()), 320), max(int(rect.height()), 240))
    except Exception:  # pragma: no cover - 无 GUI 环境
        return FALLBACK_SCREEN_SIZE


def fit_to_screen(
    preferred: tuple[int, int],
    minimum: tuple[int, int] | None = None,
    *,
    screen=None,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """把「理想尺寸 / 下限尺寸」压进屏幕可用区域，返回 ``(尺寸, 下限)``。"""
    available_w, available_h = screen_available_size(screen)
    limit_w = max(int(available_w * WINDOW_SCREEN_RATIO), 320)
    limit_h = max(int(available_h * WINDOW_SCREEN_RATIO), 240)

    floor_w, floor_h = minimum or (0, 0)
    low_w = min(max(int(floor_w), 320), limit_w)
    low_h = min(max(int(floor_h), 240), limit_h)

    width = max(min(int(preferred[0]), limit_w), low_w)
    height = max(min(int(preferred[1]), limit_h), low_h)
    return (width, height), (low_w, low_h)


def apply_window_size(
    window,
    preferred: tuple[int, int] = PREFERRED_WINDOW_SIZE,
    minimum: tuple[int, int] = MIN_WINDOW_SIZE,
) -> None:
    """按屏幕可用区域设置窗口的初始大小与最小尺寸。

    高缩放比下逻辑分辨率变小，窗口会自动收小（内容改由滚动区承载），
    而不是撑出屏幕导致底部按钮点不到、控件挤在一起。
    """
    size, floor = fit_to_screen(preferred, minimum, screen=window.screen())
    window.setMinimumSize(*floor)
    window.resize(*size)


def clamp_size(
    preferred: tuple[int, int],
    minimum: tuple[int, int] = (320, 240),
    *,
    screen=None,
) -> tuple[int, int]:
    """对话框用：只要一个已经压进屏幕的尺寸。"""
    size, _ = fit_to_screen(preferred, minimum, screen=screen)
    return size


def scrollable(widget):
    """把整页内容装进滚动区。

    窗口再小 / 字号再大，也只是出现滚动条，而不是把控件压扁、把文字裁掉——
    这是「任意缩放比例下都能正确显示」的关键兜底。
    """
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QFrame, QScrollArea

    area = QScrollArea()
    area.setObjectName("pageScroll")
    area.setWidgetResizable(True)  # 内容跟着视口宽度伸缩，尽量不出现横向滚动
    area.setFrameShape(QFrame.Shape.NoFrame)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    area.setViewportMargins(0, 0, 4, 4)  # 内容与滚动条之间留一点缝，别贴在一起
    area.setWidget(widget)
    return area


# ------------------------------------------------------------------ 样式表
def build_stylesheet(palette: Palette | None = None, metrics: Metrics | None = None) -> str:
    """按调色板生成全局 QSS。

    ``metrics`` 决定字号与几个「固定高度」控件的高度：一律由字体度量推导，
    不写死数值——否则换了字号 / 缩放比例，文字就会被裁掉或挤在一起。
    """
    c = palette or LIGHT
    m = metrics or ui_metrics()
    return f"""
QWidget {{ font-size: {m.font_px}px; color: {c.text}; }}
QMainWindow, QDialog {{ background: {c.bg}; }}
QToolTip {{
    background: {c.surface}; color: {c.text};
    border: 1px solid {c.border}; padding: 4px 6px;
}}

/* 页面滚动区：透明背景 + 无边框，视觉上与外面的窗口连成一片 */
QScrollArea, QScrollArea#pageScroll {{ border: none; background: transparent; }}
QScrollArea > QWidget, QScrollArea > QWidget > QWidget {{ background: transparent; }}

QGroupBox {{
    border: 1px solid {c.border};
    border-radius: 8px;
    margin-top: 12px;
    padding: 10px 10px 6px 10px;
    background: {c.bg};
}}
QGroupBox::title {{
    subcontrol-origin: margin; left: 12px; padding: 0 4px; color: {c.text_muted};
}}

QPushButton {{
    padding: 6px 16px; border-radius: 6px;
    border: 1px solid {c.border}; background: {c.bg}; color: {c.text};
}}
QPushButton:hover {{ background: {c.surface}; }}
QPushButton:disabled {{ color: {c.text_disabled}; border-color: {c.border_soft}; }}

QPushButton#primary {{
    background: {c.primary}; color: white; border: none; font-weight: 600;
}}
QPushButton#primary:hover {{ background: {c.primary_hover}; }}
QPushButton#primary:disabled {{ background: {c.primary_disabled}; color: {c.text_disabled}; }}

QPushButton#danger {{ color: {c.danger}; border-color: {c.danger}; }}
QPushButton#danger:hover {{ background: {c.danger_soft}; }}

QToolButton#ratioCard {{
    border: 1px solid {c.border}; border-radius: 8px; padding: 6px;
    background: {c.bg}; color: {c.text_muted};
}}
QToolButton#ratioCard:hover {{ background: {c.surface}; }}
QToolButton#ratioCard:checked {{
    border: 2px solid {c.card_border}; background: {c.card_selected}; color: {c.text};
    font-weight: 600;
}}

QLineEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit {{
    border: 1px solid {c.border}; border-radius: 6px; padding: 4px 6px;
    background: {c.bg}; color: {c.text}; selection-background-color: {c.primary};
}}
QLineEdit:focus, QPlainTextEdit:focus, QComboBox:focus, QTextEdit:focus,
QSpinBox:focus, QDoubleSpinBox:focus {{ border-color: {c.primary}; }}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
    background: {c.surface}; color: {c.text_disabled};
}}
QComboBox QAbstractItemView {{
    background: {c.bg}; color: {c.text}; border: 1px solid {c.border};
    selection-background-color: {c.card_selected}; selection-color: {c.text};
}}

QProgressBar {{
    border: 1px solid {c.border}; border-radius: 6px; text-align: center;
    height: {m.progress_height}px;  /* 由字体行高推导：文字不会被上下裁掉 */
    background: {c.surface}; color: {c.text};
}}
QProgressBar::chunk {{ background: {c.primary}; border-radius: 5px; }}

QListWidget, QTableWidget {{
    border: 1px solid {c.border}; border-radius: 6px; background: {c.bg};
    alternate-background-color: {c.surface};
}}
QTableWidget::item:selected, QListWidget::item:selected {{
    background: {c.card_selected}; color: {c.text};
}}
QHeaderView::section {{
    background: {c.surface}; border: none; border-bottom: 1px solid {c.border};
    padding: 4px 6px; color: {c.text_muted};
}}

QSlider::groove:horizontal {{ height: 4px; background: {c.border_soft}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    background: {c.primary}; width: 14px; margin: -6px 0; border-radius: 7px;
}}

QCheckBox, QRadioButton {{ color: {c.text}; }}

QPlainTextEdit#logView {{ font-family: {MONO_FONT}; }}

QScrollBar:vertical, QScrollBar:horizontal {{ background: {c.surface}; border: none; }}
QScrollBar::handle {{ background: {c.border}; border-radius: 4px;
                      min-height: {m.scrollbar_min_length}px; }}
QScrollBar::handle:hover {{ background: {c.text_muted}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}

QLabel#stepChip {{ color: {c.text_muted}; padding: 2px 8px; }}
QLabel#stepChipActive {{ color: {c.primary}; font-weight: 600; padding: 2px 8px; }}
QLabel#hint {{ color: {c.text_muted}; }}
QLabel#status {{ color: {c.text_muted}; }}

/* 首次运行的环境自检页 */
QLabel#headline {{ font-size: {m.headline_px}px; font-weight: 600; }}
QLabel#subhead {{ font-size: {m.subhead_px}px; font-weight: 600; }}
QLabel#badgeOk {{ color: {c.success}; font-weight: 600; }}
QLabel#badgeWarn {{ color: {c.warning}; font-weight: 600; }}
QLabel#badgeDanger {{ color: {c.danger}; font-weight: 600; }}
QLabel#badgeInfo {{ color: {c.primary}; font-weight: 600; }}
QLabel#mono {{ font-family: {MONO_FONT}; color: {c.text_muted}; }}
QPushButton#link {{
    border: none; background: transparent; color: {c.primary};
    padding: 2px 4px; text-decoration: underline;
}}
QPushButton#link:hover {{ color: {c.primary_hover}; }}
QProgressBar#thin {{ height: {m.thin_progress_height}px; }}
QTableWidget#checkTable {{ border: 1px solid {c.border}; }}
"""


#: 兼容旧写法的默认样式表（浅色）
STYLESHEET = build_stylesheet(LIGHT)


def enable_high_dpi() -> None:
    """开启高 DPI 适配（必须在 ``QApplication`` 创建之前调用）。

    Qt 6 默认就开着高 DPI 缩放，这里显式指定 ``PassThrough`` 是为了保住
    Windows 的 125% / 150% / 175% 这些**非整数**缩放比例：若被四舍五入成
    100%，界面就会按 100% 渲染再由系统拉伸，文字发虚、间距错乱。
    """
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QGuiApplication

        QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
    except Exception:  # noqa: BLE001 - 不同 Qt 版本 API 有差异，失败也不影响使用
        pass


def _build_qt_palette(palette: Palette):
    """生成 QPalette，让原生控件（菜单、消息框、滚动条）也跟着换色。"""
    from PySide6.QtGui import QColor, QPalette

    qt_palette = QPalette()
    role = QPalette.ColorRole
    group = QPalette.ColorGroup

    qt_palette.setColor(role.Window, QColor(palette.bg))
    qt_palette.setColor(role.WindowText, QColor(palette.text))
    qt_palette.setColor(role.Base, QColor(palette.bg))
    qt_palette.setColor(role.AlternateBase, QColor(palette.surface))
    qt_palette.setColor(role.ToolTipBase, QColor(palette.surface))
    qt_palette.setColor(role.ToolTipText, QColor(palette.text))
    qt_palette.setColor(role.Text, QColor(palette.text))
    qt_palette.setColor(role.Button, QColor(palette.bg))
    qt_palette.setColor(role.ButtonText, QColor(palette.text))
    qt_palette.setColor(role.Highlight, QColor(palette.primary))
    qt_palette.setColor(role.HighlightedText, QColor("#ffffff"))
    qt_palette.setColor(role.PlaceholderText, QColor(palette.text_muted))
    qt_palette.setColor(
        group.Disabled, role.Text, QColor(palette.text_disabled)
    )
    qt_palette.setColor(
        group.Disabled, role.ButtonText, QColor(palette.text_disabled)
    )
    qt_palette.setColor(
        group.Disabled, role.WindowText, QColor(palette.text_disabled)
    )
    return qt_palette


def _sync_module_colors(palette: Palette) -> None:
    """同步模块级 ``COLOR_*`` 常量，兼容按常量取色的旧代码。"""
    globals().update(
        {
            "STYLESHEET": build_stylesheet(palette),
            "COLOR_BG": palette.bg,
            "COLOR_SURFACE": palette.surface,
            "COLOR_BORDER": palette.border,
            "COLOR_BORDER_SOFT": palette.border_soft,
            "COLOR_TEXT": palette.text,
            "COLOR_TEXT_MUTED": palette.text_muted,
            "COLOR_TEXT_DISABLED": palette.text_disabled,
            "COLOR_PRIMARY": palette.primary,
            "COLOR_PRIMARY_HOVER": palette.primary_hover,
            "COLOR_PRIMARY_DISABLED": palette.primary_disabled,
            "COLOR_SUCCESS": palette.success,
            "COLOR_WARNING": palette.warning,
            "COLOR_DANGER": palette.danger,
            "COLOR_CARD_SELECTED": palette.card_selected,
        }
    )


def apply_theme(app, mode: str | None = None) -> Palette:
    """把样式表与调色板套用到整个应用，返回实际使用的调色板。

    ``mode`` 为 ``None`` / ``auto`` 时跟随系统，并在系统切换深浅色时自动重刷。
    """
    global _ACTIVE

    palette = resolve_palette(mode, app)
    _ACTIVE = palette
    _sync_module_colors(palette)

    with contextlib.suppress(Exception):  # pragma: no cover - 样式设置失败不影响功能
        app.setStyleSheet(build_stylesheet(palette))
    with contextlib.suppress(Exception):
        app.setPalette(_build_qt_palette(palette))

    _install_scheme_hook(app, normalize_mode(mode))
    return palette


def _install_scheme_hook(app, mode: str) -> None:
    """``auto`` 模式下监听系统深浅色切换并自动重刷。"""
    if mode != MODE_AUTO:
        return
    with contextlib.suppress(Exception):  # pragma: no cover - 老版本 Qt 无此信号
        hints = app.styleHints()
        signal = hints.colorSchemeChanged

        def _on_changed(_scheme=None) -> None:
            refreshed = resolve_palette(MODE_AUTO, app)
            apply_theme(app, refreshed.name)

        signal.connect(_on_changed)
        # 防止回调被 GC：挂到 app 上（属性名带前缀，避免与应用自身属性冲突）
        app._v2pv_scheme_hook = _on_changed
