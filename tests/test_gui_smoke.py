"""GUI 冒烟测试：向导各步骤构建、参数收集、比例对话框校验、批量路径识别。

使用 Qt 的 offscreen 后端，无显示器环境（CI）也能跑；不弹真实窗口、不跑推理。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6.QtWidgets", reason="未安装 PySide6，跳过 GUI 测试")

from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from video2personvideo.gui.ratio_dialog import RatioDialog  # noqa: E402
from video2personvideo.gui.wizard import WizardWindow  # noqa: E402


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app  # type: ignore[return-value]


@pytest.fixture(autouse=True)
def _isolated_settings(tmp_path: Path, monkeypatch) -> None:
    """把界面记忆落到临时 ini 文件，避免读写真实用户配置（注册表 / APPDATA）。"""
    from video2personvideo.gui.wizard import WizardWindow

    def _settings(_self) -> QSettings:
        return QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)

    monkeypatch.setattr(WizardWindow, "_settings", _settings)


@pytest.fixture()
def window(qapp: QApplication) -> WizardWindow:
    wizard = WizardWindow()
    wizard.input_page.set_path(None)  # 清掉可能被记忆恢复的路径
    yield wizard
    wizard.close()


@pytest.fixture()
def video_files(tmp_path: Path) -> list[Path]:
    folder = tmp_path / "videos"
    folder.mkdir()
    for name in ("a.mp4", "b.mov", "c_person.mp4", "notes.txt"):
        (folder / name).write_bytes(b"fake")
    return [folder / name for name in ("a.mp4", "b.mov")]


# ---------------------------------------------------------------- 向导结构
def test_wizard_builds_four_steps(window: WizardWindow) -> None:
    assert window.windowTitle().startswith("Video2PersonVideo")
    assert window.stack.count() == 4
    assert window.stack.currentIndex() == 0
    assert len(window._chips) == 4
    assert window._back_button.isVisible() is False
    assert window._next_button.text() == "下一步"
    assert not window._next_button.isEnabled()  # 还没选输入


def test_wizard_has_setup_entry(window: WizardWindow) -> None:
    """主界面上随时能重新做一次环境自检（首次运行也会自动弹）。"""
    assert window._setup_button.text() == "环境自检"
    assert window._setup_button.isVisibleTo(window) is True
    assert "显卡" in window._setup_button.toolTip()


def test_wizard_ratio_default_and_navigation(window: WizardWindow, video_files: list[Path]) -> None:
    window.input_page.set_path(str(video_files[0].parent))
    assert window.input_page.is_valid()
    assert window._next_button.isEnabled()

    window._go_next()
    assert window.stack.currentIndex() == 1
    assert window._next_button.text() == "开始处理"
    assert window.ratio_page.current_ratio().name == "9:16"

    window._go_back()
    assert window.stack.currentIndex() == 0


def test_input_page_discovers_videos(window: WizardWindow, video_files: list[Path]) -> None:
    window.input_page.set_path(str(video_files[0].parent))
    names = sorted(path.name for path in window.input_page.sources)
    # 忽略 *_person.* 产物与非视频文件
    assert names == ["a.mp4", "b.mov"]


def test_input_page_accepts_single_file(window: WizardWindow, video_files: list[Path]) -> None:
    window.input_page.set_path(str(video_files[0]))
    assert window.input_page.sources == [video_files[0]]


# ---------------------------------------------------------------- 参数收集
def test_collect_config_requires_input(window: WizardWindow) -> None:
    with pytest.raises(ValueError):
        window._collect_config()


def test_collect_config_builds_appconfig(window: WizardWindow, video_files: list[Path]) -> None:
    window.input_page.set_path(str(video_files[0]))
    window.ratio_page.set_ratio(window.ratio_page.current_ratio())
    cfg = window._collect_config()

    assert cfg.source == video_files[0]
    assert cfg.crop is True
    assert cfg.show is False
    assert cfg.classes == [0]
    assert cfg.resolve_ratio().name == "9:16"
    assert cfg.target_width == 1080
    assert cfg.target_height == 1920


def test_collect_config_applies_tuning(window: WizardWindow, video_files: list[Path]) -> None:
    window.input_page.set_path(str(video_files[0]))
    window.ratio_page.set_tuning(
        {"smoothing_alpha": 0.6, "min_person_height_ratio": 0.2, "headroom": 0.12}
    )
    cfg = window._collect_config()

    assert cfg.smoothing_alpha == pytest.approx(0.6)
    assert cfg.min_person_height_ratio == pytest.approx(0.2)
    assert cfg.headroom == pytest.approx(0.12)


# ---------------------------------------------------------------- 多人分屏
def test_multi_person_defaults_on_and_is_collected(
    window: WizardWindow, video_files: list[Path]
) -> None:
    """需求：多人分屏在设置里默认打开，关掉后视频同时只显示一个人。"""
    window.input_page.set_path(str(video_files[0]))

    assert window.ratio_page.multi_person() is True
    assert window._collect_config().multi_person is True

    window.ratio_page.set_multi_person(False)
    assert window._collect_config().multi_person is False


def test_multi_person_choice_is_remembered(qapp: QApplication, tmp_path: Path) -> None:
    from video2personvideo.gui.wizard import WizardWindow

    video = tmp_path / "a.mp4"
    video.write_bytes(b"fake")

    first = WizardWindow()
    first.input_page.set_path(str(video))
    first.ratio_page.set_multi_person(False)
    first.close()

    second = WizardWindow()
    try:
        assert second.ratio_page.multi_person() is False
    finally:
        second.close()


def test_ratio_page_tuning_signal_fires_for_multi_person(window: WizardWindow) -> None:
    seen: list[int] = []
    window.ratio_page.tuningChanged.connect(lambda: seen.append(1))

    window.ratio_page.set_multi_person(False)

    assert seen  # 勾选变化要通知向导（否则预览 / 保存状态会不同步）


# ------------------------------------------------------------ 比例对话框
def test_ratio_dialog_defaults(qapp: QApplication) -> None:
    dialog = RatioDialog()
    assert dialog.selected_ratio().name == "9:16"
    assert "1080" in dialog.preview_label.text()
    dialog.close()


def test_ratio_dialog_custom_ratio(qapp: QApplication) -> None:
    dialog = RatioDialog()
    dialog.width_spin.setValue(4)
    dialog.height_spin.setValue(5)
    dialog._apply_custom()
    assert dialog.selected_ratio().name == "4:5"
    assert dialog.selected_ratio().target_size == (1080, 1350)
    dialog.close()


def test_ratio_dialog_rejects_illegal_range(qapp: QApplication) -> None:
    dialog = RatioDialog()
    dialog.width_spin.setValue(1)
    dialog.height_spin.setValue(1)
    dialog._apply_custom()
    before = dialog.selected_ratio().name
    # 比例合法；再试一个超范围的：外部直接校验解析逻辑
    from video2personvideo.core.ratio import RatioError, parse_ratio

    with pytest.raises(RatioError):
        parse_ratio("30:1")
    assert dialog.selected_ratio().name == before
    dialog.close()


def test_ratio_grid_selects_preset(qapp: QApplication) -> None:
    from video2personvideo.core.ratio import PRESET_RATIOS
    from video2personvideo.gui.ratio_grid import RatioGrid

    grid = RatioGrid()
    target = next(item for item in PRESET_RATIOS if item.name == "1:1")
    grid.set_current(target)
    assert grid.current_ratio.name == "1:1"
    grid.close()


# ---------------------------------------------------------------- 界面记忆
def test_settings_are_remembered_across_windows(qapp: QApplication, tmp_path: Path) -> None:
    from video2personvideo.core.ratio import parse_ratio
    from video2personvideo.gui.wizard import WizardWindow

    video = tmp_path / "a.mp4"
    video.write_bytes(b"fake")

    first = WizardWindow()
    first.input_page.set_path(str(video))
    first.ratio_page.set_ratio(parse_ratio("4:5"))
    first.close()

    second = WizardWindow()
    try:
        assert second.ratio_page.current_ratio().name == "4:5"
        assert second.ratio_page.current_ratio().target_size == (1080, 1350)
        assert second.input_page.path == video
    finally:
        second.close()


# ------------------------------------------------------------------ 结果页
def test_result_page_renders_summary(qapp: QApplication) -> None:
    from video2personvideo.core.batch import STATUS_DONE, STATUS_FAILED, BatchResult, Task
    from video2personvideo.gui.pages.result_page import ResultPage

    page = ResultPage()
    page.set_result(
        BatchResult(
            tasks=[
                Task(source=Path("a.mp4"), output=Path("a_person.mp4"), status=STATUS_DONE),
                Task(
                    source=Path("b.mp4"),
                    output=Path("b_person.mp4"),
                    status=STATUS_FAILED,
                    error="无法打开视频",
                ),
            ],
            elapsed=1.5,
        )
    )
    assert page.table.rowCount() == 2
    assert "成功 1" in page.summary_label.text()
    assert page.output_directory == Path("b_person.mp4").parent
    page.close()


# -------------------------------------------------------------------- 主题
def test_theme_mode_normalization(qapp: QApplication, monkeypatch) -> None:
    from video2personvideo.gui import theme

    assert theme.normalize_mode("dark") == theme.MODE_DARK
    assert theme.normalize_mode("WINDOWS") == theme.MODE_AUTO
    monkeypatch.setenv(theme.THEME_ENV_VAR, "dark")
    assert theme.normalize_mode() == theme.MODE_DARK
    monkeypatch.setenv(theme.THEME_ENV_VAR, "light")
    assert theme.normalize_mode() == theme.MODE_LIGHT
    monkeypatch.delenv(theme.THEME_ENV_VAR)
    assert theme.normalize_mode() == theme.MODE_AUTO


def test_theme_follows_system_when_auto(qapp: QApplication) -> None:
    from video2personvideo.gui import theme

    palette = theme.resolve_palette(theme.MODE_AUTO, qapp)
    # 无论系统是深是浅，都必须落在一套合法调色板上
    assert palette.name in (theme.MODE_LIGHT, theme.MODE_DARK)
    assert theme.system_scheme(qapp) in (theme.MODE_LIGHT, theme.MODE_DARK)


def test_apply_theme_switches_palette_and_constants(qapp: QApplication) -> None:
    from PySide6.QtGui import QPalette

    from video2personvideo.gui import theme

    try:
        dark = theme.apply_theme(qapp, theme.MODE_DARK)
        assert dark.is_dark
        assert theme.active() is theme.DARK
        assert theme.DARK.bg == theme.COLOR_BG
        assert theme.DARK.danger == theme.COLOR_DANGER
        assert qapp.palette().color(QPalette.ColorRole.Window).name() == theme.DARK.bg
        assert theme.DARK.bg in qapp.styleSheet()
    finally:
        light = theme.apply_theme(qapp, theme.MODE_LIGHT)

    assert not light.is_dark
    assert theme.active() is theme.LIGHT
    assert theme.LIGHT.bg == theme.COLOR_BG
    assert theme.LIGHT.bg in qapp.styleSheet()


def test_stylesheets_differ_per_palette() -> None:
    from video2personvideo.gui import theme

    light_qss = theme.build_stylesheet(theme.LIGHT)
    dark_qss = theme.build_stylesheet(theme.DARK)
    assert light_qss != dark_qss
    assert theme.LIGHT.border in light_qss
    assert theme.DARK.border in dark_qss
    assert theme.LIGHT.primary in light_qss
    assert theme.DARK.primary in dark_qss


# ------------------------------------------------------------------ 预览页
def test_preview_dialog_renders_before_after(qapp: QApplication, monkeypatch, tmp_path) -> None:
    """预览对话框：注入采样帧后能渲染出「原画面 + 取景框 / 裁剪结果」对照图。"""
    import numpy as np

    from video2personvideo.config import AppConfig
    from video2personvideo.core.subject import Detection
    from video2personvideo.gui import preview_dialog as pd

    monkeypatch.setattr(pd.PreviewDialog, "_start", lambda self: None)
    dialog = pd.PreviewDialog(AppConfig(aspect_ratio="9:16"), tmp_path / "demo.mp4")

    frame = np.full((240, 320, 3), 150, dtype=np.uint8)
    dialog._samples = [
        {"index": 0, "frame": frame, "detections": [Detection(bbox=(120.0, 40.0, 200.0, 230.0))]},
        {"index": 5, "frame": frame, "detections": []},  # 无人帧走居中兜底
    ]
    dialog._on_samples(dialog._samples)

    pixmap = dialog.image_label.pixmap()
    assert pixmap is not None and not pixmap.isNull()
    assert dialog.tuning()["smoothing_alpha"] == pytest.approx(0.25)

    dialog.headroom_slider.setValue(15)
    assert dialog.tuning()["headroom"] == pytest.approx(0.15)
    dialog.close()


def test_preview_dialog_renders_multi_person(
    qapp: QApplication, monkeypatch, tmp_path
) -> None:
    """预览里可以直接试「多人分屏」：多人物帧渲染成多个小窗口，关掉立刻回到单窗口。"""
    import numpy as np

    from video2personvideo.config import AppConfig
    from video2personvideo.core.subject import Detection
    from video2personvideo.gui import preview_dialog as pd

    monkeypatch.setattr(pd.PreviewDialog, "_start", lambda self: None)
    dialog = pd.PreviewDialog(AppConfig(aspect_ratio="16:9"), tmp_path / "demo.mp4")

    frame = np.full((240, 480, 3), 150, dtype=np.uint8)
    dialog._samples = [
        {
            "index": 0,
            "frame": frame,
            "detections": [
                Detection(bbox=(40.0, 40.0, 160.0, 230.0)),
                Detection(bbox=(300.0, 40.0, 420.0, 230.0)),
            ],
        }
    ]
    dialog._on_samples(dialog._samples)
    assert dialog.image_label.pixmap() is not None
    assert dialog.multi_person() is True  # 默认跟着配置（默认开启）

    dialog.multi_box.setChecked(False)
    assert dialog.multi_person() is False
    assert dialog.image_label.pixmap() is not None  # 关掉后仍能正常渲染（单窗口）
    dialog.close()


def test_bgr_to_qimage_roundtrip() -> None:
    import numpy as np

    from video2personvideo.gui.preview_dialog import bgr_to_qimage

    image = bgr_to_qimage(np.full((10, 20, 3), 128, dtype=np.uint8))
    assert image.width() == 20 and image.height() == 10


# ------------------------------------------------- 非 100% 缩放下的布局健壮性
def _settle(app: QApplication, passes: int = 6) -> None:
    """等排版的连锁反应（重排 → 尺寸变化 → 滚动条 → 重排）收敛。"""
    for _ in range(passes):
        app.processEvents()


def test_window_size_fits_available_screen(qapp: QApplication, window: WizardWindow) -> None:
    """Windows 非 100% 缩放下桌面的逻辑分辨率会变小，窗口必须跟着收小。

    （1920×1080 @150% 只剩 1280×720 逻辑空间，写死 1000×760 就会顶出屏幕。）
    """
    from video2personvideo.gui import theme

    available_w, available_h = theme.screen_available_size()
    assert window.width() <= available_w
    assert window.height() <= available_h
    # 最小尺寸也不能比屏幕还大，否则窗口根本摆不下
    assert window.minimumWidth() <= window.width()
    assert window.minimumHeight() <= window.height()


def test_wizard_pages_scroll_instead_of_being_squeezed(
    qapp: QApplication, window: WizardWindow
) -> None:
    """窗口缩到最小时，页面内容只能滚动，不能被压到小于自身最小需求（会裁字）。"""
    from PySide6.QtWidgets import QScrollArea

    window.show()
    window.resize(window.minimumWidth(), window.minimumHeight())
    _settle(qapp)

    for index in range(window.stack.count()):
        window.stack.setCurrentIndex(index)
        window._update_nav()
        _settle(qapp)
        area = window.stack.currentWidget()
        assert isinstance(area, QScrollArea), "每个向导页都应包在滚动区里"
        page = area.widget()
        need = page.minimumSizeHint()
        assert page.width() >= need.width(), f"第 {index} 页被压窄，文字会被裁掉"
        assert page.height() >= need.height(), f"第 {index} 页被压扁，文字会被裁掉"


def test_stylesheet_heights_follow_font_metrics(qapp: QApplication) -> None:
    """进度条这类「固定高度」控件必须按字体行高推导，不能写死像素。"""
    from video2personvideo.gui import theme

    metrics = theme.ui_metrics()
    qss = theme.build_stylesheet(theme.LIGHT, metrics)

    assert metrics.progress_height >= metrics.line_height
    assert metrics.thin_progress_height >= metrics.line_height
    assert f"height: {metrics.progress_height}px" in qss
    assert f"height: {metrics.thin_progress_height}px" in qss
    assert "height: 18px" not in qss  # 旧写死值（13px 字号下会把文字挤扁）


def test_ratio_cards_fit_their_text(qapp: QApplication) -> None:
    """比例卡片要放得下「比例名 + 分辨率」两行文字。"""
    from video2personvideo.gui.ratio_grid import RatioGrid

    grid = RatioGrid()
    grid.resize(900, 400)
    _settle(qapp)

    assert grid._cards
    for card in grid._cards:
        metrics = card.fontMetrics()
        assert card.height() >= metrics.lineSpacing() * 2
        assert card.width() >= metrics.horizontalAdvance("1920×1080")
    grid.close()


def test_ratio_grid_reflows_by_available_width(qapp: QApplication) -> None:
    """网格列数按可用宽度自适应：宽了多排几列、窄了少排几列，卡片本身不变形。"""
    from PySide6.QtWidgets import QVBoxLayout, QWidget

    from video2personvideo.gui.ratio_grid import RatioGrid

    holder = QWidget()
    box = QVBoxLayout(holder)
    box.setContentsMargins(0, 0, 0, 0)
    grid = RatioGrid()
    box.addWidget(grid)
    holder.resize(900, 400)
    holder.show()
    _settle(qapp)
    wide_columns = grid._columns
    card_width = grid._cards[0].width()

    holder.resize(400, 400)
    _settle(qapp)
    narrow_columns = grid._columns

    assert wide_columns > narrow_columns >= 1
    assert grid._cards[0].width() == card_width  # 卡片尺寸不受列数影响
    # 最小宽度按「一张卡片」算，否则列数会被自己锁死、窗口再也窄不下来
    assert grid.minimumSizeHint().width() <= card_width
    holder.close()
