"""向导步骤 2：选择目标长宽比，并可微调构图参数。"""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ...core.framing import DEFAULT_FRAMING_PARAMS, DEFAULT_MIN_PERSON_HEIGHT_RATIO
from ...core.ratio import PRESET_RATIOS, AspectRatio
from ...core.smoothing import (
    CAMERA_PRESETS,
    DEFAULT_CAMERA_PRESET,
    DEFAULT_NO_PERSON_MODE,
    NO_PERSON_DESCRIPTIONS,
    NO_PERSON_LABELS,
    NO_PERSON_MODES,
    PRESET_FIELDS,
    match_preset,
    preset_values,
)
from .. import theme
from ..framing_sketch import FramingSketch
from ..ratio_dialog import RatioDialog
from ..ratio_grid import RatioGrid, make_ratio_icon

#: 预览缩略图尺寸
PREVIEW_ICON_SIZE = QSize(132, 84)

#: "自定义"档位：预览对话框回写过非档位参数时显示它
CUSTOM_FOLLOW_KEY = "custom"

#: 「镜头跟随」下拉的提示语（把参数翻译成人话）
FOLLOW_TOOLTIP = (
    "人物走动时镜头怎么跟（越靠上的档位越稳、越不容易看晕）：\n"
    "· 锁定：人物在画面内走动时镜头完全不动，只有快走出画面时才缓慢平移\n"
    "· 舒缓：镜头偶尔缓慢平移，人物可在画面内较自由地走动\n"
    "· 标准：镜头较少移动、人物基本居中\n"
    "· 跟手：默认，镜头跟得紧，画面运动感强\n"
    "· 自定义：在「预览构图效果」里细调过平滑参数"
)

#: 「多人分屏」勾选项的提示语
MULTI_PERSON_TOOLTIP = (
    "画面里出现两个及以上主要人物时，每人给一个上半身小窗口同时显示，\n"
    "而不是整段视频只跟一个人。\n"
    "· 默认开启（最多同时显示 4 人，可在配置文件里调整）\n"
    "· 关掉它 = 整段视频同时只显示一个人\n"
    "· 窗口怎么排（谁的窗口放哪、每格多大）由 configs/multi_person_layout.yaml\n"
    "  按「输出比例 + 人数」查表决定，可自行修改\n"
    "· 背景里的小人 / 路人不算主要人物，不会进窗口\n"
    "· 单人画面不受影响：输出与关闭该功能时完全一致"
)

#: 「无人物显示」下拉的提示语
NO_PERSON_TOOLTIP = (
    "画面里没有主要人物时（人都走开了、只有空镜）这一段怎么显示。\n"
    "无论选哪一档，都会先从人物的取景框平缓过渡过去，不会突然一跳。\n"
    + "\n".join(
        f"· {NO_PERSON_LABELS[mode]}：{NO_PERSON_DESCRIPTIONS[mode]}" for mode in NO_PERSON_MODES
    )
)


class RatioPage(QWidget):
    """比例选择 + 构图参数微调页。"""

    ratioChanged = Signal(object)
    tuningChanged = Signal()
    previewRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.SPACING)

        hint = QLabel("选一个长宽比；下方参数可微调构图，建议先点「预览构图效果」确认。")
        hint.setObjectName("hint")
        layout.addWidget(hint)

        ratio_group = QGroupBox("目标长宽比")
        ratio_box = QVBoxLayout(ratio_group)

        top_row = QHBoxLayout()
        self.preview_icon = QLabel()
        self.preview_icon.setFixedSize(PREVIEW_ICON_SIZE)
        self.preview_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        top_row.addWidget(self.preview_icon)

        info_box = QVBoxLayout()
        self.current_label = QLabel()
        # 字号交给 QSS 的 #subhead（由基准字号推导），不在这里写死像素
        self.current_label.setObjectName("subhead")
        self.resolution_label = QLabel()
        self.resolution_label.setObjectName("hint")
        self.choose_button = QPushButton("打开比例选择对话框…")
        self.choose_button.clicked.connect(self._open_dialog)
        info_box.addWidget(self.current_label)
        info_box.addWidget(self.resolution_label)
        info_box.addWidget(self.choose_button)
        info_box.addStretch(1)
        top_row.addLayout(info_box, stretch=1)
        ratio_box.addLayout(top_row)

        self.grid = RatioGrid()
        self.grid.ratioChanged.connect(self._on_grid_changed)
        ratio_box.addWidget(self.grid)
        layout.addWidget(ratio_group)

        layout.addWidget(self._build_tuning_group())
        layout.addStretch(1)

        #: 当前生效的平滑参数（档位展开 / 预览回写都会更新它）
        self._follow_values: dict[str, float] = preset_values(DEFAULT_CAMERA_PRESET)
        self._set_follow(DEFAULT_CAMERA_PRESET)
        self.set_ratio(PRESET_RATIOS[1])

        # 任一构图参数变化 → 立刻重画示意图，让用户直接看到改动效果
        self.tuningChanged.connect(self._refresh_sketch)
        self.ratioChanged.connect(lambda _ratio: self._refresh_sketch())
        self._refresh_sketch()

    # ------------------------------------------------------------- 子控件
    def _build_tuning_group(self) -> QGroupBox:
        group = QGroupBox("构图参数（可选微调）")
        outer = QVBoxLayout(group)
        outer.setSpacing(theme.SPACING)

        # 参数效果示意图：改动下面任一参数都会即时重画（纯几何，不读视频）
        self.sketch = FramingSketch()
        outer.addWidget(self.sketch)
        self.sketch_caption = QLabel()
        self.sketch_caption.setObjectName("hint")
        self.sketch_caption.setWordWrap(True)
        outer.addWidget(self.sketch_caption)

        form = QFormLayout()
        form.setSpacing(theme.SPACING_SMALL)

        self.follow_combo = QComboBox()
        self.follow_combo.setToolTip(FOLLOW_TOOLTIP)
        for key, preset in CAMERA_PRESETS.items():
            self.follow_combo.addItem(preset.label, key)
        self.follow_combo.addItem("自定义", CUSTOM_FOLLOW_KEY)
        self.follow_combo.currentIndexChanged.connect(self._on_follow_changed)
        form.addRow("镜头跟随", self.follow_combo)

        min_person_default = int(round(DEFAULT_MIN_PERSON_HEIGHT_RATIO * 100))
        self.min_person_slider, person_row = self._make_slider(1, 90, min_person_default)
        self.min_person_slider.setToolTip(
            "人物最小占比：低于该高度占比的人物视为不存在（背景里的路人 / 小人）。\n"
            "默认 0.73 = 只把画面里占比很大的人当主要人物，往小调可放回更多小人。"
        )
        form.addRow("人物最小占比", person_row)

        headroom_default = int(round(DEFAULT_FRAMING_PARAMS.headroom * 100))
        self.headroom_slider, headroom_row = self._make_slider(0, 30, headroom_default)
        self.headroom_slider.setToolTip("半身构图时头顶留白占取景框高度的比例")
        form.addRow("头顶留白", headroom_row)

        self.multi_person_box = QCheckBox("多人分屏 · 每人一个上半身窗口")
        self.multi_person_box.setChecked(True)
        self.multi_person_box.setToolTip(MULTI_PERSON_TOOLTIP)
        self.multi_person_box.stateChanged.connect(lambda _: self.tuningChanged.emit())
        form.addRow("多人分屏", self.multi_person_box)

        self.no_person_combo = QComboBox()
        self.no_person_combo.setToolTip(NO_PERSON_TOOLTIP)
        for mode in NO_PERSON_MODES:
            self.no_person_combo.addItem(NO_PERSON_LABELS[mode], mode)
        self.no_person_combo.currentIndexChanged.connect(lambda _: self.tuningChanged.emit())
        form.addRow("没有人物时", self.no_person_combo)

        self.speaker_box = QCheckBox("优先对准正在说话的人")
        self.speaker_box.setChecked(True)
        self.speaker_box.setToolTip(
            "画面里有多个主要人物时，镜头对准正在说话的那一位（而不是最大 / 最居中的人）。\n"
            "做法：抽音轨算语音能量 + 看每个人的嘴部运动，谁的声音与嘴型最同步就跟谁；\n"
            "带滞回，不会在两人之间来回跳。需要系统里有 ffmpeg；单人画面、无音轨或\n"
            "检测不到语音时自动退回原来的主角规则。"
        )
        self.speaker_box.stateChanged.connect(lambda _: self.tuningChanged.emit())
        form.addRow("说话人跟随", self.speaker_box)

        self.size_guard_box = QCheckBox("自动控制输出体积")
        self.size_guard_box.setChecked(True)
        self.size_guard_box.setToolTip(
            "裁剪后的视频由 OpenCV 写 mp4v，体积往往比原视频大好几倍。\n"
            "勾选后：输出一旦超过原视频体积，就自动用 ffmpeg 压成 H.264 压回去"
            "（需要系统里有 ffmpeg）。"
        )
        self.size_guard_box.stateChanged.connect(lambda _: self.tuningChanged.emit())
        form.addRow("输出体积", self.size_guard_box)

        self.preview_button = QPushButton("预览构图效果…")
        self.preview_button.setToolTip("抽取若干采样帧，先看裁剪效果再决定是否全量处理")
        self.preview_button.clicked.connect(self.previewRequested.emit)
        form.addRow("", self.preview_button)
        outer.addLayout(form)
        return group

    def _make_slider(self, minimum: int, maximum: int, value: int) -> tuple[QSlider, QWidget]:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.SPACING_SMALL)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(value)
        value_label = QLabel(f"{value / 100:.2f}")
        # 按字体实测宽度留位，换字号 / 缩放比例也不会把数字挤掉
        value_label.setMinimumWidth(theme.text_width_px("0.00", padding=8))
        slider.valueChanged.connect(lambda v: value_label.setText(f"{v / 100:.2f}"))
        slider.valueChanged.connect(lambda _: self.tuningChanged.emit())
        layout.addWidget(slider, stretch=1)
        layout.addWidget(value_label)
        return slider, row

    # ----------------------------------------------------------------- 接口
    def current_ratio(self) -> AspectRatio:
        return self.grid.current_ratio

    def set_ratio(self, ratio: AspectRatio) -> None:
        self.grid.set_current(ratio)
        self._update_labels(ratio)

    def tuning(self) -> dict[str, float]:
        """返回当前构图微调参数（含「镜头跟随」档位展开出的平滑参数）。"""
        return {
            **self._follow_values,
            "min_person_height_ratio": self.min_person_slider.value() / 100.0,
            "headroom": self.headroom_slider.value() / 100.0,
        }

    def follow_key(self) -> str:
        """当前「镜头跟随」档位名（``custom`` 表示自行细调过）。"""
        return str(self.follow_combo.currentData())

    def set_follow(self, key: str) -> None:
        """按档位名设置「镜头跟随」。"""
        if key not in CAMERA_PRESETS:
            key = DEFAULT_CAMERA_PRESET
        self._follow_values = preset_values(key)
        self._set_follow(key)
        self.tuningChanged.emit()

    def set_tuning(self, tuning: dict[str, float]) -> None:
        """应用一组构图微调参数（预览对话框确认后回写）。"""
        key = match_preset(tuning.get("smoothing_alpha"))
        if key is not None:
            self._follow_values = preset_values(key)
        else:
            # 参数不在任何档位上：记为"自定义"，只接受认识的字段
            custom = {
                name: float(tuning[name])
                for name in PRESET_FIELDS
                if tuning.get(name) is not None
            }
            if custom:
                self._follow_values.update(custom)
                key = CUSTOM_FOLLOW_KEY
        if key is not None:
            self._set_follow(key)

        for slider, name in (
            (self.min_person_slider, "min_person_height_ratio"),
            (self.headroom_slider, "headroom"),
        ):
            if tuning.get(name) is not None:
                slider.setValue(int(round(float(tuning[name]) * 100)))
        self.tuningChanged.emit()

    def size_guard(self) -> bool:
        """是否启用体积守护（输出超过原视频时自动压缩）。"""
        return self.size_guard_box.isChecked()

    def set_size_guard(self, enabled: bool) -> None:
        self.size_guard_box.setChecked(bool(enabled))

    def speaker_tracking(self) -> bool:
        """是否启用"优先对准正在说话的人"。"""
        return self.speaker_box.isChecked()

    def set_speaker_tracking(self, enabled: bool) -> None:
        self.speaker_box.setChecked(bool(enabled))

    def multi_person(self) -> bool:
        """是否启用"多人分屏"（多人时每人一个上半身窗口）。"""
        return self.multi_person_box.isChecked()

    def set_multi_person(self, enabled: bool) -> None:
        self.multi_person_box.setChecked(bool(enabled))

    def no_person_mode(self) -> str:
        """没有主要人物时的显示方式（``fit`` / ``tiles`` / ``scan`` / ``center``）。"""
        return str(self.no_person_combo.currentData())

    def set_no_person_mode(self, mode: str) -> None:
        index = self.no_person_combo.findData(str(mode))
        if index < 0:
            index = self.no_person_combo.findData(DEFAULT_NO_PERSON_MODE)
        if index >= 0 and index != self.no_person_combo.currentIndex():
            self.no_person_combo.setCurrentIndex(index)
            self.tuningChanged.emit()

    # ------------------------------------------------------------- 内部逻辑
    def _on_follow_changed(self) -> None:
        key = self.follow_key()
        if key in CAMERA_PRESETS:
            self._follow_values = preset_values(key)
        self.tuningChanged.emit()

    def _set_follow(self, key: str) -> None:
        """切换下拉项但不再触发一遍信号（值由调用方负责）。"""
        index = self.follow_combo.findData(key)
        if index < 0:
            return
        blocked = self.follow_combo.blockSignals(True)
        self.follow_combo.setCurrentIndex(index)
        self.follow_combo.blockSignals(blocked)

    def _on_grid_changed(self, ratio: AspectRatio) -> None:
        self._update_labels(ratio)
        self.ratioChanged.emit(ratio)

    def _update_labels(self, ratio: AspectRatio) -> None:
        self.current_label.setText(f"当前选择：{ratio.name}")
        self.resolution_label.setText(
            f"预估输出分辨率：{ratio.target_width} × {ratio.target_height}"
        )
        self.preview_icon.setPixmap(make_ratio_icon(ratio, PREVIEW_ICON_SIZE).pixmap(PREVIEW_ICON_SIZE))

    def _refresh_sketch(self) -> None:
        """把当前参数交给示意图重画，并用一句话说明它们的效果。"""
        self.sketch.configure(
            ratio=self.current_ratio(),
            min_person_ratio=self.min_person_slider.value() / 100.0,
            headroom=self.headroom_slider.value() / 100.0,
            follow_key=self.follow_key(),
            multi_person=self.multi_person(),
            no_person_mode=self.no_person_mode(),
        )
        self.sketch_caption.setText(self._sketch_caption())

    def _sketch_caption(self) -> str:
        """示意图下方的白话说明：每个参数在做什么、往哪调会怎样。"""
        min_ratio = self.min_person_slider.value() / 100.0
        headroom = self.headroom_slider.value() / 100.0
        key = self.follow_key()
        label = CAMERA_PRESETS[key].label if key in CAMERA_PRESETS else "自定义"
        multi = (
            "橙色编号框 = 多人分屏时每人一个小窗口。"
            if self.multi_person()
            else "「多人分屏」已关闭，只跟一位主角。"
        )
        mode = self.no_person_mode()
        return (
            f"最小占比 {min_ratio:.2f}：比虚线更矮的人算背景，会被忽略；"
            f"头顶留白 {headroom:.2f}：半身时头顶到画面上沿的距离，越大头顶越空、人越小；"
            f"镜头跟随「{label}」：曲线越平 = 画面越稳；"
            f"没有人时「{NO_PERSON_LABELS[mode]}」：{NO_PERSON_DESCRIPTIONS[mode]}。{multi}"
        )

    def _open_dialog(self) -> None:
        dialog = RatioDialog(self.current_ratio(), self)
        if dialog.exec():
            self.set_ratio(dialog.selected_ratio())
            self.ratioChanged.emit(dialog.selected_ratio())
