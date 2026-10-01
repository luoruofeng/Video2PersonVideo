"""处理前预览：抽样若干帧跑一遍构图算法，对照展示裁剪前后的效果。

长视频跑完才发现构图不对代价太高，因此在全量处理前先给用户一次确认机会
（M3-4）。检测结果只算一次，之后调整参数是纯几何重算，交互即时。
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QObject, Qt, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ..config import AppConfig
from ..core.crop import DEFAULT_BACKGROUND, WindowSlice, compose_multi_frame, crop_frame
from ..core.framing import MODE_TILES, compute_target_box, mode_label
from ..core.layout import TransitionFrame, fit_rect
from ..core.multi import plan_static_multi
from ..core.noperson import TilesPlanner, whole_frame_box
from ..core.ratio import AspectRatio
from ..core.sharpness import annotate_sharpness
from ..core.smoothing import (
    CAMERA_PRESETS,
    DEFAULT_NO_PERSON_MODE,
    NO_PERSON_LABELS,
    NO_PERSON_MODES,
    PRESET_FIELDS,
    match_preset,
    no_person_mode_label,
    preset_values,
)
from ..core.subject import (
    DEFAULT_MIN_PERSON_SHARPNESS,
    Detection,
    select_subject,
)
from ..core.video_io import VideoReader
from . import theme
from .pages.ratio_page import (
    CUSTOM_FOLLOW_KEY,
    FOLLOW_TOOLTIP,
    MULTI_PERSON_TOOLTIP,
    NO_PERSON_TOOLTIP,
)

#: 抽样帧数
SAMPLE_COUNT = 5
#: 单行预览图高度
TILE_HEIGHT = 200


def bgr_to_qimage(image: np.ndarray) -> QImage:
    """OpenCV 的 BGR ndarray → Qt 的 QImage。"""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb.shape
    return QImage(
        rgb.data, width, height, channels * width, QImage.Format.Format_RGB888
    ).copy()


class PreviewSampler(QObject):
    """在子线程里抽样并对每帧跑一次检测。"""

    ready = Signal(object)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(self, cfg: AppConfig, video: Path) -> None:
        super().__init__()
        self._cfg = cfg
        self._video = Path(video)

    @Slot()
    def run(self) -> None:
        try:
            samples = self._sample()
        except Exception as exc:  # noqa: BLE001 - 统一上报
            self.failed.emit(str(exc))
            return
        self.ready.emit(samples)

    def _sample(self) -> list[dict]:
        from ..core.detector import PersonDetector

        detector = PersonDetector(
            model=self._cfg.model,
            conf=self._cfg.conf,
            iou=self._cfg.iou,
            imgsz=self._cfg.imgsz,
            classes=self._cfg.classes or None,
            max_det=self._cfg.max_det,
            device=self._cfg.device,
            use_keypoints=self._cfg.use_keypoints,
        )

        self.progress.emit("正在读取采样帧…")
        with VideoReader(self._video) as reader:
            total = reader.meta.frame_count or 0
            frames: list[tuple[int, np.ndarray]] = []
            for index, frame in reader.frames():
                frames.append((index, frame))
                if total and len(frames) >= total:
                    break
        if not frames:
            raise RuntimeError(f"无法从视频中读取任何帧：{self._video}")

        if len(frames) <= SAMPLE_COUNT:
            picked = frames
        else:
            step = (len(frames) - 1) / (SAMPLE_COUNT - 1)
            picked = [frames[int(round(step * i))] for i in range(SAMPLE_COUNT)]

        samples: list[dict] = []
        for index, frame in picked:
            self.progress.emit(f"正在对第 {index} 帧做构图分析…")
            detections: list[Detection] = detector.detect_boxes(frame)
            samples.append({"index": index, "frame": frame, "detections": detections})
        return samples


class PreviewDialog(QDialog):
    """构图效果对照对话框。"""

    def __init__(self, cfg: AppConfig, video: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"构图预览 · {Path(video).name}")
        # 尺寸按屏幕可用区域收缩：高缩放比下窗口（逻辑像素）会小很多，
        # 写死 860×620 会撑出屏幕。图片区本身在滚动区里，小窗口也能看全。
        self.resize(*theme.clamp_size((860, 620), (520, 400), screen=self.screen()))
        self.setMinimumSize(*theme.clamp_size((640, 480), (420, 320), screen=self.screen()))

        self._cfg = cfg
        self._video = Path(video)
        self._samples: list[dict] = []
        self._thread = None
        self._worker: PreviewSampler | None = None
        self._follow_key = match_preset(cfg.smoothing_alpha) or CUSTOM_FOLLOW_KEY
        self._follow_values = {name: float(getattr(cfg, name)) for name in PRESET_FIELDS}
        self._render_params = {
            **self._follow_values,
            "min_person_height_ratio": cfg.min_person_height_ratio,
            "min_person_sharpness": cfg.min_person_sharpness,
            "headroom": cfg.headroom,
            "multi_person": bool(cfg.multi_person),
        }
        self._no_person = (
            cfg.no_person_mode if cfg.no_person_mode in NO_PERSON_MODES else DEFAULT_NO_PERSON_MODE
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(theme.SPACING_LARGE, theme.SPACING_LARGE,
                                  theme.SPACING_LARGE, theme.SPACING_LARGE)

        self.status_label = QLabel("正在准备预览…")
        self.status_label.setObjectName("status")
        layout.addWidget(self.status_label)

        layout.addWidget(self._build_tuning_group())

        self.image_label = QLabel("正在抽取采样帧并分析构图…")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.image_label)
        layout.addWidget(scroll, stretch=1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("使用当前参数")
        buttons.button(QDialogButtonBox.StandardButton.Ok).setObjectName("primary")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("关闭")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._start()

    # ------------------------------------------------------------- 子控件
    def _build_tuning_group(self) -> QGroupBox:
        group = QGroupBox("可调参数（实时重算，无需重新检测）")
        form = QFormLayout(group)
        form.setSpacing(theme.SPACING_SMALL)

        self.follow_combo = QComboBox()
        self.follow_combo.setToolTip(
            FOLLOW_TOOLTIP + "\n\n（静态预览看不出镜头运动的差异，它决定正片里镜头跟得有多紧）"
        )
        for key, preset in CAMERA_PRESETS.items():
            self.follow_combo.addItem(preset.label, key)
        self.follow_combo.addItem("自定义", CUSTOM_FOLLOW_KEY)
        self.follow_combo.setCurrentIndex(max(self.follow_combo.findData(self._follow_key), 0))
        self.follow_combo.currentIndexChanged.connect(self._on_follow_changed)
        form.addRow("镜头跟随", self.follow_combo)

        self.person_slider, person_row = self._slider(
            1, 90, int(self._render_params["min_person_height_ratio"] * 100)
        )
        form.addRow("人物最小占比", person_row)

        sharpness = float(
            self._render_params.get("min_person_sharpness", DEFAULT_MIN_PERSON_SHARPNESS)
        )
        self.sharpness_slider, sharpness_row = self._slider(0, 100, int(round(sharpness * 100)))
        self.sharpness_slider.setToolTip(
            "人物清晰度下限：与「人物最小占比」并列，两者都达标才算主要人物。\n"
            "被镜头虚化的人（背景里的路人、远处海报 / 屏幕里的人）即便占比很大，\n"
            "清晰度不达标也会被当成背景人物。默认 0.30；0 = 关闭这项判定。"
        )
        form.addRow("人物清晰度下限", sharpness_row)

        self.headroom_slider, headroom_row = self._slider(
            0, 30, int(self._render_params["headroom"] * 100)
        )
        form.addRow("头顶留白", headroom_row)

        self.multi_box = QCheckBox("多人分屏 · 每人一个上半身窗口")
        self.multi_box.setChecked(bool(self._render_params["multi_person"]))
        self.multi_box.setToolTip(MULTI_PERSON_TOOLTIP)
        self.multi_box.stateChanged.connect(self._on_tuning_changed)
        form.addRow("多人分屏", self.multi_box)

        self.no_person_combo = QComboBox()
        self.no_person_combo.setToolTip(NO_PERSON_TOOLTIP)
        for mode in NO_PERSON_MODES:
            self.no_person_combo.addItem(NO_PERSON_LABELS[mode], mode)
        self.no_person_combo.setCurrentIndex(
            max(self.no_person_combo.findData(self._no_person), 0)
        )
        self.no_person_combo.currentIndexChanged.connect(self._on_tuning_changed)
        form.addRow("没有人物时", self.no_person_combo)
        return group

    def _slider(self, minimum: int, maximum: int, value: int) -> tuple[QSlider, QWidget]:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.SPACING_SMALL)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(value)
        label = QLabel(f"{value / 100:.2f}")
        label.setMinimumWidth(theme.text_width_px("0.00", padding=8))
        slider.valueChanged.connect(lambda v: label.setText(f"{v / 100:.2f}"))
        slider.valueChanged.connect(self._on_tuning_changed)
        layout.addWidget(slider, stretch=1)
        layout.addWidget(label)
        return slider, row

    # ----------------------------------------------------------------- 接口
    def tuning(self) -> dict[str, float]:
        return {
            **self._follow_values,
            "min_person_height_ratio": self.person_slider.value() / 100.0,
            "min_person_sharpness": self.sharpness_slider.value() / 100.0,
            "headroom": self.headroom_slider.value() / 100.0,
        }

    def multi_person(self) -> bool:
        """预览里试出来的"多人分屏"开关（确认后回写到设置页）。"""
        return self.multi_box.isChecked()

    def no_person_mode(self) -> str:
        """预览里试出来的"没有人物时"显示方式（确认后回写到设置页）。"""
        return str(self.no_person_combo.currentData())

    # ------------------------------------------------------------- 后台任务
    def _start(self) -> None:
        from PySide6.QtCore import QThread

        self._thread = QThread(self)
        self._worker = PreviewSampler(self._cfg, self._video)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.status_label.setText)
        self._worker.ready.connect(self._on_samples)
        self._worker.failed.connect(self._on_failed)
        self._thread.start()

    @Slot(object)
    def _on_samples(self, samples: list[dict]) -> None:
        # 清晰度只跟"帧 + 人物框"有关、与阈值无关：先算一次存进样本里，
        # 之后拖动滑块重算构图就不必重复算 Laplacian。
        for sample in samples:
            sample["detections"] = annotate_sharpness(
                sample["frame"], sample["detections"]
            )
        self._samples = samples
        self._teardown()
        self.status_label.setText(
            f"已分析 {len(samples)} 帧（左：原画面 + 取景框 / 右：裁剪结果）"
        )
        self._render()

    @Slot(str)
    def _on_failed(self, message: str) -> None:
        self._teardown()
        self.status_label.setText(f"预览失败：{message}")
        self.image_label.setText(message)

    def _teardown(self) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(5000)
            self._thread = None
        self._worker = None

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt 命名约定
        self._teardown()
        super().closeEvent(event)

    # ------------------------------------------------------------- 渲染
    def _on_follow_changed(self) -> None:
        key = str(self.follow_combo.currentData())
        if key in CAMERA_PRESETS:
            self._follow_values = preset_values(key)
        self._on_tuning_changed()

    def _on_tuning_changed(self) -> None:
        self._render_params = {**self.tuning(), "multi_person": self.multi_box.isChecked()}
        self._no_person = str(self.no_person_combo.currentData())
        if self._samples:
            self._render()

    def _render(self) -> None:
        if not self._samples:
            return

        cfg = replace(
            self._cfg, **self._render_params, no_person_mode=self._no_person
        )
        params = cfg.framing_params()
        weights = cfg.subject_weights()
        ratio = cfg.resolve_ratio()
        # 布局外观（底色 / 缝隙）无论是否开启多人分屏都要用，故单独取一份
        display_policy = cfg.multi_person_policy()
        policy = display_policy if cfg.multi_person else None

        rows = [
            self._render_row(
                sample, cfg, ratio, params, weights, policy, display_policy
            )
            for sample in self._samples
        ]
        max_width = max(row.shape[1] for row in rows)
        padded = [
            cv2.copyMakeBorder(
                row,
                0,
                0,
                0,
                max_width - row.shape[1],
                cv2.BORDER_CONSTANT,
                value=(255, 255, 255),
            )
            for row in rows
        ]
        separator = np.full((6, max_width, 3), 210, dtype=np.uint8)
        composite = padded[0]
        for row in padded[1:]:
            composite = np.vstack([composite, separator, row])

        self.image_label.setPixmap(QPixmap.fromImage(bgr_to_qimage(composite)))

    def _render_row(
        self,
        sample: dict,
        cfg: AppConfig,
        ratio: AspectRatio,
        params,
        weights,
        policy=None,
        display_policy=None,
    ) -> np.ndarray:
        frame = sample["frame"]
        detections = sample["detections"]
        frame_size = (float(frame.shape[1]), float(frame.shape[0]))
        min_person_ratio = cfg.min_person_height_ratio
        min_person_sharpness = cfg.min_person_sharpness

        windows: list[WindowSlice] | None = (
            plan_static_multi(
                detections,
                frame_size,
                ratio,
                params=params,
                policy=policy,
                min_person_height_ratio=min_person_ratio,
                min_person_sharpness=min_person_sharpness,
            )
            if policy is not None
            else None
        )

        if windows:
            # 多人分屏：每个窗口按自己的比例取景，拼成成品的样子
            cropped = compose_multi_frame(
                frame, windows, ratio.target_size, background=policy.background
            )
            boxes = [window.box for window in windows]
            title = f"多人分屏 ×{len(windows)}"
        else:
            subject = select_subject(
                detections,
                frame_size,
                weights=weights,
                min_height_ratio=min_person_ratio,
                min_sharpness=min_person_sharpness,
            )
            if subject is None:
                # 没有主要人物：按"没有人物时"的档位预览（整幅画面 / 全景 + 特写）
                windows = _no_person_windows(
                    cfg, frame_size, ratio, detections, params, display_policy
                )
                background = (
                    display_policy.background if display_policy is not None else DEFAULT_BACKGROUND
                )
                cropped = compose_multi_frame(
                    frame,
                    windows,
                    ratio.target_size,
                    background=background,
                    annotate=False,
                    blur=cfg.no_person_blur,
                )
                boxes = [window.box for window in windows]
                title = f"无人物：{no_person_mode_label(cfg.no_person_mode)}"
            else:
                box = compute_target_box(subject.bbox, frame_size, ratio, params)
                cropped = crop_frame(frame, box, ratio.target_size)
                boxes = [box]
                title = mode_label(box.mode)

        source_view = _fit_height(_draw_boxes(frame, boxes), TILE_HEIGHT)
        crop_view = _fit_height(cropped, TILE_HEIGHT)
        cv2.putText(
            source_view,
            f"#{sample['index']} {title}",
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )
        gap = np.full((TILE_HEIGHT, 8, 3), 255, dtype=np.uint8)
        return np.hstack([source_view, gap, crop_view])


def _no_person_windows(
    cfg: AppConfig,
    frame_size: tuple[float, float],
    ratio: AspectRatio,
    detections: list[Detection],
    params,
    policy,
) -> list[WindowSlice]:
    """画面里没有主要人物时，这一帧会贴哪些窗口（预览用）。

    直接复用正片的排布逻辑（:class:`TilesPlanner` + ``fit_rect``），
    于是"预览看到的"与"正片输出的"是同一套几何，不会各说各话。
    """
    if cfg.no_person_mode == MODE_TILES and policy is not None:
        planner = TilesPlanner(
            ratio,
            params=params,
            policy=policy,
            min_person_height_ratio=cfg.min_person_height_ratio,
            secondary_ratio=cfg.no_person_secondary_ratio,
            max_details=cfg.no_person_tiles_max,
        )
        planner.update(detections, frame_size)
        cell = planner.dst_rect(frame_size)
        if cell is not None:
            prepared = planner.windows(
                frame_size,
                TransitionFrame(
                    src=(0.0, 0.0, float(frame_size[0]), float(frame_size[1])),
                    dst=cell,
                    progress=1.0,
                ),
            )
            if prepared:
                return prepared

    return [
        WindowSlice(
            whole_frame_box(frame_size), fit_rect(frame_size, ratio.target_size), fit=True
        )
    ]


def _draw_boxes(frame: np.ndarray, boxes) -> np.ndarray:
    """在源帧上画出取景框；多人分屏时每个窗口一个，并标上窗口编号。"""
    view = frame.copy()
    multiple = len(boxes) > 1
    for order, box in enumerate(boxes, start=1):
        x, y, width, height = box.as_int_even(frame.shape[1], frame.shape[0])
        cv2.rectangle(view, (x, y), (x + width, y + height), (0, 200, 255), 2)
        if multiple:
            cv2.putText(
                view,
                str(order),
                (x + 6, y + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 200, 255),
                2,
            )
    return view


def _fit_height(image: np.ndarray, height: int) -> np.ndarray:
    """等比缩放到指定高度。"""
    scale = height / image.shape[0]
    width = max(int(round(image.shape[1] * scale)), 2)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
