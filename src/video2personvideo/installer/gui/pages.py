"""安装向导的各个页面（纯界面，不含任何安装逻辑）。

五个页面串起一次安装：欢迎 → 安装位置 → 组件 → 安装 → 完成。
每个页面只做两件事：把用户的选择暴露成信号 / 属性，把后台状态渲染出来。
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ... import __version__
from ...gui import theme
from ...utils.gpu_probe import HardwareProfile
from ...utils.torch_backends import BackendRecommendation
from ...utils.torch_install import YOLO_WEIGHTS
from .. import paths
from ..journal import InstallProbe
from ..options import OPTIONAL_WEIGHTS, InstallOptions
from ..stages import ORDERED_STAGES, Stage

#: 步骤标题（左侧步骤条）
STEP_TITLES = ("欢迎", "选择安装位置", "选择组件", "正在安装", "完成")
#: 每一步的副标题
STEP_SUBTITLES = (
    "先看一遍说明，然后开始",
    "装到哪里由你决定",
    "按显卡挑一套 PyTorch",
    "全部下载与安装都在这一步",
    "可以开始用了",
)

#: 许可摘要（正文在仓库 LICENSE 里，界面只给要点）
LICENSE_SUMMARY = (
    "本程序以 Apache-2.0 许可发布，可自由使用、修改与分发。\n\n"
    "安装过程中会下载以下第三方组件，它们各自遵循自己的许可：\n"
    "  · PyTorch（BSD-3-Clause）\n"
    "  · Ultralytics YOLO 与官方权重（AGPL-3.0，商用需向 Ultralytics 取得授权）\n"
    "  · 内嵌 Python 运行时（PSF License）\n"
    "  · ffmpeg（可选，LGPL / GPL，取决于所用构建）\n\n"
    "继续安装即表示你接受上述条款。"
)

#: 阶段状态的显示文字与配色键
_STAGE_STATUS = {
    "pending": ("待执行", "text_muted"),
    "running": ("进行中…", "primary"),
    "done": ("完成", "success"),
    "skipped": ("已跳过", "text_muted"),
    "failed": ("失败", "danger"),
    "cancelled": ("已暂停", "warning"),
}

#: 日志区最多保留多少行
MAX_LOG_LINES = 3000


# --------------------------------------------------------------------- 小工具
def _label(text: str, *, object_name: str = "", wrap: bool = False) -> QLabel:
    widget = QLabel(text)
    if object_name:
        widget.setObjectName(object_name)
    widget.setWordWrap(wrap)
    widget.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
    return widget


def _paragraph(text: str) -> QLabel:
    widget = _label(text, wrap=True)
    widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
    return widget


def _card(title: str) -> tuple[QGroupBox, QVBoxLayout]:
    box = QGroupBox(title)
    layout = QVBoxLayout(box)
    layout.setSpacing(theme.SPACING_SMALL)
    return box, layout


def _color_for(key: str) -> QColor:
    palette = theme.active()
    return QColor(getattr(palette, key, palette.text))


def restyle(widget: QWidget, object_name: str) -> None:
    """改控件的 ``objectName`` 并让样式表立即生效。

    QSS 是按 ``#名字`` 选中的，换名字之后 Qt 不会自动重新套用样式，
    必须手动 unpolish / polish 一次，否则"成功用绿字、失败用红字"这类
    状态切换看起来像没生效。
    """
    widget.setObjectName(object_name)
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)


# --------------------------------------------------------------------- 欢迎页
class WelcomePage(QWidget):
    """欢迎 + 许可 + "这台机器上装过没有"。"""

    agreementChanged = Signal(bool)
    resumeRequested = Signal()
    reinstallRequested = Signal()
    uninstallRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(theme.SPACING)

        layout.addWidget(_label(f"欢迎使用 Video2PersonVideo 安装向导 {__version__}", object_name="headline"))
        layout.addWidget(
            _paragraph(
                "这个向导会把程序装成一台机器上的普通软件：自带一份独立的 Python 运行环境，"
                "按你的显卡挑一套合适的 PyTorch，并把 YOLO 权重一起下好 —— "
                "装完之后双击快捷方式就能用，不需要再自己配环境。"
            )
        )
        layout.addWidget(
            _paragraph(
                "安装过程需要联网下载 PyTorch（CPU 版约 260 MB，CUDA 版 1~3 GB）。"
                "所有下载都支持断点续传：中途暂停、断网甚至关机，下次都会从断点接着下，不会白下。"
            )
        )

        # ---- 已安装 / 未完成的安装 ----
        self.state_box, self.state_layout = _card("安装状态")
        self.state_label = _paragraph("")
        self.state_layout.addWidget(self.state_label)
        self.state_hint = _label("", object_name="hint", wrap=True)
        self.state_layout.addWidget(self.state_hint)
        row = QHBoxLayout()
        self.resume_button = QPushButton("继续上次的安装")
        self.resume_button.setObjectName("primary")
        self.resume_button.clicked.connect(self.resumeRequested.emit)
        self.reinstall_button = QPushButton("重新安装")
        self.reinstall_button.clicked.connect(self.reinstallRequested.emit)
        self.uninstall_button = QPushButton("卸载")
        self.uninstall_button.setObjectName("danger")
        self.uninstall_button.clicked.connect(self.uninstallRequested.emit)
        row.addWidget(self.resume_button)
        row.addWidget(self.reinstall_button)
        row.addWidget(self.uninstall_button)
        row.addStretch(1)
        self.state_layout.addLayout(row)
        self.state_box.setVisible(False)
        layout.addWidget(self.state_box)

        # ---- 许可 ----
        license_box, license_layout = _card("许可条款")
        license_layout.addWidget(_paragraph(LICENSE_SUMMARY))
        self.view_license_button = QPushButton("查看许可全文")
        self.view_license_button.setObjectName("link")
        self.view_license_button.clicked.connect(self._show_license)
        license_layout.addWidget(self.view_license_button, alignment=Qt.AlignmentFlag.AlignLeft)
        self.agree_check = QCheckBox("我已阅读并同意上述许可条款")
        self.agree_check.toggled.connect(self.agreementChanged.emit)
        license_layout.addWidget(self.agree_check)
        layout.addWidget(license_box)
        layout.addStretch(1)

    def _show_license(self) -> None:
        dialog = TextDialog("许可全文", _read_license_text(), self)
        dialog.exec()

    def set_install_state(self, probe: InstallProbe) -> None:
        """根据"这台机器上装到哪了"决定状态卡片显示什么。"""
        if probe.journal is None and probe.registered_dir is None:
            self.state_box.setVisible(False)
            return
        self.state_box.setVisible(True)
        self.state_label.setText(probe.describe())

        journal = probe.journal
        resumable = bool(journal and not journal.complete and not probe.missing_files)
        self.resume_button.setVisible(resumable)
        self.reinstall_button.setVisible(journal is not None and not probe.missing_files)
        self.uninstall_button.setVisible(journal is not None or probe.registered_dir is not None)

        if probe.missing_files:
            self.state_hint.setText("先卸载残留的安装记录，再重新安装即可。")
        elif resumable and journal is not None:
            stage = journal.interrupted_stage()
            hint = journal.summary()
            if stage is not None:
                from ..stages import stage_info  # noqa: PLC0415 - 只在需要时导入

                hint += f"　·　上次退出的位置：{stage_info(stage).title}"
            self.state_hint.setText(hint)
        else:
            self.state_hint.setText("重新安装会复用已经下载好的安装包，不会重新下载一遍。")

    def agreed(self) -> bool:
        return self.agree_check.isChecked()

    def set_agreed(self, value: bool) -> None:
        self.agree_check.setChecked(bool(value))


class TextDialog(QDialog):
    """只读文本查看器（许可全文、安装日志）。"""

    def __init__(self, title: str, text: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        layout = QVBoxLayout(self)
        view = QPlainTextEdit(text)
        view.setReadOnly(True)
        view.setObjectName("logView")
        view.setMinimumSize(520, 360)
        layout.addWidget(view)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)
        theme.apply_window_size(self, preferred=(760, 580), minimum=(520, 400))


def _read_license_text() -> str:
    """读取随包的 LICENSE（找不到就退回许可摘要）。

    已经在机器上装过时优先读安装目录里的那份（它一定存在），
    否则从安装器自带的 payload 里读。
    """
    from ...utils.app_paths import install_root  # noqa: PLC0415
    from .. import payload  # noqa: PLC0415

    candidates: list[Path] = []
    root = install_root()
    if root is not None:
        candidates.append(root / "LICENSE")
    payload_dir = payload.payload_root()
    if payload_dir is not None:
        candidates.append(payload_dir / "LICENSE")

    for target in candidates:
        try:
            if target.is_file():
                return target.read_text(encoding="utf-8", errors="replace")
        except OSError:  # pragma: no cover
            continue
    return LICENSE_SUMMARY


# --------------------------------------------------------------------- 位置页
class LocationPage(QWidget):
    """选择安装目录 + 快捷方式 / 缓存等选项。"""

    optionsChanged = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(theme.SPACING)

        layout.addWidget(_label("安装位置", object_name="headline"))
        layout.addWidget(
            _paragraph(
                "安装目录会包含程序本体、一份独立的内嵌 Python 运行环境和 YOLO 权重。"
                "默认装在当前用户目录下，**不需要管理员权限**，也不会影响系统里的 Python。"
            )
        )

        row = QHBoxLayout()
        self.path_edit = QLineEdit(str(paths.default_install_dir()))
        self.path_edit.textChanged.connect(self._on_changed)
        self.browse_button = QPushButton("浏览…")
        self.browse_button.clicked.connect(self._browse)
        row.addWidget(self.path_edit, stretch=1)
        row.addWidget(self.browse_button)
        layout.addLayout(row)

        self.path_hint = _label("", object_name="hint", wrap=True)
        layout.addWidget(self.path_hint)
        self.space_label = _label("", object_name="status", wrap=True)
        layout.addWidget(self.space_label)

        options_box, options_layout = _card("其它选项")
        self.desktop_check = QCheckBox("创建桌面快捷方式")
        self.desktop_check.setChecked(True)
        self.start_menu_check = QCheckBox("添加到开始菜单（含卸载入口）")
        self.start_menu_check.setChecked(True)
        self.launch_check = QCheckBox("安装完成后立即启动程序")
        self.launch_check.setChecked(True)
        self.delete_downloads_check = QCheckBox("安装完成后删除下载的安装包（可省下数 GB，但重装需重新下载）")
        for widget in (
            self.desktop_check,
            self.start_menu_check,
            self.launch_check,
            self.delete_downloads_check,
        ):
            widget.toggled.connect(self._on_changed)
            options_layout.addWidget(widget)
        layout.addWidget(options_box)
        layout.addStretch(1)

    # ------------------------------------------------------------------ 交互
    def _browse(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, "选择安装目录", str(self.install_dir().parent)
        )
        if chosen:
            self.path_edit.setText(str(Path(chosen) / paths.APP_DIR_NAME))

    def _on_changed(self) -> None:
        self.refresh()
        self.optionsChanged.emit()

    # ------------------------------------------------------------------ 读写
    def install_dir(self) -> Path:
        text = self.path_edit.text().strip() or str(paths.default_install_dir())
        return Path(text).expanduser()

    def apply(self, options: InstallOptions) -> None:
        """把界面上的选择写回选项对象。"""
        options.install_dir = self.install_dir()
        options.desktop_shortcut = self.desktop_check.isChecked()
        options.start_menu_shortcut = self.start_menu_check.isChecked()
        options.launch_after_install = self.launch_check.isChecked()
        options.delete_downloads = self.delete_downloads_check.isChecked()

    def load(self, options: InstallOptions) -> None:
        self.path_edit.setText(str(options.install_dir))
        self.desktop_check.setChecked(options.desktop_shortcut)
        self.start_menu_check.setChecked(options.start_menu_shortcut)
        self.launch_check.setChecked(options.launch_after_install)
        self.delete_downloads_check.setChecked(options.delete_downloads)
        self.refresh()

    def refresh(self) -> None:
        """刷新磁盘空间与目录可用性提示。"""
        from ..engine import estimate_install_bytes  # noqa: PLC0415
        from .. import windows  # noqa: PLC0415

        target = self.install_dir()
        free = windows.free_space(target)
        needed = estimate_install_bytes(InstallOptions(install_dir=target))

        parts = [f"预计占用约 {paths.format_size(needed)}"]
        if free is not None:
            parts.append(f"目标磁盘剩余 {paths.format_size(free)}")
        self.space_label.setText("　·　".join(parts))
        restyle(
            self.space_label,
            "badgeDanger" if (free is not None and free < needed) else "status",
        )

    def validate(self) -> tuple[bool, str]:
        """返回 ``(是否可用, 说明)``；说明会显示在 hint 上。"""
        from .. import windows  # noqa: PLC0415

        target = self.install_dir()
        if not str(target).strip():
            return False, "请选择安装目录。"
        if target.is_file():
            return False, f"{target} 是一个文件，不能作为安装目录。"

        existing = False
        if target.is_dir():
            try:
                entries = [item for item in target.iterdir()]
            except OSError as exc:
                return False, f"无法访问该目录：{exc}"
            existing = bool(entries)
            if existing:
                marker = target / "install.json"
                if not marker.is_file():
                    return (
                        False,
                        "该目录里已经有别的东西（且不是本程序的安装目录）。"
                        "请换一个空目录，或先清空它。",
                    )

        parent = target if target.exists() else target.parent
        free = windows.free_space(parent)
        needed = windows.directory_size(target) if existing else 0
        if free is not None and free + needed < 1024 * 1024 * 1024:
            return False, f"目标磁盘剩余空间不足（{paths.format_size(free)}）。请换一个分区。"

        note = "该目录已存在本程序的安装文件，本次会继续 / 覆盖安装。" if existing else "目录可用。"
        return True, note


# --------------------------------------------------------------------- 组件页
class ComponentsPage(QWidget):
    """按硬件推荐 PyTorch 构建，并选择要一起装的东西。"""

    optionsChanged = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._profile: HardwareProfile | None = None
        self._recommendation: BackendRecommendation | None = None

        layout = QVBoxLayout(self)
        layout.setSpacing(theme.SPACING)

        layout.addWidget(_label("组件选择", object_name="headline"))
        self.summary_label = _label("正在检测这台电脑的硬件…", object_name="subhead", wrap=True)
        layout.addWidget(self.summary_label)

        hardware_box, hardware_layout = _card("硬件明细")
        self.hardware_table = QTableWidget(0, 2)
        self.hardware_table.horizontalHeader().setVisible(False)
        self.hardware_table.verticalHeader().setVisible(False)
        self.hardware_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.hardware_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.hardware_table.setShowGrid(False)
        self.hardware_table.setObjectName("checkTable")
        self.hardware_table.setMinimumHeight(150)
        hardware_layout.addWidget(self.hardware_table)
        layout.addWidget(hardware_box)

        backend_box, backend_layout = _card("PyTorch")
        self.backend_combo = QComboBox()
        self.backend_combo.currentIndexChanged.connect(self._on_backend_changed)
        backend_layout.addWidget(self.backend_combo)
        self.reason_label = _paragraph("")
        backend_layout.addWidget(self.reason_label)
        self.backend_detail = _label("", object_name="hint", wrap=True)
        backend_layout.addWidget(self.backend_detail)
        self.torch_check = QCheckBox("下载并安装 PyTorch（取消勾选表示复用本机已装的版本）")
        self.torch_check.setChecked(True)
        self.torch_check.toggled.connect(self._on_option_toggled)
        backend_layout.addWidget(self.torch_check)
        layout.addWidget(backend_box)

        extra_box, extra_layout = _card("其它组件")
        self.weight_checks: dict[str, QCheckBox] = {}
        for name, (label, approx) in YOLO_WEIGHTS.items():
            text = f"YOLO 权重 {name}（{label}，约 {paths.format_size(approx)}）"
            check = QCheckBox(text)
            check.setChecked(name not in OPTIONAL_WEIGHTS)
            if name not in OPTIONAL_WEIGHTS:
                check.setEnabled(False)  # 默认检测模型是必须的
                check.setToolTip("默认检测模型，安装后会跳过首次运行时的下载步骤。")
            check.toggled.connect(self._on_option_toggled)
            self.weight_checks[name] = check
            extra_layout.addWidget(check)
        self.ffmpeg_check = QCheckBox("随包安装 ffmpeg（保留音轨 / H.264 重编码 / 说话人跟随，约 90 MB）")
        self.ffmpeg_check.setChecked(True)
        self.ffmpeg_check.toggled.connect(self._on_option_toggled)
        extra_layout.addWidget(self.ffmpeg_check)
        layout.addWidget(extra_box)

        self.size_label = _label("", object_name="status", wrap=True)
        layout.addWidget(self.size_label)
        layout.addStretch(1)

    # ------------------------------------------------------------------ 自检
    def set_profile(self, profile: HardwareProfile) -> None:
        self._profile = profile
        rows = profile.summary_lines()
        self.hardware_table.setRowCount(len(rows))
        for index, (title, value) in enumerate(rows):
            self.hardware_table.setItem(index, 0, QTableWidgetItem(title))
            self.hardware_table.setItem(index, 1, QTableWidgetItem(value))
        self.hardware_table.resizeColumnsToContents()
        self.hardware_table.setColumnWidth(0, max(self.hardware_table.columnWidth(0), 110))

    def set_problem(self, message: str, *, level: str = "badgeWarn") -> None:
        """自检之类的"没成功但也不算失败"的提示。"""
        self.summary_label.setText(message)
        restyle(self.summary_label, level)

    def set_recommendation(self, recommendation: BackendRecommendation) -> None:
        self._recommendation = recommendation
        self.summary_label.setText(f"这台电脑建议安装：{recommendation.backend.display}")
        restyle(self.summary_label, "badgeInfo")
        self._fill_backends(recommendation)
        self.refresh()

    def _fill_backends(self, recommendation: BackendRecommendation) -> None:
        self.backend_combo.blockSignals(True)
        self.backend_combo.clear()
        options = {option.backend.key: option for option in recommendation.options}
        from ...utils.torch_backends import TORCH_BACKENDS  # noqa: PLC0415

        selected = -1
        for item in TORCH_BACKENDS:
            option = options.get(item.key)
            label = item.label
            if item.key == recommendation.backend.key:
                label += "　★ 推荐"
            if option is not None and not option.usable:
                label += f"（不可用：{option.reason}）"
            elif option is None:
                label += "（当前系统不提供）"
            self.backend_combo.addItem(label, item.key)
            index = self.backend_combo.count() - 1
            if option is not None and not option.usable:
                self.backend_combo.model().item(index).setEnabled(False)
            if item.key == recommendation.backend.key:
                selected = index
        if selected >= 0:
            self.backend_combo.setCurrentIndex(selected)
        self.backend_combo.blockSignals(False)
        self._refresh_reason()

    def _refresh_reason(self) -> None:
        recommendation = self._recommendation
        key = self.current_backend_key()
        if recommendation is None:
            self.reason_label.setText("")
            return
        if key == recommendation.backend.key:
            text = recommendation.reason
            if recommendation.warnings:
                text += "\n" + "\n".join(f"· {item}" for item in recommendation.warnings)
        else:
            option = next(
                (item for item in recommendation.options if item.backend.key == key), None
            )
            if option is not None and not option.usable:
                text = f"手动选择：{option.backend.display}。注意：{option.reason}"
            else:
                text = f"手动选择：{option.backend.display if option else key}。"
        self.reason_label.setText(text)

        try:
            from ...utils.torch_backends import backend_by_key  # noqa: PLC0415

            backend = backend_by_key(key)
            self.backend_detail.setText(
                f"官方索引：{backend.index_url}　·　断点续传下载后装入安装目录里的独立环境。"
            )
        except Exception:  # noqa: BLE001
            self.backend_detail.setText("")

    # ------------------------------------------------------------------ 交互
    def _on_backend_changed(self) -> None:
        self._refresh_reason()
        self.refresh()
        self.optionsChanged.emit()

    def _on_option_toggled(self) -> None:
        self.refresh()
        self.optionsChanged.emit()

    def current_backend_key(self) -> str:
        key = self.backend_combo.currentData()
        return str(key) if key else "cpu"

    def recommendation(self) -> BackendRecommendation | None:
        """自检给出的推荐结论（没跑完自检时为 ``None``）。"""
        return self._recommendation

    def profile(self) -> HardwareProfile | None:
        """自检得到的硬件信息（没跑完自检时为 ``None``）。"""
        return self._profile

    def selected_weights(self) -> list[str]:
        return [name for name, check in self.weight_checks.items() if check.isChecked()]

    # ------------------------------------------------------------------ 读写
    def apply(self, options: InstallOptions) -> None:
        options.backend_key = self.current_backend_key()
        options.install_torch = self.torch_check.isChecked()
        options.weights = self.selected_weights()
        options.install_ffmpeg = self.ffmpeg_check.isChecked()

    def load(self, options: InstallOptions) -> None:
        index = self.backend_combo.findData(options.backend_key)
        if index >= 0:
            self.backend_combo.setCurrentIndex(index)
        self.torch_check.setChecked(options.install_torch)
        for name, check in self.weight_checks.items():
            check.setChecked(name in options.weights)
        self.ffmpeg_check.setChecked(options.install_ffmpeg)
        self.refresh()

    def refresh(self) -> None:
        """刷新"本次要下多少、装完占多少"。"""
        from ..engine import estimate_download_bytes, estimate_install_bytes  # noqa: PLC0415
        from .. import windows  # noqa: PLC0415

        options = InstallOptions(
            install_dir=paths.default_install_dir(),
            backend_key=self.current_backend_key(),
            install_torch=self.torch_check.isChecked(),
            weights=self.selected_weights(),
            install_ffmpeg=self.ffmpeg_check.isChecked(),
        )
        download = estimate_download_bytes(options)
        install = estimate_install_bytes(options)
        target = options.install_dir
        free = windows.free_space(target)
        text = f"预计下载约 {paths.format_size(download)}　·　装完约占用 {paths.format_size(install)}"
        if free is not None:
            text += f"　·　目标磁盘剩余 {paths.format_size(free)}"
        self.size_label.setText(text)


# --------------------------------------------------------------------- 进度页
class ProgressPage(QWidget):
    """安装进度：阶段列表 + 两档进度条 + 实时日志。"""

    cancelRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(theme.SPACING)

        layout.addWidget(_label("正在安装", object_name="headline"))
        self.stage_summary = _label("准备开始…", object_name="hint", wrap=True)
        layout.addWidget(self.stage_summary)

        self.stage_table = QTableWidget(len(ORDERED_STAGES), 2)
        self.stage_table.horizontalHeader().setVisible(True)
        self.stage_table.setHorizontalHeaderLabels(["步骤", "状态"])
        self.stage_table.verticalHeader().setVisible(False)
        self.stage_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.stage_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.stage_table.setObjectName("checkTable")
        for row, info in enumerate(ORDERED_STAGES):
            self.stage_table.setItem(row, 0, QTableWidgetItem(info.title))
            self.stage_table.setItem(row, 1, QTableWidgetItem(_STAGE_STATUS["pending"][0]))
            self._color_stage(row, "pending")
        self.stage_table.resizeColumnsToContents()
        self.stage_table.setColumnWidth(0, max(self.stage_table.columnWidth(0), 190))
        self.stage_table.setColumnWidth(1, max(self.stage_table.columnWidth(1), 90))
        self.stage_table.setMinimumHeight(240)
        layout.addWidget(self.stage_table)

        self.total_bar = QProgressBar()
        self.total_bar.setRange(0, 100)
        self.total_bar.setValue(0)
        layout.addWidget(self.total_bar)
        self.total_label = _label("总进度", object_name="status", wrap=True)
        layout.addWidget(self.total_label)

        self.stage_bar = QProgressBar()
        self.stage_bar.setObjectName("thin")
        self.stage_bar.setRange(0, 100)
        self.stage_bar.setValue(0)
        layout.addWidget(self.stage_bar)
        self.stage_label = _label("", object_name="hint", wrap=True)
        layout.addWidget(self.stage_label)

        log_box, log_layout = _card("安装日志")
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setObjectName("logView")
        self.log_view.setMaximumBlockCount(MAX_LOG_LINES)
        self.log_view.setMinimumHeight(140)
        log_layout.addWidget(self.log_view)
        layout.addWidget(log_box)

        row = QHBoxLayout()
        self.cancel_button = QPushButton("取消安装…")
        self.cancel_button.setObjectName("danger")
        self.cancel_button.clicked.connect(self.cancelRequested.emit)
        row.addStretch(1)
        row.addWidget(self.cancel_button)
        layout.addLayout(row)

    # ------------------------------------------------------------------ 渲染
    def reset(self) -> None:
        for row in range(self.stage_table.rowCount()):
            self._color_stage(row, "pending")
            self.stage_table.item(row, 1).setText(_STAGE_STATUS["pending"][0])
        self.total_bar.setValue(0)
        self.stage_bar.setValue(0)
        self.total_label.setText("总进度")
        self.stage_label.setText("")
        self.stage_summary.setText("准备开始…")

    def _color_stage(self, row: int, status: str) -> None:
        _text, color_key = _STAGE_STATUS.get(status, _STAGE_STATUS["pending"])
        color = _color_for(color_key)
        for column in (0, 1):
            item = self.stage_table.item(row, column)
            if item is not None:
                item.setForeground(color)
        item = self.stage_table.item(row, 1)
        if item is not None:
            font = item.font()
            font.setBold(status in {"running", "failed"})
            item.setFont(font)

    def stage_row(self, stage: Stage | str) -> int:
        key = stage.value if isinstance(stage, Stage) else str(stage)
        for row, info in enumerate(ORDERED_STAGES):
            if info.key == key:
                return row
        return -1

    def set_stage(self, stage: Stage, status: str, message: str = "") -> None:
        row = self.stage_row(stage)
        if row < 0:
            return
        text, _color = _STAGE_STATUS.get(status, _STAGE_STATUS["pending"])
        self.stage_table.item(row, 1).setText(text)
        if message:
            self.stage_table.item(row, 0).setToolTip(message)
        self._color_stage(row, status)
        if status in {"running", "failed", "cancelled", "done"} and message:
            self.stage_summary.setText(message if status != "running" else f"{message}…")

    def set_stage_progress(self, fraction: float, text: str = "") -> None:
        self.stage_bar.setValue(int(max(0.0, min(fraction, 1.0)) * 100))
        if text:
            self.stage_label.setText(text)

    def set_total_progress(self, fraction: float, text: str = "") -> None:
        value = int(max(0.0, min(fraction, 1.0)) * 100)
        self.total_bar.setValue(value)
        self.total_bar.setFormat(f"{value}%")
        if text:
            self.total_label.setText(text)

    def set_download_stats(self, event) -> None:
        """显示速度 / 剩余时间 / 续传字节（下载阶段的额外信息）。"""
        if event is None:
            return
        parts: list[str] = []
        total = getattr(event, "total", None)
        downloaded = getattr(event, "downloaded", 0)
        speed = getattr(event, "speed_bps", 0.0) or 0.0
        parts.append(f"{paths.format_size(downloaded)}" + (f" / {paths.format_size(total)}" if total else ""))
        if speed > 0:
            parts.append(f"{paths.format_size(speed)}/s")
        eta = getattr(event, "eta", 0.0) or 0.0
        if eta > 0:
            parts.append(f"剩余 {_format_seconds(eta)}")
        resumed = getattr(event, "resumed_from", 0) or 0
        if resumed > 0:
            parts.append(f"续传自 {paths.format_size(resumed)}")
        interval = getattr(event, "index", 1)
        count = getattr(event, "count", 1)
        parts.append(f"第 {interval}/{count} 个文件")
        self.stage_label.setText("　·　".join(parts))

    def append_log(self, text: str) -> None:
        for line in str(text).splitlines() or [""]:
            self.log_view.appendPlainText(line)

    def log_text(self) -> str:
        return self.log_view.toPlainText()

    def set_finished(self, *, cancelled: bool = False, failed: bool = False) -> None:
        self.cancel_button.setVisible(False)
        if cancelled:
            self.stage_summary.setText("安装已暂停：断点与已下载的文件都已保留，下次继续即可。")
        elif failed:
            self.stage_summary.setText("安装失败：请查看下方日志，修复后可以「继续安装」重试。")


def _format_seconds(seconds: float) -> str:
    total = int(max(seconds, 0))
    if total < 60:
        return f"{total} 秒"
    minutes, sec = divmod(total, 60)
    if minutes < 60:
        return f"{minutes} 分 {sec} 秒"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} 小时 {minutes} 分"


# --------------------------------------------------------------------- 完成页
class FinishPage(QWidget):
    """安装结果 + 后续动作。"""

    launchRequested = Signal()
    openFolderRequested = Signal()
    viewLogRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(theme.SPACING)

        self.headline = _label("安装完成", object_name="headline")
        layout.addWidget(self.headline)
        self.summary = _paragraph("")
        layout.addWidget(self.summary)

        self.detail_box, self.detail_layout = _card("安装明细")
        self.detail_label = _paragraph("")
        self.detail_layout.addWidget(self.detail_label)
        layout.addWidget(self.detail_box)

        row = QHBoxLayout()
        self.launch_button = QPushButton("立即启动 Video2PersonVideo")
        self.launch_button.setObjectName("primary")
        self.launch_button.clicked.connect(self.launchRequested.emit)
        self.folder_button = QPushButton("打开安装目录")
        self.folder_button.clicked.connect(self.openFolderRequested.emit)
        self.log_button = QPushButton("查看安装日志")
        self.log_button.clicked.connect(self.viewLogRequested.emit)
        row.addWidget(self.launch_button)
        row.addWidget(self.folder_button)
        row.addWidget(self.log_button)
        row.addStretch(1)
        layout.addLayout(row)
        layout.addStretch(1)

    def set_success(self, install_dir: Path, details: str, *, can_launch: bool) -> None:
        self.headline.setText("安装完成")
        restyle(self.headline, "badgeOk")
        self.summary.setText(
            f"Video2PersonVideo 已安装到 {install_dir}。\n"
            "从开始菜单或桌面快捷方式都能启动；"
            "首次启动会直接进入主界面，不需要再下载任何东西。"
        )
        self.detail_label.setText(details or "—")
        self.launch_button.setVisible(can_launch)
        self.folder_button.setVisible(True)

    def set_cancelled(self, install_dir: Path, stage_text: str) -> None:
        self.headline.setText("安装已暂停")
        restyle(self.headline, "badgeWarn")
        self.summary.setText(
            f"安装停在了「{stage_text}」。\n"
            "已经下载的内容都保留着，下次打开安装程序点「继续安装」就会从断点接着走，"
            "不会重复下载。"
        )
        self.detail_label.setText(f"安装目录：{install_dir}")
        self.launch_button.setVisible(False)

    def set_failed(self, message: str, install_dir: Path) -> None:
        self.headline.setText("安装未完成")
        restyle(self.headline, "badgeDanger")
        self.summary.setText(
            f"{message}\n\n可以查看日志定位原因，或点「继续安装」重试"
            "（已下载的安装包会被复用）。"
        )
        self.detail_label.setText(f"安装目录：{install_dir}")
        self.launch_button.setVisible(False)


# --------------------------------------------------------------------- 退出对话框
class ExitDialog(QDialog):
    """安装中关窗口 / 点取消时的三选一。

    这是需求里最关键的一处交互：安装可能要下一个多小时，
    用户中途退出时必须**说清楚现在在哪一步、退了会怎样、下次怎么继续**。
    """

    KEEP = "keep"
    ROLLBACK = "rollback"
    CONTINUE = "continue"

    def __init__(self, text: str, *, can_rollback: bool = True, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("要退出安装吗？")
        self._choice = self.CONTINUE

        layout = QVBoxLayout(self)
        layout.setSpacing(theme.SPACING)
        layout.addWidget(_label("安装正在进行中", object_name="subhead"))

        view = QPlainTextEdit(text)
        view.setReadOnly(True)
        view.setMinimumHeight(200)
        layout.addWidget(view)

        layout.addWidget(
            _paragraph(
                "提示：暂停后再次运行安装程序，会从断点继续；"
                "CUDA 版 PyTorch 体积很大，建议不要「回滚」除非确定不再安装。"
            )
        )

        row = QHBoxLayout()
        self.continue_button = QPushButton("继续安装")
        self.continue_button.setObjectName("primary")
        self.continue_button.clicked.connect(lambda: self._finish(self.CONTINUE))
        self.keep_button = QPushButton("暂停并退出（保留进度）")
        self.keep_button.clicked.connect(lambda: self._finish(self.KEEP))
        self.rollback_button = QPushButton("回滚并退出（删除已写入的文件）")
        self.rollback_button.setObjectName("danger")
        self.rollback_button.setEnabled(can_rollback)
        self.rollback_button.clicked.connect(lambda: self._finish(self.ROLLBACK))
        row.addWidget(self.continue_button)
        row.addStretch(1)
        row.addWidget(self.keep_button)
        row.addWidget(self.rollback_button)
        layout.addLayout(row)

        theme.apply_window_size(self, preferred=(780, 600), minimum=(560, 420))

    def _finish(self, choice: str) -> None:
        self._choice = choice
        if choice == self.CONTINUE:
            self.reject()
        else:
            self.accept()

    def choice(self) -> str:
        return self._choice


__all__ = [
    "LICENSE_SUMMARY",
    "MAX_LOG_LINES",
    "STEP_SUBTITLES",
    "STEP_TITLES",
    "ComponentsPage",
    "ExitDialog",
    "FinishPage",
    "LocationPage",
    "ProgressPage",
    "TextDialog",
    "WelcomePage",
    "restyle",
    "_card",
    "_format_seconds",
    "_label",
    "_paragraph",
]
