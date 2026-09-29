"""首次运行页：三步向导（检查电脑 → 确认方案 → 下载安装）。

只负责展示与事件转发：探测、下载、安装全部在 ``gui.controller.SetupWorker`` 里跑，
页面通过下面这些 ``set_*`` 方法被"喂"数据，自己不发出一条阻塞调用。

页面只对用户露出三步，每步底部都是同一排按钮（上一步 / 下一步），
点「下一步」就能一路走完；硬件明细、文件清单、安装版本这些"细节"都收在
「查看详情」里，默认不出现：

1. **① 检查电脑** —— 一句话结论（能不能用显卡加速）+ 可展开的硬件明细；
2. **② 确认方案** —— 装哪一套、还要下多少、下到哪，改目录 / 换版本都收在"高级"里；
3. **③ 下载安装** —— 两个进度条 + 实时日志，主按钮就是「开始下载并安装」。

开始下载后就自动停在第 ③ 步（``set_running(True)`` 会切过去），
用户不会出现"后台在下载、眼前却是别的页面"的困惑。
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...core.processor import format_bytes
from ...utils.downloader import (
    STATE_MISMATCH,
    STATE_PARTIAL,
    STATE_READY,
    CacheState,
    DownloadItem,
    DownloadProgress,
    FileState,
)
from ...utils.env_check import environment_summary, torch_installed, torch_version
from ...utils.gpu_probe import BACKEND_CPU, HardwareProfile
from ...utils.torch_backends import BackendRecommendation, backend_by_key
from ...utils.torch_install import InstallPlan, is_frozen, is_virtual_env, yolo_items
from .. import theme
from ..controller import SetupRequest, SetupResult

#: 「重新自检」会把探测缓存丢掉重来
REFRESH_TOOLTIP = "重新检测硬件（换过显卡 / 更新过驱动后点它）"
#: 主按钮在没有断点时的文案
START_TEXT = "开始下载并安装"
#: 主按钮在有断点时的文案（这就是"断点续传"对用户的样子）
RESUME_TEXT = "继续下载"
#: 丢弃断点重来的二次确认
RESTART_CONFIRM = (
    "确定要丢弃已有的下载断点，从头重新下载吗？\n\n"
    "已经下载完成并校验通过的文件会被保留，只有下到一半的部分会重来。"
)
#: 换目录的说明
DIRECTORY_TITLE = "选择下载目录（放 PyTorch 安装包）"
#: 本机已经装过 PyTorch 时，勾选项要说清楚"勾上才会重新下载"
TORCH_INSTALLED_HINT = "要换别的 CUDA 版本时再勾上它"
#: 两个勾选项都空着时的主按钮文案（一次"确认已就绪"而已，不是在下载）
READY_TEXT = "确认已就绪（无需下载）"
#: 勾选项的说明
TORCH_TOOLTIP = "PyTorch 是推理引擎。本机已经装过时默认不重复下载，沿用现有版本即可。"
YOLO_TOOLTIP = "YOLO 权重只有几 MB，放在本地后断网也能用；本地已有时取消勾选即可。"

#: 三步向导的小标题（同时用作顶部步骤条）
STEP_TITLES = ("① 检查电脑", "② 确认方案", "③ 下载安装")
#: 最后一步的下标
LAST_STEP = len(STEP_TITLES) - 1
#: 「下一步」在各步的文案（最后一步没有"下一步"）
NEXT_TEXTS = ("下一步：确认方案", "下一步：开始下载")


def format_seconds(seconds: float) -> str:
    """把秒数格式化成 ``1小时2分`` / ``3分12秒`` / ``45秒``。"""
    if seconds <= 0:
        return "—"
    minutes, remain = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}小时{minutes}分"
    if minutes:
        return f"{minutes}分{remain}秒"
    return f"{remain}秒"


def format_speed(bytes_per_second: float) -> str:
    """下载速度的易读写法。"""
    if bytes_per_second <= 0:
        return "—"
    return f"{format_bytes(int(bytes_per_second))}/s"


def torch_check_text() -> str:
    """「下载 PyTorch 安装包」的文案：本机装过就写清楚版本，别让人糊里糊涂重下一遍。"""
    version = torch_version()
    if version:
        return f"下载 PyTorch 安装包（本机已安装 {version}，{TORCH_INSTALLED_HINT}）"
    return "下载 PyTorch 安装包（本机未检测到）"


def _table_height(table: QTableWidget) -> int:
    """算出"刚好装下所有行"的表格高度，避免出现嵌套滚动条。"""
    header = table.horizontalHeader().height()
    rows = sum(table.rowHeight(row) for row in range(table.rowCount()))
    frame = table.frameWidth() * 2
    return header + rows + frame + 4


def _environment_note() -> str:
    """告诉用户这套依赖会装到哪个 Python 里（装错地方是最常见的坑）。"""
    if is_frozen():
        return "当前是打包版程序：只会把安装包下载到本地，不会自动安装到系统里。"
    if is_virtual_env():
        return f"依赖会安装到当前虚拟环境：{sys.executable}"
    return (
        f"提醒：当前用的是系统 Python（{sys.executable}），"
        "建议在虚拟环境（.venv）里安装，避免污染系统环境。"
    )


def describe_file_state(state: FileState | None) -> str:
    """把文件状态翻译成一句人话（文件清单里的「状态」列）。"""
    if state is None:
        return "待下载"
    if state.state == STATE_READY:
        return "已下载"
    if state.state == STATE_PARTIAL:
        done = format_bytes(state.size)
        if state.expected:
            return f"可续传（已有 {done}，{state.fraction * 100:.0f}%）"
        return f"可续传（已有 {done}）"
    if state.state == STATE_MISMATCH:
        return "大小不符，将重新下载"
    return "待下载"


def describe_cache(state: CacheState) -> str:
    """汇总一句话：已下好多少、断点能省多少、还要下多少、磁盘够不够。"""
    parts = []
    if state.ready_bytes:
        parts.append(f"已下载 {format_bytes(state.ready_bytes)}")
    if state.partial_bytes:
        parts.append(f"可续传 {format_bytes(state.partial_bytes)}")
    needed = f"需下载 {format_bytes(state.needed_bytes)}"
    if state.unknown_sizes:
        needed += f"（另有 {state.unknown} 个体积待探测）"
    parts.append(needed)
    text = " · ".join(parts)
    if state.free_bytes is not None:
        free = format_bytes(state.free_bytes)
        if state.enough_space is False:
            text += f"　｜　磁盘剩余 {free}（不足，建议清理或更换目录）"
        else:
            text += f"　｜　磁盘剩余 {free}"
    return text


class SetupPage(QWidget):
    """三步式的环境自检与依赖安装页。"""

    startRequested = Signal(object)  # SetupRequest
    pauseRequested = Signal()
    refreshRequested = Signal()
    backendChanged = Signal(str)
    directoryRequested = Signal(object)  # Path（打开下载目录）
    directoryChanged = Signal(object)  # Path（换了下载目录，需要重新解析地址）

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._plan: InstallPlan | None = None
        self._cache: CacheState | None = None
        self._directory: Path | None = None
        self._running = False
        self._finished = False
        #: 是否已经拿到推荐结论（拿到之后"状态行"不再被"正在…"覆盖）
        self._concluded = False
        #: 文件清单里每一行的状态格（按"待下载目标的路径"索引，下载时实时刷新）
        self._status_cells: dict[str, QTableWidgetItem] = {}
        self._rows: list[DownloadItem] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.SPACING)

        layout.addWidget(self._build_header())
        layout.addLayout(self._build_step_bar())

        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_check_step())
        self.stack.addWidget(self._build_plan_step())
        self.stack.addWidget(self._build_download_step())
        layout.addWidget(self.stack, stretch=1)

        layout.addLayout(self._build_nav_row())
        self._sync_step()

    # ------------------------------------------------------------ 界面构建
    def _build_header(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.SPACING_SMALL)

        title = QLabel("环境自检")
        title.setObjectName("headline")
        layout.addWidget(title)

        hint = QLabel(
            "程序要装一套 PyTorch（官方有好几个版本，装错了显卡就用不上），"
            "所以先看看这台电脑是什么显卡。\n"
            "全程只要点「下一步」：检查电脑 → 确认方案 → 开始下载。"
        )
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        return box

    def _build_step_bar(self) -> QHBoxLayout:
        """顶部的三步进度条（只做提示，不可点击）。"""
        row = QHBoxLayout()
        row.setSpacing(theme.SPACING_LARGE)
        self._chips: list[QLabel] = []
        for title in STEP_TITLES:
            chip = QLabel(title)
            chip.setObjectName("stepChip")
            self._chips.append(chip)
            row.addWidget(chip)
        row.addStretch(1)
        return row

    # ------------------------------------------------------- 第 ① 步：检查电脑
    def _build_check_step(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.SPACING)

        group = QGroupBox("这台电脑")
        box = QVBoxLayout(group)
        box.setSpacing(theme.SPACING_SMALL)

        self.hardware_badge = QLabel("正在检测…")
        self.hardware_badge.setObjectName("badgeInfo")
        self.hardware_badge.setWordWrap(True)
        box.addWidget(self.hardware_badge)

        self.check_status = QLabel("正在检测 CPU / 显卡…")
        self.check_status.setObjectName("hint")
        self.check_status.setWordWrap(True)
        box.addWidget(self.check_status)

        # 硬件明细默认收起：先给结论，想看细节再点开
        self.detail_button = QPushButton("查看硬件详情")
        self.detail_button.setObjectName("link")
        self.detail_button.setCheckable(True)
        self.detail_button.toggled.connect(self._on_hardware_detail_toggled)
        box.addWidget(self.detail_button)

        self.hardware_table = QTableWidget(0, 2)
        self.hardware_table.setObjectName("checkTable")
        self.hardware_table.setHorizontalHeaderLabels(["项目", "检测结果"])
        self.hardware_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.hardware_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.hardware_table.verticalHeader().setVisible(False)
        self.hardware_table.setMinimumHeight(170)
        header = self.hardware_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.hardware_table.setVisible(False)
        box.addWidget(self.hardware_table)

        layout.addWidget(group)
        layout.addStretch(1)
        return page

    # ------------------------------------------------------- 第 ② 步：确认方案
    def _build_plan_step(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.SPACING)

        plan_group = QGroupBox("推荐方案")
        box = QVBoxLayout(plan_group)
        box.setSpacing(theme.SPACING_SMALL)

        # 「推荐」结论一行说清：左边小标题、右边结论，理由换行跟在下面
        head = QHBoxLayout()
        head.setSpacing(theme.SPACING_SMALL)
        caption = QLabel("推荐安装")
        caption.setObjectName("hint")
        head.addWidget(caption)

        self.recommend_label = QLabel("等待自检…")
        self.recommend_label.setObjectName("headline")
        self.recommend_label.setWordWrap(True)
        head.addWidget(self.recommend_label, stretch=1)
        box.addLayout(head)

        self.reason_label = QLabel("—")
        self.reason_label.setObjectName("hint")
        self.reason_label.setWordWrap(True)
        box.addWidget(self.reason_label)

        self.warning_label = QLabel()
        self.warning_label.setObjectName("badgeWarn")
        self.warning_label.setWordWrap(True)
        self.warning_label.setVisible(False)
        box.addWidget(self.warning_label)

        # 三个勾选项竖着排：文案都不短，挤成一行在高缩放下会被截断
        option_box = QVBoxLayout()
        option_box.setSpacing(theme.SPACING_SMALL)

        self.torch_check = QCheckBox(torch_check_text())
        # 本机已经装过 PyTorch 就不默认重下（想换 CUDA 版本时自己勾上）
        self.torch_check.setChecked(not torch_installed())
        self.torch_check.setToolTip(TORCH_TOOLTIP)
        self.torch_check.toggled.connect(self._on_torch_toggled)
        option_box.addWidget(self.torch_check)

        self.install_check = QCheckBox("下载完成后自动安装（推荐）")
        self.install_check.setChecked(not is_frozen())
        if is_frozen():
            self.install_check.setToolTip(
                "打包运行的程序无法把依赖装进自身，只会把安装包下载到本地。"
            )
        option_box.addWidget(self.install_check)

        self.yolo_check = QCheckBox("同时下载 YOLO 权重（2 个文件，共约 12 MB，离线也能用）")
        self.yolo_check.setChecked(True)
        self.yolo_check.setToolTip(YOLO_TOOLTIP)
        self.yolo_check.toggled.connect(self._on_yolo_toggled)
        option_box.addWidget(self.yolo_check)
        box.addLayout(option_box)

        # 本机已经有什么（装了 torch / 已有权重）：拿这个说话，比让用户自己猜强
        summary = environment_summary()
        self.local_label = QLabel(summary or "")
        self.local_label.setObjectName("hint")
        self.local_label.setWordWrap(True)
        self.local_label.setVisible(bool(summary))
        box.addWidget(self.local_label)

        # 换安装版本是"少数人才要做的事"，收起来，避免一上来就被版本号砸晕
        self.advanced_button = QPushButton("我要自己选安装版本（一般不用改）")
        self.advanced_button.setObjectName("link")
        self.advanced_button.setCheckable(True)
        self.advanced_button.toggled.connect(self._on_advanced_toggled)
        box.addWidget(self.advanced_button)

        self.backend_row = QWidget()
        backend_row = QHBoxLayout(self.backend_row)
        backend_row.setContentsMargins(0, 0, 0, 0)
        backend_row.setSpacing(theme.SPACING_SMALL)
        self.backend_combo = QComboBox()
        self.backend_combo.setToolTip("自动推荐之外，也可以手动指定要安装的 PyTorch 构建")
        self.backend_combo.currentIndexChanged.connect(self._on_backend_changed)
        backend_row.addWidget(QLabel("安装版本"))
        backend_row.addWidget(self.backend_combo, stretch=1)
        self.backend_row.setVisible(False)
        box.addWidget(self.backend_row)

        layout.addWidget(plan_group)

        size_group = QGroupBox("要下载的东西")
        size_box = QVBoxLayout(size_group)
        size_box.setSpacing(theme.SPACING_SMALL)

        # 一句话说清"还要下多少 / 磁盘够不够"
        self.cache_label = QLabel("等待自检…")
        self.cache_label.setObjectName("hint")
        self.cache_label.setWordWrap(True)
        size_box.addWidget(self.cache_label)

        self.total_label = QLabel("—")
        self.total_label.setObjectName("hint")
        self.total_label.setWordWrap(True)
        size_box.addWidget(self.total_label)

        dir_row = QHBoxLayout()
        dir_row.addStretch(1)
        self.change_dir_button = QPushButton("更换目录")
        self.change_dir_button.setObjectName("link")
        self.change_dir_button.setToolTip("把 PyTorch 安装包放到别的磁盘（例如 C 盘空间不足时）")
        self.change_dir_button.clicked.connect(self._choose_directory)
        dir_row.addWidget(self.change_dir_button)
        self.open_dir_button = QPushButton("打开目录")
        self.open_dir_button.setObjectName("link")
        self.open_dir_button.clicked.connect(self._emit_directory)
        dir_row.addWidget(self.open_dir_button)
        size_box.addLayout(dir_row)

        # 文件清单默认收起：体积 / 文件名这些细节按需再看
        self.files_button = QPushButton("查看文件清单")
        self.files_button.setObjectName("link")
        self.files_button.setCheckable(True)
        self.files_button.toggled.connect(self._on_files_toggled)
        size_box.addWidget(self.files_button)

        self.files_table = QTableWidget(0, 4)
        self.files_table.setObjectName("checkTable")
        self.files_table.setHorizontalHeaderLabels(["文件", "版本", "大小", "状态"])
        self.files_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.files_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.files_table.verticalHeader().setVisible(False)
        self.files_table.setWordWrap(False)
        self.files_table.setMinimumHeight(120)
        files_header = self.files_table.horizontalHeader()
        files_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        files_header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        files_header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        files_header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.files_table.setVisible(False)
        size_box.addWidget(self.files_table)

        self.env_label = QLabel(_environment_note())
        self.env_label.setObjectName("hint")
        self.env_label.setWordWrap(True)
        size_box.addWidget(self.env_label)

        layout.addWidget(size_group)
        layout.addStretch(1)
        return page

    # ------------------------------------------------------- 第 ③ 步：下载安装
    def _build_download_step(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.SPACING)

        group = QGroupBox("下载与安装")
        box = QVBoxLayout(group)
        box.setSpacing(theme.SPACING_SMALL)

        self.total_bar = QProgressBar()
        self.total_bar.setObjectName("thin")
        self.total_bar.setRange(0, 100)
        self.total_bar.setValue(0)
        self.total_bar.setFormat("%p% · 整体进度")
        box.addWidget(self.total_bar)

        self.file_bar = QProgressBar()
        self.file_bar.setRange(0, 100)
        self.file_bar.setValue(0)
        self.file_bar.setFormat("%p% · 当前文件")
        box.addWidget(self.file_bar)

        info = QGridLayout()
        info.setHorizontalSpacing(theme.SPACING_LARGE)
        info.setVerticalSpacing(theme.SPACING_SMALL)
        self.stage_label = QLabel("等待开始")
        self.file_label = QLabel("—")
        self.size_label = QLabel("—")
        self.speed_label = QLabel("—")
        for column, (caption, widget) in enumerate(
            (
                ("当前阶段", self.stage_label),
                ("当前文件", self.file_label),
                ("已下载", self.size_label),
                ("速度 / 剩余", self.speed_label),
            )
        ):
            label = QLabel(caption)
            label.setObjectName("hint")
            info.addWidget(label, 0, column)
            info.addWidget(widget, 1, column)
        box.addLayout(info)

        self.resume_label = QLabel("断点续传：已就绪（中断后重新开始会自动接着下）")
        self.resume_label.setObjectName("hint")
        self.resume_label.setWordWrap(True)
        box.addWidget(self.resume_label)

        layout.addWidget(group)

        log_group = QGroupBox("日志")
        log_box = QVBoxLayout(log_group)
        log_box.setSpacing(theme.SPACING_SMALL)
        self.log_view = QPlainTextEdit()
        self.log_view.setObjectName("logView")
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(3000)
        self.log_view.setMinimumHeight(120)
        self.log_view.setPlaceholderText("下载 / 安装日志会显示在这里")
        log_box.addWidget(self.log_view)
        layout.addWidget(log_group, stretch=1)
        return page

    # ------------------------------------------------------------ 底部按钮排
    def _build_nav_row(self) -> QHBoxLayout:
        row = QHBoxLayout()

        self.refresh_button = QPushButton("重新自检")
        self.refresh_button.setToolTip(REFRESH_TOOLTIP)
        self.refresh_button.clicked.connect(self._emit_refresh)
        row.addWidget(self.refresh_button)

        # 有断点时才会出现：想从头下的人有路可走
        self.restart_button = QPushButton("重新下载（丢弃断点）")
        self.restart_button.setObjectName("link")
        self.restart_button.setToolTip(
            "丢弃下到一半的断点，从零开始下载（已经下完并校验通过的文件不会重下）"
        )
        self.restart_button.setVisible(False)
        self.restart_button.clicked.connect(self._emit_restart)
        row.addWidget(self.restart_button)

        row.addStretch(1)

        self.back_button = QPushButton("上一步")
        self.back_button.clicked.connect(self.previous_step)
        row.addWidget(self.back_button)

        self.pause_button = QPushButton("暂停（保留断点）")
        self.pause_button.setObjectName("danger")
        self.pause_button.setVisible(False)
        self.pause_button.clicked.connect(self._emit_pause)
        row.addWidget(self.pause_button)

        self.next_button = QPushButton(NEXT_TEXTS[0])
        self.next_button.setObjectName("primary")
        self.next_button.clicked.connect(self.next_step)
        row.addWidget(self.next_button)

        self.start_button = QPushButton(START_TEXT)
        self.start_button.setObjectName("primary")
        self.start_button.clicked.connect(self._emit_start)
        row.addWidget(self.start_button)
        return row

    # ----------------------------------------------------------------- 接口
    @property
    def plan(self) -> InstallPlan | None:
        return self._plan

    @property
    def finished(self) -> bool:
        """是不是"下载 / 安装都走完了"（暂停与失败时为 ``False``）。"""
        return self._finished

    def backend_key(self) -> str:
        return str(self.backend_combo.currentData() or BACKEND_CPU)

    @property
    def cache(self) -> CacheState | None:
        """当前这批文件的缓存现状（界面据此显示"能不能续传"）。"""
        return self._cache

    @property
    def directory(self) -> Path | None:
        return self._directory

    def request(self, *, resume: bool = True) -> SetupRequest:
        """把界面上的选择整理成后台任务参数。"""
        directory = self._directory or (self._plan.directory if self._plan else None)
        return SetupRequest(
            backend_key=self.backend_key(),
            skip_torch=not self.torch_check.isChecked(),
            download_yolo=self.yolo_check.isChecked(),
            install=self.install_check.isChecked() and self.torch_check.isChecked(),
            wheel_dir=directory,
            resume=resume,
        )

    def set_running(self, running: bool) -> None:
        self._running = running
        self.backend_combo.setEnabled(not running)
        self.torch_check.setEnabled(not running)
        self.install_check.setEnabled(not running and self.torch_check.isChecked())
        self.yolo_check.setEnabled(not running)
        self.change_dir_button.setEnabled(not running)
        self.refresh_button.setEnabled(not running)
        if running:
            # 开始下载就切到第 ③ 步：进度、日志都在那儿，不给用户"下载跑到哪去了"的困惑
            self.stack.setCurrentIndex(LAST_STEP)
            self._sync_step()
            return
        self._sync_step()

    def _update_action_text(self) -> None:
        """主按钮文案跟着"有没有断点 / 下没下完"走，这就是断点续传的入口。"""
        if self._running or self._plan is None:
            return
        if not self._rows:
            # 两个勾选项都空着：点下去只是确认一遍"本机已就绪"
            self.start_button.setText(READY_TEXT)
            self.start_button.setToolTip("没有勾选任何要下载的内容，直接点「完成」也行。")
            self.restart_button.setVisible(False)
            return
        state = self._visible_state()
        if state is not None and state.has_partials:
            self.start_button.setText(RESUME_TEXT)
            self.start_button.setToolTip(
                f"从上次的断点接着下（已有 {format_bytes(state.partial_bytes)} 可直接复用）"
            )
            self.restart_button.setVisible(True)
        else:
            self.start_button.setText(START_TEXT)
            self.start_button.setToolTip("")
            self.restart_button.setVisible(False)

    def set_stage(self, text: str) -> None:
        value = text or "—"
        self.stage_label.setText(value)
        # 还没出结论时，"正在检测…"这类即时状态显示在第 ① 步，用户看得见
        if not self._concluded:
            self.check_status.setText(value)

    def set_profile(self, profile: HardwareProfile) -> None:
        """渲染硬件明细表（整表一次性展开，不出现嵌套滚动条）。"""
        lines = profile.summary_lines()
        self.hardware_table.setRowCount(len(lines))
        for row, (key, value) in enumerate(lines):
            for column, text in enumerate((key, value)):
                item = QTableWidgetItem(text)
                item.setToolTip(text)
                self.hardware_table.setItem(row, column, item)
        self.hardware_table.resizeRowsToContents()
        self.hardware_table.setFixedHeight(_table_height(self.hardware_table))

    def set_recommendation(self, recommendation: BackendRecommendation) -> None:
        """渲染推荐结论 + 可选方案下拉，并把一句话结论写在第 ① 步。"""
        self.recommend_label.setText(recommendation.backend.display)
        self.reason_label.setText(recommendation.reason)

        if recommendation.warnings:
            self.warning_label.setText("· " + "\n· ".join(recommendation.warnings))
            self.warning_label.setVisible(True)
        else:
            self.warning_label.setVisible(False)

        blocked = self.backend_combo.blockSignals(True)
        self.backend_combo.clear()
        for option in recommendation.options:
            label = option.backend.display
            if not option.usable:
                label = f"{label} —— {option.reason}"
            self.backend_combo.addItem(label, option.backend.key)
            index = self.backend_combo.count() - 1
            if not option.usable:
                # 不可用的档位仍然列出（用户可能自己知道驱动怎么升），但标注出来
                item = self.backend_combo.model().item(index)
                if item is not None:
                    item.setEnabled(False)
        position = self.backend_combo.findData(recommendation.backend.key)
        if position >= 0:
            self.backend_combo.setCurrentIndex(position)
        self.backend_combo.blockSignals(blocked)

        self._update_badge(recommendation)
        # 结论出来了：状态行换成一句人话，不再被"正在…"覆盖
        self._concluded = True
        if recommendation.is_gpu:
            self.check_status.setText("这台电脑可以用显卡加速，处理速度会快很多。")
        else:
            self.check_status.setText("这台电脑没检测到可用的 NVIDIA 显卡，会用 CPU 运行：更稳，但慢一些。")

    def set_plan(self, plan: InstallPlan) -> None:
        """渲染下载清单（PyTorch + 可选 YOLO 权重）。"""
        self._plan = plan
        if self._directory is None:
            self._directory = Path(plan.directory)
        self._render_files()

        # 体积按**界面上列出来的**文件算：跳过 PyTorch / 取消 YOLO 后"合计"要跟着变小
        total = sum(item.size or 0 for item in self._rows)
        summary = f"共 {len(self._rows)} 个文件"
        summary += f" · 合计约 {format_bytes(total)}" if total else " · 体积待探测"
        lines = [summary, f"下载到：{self._directory}"]
        if plan.problems:
            lines.append("问题：" + "；".join(plan.problems))
        self.total_label.setText("\n".join(lines))

        self.open_dir_button.setEnabled(bool(plan.directory))

    def set_cache_state(self, state: CacheState) -> None:
        """渲染"还要下多少 / 磁盘够不够"，并刷新每个文件的状态。"""
        self._cache = state
        self._render_files()
        self._update_action_text()

    # --------------------------------------------------------- 文件清单渲染
    def visible_items(self) -> list[DownloadItem]:
        """本次会下载的全部文件（PyTorch / YOLO 都按各自的勾选状态决定要不要算进来）。"""
        items: list[DownloadItem] = []
        if self._plan is not None and self.torch_check.isChecked():
            items += list(self._plan.items)
        if self.yolo_check.isChecked():
            items += yolo_items()
        return items

    def _visible_state(self) -> CacheState | None:
        """只看界面上列出来的那些文件的缓存现状（取消勾选后汇总也要跟着变）。"""
        if self._cache is None:
            return None
        return self._cache.subset([item.path for item in self._rows])

    def _render_files(self) -> None:
        """按当前勾选与缓存现状重建文件清单（纯本地渲染，不碰网络 / 磁盘）。"""
        items = self.visible_items()
        self._rows = items
        known = {item.path: item for item in (self._cache.files if self._cache else [])}

        plan_paths = {item.path for item in (self._plan.items if self._plan else [])}
        self._status_cells.clear()
        self.files_table.setRowCount(len(items))
        for row, item in enumerate(items):
            size = format_bytes(item.size) if item.size else "待探测"
            optional = "" if item.path in plan_paths else "（可选）"
            cells = (
                item.path.name + optional,
                item.label or "—",
                size,
                describe_file_state(known.get(item.path)),
            )
            for column, text in enumerate(cells):
                cell = QTableWidgetItem(text)
                cell.setToolTip(f"{item.path.name}\n{item.label}\n{item.url}")
                self.files_table.setItem(row, column, cell)
            self._status_cells[str(item.path)] = self.files_table.item(row, 3)

        self.files_table.resizeRowsToContents()
        self.files_table.setFixedHeight(min(max(_table_height(self.files_table), 96), 240))
        self._render_cache_row()

    def _render_cache_row(self) -> None:
        """"还要下多少 / 磁盘够不够"那一行。"""
        if self._plan is not None and not self._rows:
            # 本机已就绪（PyTorch 装过、权重也有），用户把勾都取消了：没什么可下的
            self.cache_label.setText("本机已就绪：这次没有需要下载的文件。")
            self._restyle_cache_label("hint")
            return
        state = self._visible_state()
        if state is None:
            self.cache_label.setText("等待自检…")
            return
        if state.complete:
            self.cache_label.setText("全部文件已下载完成，点「开始」只会做一次校验。")
        else:
            self.cache_label.setText("断点续传　" + describe_cache(state))
        self._restyle_cache_label("badgeWarn" if state.enough_space is False else "hint")

    def _restyle_cache_label(self, object_name: str) -> None:
        """换样式表对象名后重新走一遍 polish，颜色才会跟着变。"""
        self.cache_label.setObjectName(object_name)
        self.cache_label.style().unpolish(self.cache_label)
        self.cache_label.style().polish(self.cache_label)

    def set_progress(self, event: DownloadProgress) -> None:
        self.file_bar.setValue(int(round(event.fraction * 100)))
        self.file_label.setText(f"{event.path.name.removesuffix('.part')}（{event.index}/{event.count}）")
        downloaded = format_bytes(event.downloaded)
        if event.total:
            self.size_label.setText(f"{downloaded} / {format_bytes(event.total)}")
        else:
            self.size_label.setText(downloaded)
        self.speed_label.setText(
            f"{format_speed(event.speed_bps)} · 剩余 {format_seconds(event.eta)}"
        )
        self._mark_row_downloading(event)
        if event.resumed_from:
            self.resume_label.setText(
                f"断点续传：本次已从 {format_bytes(event.resumed_from)} 处继续下载"
            )
        else:
            self.resume_label.setText("断点续传：已就绪（中断后重新开始会自动接着下）")

    def _mark_row_downloading(self, event: DownloadProgress) -> None:
        """把文件清单里"正在下"的那一行状态改成实时进度。"""
        cell = self._status_cells.get(str(event.path).removesuffix(".part"))
        if cell is None:
            return
        text = f"下载中 {event.fraction * 100:.0f}%"
        if event.speed_bps > 0:
            text += f" · {format_speed(event.speed_bps)}"
        if event.resumed_from:
            text += f"（续传自 {format_bytes(event.resumed_from)}）"
        cell.setText(text)

    def set_total_progress(self, done: int, total: int, text: str) -> None:
        self.total_bar.setValue(int(round(done / max(total, 1) * 100)))
        if text:
            self.stage_label.setText(text)

    def set_finished(self, result: SetupResult) -> None:
        """收尾：切换按钮文案并给出人话结论。"""
        self._finished = result.ok or result.skipped_install
        self.set_running(False)

        if result.discarded:
            self.append_log(f"本次已丢弃 {len(result.discarded)} 个断点。")
        if result.resumed_bytes:
            self.append_log(
                f"断点续传本次省下约 {format_bytes(result.resumed_bytes)} 的重复下载。"
            )

        if result.cancelled:
            self.start_button.setText(RESUME_TEXT)
            self.stage_label.setText("已暂停，断点已保留")
            self.append_log("已暂停：下次点「继续下载」会从断点接着下。")
            return

        if result.failed:
            self.start_button.setText("重试下载")
            self.stage_label.setText("部分文件下载失败")
            for name, error in result.failed:
                self.append_log(f"失败：{name} —— {error}")
            self.append_log("已下好的部分不会重复下载，重试时会接着下。")
            return

        if result.skipped_install:
            self.start_button.setText("重新下载")
            self.stage_label.setText("下载完成")
            if result.install_message:
                self.append_log(result.install_message)
            return

        if result.installed:
            self.start_button.setText("重新下载")
            self.stage_label.setText("安装完成，重启程序即可使用 GPU 加速")
            self.append_log(result.install_message or "安装完成。")
            return

        self.start_button.setText("重试下载")
        self.stage_label.setText("安装失败")
        self.append_log(result.install_message or "安装失败")

    def append_log(self, text: str) -> None:
        self.log_view.appendPlainText(str(text))

    def reset_progress(self) -> None:
        self.total_bar.setValue(0)
        self.file_bar.setValue(0)
        self.file_label.setText("—")
        self.size_label.setText("—")
        self.speed_label.setText("—")
        self.resume_label.setText("断点续传：正在复用已有断点…")

    # --------------------------------------------------------------- 步骤导航
    def current_step(self) -> int:
        return self.stack.currentIndex()

    def next_step(self) -> None:
        """「下一步」：只翻页，不替用户按"开始下载"。"""
        if self._running:
            return
        index = self.stack.currentIndex()
        if index >= LAST_STEP:
            return
        self.stack.setCurrentIndex(index + 1)
        self._sync_step()
        self._scroll_to_top()

    def previous_step(self) -> None:
        """「上一步」：下载中不给退，避免"后台在下载、眼前是别的页"。"""
        if self._running:
            return
        index = self.stack.currentIndex()
        if index <= 0:
            return
        self.stack.setCurrentIndex(index - 1)
        self._sync_step()
        self._scroll_to_top()

    def _sync_step(self) -> None:
        """步骤条高亮 + 底部按钮跟着当前步骤切换。"""
        index = self.stack.currentIndex()
        for position, chip in enumerate(self._chips):
            chip.setObjectName("stepChipActive" if position == index else "stepChip")
            chip.style().unpolish(chip)
            chip.style().polish(chip)
        self.next_button.setText(NEXT_TEXTS[min(index, LAST_STEP - 1)])
        self._sync_buttons()

    def _sync_buttons(self) -> None:
        """按"第几步 + 有没有在下载"决定底部三个按钮谁露面。"""
        index = self.stack.currentIndex()
        running = self._running

        self.pause_button.setVisible(running)
        self.start_button.setVisible(not running and index == LAST_STEP)
        self.next_button.setVisible(not running and index < LAST_STEP)
        self.back_button.setVisible(not running and index > 0)
        self.refresh_button.setVisible(not running and index == 0)
        if running:
            self.restart_button.setVisible(False)
            return
        self._update_action_text()

    def _scroll_to_top(self) -> None:
        """翻页后把外面那层滚动区拉回顶部，免得看到的是上一页的中间位置。"""
        widget = self.parentWidget()
        while widget is not None:
            if isinstance(widget, QScrollArea):
                widget.verticalScrollBar().setValue(0)
                return
            widget = widget.parentWidget()

    # ------------------------------------------------------------- 内部逻辑
    def _update_badge(self, recommendation: BackendRecommendation) -> None:
        if recommendation.is_gpu:
            self.hardware_badge.setObjectName("badgeOk")
            self.hardware_badge.setText(
                f"检测完成：可用 GPU 加速（{recommendation.backend.label}）"
            )
        else:
            self.hardware_badge.setObjectName("badgeWarn")
            self.hardware_badge.setText("检测完成：按 CPU 运行（稳定，但比 GPU 慢）")
        self.hardware_badge.style().unpolish(self.hardware_badge)
        self.hardware_badge.style().polish(self.hardware_badge)

    def _on_hardware_detail_toggled(self, visible: bool) -> None:
        self.hardware_table.setVisible(visible)
        self.detail_button.setText("收起硬件详情" if visible else "查看硬件详情")

    def _on_files_toggled(self, visible: bool) -> None:
        self.files_table.setVisible(visible)
        self.files_button.setText("收起文件清单" if visible else "查看文件清单")

    def _on_advanced_toggled(self, visible: bool) -> None:
        self.backend_row.setVisible(visible)
        self.advanced_button.setText("收起安装版本" if visible else "我要自己选安装版本（一般不用改）")

    def _on_yolo_toggled(self) -> None:
        """勾选 / 取消 YOLO 权重：清单、汇总、按钮文案一起跟着变。"""
        self._render_files()
        self._update_action_text()

    def _on_torch_toggled(self) -> None:
        """勾选 / 取消 PyTorch 安装包：清单、汇总、"自动安装"跟着一起变。"""
        self.install_check.setEnabled(self.torch_check.isChecked() and not self._running)
        self._render_files()
        self._update_action_text()

    def _emit_start(self) -> None:
        cache = self._visible_state()
        if cache is not None and cache.has_partials:
            self.append_log(
                f"继续下载：复用 {cache.partial_files} 个断点（约 "
                f"{format_bytes(cache.partial_bytes)}）…"
            )
        else:
            self.append_log("开始下载…")
        self.startRequested.emit(self.request(resume=True))

    def _emit_restart(self) -> None:
        """丢弃断点重下（二次确认，避免误点）。"""
        answer = QMessageBox.question(
            self,
            "重新下载",
            RESTART_CONFIRM,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.append_log("已选择重新下载：丢弃断点，从头开始。")
        self.startRequested.emit(self.request(resume=False))

    def _emit_pause(self) -> None:
        self.append_log("已请求暂停，正在等当前数据块写完（断点会保留）…")
        self.pauseRequested.emit()

    def _emit_refresh(self) -> None:
        self.append_log("重新自检…")
        self._concluded = False
        self.check_status.setText("正在重新检测 CPU / 显卡…")
        self.refreshRequested.emit()

    def _choose_directory(self) -> None:
        """换一个下载目录（C 盘不够用时最有用的一个按钮）。"""
        directory = QFileDialog.getExistingDirectory(
            self,
            DIRECTORY_TITLE,
            str(self._directory or Path.home()),
        )
        if not directory:
            return
        chosen = Path(directory)
        if chosen == self._directory:
            return
        self._directory = chosen
        self.append_log(f"下载目录已改为：{chosen}，正在重新解析下载地址…")
        self.directoryChanged.emit(chosen)

    def _emit_directory(self) -> None:
        directory = self._plan.directory if self._plan else None
        if directory is not None:
            self.directoryRequested.emit(Path(directory))

    def _on_backend_changed(self) -> None:
        key = self.backend_key()
        backend = backend_by_key(key)
        self.recommend_label.setText(backend.display)
        self.reason_label.setText(backend.note or "—")
        if not self._running:
            self.backendChanged.emit(key)


__all__ = [
    "LAST_STEP",
    "NEXT_TEXTS",
    "READY_TEXT",
    "RESUME_TEXT",
    "START_TEXT",
    "STEP_TITLES",
    "SetupPage",
    "describe_cache",
    "describe_file_state",
    "format_seconds",
    "format_speed",
    "torch_check_text",
]
