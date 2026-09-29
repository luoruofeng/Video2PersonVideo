"""首次运行自检页的 GUI 冒烟测试（offscreen，不联网、不弹窗）。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6.QtWidgets", reason="未安装 PySide6，跳过 GUI 测试")

from PySide6.QtCore import QSettings, QThread  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from video2personvideo.gui import theme  # noqa: E402
from video2personvideo.gui.controller import SetupRequest, SetupResult  # noqa: E402
from video2personvideo.gui.pages import setup_page as setup_page_module  # noqa: E402
from video2personvideo.gui.pages.setup_page import (  # noqa: E402
    READY_TEXT,
    RESUME_TEXT,
    START_TEXT,
    SetupPage,
    format_seconds,
    format_speed,
)
from video2personvideo.gui.setup_dialog import (  # noqa: E402
    SetupDialog,
    mark_setup_completed,
    setup_completed,
)
from video2personvideo.utils.downloader import (  # noqa: E402
    DownloadItem,
    DownloadProgress,
    build_cache_state,
)
from video2personvideo.utils.gpu_probe import (  # noqa: E402
    BACKEND_CUDA,
    VENDOR_NVIDIA,
    GpuInfo,
    HardwareProfile,
)
from video2personvideo.utils.torch_backends import backend_by_key, recommend_backend  # noqa: E402
from video2personvideo.utils.torch_install import InstallPlan, yolo_items  # noqa: E402


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    theme.apply_theme(app, theme.MODE_LIGHT)
    return app  # type: ignore[return-value]


@pytest.fixture(autouse=True)
def _blank_machine(monkeypatch) -> None:
    """默认按"这台电脑什么都没装"渲染页面。

    页面会去问本机有没有 torch、有没有本地权重，跑测试的机器上这两样通常都有，
    行数 / 勾选状态就会跟着变。这里统一压成"全新机器"，要验证"已经装好"的用例
    自己把这几项覆盖回去。
    """
    monkeypatch.setattr(setup_page_module, "torch_installed", lambda: False)
    monkeypatch.setattr(setup_page_module, "torch_version", lambda: None)
    monkeypatch.setattr(setup_page_module, "environment_summary", lambda: None)


def _profile() -> HardwareProfile:
    return HardwareProfile(
        os_name="Windows",
        os_release="11",
        arch="AMD64",
        python_version="3.13.3",
        is_windows=True,
        cpu_name="Intel Core i7-13700K",
        cpu_cores=24,
        ram_gb=32.0,
        gpus=[
            GpuInfo(
                name="NVIDIA GeForce RTX 4070 Ti",
                vendor=VENDOR_NVIDIA,
                memory_mb=12288,
                driver_version="560.94",
                compute=(8, 9),
                backend=BACKEND_CUDA,
                family="GeForce RTX 40 系",
                architecture="Ada Lovelace",
            )
        ],
        cuda_driver_version="12.6",
    )


def _plan(tmp_path: Path, *, size: int = 2_500_000_000) -> InstallPlan:
    return InstallPlan(
        backend=backend_by_key("cu126"),
        directory=tmp_path,
        items=[
            DownloadItem(
                url="https://download.pytorch.org/whl/cu126/torch/x.whl",
                path=tmp_path / "torch-2.14.0+cu126-cp313-cp313-win_amd64.whl",
                sha256="ab",
                label="torch 2.14.0",
                size=size,
            )
        ],
    )


def _all_items(plan: InstallPlan) -> list:
    """计划里的 torch 文件 + 两个 YOLO 权重（和界面上看到的清单一致）。"""
    return list(plan.items) + list(yolo_items())


# ------------------------------------------------------------------ 页面渲染
def test_setup_page_renders_hardware_and_recommendation(qapp: QApplication) -> None:
    page = SetupPage()
    profile = _profile()
    recommendation = recommend_backend(profile)

    page.set_profile(profile)
    page.set_recommendation(recommendation)

    assert page.hardware_table.rowCount() == len(profile.summary_lines())
    assert "RTX 4070 Ti" in page.hardware_table.item(5, 1).text()
    assert page.recommend_label.text().startswith("CUDA 12.6")
    assert page.backend_key() == "cu126"
    assert page.hardware_badge.objectName() == "badgeOk"
    page.close()


def test_setup_page_lists_options_and_marks_unusable(qapp: QApplication) -> None:
    page = SetupPage()
    page.set_recommendation(recommend_backend(_profile()))

    keys = [page.backend_combo.itemData(index) for index in range(page.backend_combo.count())]
    assert "cpu" in keys
    model = page.backend_combo.model()
    cu126_index = keys.index("cu126")
    assert model.item(cu126_index).isEnabled() is True
    # cu130 需要 13.0 的驱动，这台机器上不可用但依然列出来（标注原因）
    cu130_index = keys.index("cu130")
    assert model.item(cu130_index).isEnabled() is False
    assert "CUDA 13.0" in page.backend_combo.itemText(cu130_index)
    page.close()


def test_setup_page_cpu_badge_is_warning(qapp: QApplication) -> None:
    page = SetupPage()
    page.set_recommendation(recommend_backend(HardwareProfile(os_name="Windows")))
    assert page.hardware_badge.objectName() == "badgeWarn"
    assert page.backend_key() == "cpu"
    page.close()


def test_setup_page_plan_table_and_total(qapp: QApplication, tmp_path: Path) -> None:
    page = SetupPage()
    plan = _plan(tmp_path)
    page.set_plan(plan)

    # 清单 = torch 的 wheel + 两个可选 YOLO 权重
    assert page.files_table.rowCount() == 1 + len(yolo_items())
    assert page.files_table.item(0, 1).text() == "torch 2.14.0"
    assert "GB" in page.files_table.item(0, 2).text()
    assert f"共 {1 + len(yolo_items())} 个文件" in page.total_label.text()
    assert str(tmp_path) in page.total_label.text()
    assert page.plan is plan
    page.close()


def test_setup_page_progress_updates(qapp: QApplication, tmp_path: Path) -> None:
    page = SetupPage()
    page.set_progress(
        DownloadProgress(
            path=tmp_path / "torch.whl",
            url="https://x",
            downloaded=250_000_000,
            total=500_000_000,
            resumed_from=120_000_000,
            speed_bps=10_000_000,
            index=1,
            count=2,
        )
    )
    assert page.file_bar.value() == 50
    assert "1/2" in page.file_label.text()
    assert "MB" in page.size_label.text()
    assert "剩余" in page.speed_label.text()
    assert "断点续传" in page.resume_label.text()

    page.set_total_progress(1, 2, "PyTorch 完成")
    assert page.total_bar.value() == 50
    assert page.stage_label.text() == "PyTorch 完成"
    page.close()


def test_setup_page_finished_states(qapp: QApplication) -> None:
    page = SetupPage()

    page.set_finished(SetupResult(backend_key="cu126", cancelled=True))
    assert page.start_button.text() == "继续下载"
    assert "暂停" in page.stage_label.text()

    page.set_finished(SetupResult(backend_key="cu126", failed=[("a.whl", "网络超时")]))
    assert page.start_button.text() == "重试下载"
    assert "a.whl" in page.log_view.toPlainText()

    page.set_finished(SetupResult(backend_key="cu126", installed=True, downloaded=[]))
    assert "重启" in page.stage_label.text()
    assert page.finished is True

    page.set_finished(SetupResult(backend_key="cu126", cancelled=True))
    assert page.finished is False
    page.close()


def test_setup_page_request_collects_choices(qapp: QApplication, tmp_path: Path) -> None:
    page = SetupPage()
    page.set_plan(_plan(tmp_path))
    page.set_recommendation(recommend_backend(_profile()))
    page.install_check.setChecked(True)
    page.yolo_check.setChecked(False)

    request = page.request()
    assert isinstance(request, SetupRequest)
    assert request.backend_key == "cu126"
    assert request.install is True
    assert request.download_yolo is False
    assert request.wheel_dir == tmp_path
    assert request.backend.key == "cu126"
    page.close()


def test_setup_page_emits_signals(qapp: QApplication) -> None:
    page = SetupPage()
    seen: list[object] = []
    page.startRequested.connect(seen.append)
    page.refreshRequested.connect(lambda: seen.append("refresh"))
    page.pauseRequested.connect(lambda: seen.append("pause"))
    page.directoryRequested.connect(seen.append)

    page._emit_start()
    page._emit_refresh()
    page._emit_pause()
    page._emit_directory()

    assert isinstance(seen[0], SetupRequest)
    assert "refresh" in seen and "pause" in seen
    page.close()


def test_setup_page_running_toggles_buttons(qapp: QApplication) -> None:
    page = SetupPage()
    page.set_running(True)
    assert page.pause_button.isVisibleTo(page) is True
    assert page.start_button.isVisibleTo(page) is False
    assert page.backend_combo.isEnabled() is False
    page.set_running(False)
    assert page.start_button.isVisibleTo(page) is True
    page.close()


# ------------------------------------------------------- 断点续传交互（核心）
def _write_partial(plan: InstallPlan, size: int = 400) -> Path:
    """给计划里的第一个文件造一个"下到一半"的断点。"""
    part = Path(f"{plan.items[0].path}.part")
    part.write_bytes(b"x" * size)
    return part


def test_status_column_reflects_cache_state(qapp: QApplication, tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    _write_partial(plan)
    page = SetupPage()
    page.set_plan(plan)
    page.set_cache_state(build_cache_state(_all_items(plan), directory=tmp_path))

    rows = page.files_table.rowCount()
    assert rows == len(plan.items) + len(yolo_items())
    statuses = [page.files_table.item(row, 3).text() for row in range(rows)]
    assert "可续传" in statuses[0]
    assert statuses[1:] == ["待下载"] * (rows - 1)
    # YOLO 权重是"可选项"，要标出来
    assert "（可选）" in page.files_table.item(rows - 1, 0).text()
    page.close()


def test_ready_file_shows_downloaded(qapp: QApplication, tmp_path: Path) -> None:
    plan = _plan(tmp_path, size=1024)
    plan.items[0].path.write_bytes(b"x" * 1024)  # 大小与官方索引一致 → 已下载
    page = SetupPage()
    page.set_plan(plan)
    page.set_cache_state(build_cache_state(list(plan.items), directory=tmp_path))
    assert page.files_table.item(0, 3).text() == "已下载"
    assert "已下载" in page.cache_label.text()
    page.close()


def test_mismatched_size_is_flagged(qapp: QApplication, tmp_path: Path) -> None:
    plan = _plan(tmp_path, size=1024)
    plan.items[0].path.write_bytes(b"x" * 5)  # 大小不对 → 会被重新下载
    page = SetupPage()
    page.set_plan(plan)
    page.set_cache_state(build_cache_state(list(plan.items), directory=tmp_path))
    assert "重新下载" in page.files_table.item(0, 3).text()
    page.close()


def test_action_button_becomes_resume_when_partial_exists(
    qapp: QApplication, tmp_path: Path
) -> None:
    plan = _plan(tmp_path)
    _write_partial(plan)
    page = SetupPage()
    page.set_plan(plan)

    assert page.start_button.text() == START_TEXT
    assert page.restart_button.isVisibleTo(page) is False

    page.set_cache_state(build_cache_state(_all_items(plan), directory=tmp_path))
    assert page.start_button.text() == RESUME_TEXT
    assert page.restart_button.isVisibleTo(page) is True
    assert "断点续传" in page.cache_label.text()

    page.set_running(True)
    assert page.restart_button.isVisibleTo(page) is False  # 下载中不给误点
    page.close()


def test_request_carries_resume_flag(qapp: QApplication, tmp_path: Path) -> None:
    page = SetupPage()
    page.set_plan(_plan(tmp_path))
    assert page.request().resume is True
    assert page.request(resume=False).resume is False
    page.close()


def test_restart_asks_before_discarding(
    qapp: QApplication, monkeypatch, tmp_path: Path
) -> None:
    page = SetupPage()
    page.set_plan(_plan(tmp_path))
    seen: list[SetupRequest] = []
    page.startRequested.connect(seen.append)

    monkeypatch.setattr(
        "video2personvideo.gui.pages.setup_page.QMessageBox.question",
        lambda *args, **kwargs: QMessageBox.StandardButton.No,
    )
    page._emit_restart()
    assert seen == []  # 用户点了"否"就不该动断点

    monkeypatch.setattr(
        "video2personvideo.gui.pages.setup_page.QMessageBox.question",
        lambda *args, **kwargs: QMessageBox.StandardButton.Yes,
    )
    page._emit_restart()
    assert len(seen) == 1
    assert seen[0].resume is False
    assert "丢弃断点" in page.log_view.toPlainText()
    page.close()


def test_progress_marks_current_row(qapp: QApplication, tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    page = SetupPage()
    page.set_plan(plan)
    page.set_cache_state(build_cache_state(list(plan.items), directory=tmp_path))

    page.set_progress(
        DownloadProgress(
            path=Path(f"{plan.items[0].path}.part"),
            url="u",
            downloaded=500,
            total=1000,
            resumed_from=200,
            speed_bps=1024,
            index=1,
            count=2,
        )
    )
    status = page.files_table.item(0, 3).text()
    assert status.startswith("下载中 50%")
    assert "续传自" in status
    assert ".part" not in page.file_label.text()
    page.close()


def test_cache_label_warns_when_disk_is_short(qapp: QApplication, tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    state = build_cache_state(list(plan.items), directory=tmp_path)
    state.free_bytes = 1  # 假装磁盘只剩 1 字节

    page = SetupPage()
    page.set_plan(plan)
    page.set_cache_state(state)

    assert state.enough_space is False
    assert "不足" in page.cache_label.text()
    assert page.cache_label.objectName() == "badgeWarn"
    page.close()


def test_yolo_checkbox_rebuilds_table_and_request(
    qapp: QApplication, tmp_path: Path
) -> None:
    page = SetupPage()
    page.set_plan(_plan(tmp_path))
    page.set_cache_state(build_cache_state(list(page.visible_items()), directory=tmp_path))

    with_yolo = page.files_table.rowCount()
    page.yolo_check.setChecked(False)
    assert page.files_table.rowCount() == with_yolo - len(yolo_items())
    assert page.request().download_yolo is False

    page.yolo_check.setChecked(True)
    assert page.files_table.rowCount() == with_yolo
    page.close()


def test_cache_summary_follows_yolo_checkbox(qapp: QApplication, tmp_path: Path) -> None:
    """汇总行只统计界面上列出来的文件：取消 YOLO 后"需下载"要跟着变小。"""
    plan = _plan(tmp_path, size=1000)
    page = SetupPage()
    page.set_plan(plan)
    page.set_cache_state(build_cache_state(_all_items(plan), directory=tmp_path))

    with_yolo = page.cache_label.text()
    page.yolo_check.setChecked(False)
    without_yolo = page.cache_label.text()

    assert with_yolo != without_yolo
    assert "5.4 MB" not in without_yolo  # YOLO 权重不再计入
    page.close()


def test_change_directory_emits_signal(qapp: QApplication, monkeypatch, tmp_path: Path) -> None:
    page = SetupPage()
    page.set_plan(_plan(tmp_path))
    seen: list[Path] = []
    page.directoryChanged.connect(seen.append)

    monkeypatch.setattr(
        "video2personvideo.gui.pages.setup_page.QFileDialog.getExistingDirectory",
        lambda *args, **kwargs: "",
    )
    page._choose_directory()
    assert seen == []  # 取消选择就什么都不做

    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.setattr(
        "video2personvideo.gui.pages.setup_page.QFileDialog.getExistingDirectory",
        lambda *args, **kwargs: str(other),
    )
    page._choose_directory()
    assert seen == [other]
    assert page.directory == other
    assert page.request().wheel_dir == other
    page.close()


def test_finished_reports_resumed_savings(qapp: QApplication, tmp_path: Path) -> None:
    page = SetupPage()
    page.set_finished(
        SetupResult(
            backend_key="cu126",
            cancelled=True,
            resumed_bytes=1_200_000_000,
            discarded=[tmp_path / "a.part"],
        )
    )
    log = page.log_view.toPlainText()
    assert "断点续传本次省下" in log
    assert "已丢弃 1 个断点" in log
    assert page.start_button.text() == RESUME_TEXT
    page.close()


# ------------------------------------------------------------------ 对话框
def test_setup_dialog_builds_without_autodetect(qapp: QApplication) -> None:
    dialog = SetupDialog(auto_detect=False, first_run=True)
    assert dialog.windowTitle().startswith("首次运行")
    assert dialog.page is not None
    assert dialog.done_button.isEnabled() is False
    dialog.close()


def test_setup_dialog_reacts_to_results(qapp: QApplication, tmp_path: Path) -> None:
    """直接驱动对话框的槽，验证状态流转（不起线程）。"""
    dialog = SetupDialog(auto_detect=False)
    profile = _profile()
    dialog.page.set_profile(profile)
    dialog.page.set_recommendation(recommend_backend(profile))
    dialog.page.set_plan(_plan(tmp_path))
    assert dialog.page.backend_key() == "cu126"

    dialog._on_finished(SetupResult(backend_key="cu126", installed=True))
    assert dialog.done_button.isEnabled() is True
    assert "重启" in dialog.footer_hint.text()
    assert "重启" in dialog.done_button.text()

    # 没有线程时暂停 / 关闭都不该崩
    dialog._on_pause()
    dialog.close()


def test_setup_dialog_replans_after_directory_change(
    qapp: QApplication, monkeypatch, tmp_path: Path
) -> None:
    """换下载目录必须重新解析地址，否则文件还是落到旧目录。"""
    from video2personvideo.gui.controller import SetupWorker

    seen: list[tuple[str, object]] = []

    def fake_load_plan(self: SetupWorker, backend_key: str) -> None:
        seen.append((backend_key, self.directory))
        self.taskFinished.emit("plan")

    monkeypatch.setattr(SetupWorker, "load_plan", fake_load_plan)

    dialog = SetupDialog(auto_detect=False)
    dialog.page.set_recommendation(recommend_backend(_profile()))
    dialog.page.set_plan(_plan(tmp_path))

    other = tmp_path / "other"
    other.mkdir()
    dialog.page._directory = other
    dialog._on_directory_changed(other)

    for _ in range(300):
        qapp.processEvents()
        if dialog._thread is None and seen:
            break
        QThread.msleep(10)

    assert seen == [("cu126", other)]
    assert dialog._thread is None  # 解析完地址后线程要收干净
    dialog.close()


def test_setup_dialog_spawns_and_reaps_thread(qapp: QApplication, monkeypatch) -> None:
    """自检任务结束后线程必须被回收（否则退出时可能崩）。"""
    from video2personvideo.gui.controller import SetupWorker

    monkeypatch.setattr(
        SetupWorker,
        "detect",
        lambda self, force=False: self.taskFinished.emit("detect"),
    )

    dialog = SetupDialog(auto_detect=False)
    dialog._start_detect(force=True)
    assert dialog._thread is not None
    for _ in range(300):
        qapp.processEvents()
        if dialog._thread is None:
            break
        QThread.msleep(10)
    assert dialog._thread is None
    dialog.close()


def test_setup_dialog_pending_request_flows_after_plan(qapp: QApplication, monkeypatch) -> None:
    """用户换方案后直接点开始：解析完地址要自动接着下载。"""
    from video2personvideo.gui.controller import SetupWorker

    calls: list[str] = []

    def fake_load_plan(self: SetupWorker, backend_key: str) -> None:
        calls.append(f"plan:{backend_key}")
        self.taskFinished.emit("plan")

    def fake_run(self: SetupWorker, request) -> None:
        calls.append("run")
        self.taskFinished.emit("run")

    monkeypatch.setattr(SetupWorker, "load_plan", fake_load_plan)
    monkeypatch.setattr(SetupWorker, "run", fake_run)

    dialog = SetupDialog(auto_detect=False)
    dialog._on_start(SetupRequest(backend_key="cu128"))

    for _ in range(300):
        qapp.processEvents()
        if "run" in calls:
            break
        QThread.msleep(10)
    # 顺序必须是"先解析地址，再下载"，不能反
    assert calls == ["plan:cu128", "run"]
    assert dialog._thread is None
    dialog.close()


def test_setup_settings_roundtrip(tmp_path: Path, monkeypatch) -> None:
    from video2personvideo.gui import setup_dialog as module

    store = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(module, "setup_settings", lambda: store)

    assert setup_completed() is False
    mark_setup_completed()
    assert setup_completed() is True


def test_maybe_show_setup_skips_dialog_when_already_done(
    tmp_path: Path, monkeypatch
) -> None:
    """已经自检过就不该再弹窗（首次运行只弹一次）。"""
    from video2personvideo.gui import setup_dialog as module

    store = QSettings(str(tmp_path / "s2.ini"), QSettings.Format.IniFormat)
    store.setValue(module.SETTINGS_KEY, True)
    store.sync()
    monkeypatch.setattr(module, "setup_settings", lambda: store)

    called: list[int] = []
    monkeypatch.setattr(
        module, "SetupDialog", lambda *args, **kwargs: called.append(1) or None
    )
    assert module.maybe_show_setup() is True
    assert called == []


def _settings_at(tmp_path: Path, name: str, monkeypatch, module):
    """一个干净的 QSettings（``setup/completed`` 未写入）。"""
    store = QSettings(str(tmp_path / name), QSettings.Format.IniFormat)
    monkeypatch.setattr(module, "setup_settings", lambda: store)
    return store


def test_maybe_show_setup_skips_when_environment_ready(tmp_path: Path, monkeypatch) -> None:
    """本机已经装好 torch + 有 YOLO 权重：不该再弹下载页。"""
    from video2personvideo.gui import setup_dialog as module

    _settings_at(tmp_path, "ready.ini", monkeypatch, module)
    monkeypatch.setattr(module, "environment_ready", lambda: True)

    called: list[int] = []
    monkeypatch.setattr(
        module, "SetupDialog", lambda *args, **kwargs: called.append(1) or None
    )

    assert module.maybe_show_setup() is True
    assert called == []
    # 不写标记：以后依赖被卸了，下一次启动还会重新自检
    assert setup_completed() is False


def test_maybe_show_setup_shows_dialog_when_environment_missing(
    tmp_path: Path, monkeypatch
) -> None:
    from video2personvideo.gui import setup_dialog as module

    _settings_at(tmp_path, "missing.ini", monkeypatch, module)
    monkeypatch.setattr(module, "environment_ready", lambda: False)

    created: list[dict] = []

    class _Dialog:
        def __init__(self, *args, **kwargs) -> None:
            created.append(kwargs)

        def exec(self) -> int:
            return 0  # 等价于"稍后再说"

    monkeypatch.setattr(module, "SetupDialog", _Dialog)

    assert module.maybe_show_setup() is False
    assert created == [{"first_run": True}]
    # force=True（主界面「环境自检」按钮）任何时候都要弹
    assert module.maybe_show_setup(force=True) is False
    assert created[-1] == {"first_run": False}


# ------------------------------------------------- 本机已就绪时的界面表现
def test_page_unchecks_torch_when_already_installed(
    qapp: QApplication, monkeypatch, tmp_path: Path
) -> None:
    """本机装过 PyTorch：默认不重下，清单里只剩 YOLO 权重。"""
    monkeypatch.setattr(setup_page_module, "torch_installed", lambda: True)
    monkeypatch.setattr(setup_page_module, "torch_version", lambda: "2.14.0+cpu")
    monkeypatch.setattr(
        setup_page_module, "environment_summary", lambda: "本机环境：已安装 PyTorch 2.14.0+cpu"
    )

    page = SetupPage()
    assert page.torch_check.isChecked() is False
    assert "2.14.0+cpu" in page.torch_check.text()
    # "本机已经有什么"那一行露出来（不是被 setVisible(False) 收起来的状态）
    assert "已安装 PyTorch" in page.local_label.text()
    assert page.local_label.isHidden() is False

    page.set_plan(_plan(tmp_path))
    assert page.files_table.rowCount() == len(yolo_items())
    request = page.request()
    assert request.skip_torch is True
    assert request.install is False

    # 想换 CUDA 版本的人自己勾回来：清单和请求都要跟着变
    page.torch_check.setChecked(True)
    assert page.files_table.rowCount() == 1 + len(yolo_items())
    assert page.request().skip_torch is False
    page.close()


def test_page_says_nothing_to_download_when_all_unchecked(
    qapp: QApplication, monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(setup_page_module, "torch_installed", lambda: True)

    page = SetupPage()
    page.set_plan(_plan(tmp_path))
    page.yolo_check.setChecked(False)

    assert page.files_table.rowCount() == 0
    assert page.start_button.text() == READY_TEXT
    assert "没有需要下载的文件" in page.cache_label.text()
    assert page.request().download_yolo is False
    # 什么都没装的环境不提"本机已有什么"
    assert page.local_label.isHidden() is True
    page.close()


# ------------------------------------------------------------------ 格式化
def test_format_helpers() -> None:
    assert format_seconds(0) == "—"
    assert format_seconds(45) == "45秒"
    assert format_seconds(125) == "2分5秒"
    assert format_seconds(3725) == "1小时2分"
    assert format_speed(0) == "—"
    assert format_speed(1024 * 1024).endswith("/s")
