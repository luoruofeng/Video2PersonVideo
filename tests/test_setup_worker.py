"""``SetupWorker`` 的后台逻辑：缓存状态上报、续传 / 丢弃断点、进度与结果汇总。

这些用例不起线程、不联网：把 ``build_plan`` / ``download_many`` 换成假实现，
只验证"界面拿到的东西"对不对。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6.QtWidgets", reason="未安装 PySide6，跳过 GUI 测试")

from PySide6.QtWidgets import QApplication  # noqa: E402

from video2personvideo.gui import controller as controller_module  # noqa: E402
from video2personvideo.gui.controller import (  # noqa: E402
    SetupRequest,
    SetupWorker,
)
from video2personvideo.utils.downloader import (  # noqa: E402
    DownloadBatchResult,
    DownloadItem,
    DownloadResult,
)
from video2personvideo.utils.torch_backends import backend_by_key  # noqa: E402
from video2personvideo.utils.torch_install import InstallPlan  # noqa: E402


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app  # type: ignore[return-value]


def _plan(tmp_path: Path, *, size: int = 1000) -> InstallPlan:
    return InstallPlan(
        backend=backend_by_key("cu126"),
        directory=tmp_path,
        notes=["使用 CUDA 12.6 官方源"],
        items=[
            DownloadItem(
                url="https://x/torch.whl",
                path=tmp_path / "torch-2.14.0+cu126-cp313-cp313-win_amd64.whl",
                label="torch 2.14.0",
                size=size,
            ),
            DownloadItem(
                url="https://x/tv.whl",
                path=tmp_path / "torchvision-0.29.0+cu126-cp313-cp313-win_amd64.whl",
                label="torchvision 0.29.0",
                size=size,
            ),
        ],
    )


def _worker(monkeypatch, plan: InstallPlan) -> SetupWorker:
    """造一个不联网的 worker：计划固定，YOLO 权重不参与。"""
    monkeypatch.setattr(controller_module, "build_plan", lambda *args, **kwargs: plan)
    monkeypatch.setattr(controller_module, "yolo_items", lambda *args, **kwargs: [])
    return SetupWorker()


def _fake_download_many(calls: list[dict], *, resumed_from: int = 0):
    def fake(items, *, progress=None, stop=None, timeout=None, retries=None, resume=True):
        calls.append({"count": len(items), "resume": resume})
        batch = DownloadBatchResult()
        for item in items:
            batch.items.append(
                DownloadResult(
                    path=item.path, url=item.url, resumed_from=resumed_from, bytes_total=1000
                )
            )
        return batch

    return fake


# --------------------------------------------------------------- 缓存状态上报
def test_publish_plan_emits_cache_state(qapp: QApplication, monkeypatch, tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    worker = _worker(monkeypatch, plan)
    Path(f"{plan.items[0].path}.part").write_bytes(b"x" * 400)

    states: list = []
    logs: list[str] = []
    worker.cacheState.connect(states.append)
    worker.log.connect(logs.append)

    worker._publish_plan(plan)

    assert len(states) == 1
    state = states[0]
    assert state.partial_bytes == 400
    assert state.has_partials
    assert state.total == 2000
    assert any("断点" in line for line in logs)


def test_publish_plan_warns_when_disk_short(
    qapp: QApplication, monkeypatch, tmp_path: Path
) -> None:
    plan = _plan(tmp_path)
    worker = _worker(monkeypatch, plan)
    real_build = controller_module.build_cache_state

    def short(items, *, directory=None):
        state = real_build(items, directory=directory)
        state.free_bytes = 1
        return state

    monkeypatch.setattr(controller_module, "build_cache_state", short)
    logs: list[str] = []
    worker.log.connect(logs.append)

    worker._publish_plan(plan)

    assert any("空间" in line for line in logs)


# --------------------------------------------------------------- 下载计划
def test_make_plan_passes_directory_override(
    qapp: QApplication, monkeypatch, tmp_path: Path
) -> None:
    seen: dict = {}

    def fake_build(backend, **kwargs):
        seen.update(kwargs)
        return _plan(tmp_path)

    monkeypatch.setattr(controller_module, "build_plan", fake_build)
    monkeypatch.setattr(controller_module, "yolo_items", lambda *args, **kwargs: [])

    worker = SetupWorker()
    other = tmp_path / "other"
    worker.directory = other
    worker._make_plan("cu126")

    assert seen["directory"] == other


def test_make_plan_reuses_plan_for_same_directory(
    qapp: QApplication, monkeypatch, tmp_path: Path
) -> None:
    """同一个构建 + 同一个目录不该重复联网抓索引（点开始下载时不能再卡一下）。"""
    calls: list = []
    plan = _plan(tmp_path)

    def fake_build(backend, **kwargs):
        calls.append(kwargs.get("directory"))
        return plan

    monkeypatch.setattr(controller_module, "build_plan", fake_build)
    worker = SetupWorker()
    worker.directory = tmp_path

    assert worker._make_plan("cu126") is plan
    assert worker._make_plan("cu126") is plan  # 命中缓存
    assert calls == [tmp_path]

    # 换目录 → 必须重新解析
    other = tmp_path / "other"
    worker.directory = other
    worker._make_plan("cu126")
    assert calls == [tmp_path, other]


# --------------------------------------------------------------- 下载执行
def test_run_inner_continues_from_partial(
    qapp: QApplication, monkeypatch, tmp_path: Path
) -> None:
    plan = _plan(tmp_path)
    worker = _worker(monkeypatch, plan)
    Path(f"{plan.items[0].path}.part").write_bytes(b"x" * 400)
    calls: list[dict] = []
    monkeypatch.setattr(controller_module, "download_many", _fake_download_many(calls, resumed_from=400))
    logs: list[str] = []
    worker.log.connect(logs.append)

    result = worker._run_inner(SetupRequest(backend_key="cu126", download_yolo=False, install=False))

    assert calls == [{"count": 2, "resume": True}]
    assert result.resumed_bytes == 800  # 两个文件各续传 400
    assert result.discarded == []
    assert any("接着上次的进度" in line for line in logs)
    assert Path(f"{plan.items[0].path}.part").exists()  # 续传不会丢断点


def test_run_inner_discards_partials_on_restart(
    qapp: QApplication, monkeypatch, tmp_path: Path
) -> None:
    plan = _plan(tmp_path)
    worker = _worker(monkeypatch, plan)
    part = Path(f"{plan.items[0].path}.part")
    part.write_bytes(b"x" * 400)
    calls: list[dict] = []
    monkeypatch.setattr(controller_module, "download_many", _fake_download_many(calls))
    logs: list[str] = []
    worker.log.connect(logs.append)

    result = worker._run_inner(
        SetupRequest(
            backend_key="cu126", download_yolo=False, install=False, resume=False
        )
    )

    assert calls == [{"count": 2, "resume": False}]
    assert [item.name for item in result.discarded] == [part.name]
    assert not part.exists()
    assert any("丢弃" in line for line in logs)


def test_run_inner_skips_already_downloaded(
    qapp: QApplication, monkeypatch, tmp_path: Path
) -> None:
    plan = _plan(tmp_path)
    plan.items[0].path.write_bytes(b"x" * 1000)  # 大小与索引一致 → 已下好
    worker = _worker(monkeypatch, plan)
    monkeypatch.setattr(controller_module, "download_many", _fake_download_many([]))
    logs: list[str] = []
    worker.log.connect(logs.append)

    worker._run_inner(SetupRequest(backend_key="cu126", download_yolo=False, install=False))

    assert any("只做校验不重复下载" in line for line in logs)


def test_run_inner_reports_failures_without_stopping(
    qapp: QApplication, monkeypatch, tmp_path: Path
) -> None:
    plan = _plan(tmp_path)
    worker = _worker(monkeypatch, plan)

    def fake(items, **kwargs):
        batch = DownloadBatchResult()
        batch.items.append(DownloadResult(path=items[0].path, url="u", error="连接超时"))
        batch.items.append(DownloadResult(path=items[1].path, url="u", bytes_total=10))
        return batch

    monkeypatch.setattr(controller_module, "download_many", fake)

    result = worker._run_inner(SetupRequest(backend_key="cu126", download_yolo=False, install=False))

    assert result.failed == [(plan.items[0].path.name, "连接超时")]
    assert result.downloaded == [plan.items[1].path]
    assert not result.ok


def test_run_inner_skips_install_when_not_requested(
    qapp: QApplication, monkeypatch, tmp_path: Path
) -> None:
    plan = _plan(tmp_path)
    worker = _worker(monkeypatch, plan)
    monkeypatch.setattr(controller_module, "download_many", _fake_download_many([]))
    monkeypatch.setattr(
        controller_module,
        "install_plan",
        lambda *args, **kwargs: pytest.fail("不该调用安装"),
    )

    result = worker._run_inner(SetupRequest(backend_key="cu126", download_yolo=False, install=False))

    assert result.skipped_install
    assert str(tmp_path) in result.install_message


def test_run_inner_skips_torch_when_already_installed(
    qapp: QApplication, monkeypatch, tmp_path: Path
) -> None:
    """本机装好了 PyTorch：只补下 YOLO 权重，不再碰 wheel、也不安装。"""
    plan = _plan(tmp_path)
    worker = _worker(monkeypatch, plan)
    weight = DownloadItem(
        url="https://x/yolo11n.pt", path=tmp_path / "yolo11n.pt", label="检测模型", size=100
    )
    monkeypatch.setattr(controller_module, "yolo_items", lambda *args, **kwargs: [weight])
    calls: list[dict] = []
    monkeypatch.setattr(controller_module, "download_many", _fake_download_many(calls))
    monkeypatch.setattr(
        controller_module, "install_plan", lambda *args, **kwargs: pytest.fail("不该调用安装")
    )
    logs: list[str] = []
    worker.log.connect(logs.append)

    result = worker._run_inner(
        SetupRequest(backend_key="cu126", skip_torch=True, install=True)
    )

    assert calls == [{"count": 1, "resume": True}]  # 只有 YOLO 那一批
    assert result.downloaded == []  # torch 的 wheel 完全没参与
    assert result.yolo_paths == [weight.path]
    assert result.skipped_install is True
    assert any("跳过下载" in line for line in logs)


def test_run_inner_with_nothing_selected_is_not_a_failure(
    qapp: QApplication, monkeypatch, tmp_path: Path
) -> None:
    """PyTorch 和 YOLO 都跳过 = 一次"确认已就绪"，不是报错。"""
    worker = _worker(monkeypatch, _plan(tmp_path))

    result = worker._run_inner(
        SetupRequest(backend_key="cu126", skip_torch=True, download_yolo=False)
    )

    assert result.ok and result.skipped_install
    assert "无需下载" in result.install_message


def test_start_run_without_request_reports_error(qapp: QApplication) -> None:
    worker = SetupWorker()
    errors: list[str] = []
    done: list[str] = []
    worker.failed.connect(errors.append)
    worker.taskFinished.connect(done.append)

    worker.start_run()  # 没有请求就调用，不该崩

    assert errors and "没有可执行" in errors[0]
    assert done == ["run"]


def test_setup_request_backend_falls_back_to_cpu() -> None:
    assert SetupRequest(backend_key="不存在").backend.key == "cpu"
    assert SetupRequest().backend.key == "cpu"
    assert SetupRequest(backend_key="cu126").backend.key == "cu126"
