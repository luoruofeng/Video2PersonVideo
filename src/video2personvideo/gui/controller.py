"""GUI ↔ 后台任务桥接：只做信号槽转发，不写业务逻辑。

所有长耗时操作都跑在 ``QThread`` 里，界面线程只负责刷新控件（M3-5）。
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from ..config import AppConfig
from ..core.batch import BatchResult, run_batch
from ..utils.downloader import (
    CacheState,
    DownloadCancelled,
    DownloadItem,
    build_cache_state,
    discard_partials,
    download_many,
)
from ..utils.gpu_probe import HardwareProfile, probe_hardware
from ..utils.logger import attach_handler, detach_handler, get_logger
from ..utils.torch_backends import BackendRecommendation, backend_by_key, recommend_backend
from ..utils.torch_install import (
    InstallPlan,
    build_plan,
    install_plan,
    is_frozen,
    yolo_items,
)

logger = get_logger(__name__)


class _SignalLogHandler(logging.Handler):
    """把 logging 记录转发到 Qt 信号（供界面日志区显示）。"""

    def __init__(self, signal) -> None:  # type: ignore[valid-type]
        super().__init__()
        self._signal = signal
        self.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
        )

    def emit(self, record: logging.LogRecord) -> None:
        with contextlib.suppress(Exception):  # 界面销毁时可能失败
            self._signal.emit(self.format(record))


@dataclass(slots=True)
class SetupRequest:
    """首次运行自检页点「开始下载」时提交的参数。"""

    backend_key: str | None = None
    #: 跳过 PyTorch 下载（本机已经装好了，沿用现有版本）
    skip_torch: bool = False
    #: 是否顺带下载 YOLO 权重
    download_yolo: bool = True
    #: 下载完成后是否直接 pip 安装
    install: bool = True
    #: wheel / 模型的落地目录（``None`` = 用默认位置）
    wheel_dir: Path | None = None
    model_dir: Path | None = None
    #: ``False`` = 丢弃已有断点，从头下载（用户点了「重新下载」）
    resume: bool = True

    @property
    def backend(self):
        return backend_by_key(self.backend_key)


@dataclass(slots=True)
class SetupResult:
    """自检 + 下载 + 安装的结果。"""

    backend_key: str
    downloaded: list[Path] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    installed: bool = False
    install_message: str = ""
    yolo_paths: list[Path] = field(default_factory=list)
    cancelled: bool = False
    #: 本次下载中来自断点续传的字节数
    resumed_bytes: int = 0
    #: 只下载不安装（打包环境 / 用户选择）
    skipped_install: bool = False
    #: 本次丢弃的断点文件（用户点了「重新下载」）
    discarded: list[Path] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failed and not self.cancelled


class SetupWorker(QObject):
    """在子线程里跑「硬件自检 → 生成下载计划 → 断点续传下载 → 安装」。

    界面只接信号，不做任何阻塞调用；下载可随时暂停（``.part`` 保留，下次续传）。
    """

    detected = Signal(object)  # HardwareProfile
    recommended = Signal(object)  # BackendRecommendation
    planReady = Signal(object)  # InstallPlan
    cacheState = Signal(object)  # downloader.CacheState（还有多少要下、磁盘够不够）
    stage = Signal(str)  # 阶段文字
    progress = Signal(object)  # DownloadProgress
    totalProgress = Signal(int, int, str)  # 已完成文件数 / 总数 / 描述
    log = Signal(str)
    finished = Signal(object)  # SetupResult
    failed = Signal(str)
    #: 每个任务跑完都会发一次（``detect`` / ``plan`` / ``run``），界面据此收线程
    taskFinished = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._stop = False
        self._profile: HardwareProfile | None = None
        self._plan: InstallPlan | None = None

        #: 这几个是给「无参槽」读的入参（QThread.started 不带参数）
        self.force = False
        self.backend_key = ""
        self.request: SetupRequest | None = None
        #: 下载目录覆盖（用户在界面上改了「下载目录」时用）
        self.directory: Path | None = None

    # ------------------------------------------------------------- 控制
    def request_stop(self) -> None:
        """线程安全地请求暂停 / 取消（已下载的部分会保留下来）。"""
        self._stop = True

    def _stopped(self) -> bool:
        return self._stop

    # --------------------------------------------------- 供 QThread 调用的无参槽
    @Slot()
    def start_detect(self) -> None:
        """``QThread.started`` 是无参信号，用它跑一次自检。"""
        self.detect(self.force)

    @Slot()
    def start_load_plan(self) -> None:
        self.load_plan(self.backend_key)

    @Slot()
    def start_run(self) -> None:
        if self.request is None:
            self.failed.emit("没有可执行的下载任务")
            self.taskFinished.emit("run")
            return
        self.run(self.request)

    # ------------------------------------------------------------- 自检
    def detect(self, force: bool = False) -> None:
        """硬件自检 + 推荐 + 生成下载计划（计划需要联网抓索引）。"""
        handler = _SignalLogHandler(self.log)
        attach_handler(handler)
        try:
            self.stage.emit("正在检测 CPU / 显卡型号…")
            profile = probe_hardware(force=force)
            self._profile = profile
            self.detected.emit(profile)

            self.stage.emit("正在匹配 PyTorch 构建…")
            recommendation: BackendRecommendation = recommend_backend(profile)
            self.recommended.emit(recommendation)

            self.stage.emit("正在解析官方下载地址…")
            plan = self._make_plan(recommendation.backend.key, force=force)
            if plan is not None:
                self._publish_plan(plan)
            self.stage.emit("自检完成，确认方案后点「开始下载」")
        except Exception as exc:  # noqa: BLE001 - 统一上报界面
            logger.exception("环境自检失败")
            self.failed.emit(str(exc))
        finally:
            detach_handler(handler)
            self.taskFinished.emit("detect")

    def load_plan(self, backend_key: str) -> None:
        """用户手动切换方案（或改了下载目录）后重新解析下载地址。"""
        handler = _SignalLogHandler(self.log)
        attach_handler(handler)
        try:
            self.stage.emit(f"正在解析 {backend_key} 的下载地址…")
            plan = self._make_plan(backend_key, force=True)
            if plan is not None:
                self._publish_plan(plan)
            self.stage.emit("就绪")
        except Exception as exc:  # noqa: BLE001
            logger.exception("解析下载计划失败")
            self.failed.emit(str(exc))
        finally:
            detach_handler(handler)
            self.taskFinished.emit("plan")

    def _make_plan(self, backend_key: str, *, force: bool = False) -> InstallPlan | None:
        cached = self._plan
        if (
            cached is not None
            and not force
            and cached.backend.key == backend_key
            # 目录也要一致：换了目录相当于换了一份计划
            and (self.directory is None or Path(cached.directory) == Path(self.directory))
        ):
            return cached
        try:
            plan = build_plan(
                backend_by_key(backend_key),
                profile=self._profile,
                with_sizes=True,
                directory=self.directory,
            )
        except Exception as exc:  # noqa: BLE001 - 网络问题不该让界面崩
            logger.exception("生成下载计划失败")
            self.failed.emit(f"无法连接 PyTorch 官方源：{exc}")
            return None
        self._plan = plan
        if plan.problems:
            for problem in plan.problems:
                self.log.emit(problem)
        return plan

    def _publish_plan(self, plan: InstallPlan) -> None:
        """把下载计划 + 缓存现状一起交给界面。"""
        self.planReady.emit(plan)
        state = self._cache_state(plan)
        if state.has_partials:
            self.log.emit(
                f"发现未完成的下载：{state.partial_files} 个断点，"
                "可以接着下，也可以点「重新下载」丢弃断点重来。"
            )
        if state.enough_space is False:
            self.log.emit("警告：下载目录所在磁盘剩余空间可能不足，建议先清理或换目录。")
        self.cacheState.emit(state)

    def _cache_state(self, plan: InstallPlan) -> CacheState:
        """汇总"本次要下的所有文件"的缓存现状（只 stat，秒回）。"""
        items: list[DownloadItem] = list(plan.items) + list(yolo_items())
        directory = plan.directory or self.directory
        return build_cache_state(items, directory=directory)

    # ------------------------------------------------------------- 执行
    def run(self, request: SetupRequest) -> None:
        """下载（+ 可选安装）。可以在自检之后单独调用。"""
        handler = _SignalLogHandler(self.log)
        attach_handler(handler)
        self._stop = False
        try:
            result = self._run_inner(request)
        except DownloadCancelled:
            self.finished.emit(
                SetupResult(backend_key=request.backend.key, cancelled=True)
            )
        except Exception as exc:  # noqa: BLE001 - 统一上报界面
            logger.exception("下载 / 安装失败")
            self.failed.emit(str(exc))
        else:
            if self._plan is not None:
                # 刷新缓存现状：刚下完的文件这次应该显示"已下载"
                self.cacheState.emit(self._cache_state(self._plan))
            self.finished.emit(result)
        finally:
            detach_handler(handler)
            self.taskFinished.emit("run")

    def _run_inner(self, request: SetupRequest) -> SetupResult:
        result = SetupResult(backend_key=request.backend.key)
        plan = self._make_plan(request.backend.key)

        batches: list[tuple[str, list]] = []
        #: 哪些文件属于 PyTorch（结果归类用；跳过 torch 时 YOLO 批次会变成第 1 批）
        torch_paths: set[Path] = set()
        if request.skip_torch:
            self.log.emit("本机已安装 PyTorch：跳过下载，沿用现有版本。")
        elif plan is not None and plan.items:
            batches.append((f"PyTorch · {request.backend.label}", plan.items))
            torch_paths.update(item.path for item in plan.items)
        if request.download_yolo:
            items = yolo_items(request.model_dir)
            if items:
                batches.append(("YOLO 权重", items))

        if not batches:
            # 勾掉的都被跳过了：这是一次"确认本机已就绪"，不是失败
            result.skipped_install = True
            result.install_message = "无需下载：PyTorch 与 YOLO 权重都已就绪。"
            self.log.emit(result.install_message)
            return result

        all_items = [item for _, items in batches for item in items]
        if not request.resume:
            removed = discard_partials([item.path for item in all_items])
            result.discarded = removed
            if removed:
                self.log.emit(f"已丢弃 {len(removed)} 个断点，从头开始下载。")
        else:
            state = build_cache_state(
                all_items, directory=plan.directory if plan is not None else request.wheel_dir
            )
            if state.has_partials:
                self.log.emit(
                    f"检测到 {state.partial_files} 个未完成的下载，将接着上次的进度继续"
                    f"（约 {state.partial_bytes / 1024 / 1024:.0f} MB 可直接复用）。"
                )
            if state.ready_files:
                self.log.emit(f"检测到 {state.ready_files} 个已下载的文件，只做校验不重复下载。")

        for index, (title, items) in enumerate(batches, start=1):
            self.stage.emit(f"正在下载 {title}…")
            self.log.emit(f"开始下载 {title}（{len(items)} 个文件）")
            batch = download_many(
                items,
                progress=self.progress.emit,
                stop=self._stopped,
                resume=request.resume,
            )
            for item in batch.items:
                result.resumed_bytes += item.resumed_from
                if item.error:
                    result.failed.append((item.path.name, item.error))
                elif item.skipped:
                    self.log.emit(f"已存在，跳过：{item.path.name}")
                    if item.path not in torch_paths:
                        result.yolo_paths.append(item.path)
                else:
                    self.log.emit(f"下载完成：{item.path.name}")
                    if item.path in torch_paths:
                        result.downloaded.append(item.path)
                    else:
                        result.yolo_paths.append(item.path)
            if batch.cancelled:
                result.cancelled = True
                self.totalProgress.emit(index - 1, len(batches), "已暂停（断点已保留）")
                return result
            self.totalProgress.emit(index, len(batches), f"{title} 完成")

        if request.skip_torch:
            result.skipped_install = True
            result.install_message = "已跳过 PyTorch 下载：沿用本机已安装的版本。"
            self.log.emit(result.install_message)
            return result

        if not request.install:
            result.skipped_install = True
            directory = plan.directory if plan is not None else request.wheel_dir
            result.install_message = f"安装包已下载到 {directory}，可自行用 pip 安装。"
            self.log.emit(result.install_message)
            if plan is not None:
                self.log.emit(plan.pip_command())
            return result

        if is_frozen():
            result.skipped_install = True
            result.install_message = (
                "当前运行在打包好的 exe 里，无法自动安装依赖。\n"
                "请把下载好的 wheel 用 pip 安装（命令已显示在日志里）。"
            )
            self.log.emit(result.install_message)
            if plan is not None:
                self.log.emit(plan.pip_command())
            return result

        if plan is None or not plan.items:
            result.skipped_install = True
            return result

        self.stage.emit("正在安装 PyTorch（pip）…")
        self.log.emit("开始安装，这可能需要几分钟，请勿关闭窗口…")
        ok, message = install_plan(plan, on_output=self.log.emit)
        result.installed = ok
        result.install_message = (
            "安装完成，重启程序后生效。" if ok else "安装失败，详见日志。"
        )
        self.log.emit(message.strip().splitlines()[-1] if message.strip() else result.install_message)
        return result


class BatchWorker(QObject):
    """在子线程里跑一批（或单个）视频的裁剪任务。"""

    progress = Signal(object)  # BatchProgress
    taskUpdated = Signal(object)  # Task
    finished = Signal(object)  # BatchResult
    failed = Signal(str)
    log = Signal(str)

    def __init__(self, cfg: AppConfig, sources: Sequence[str | Path]) -> None:
        super().__init__()
        self._cfg = cfg
        self._sources = [Path(item) for item in sources]
        self._stop = False

    def request_stop(self) -> None:
        """线程安全地请求中止（处理循环会把当前帧跑完后退出）。"""
        self._stop = True

    @Slot()
    def run(self) -> None:
        handler = _SignalLogHandler(self.log)
        attach_handler(handler)
        try:
            result = run_batch(
                self._sources,
                self._cfg,
                progress_cb=self.progress.emit,
                stop_cb=lambda: self._stop,
                task_cb=self.taskUpdated.emit,
            )
        except Exception as exc:  # noqa: BLE001 - 统一上报到界面，禁止裸异常崩溃
            logger.exception("批量处理失败")
            self.failed.emit(str(exc))
        else:
            self.finished.emit(result)
        finally:
            detach_handler(handler)


__all__ = [
    "BatchResult",
    "BatchWorker",
    "SetupRequest",
    "SetupResult",
    "SetupWorker",
]
