"""安装引擎：真正"把软件装上去"的那部分（不依赖 Qt，可单独测）。

执行顺序（每步都会落盘，任何一步中断都能接着走）：

``prepare`` 建目录 → ``runtime`` 部署内嵌 Python → ``pip`` 引导 pip →
``torch`` 按显卡下载并安装 PyTorch → ``deps`` 安装其余依赖 → ``app`` 安装程序本体 →
``yolo`` 下载权重 → ``ffmpeg``（可选）→ ``finalize`` 快捷方式 / 注册表 / 自检。

设计上刻意把三件事分清楚：

* **谁在下载**：一律走 :mod:`video2personvideo.utils.downloader`，
  所以全部文件都自带断点续传与 sha256 校验；
* **谁在安装**：一律走 ``<安装目录>/python/python.exe -m pip``，
  所有东西都装进这份独立环境，不碰用户的 Python / conda；
* **谁在记账**：一律走 :class:`~video2personvideo.installer.journal.InstallJournal`，
  界面只读状态文件就知道"现在该从哪儿接着走"。
"""

from __future__ import annotations

import shutil
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .. import __version__
from ..utils.downloader import (
    CacheState,
    DownloadBatchResult,
    DownloadItem,
    DownloadProgress,
    build_cache_state,
    download_many,
)
from ..utils.gpu_probe import HardwareProfile
from ..utils.logger import get_logger
from ..utils.torch_backends import backend_by_key
from ..utils.torch_install import (
    InstallPlan,
    build_plan,
    platform_tags,
    run_process_streaming,
    yolo_items,
)
from . import payload, paths, windows
from .journal import InstallJournal
from .options import FFMPEG_APPROX_BYTES, FFMPEG_MEMBERS, FFMPEG_URL, InstallOptions
from .runtime import (
    extract_runtime,
    get_pip_download_item,
    runtime_download_item,
    runtime_present,
    write_pth,
)
from .stages import ORDERED_STAGES, Stage, stage_info, weight_before, weight_span

logger = get_logger(__name__)

#: 下载缓存大小上限提示：低于这个剩余空间会明确告警
MIN_FREE_SPACE_BYTES = 3 * 1024 * 1024 * 1024

#: PyTorch 构建的大致下载体积（界面预估用，不联网；真实体积以官方索引为准）
TORCH_APPROX_BYTES = {
    "cpu": 260_000_000,
    "cuda": 2_600_000_000,
    "rocm": 3_100_000_000,
    "xpu": 900_000_000,
    "mps": 260_000_000,
}


class InstallError(RuntimeError):
    """安装过程中的可预期失败（消息直接给用户看）。"""


class InstallCancelled(Exception):
    """用户中止了安装（断点与状态都已保留）。"""


# --------------------------------------------------------------------- 事件上报
@dataclass(slots=True)
class Reporter:
    """安装过程中的回调集合（全部可选；回调抛异常不影响安装）。"""

    on_log: Callable[[str], None] | None = None
    #: (阶段, 状态 running/done/skipped/failed, 说明)
    on_stage: Callable[[Stage, str, str], None] | None = None
    #: (阶段内进度 0~1, 说明)
    on_stage_progress: Callable[[float, str], None] | None = None
    #: (总进度 0~1, 说明)
    on_total_progress: Callable[[float, str], None] | None = None
    #: 原始下载进度（速度 / 剩余时间 / 续传字节）
    on_download: Callable[[DownloadProgress], None] | None = None

    def log(self, message: str) -> None:
        logger.info("%s", message)
        self._safe(self.on_log, message)

    def stage(self, stage: Stage, status: str, message: str = "") -> None:
        self._safe(self.on_stage, stage, status, message)

    def progress(self, fraction: float, text: str = "") -> None:
        self._safe(self.on_stage_progress, max(0.0, min(fraction, 1.0)), text)

    def total(self, fraction: float, text: str = "") -> None:
        self._safe(self.on_total_progress, max(0.0, min(fraction, 1.0)), text)

    def download(self, event: DownloadProgress) -> None:
        self._safe(self.on_download, event)

    @staticmethod
    def _safe(callback, *args) -> None:
        if callback is None:
            return
        try:
            callback(*args)
        except Exception as exc:  # noqa: BLE001 - 界面回调出错不该影响安装
            logger.debug("安装回调异常：%s", exc)


@dataclass(slots=True)
class InstallOutcome:
    """一次安装执行的结果。"""

    ok: bool
    cancelled: bool
    stage: Stage | None
    message: str
    journal: InstallJournal

    @property
    def complete(self) -> bool:
        return self.ok and self.journal.complete


@dataclass(slots=True)
class RollbackReport:
    """回滚结果。"""

    removed_paths: list[Path] = field(default_factory=list)
    scheduled: list[Path] = field(default_factory=list)
    kept_cache: bool = True
    cache_path: Path | None = None
    message: str = ""


def backend_approx_bytes(backend_key: str) -> int:
    """某个 PyTorch 构建的大致下载体积（按 backend 前缀估算）。"""
    key = (backend_key or "cpu").lower()
    if key.startswith("cu"):
        return TORCH_APPROX_BYTES["cuda"]
    if key.startswith("rocm"):
        return TORCH_APPROX_BYTES["rocm"]
    if key.startswith("xpu"):
        return TORCH_APPROX_BYTES["xpu"]
    if key.startswith("mps"):
        return TORCH_APPROX_BYTES["mps"]
    return TORCH_APPROX_BYTES["cpu"]


def estimate_download_bytes(options: InstallOptions, *, backend_key: str | None = None) -> int:
    """本次安装大概要下多少字节（不联网的粗估，界面用来提示"还需要多大空间"）。"""
    total = 0
    key = backend_key or options.backend_key
    if options.install_torch:
        total += backend_approx_bytes(key)
    if "yolo11n.pt" in options.weights:
        total += 5_400_000
    if "yolo11n-pose.pt" in options.weights:
        total += 6_100_000
    if options.install_ffmpeg:
        total += FFMPEG_APPROX_BYTES
    total += 12_000_000  # 内嵌 Python + get-pip
    return total


def estimate_install_bytes(options: InstallOptions, *, backend_key: str | None = None) -> int:
    """装完之后大约占多少磁盘（下载体积 + 解压后的体积）。"""
    download = estimate_download_bytes(options, backend_key=backend_key)
    return download + int(download * 1.6) + 300_000_000


def describe_exit(
    stage: Stage | None,
    *,
    downloaded_bytes: int = 0,
    total_bytes: int | None = None,
    install_dir_exists: bool = True,
) -> str:
    """回答"现在退出会怎样、下次怎么继续"（界面关闭时的确认框用它）。

    这是需求里最容易做错、也最影响体验的一点：安装可能要下一个多小时，
    用户随时可能关窗口，必须清楚告诉他"退了会损失什么、下次能不能接着来"。
    """
    if stage is None:
        return "安装尚未开始，退出不会留下任何文件。"
    info = stage_info(stage)
    lines: list[str] = []

    if stage is Stage.PREPARE and not install_dir_exists:
        lines.append("还没有往磁盘上写任何东西，退出不会有残留。")
    elif info.resumable:
        lines.append(f"当前位置：{info.title}（支持断点续传）。")
        if downloaded_bytes:
            lines.append(f"已经下载 {paths.format_size(downloaded_bytes)}，退出后会保留，不会白下。")
    else:
        lines.append(f"当前位置：{info.title}。")
        lines.append("这一步不能只做一半：退出后下次会重跑这一步，但已经下载的安装包会复用。")

    if info.exit_note:
        lines.append(info.exit_note)
    if total_bytes:
        remaining = max(total_bytes - downloaded_bytes, 0)
        lines.append(f"预计还要下载约 {paths.format_size(remaining)}。")
    lines.append("退出方式可选：① 暂停并退出（保留进度，下次继续）；② 回滚并退出（删掉已写入的文件）。")
    return "\n".join(lines)


class InstallEngine:
    """一次安装的执行者。线程不亲和：在后台线程里调用 :meth:`run` 即可。"""

    def __init__(
        self,
        options: InstallOptions,
        *,
        journal: InstallJournal | None = None,
        profile: HardwareProfile | None = None,
        reporter: Reporter | None = None,
        downloader: Callable[..., DownloadBatchResult] | None = None,
        process_runner: Callable[..., tuple[int, str]] | None = None,
        plan_builder: Callable[..., InstallPlan] | None = None,
        payload_root: Path | None = None,
        app_version: str | None = None,
    ) -> None:
        self.options = options
        self.install_dir = options.resolved_install_dir()
        self.profile = profile
        self.reporter = reporter or Reporter()
        self.app_version = app_version or __version__
        self.journal = journal
        self._stop = False

        self._downloader = downloader or download_many
        self._process_runner = process_runner or run_process_streaming
        self._plan_builder = plan_builder or build_plan
        self._payload_root = payload_root
        #: 本次执行中"已经下载了多少字节"（界面显示与退出提示用）
        self.downloaded_bytes = 0
        self.total_bytes: int | None = None
        self._current_stage: Stage | None = None
        self._plan: InstallPlan | None = None

    # ------------------------------------------------------------- 控制
    def cancel(self) -> None:
        """请求中止（下载断点与已下载的文件都会保留）。"""
        self._stop = True

    def cancelled(self) -> bool:
        return self._stop

    def _check_stop(self) -> None:
        if self._stop:
            raise InstallCancelled()

    @property
    def current_stage(self) -> Stage | None:
        return self._current_stage

    # ------------------------------------------------------------- 查询
    def pending_download_items(self) -> list[DownloadItem]:
        """还没下完的文件（界面显示"还要下多少、能不能续传"）。"""
        items: list[DownloadItem] = []
        if self.options.install_torch and self._plan is not None:
            items.extend(self._plan.items)
        items.extend(self._yolo_download_items())
        if self.options.install_ffmpeg:
            items.append(self._ffmpeg_item())
        items.append(runtime_download_item(paths.downloads_dir(), version=self.options.python_version))
        items.append(get_pip_download_item(paths.downloads_dir()))
        return items

    def cache_state(self) -> CacheState:
        """当前所有待下载文件在磁盘上的现状（已下载 / 可续传 / 待下载）。"""
        return build_cache_state(self.pending_download_items())

    def resume_hint(self) -> str:
        """一句"能接着下多少"的说明。"""
        try:
            state = self.cache_state()
        except OSError:  # pragma: no cover
            return ""
        if state.has_partials:
            return (
                f"检测到断点：{state.partial_files} 个文件已下 "
                f"{paths.format_size(state.partial_bytes)}，可以接着下。"
            )
        if state.ready_files:
            return f"已有 {state.ready_files} 个安装包下好了，本次只做校验不重复下载。"
        return "尚未开始下载。"

    # ------------------------------------------------------------- 执行
    def prepare_journal(self) -> InstallJournal:
        """准备好状态文件（已存在就接着用，按新选项做必要的一致性修正）。"""
        journal = self.journal or InstallJournal.load_for(self.install_dir)
        if journal is None:
            journal = InstallJournal.create(
                install_dir=self.install_dir,
                app_version=self.app_version,
                python_version=self.options.python_version,
                backend_key=self.options.backend_key,
                options=self.options.to_dict(),
            )
        else:
            # 换了 PyTorch 构建 → 这一阶段及其之后必须重做（否则装的还是旧构建）
            if journal.backend_key and journal.backend_key != self.options.backend_key:
                journal.reset_from(Stage.TORCH)
            journal.backend_key = self.options.backend_key
            journal.python_version = self.options.python_version
            journal.app_version = self.app_version
        journal.options = self.options.to_dict()
        self.journal = journal
        return journal

    def run(self) -> InstallOutcome:
        """从当前未完成的阶段开始，一路装到底。"""
        journal = self.prepare_journal()
        journal.begin_run()
        journal.save()
        self.reporter.log(f"开始安装：{self.options.describe()}")

        if not self.options.install_torch:
            self.reporter.log("按用户选择跳过 PyTorch 下载，将复用本机环境中已有的版本。")
        self.reporter.log(f"下载缓存：{paths.downloads_dir()}")

        try:
            for info in ORDERED_STAGES:
                if journal.is_done(info.stage):
                    continue
                if self._should_skip(info.stage):
                    journal.mark_skipped(info.stage, self._skip_reason(info.stage))
                    journal.save()
                    self.reporter.stage(info.stage, "skipped", self._skip_reason(info.stage))
                    self._emit_total(info.stage, 1.0, self._skip_reason(info.stage))
                    continue
                self._run_stage(info.stage)
            journal.finish()
            journal.save()
        except InstallCancelled:
            stage = self._current_stage
            journal.mark_interrupted(stage, "用户中止安装")
            journal.save()
            reason = describe_exit(
                stage,
                downloaded_bytes=self.downloaded_bytes,
                total_bytes=self.total_bytes,
                install_dir_exists=self.install_dir.exists(),
            )
            self.reporter.log("安装已暂停，进度与断点都已保留。")
            self.reporter.stage(stage or Stage.PREPARE, "cancelled", reason)
            return InstallOutcome(
                ok=False,
                cancelled=True,
                stage=stage,
                message="安装已暂停（可以下次继续，已下载的内容不会重复下载）",
                journal=journal,
            )
        except InstallError as exc:
            stage = self._current_stage or Stage.PREPARE
            journal.mark_failed(stage, str(exc))
            journal.save()
            self.reporter.log(f"安装失败：{exc}")
            self.reporter.stage(stage, "failed", str(exc))
            return InstallOutcome(
                ok=False, cancelled=False, stage=stage, message=str(exc), journal=journal
            )
        except Exception as exc:  # noqa: BLE001 - 兜底：绝不让安装器自己崩掉
            stage = self._current_stage or Stage.PREPARE
            logger.exception("安装过程中出现未预期的错误")
            journal.mark_failed(stage, f"未预期的错误：{exc}")
            journal.save()
            self.reporter.stage(stage, "failed", str(exc))
            return InstallOutcome(
                ok=False,
                cancelled=False,
                stage=stage,
                message=f"安装过程中出现未预期的错误：{exc}",
                journal=journal,
            )

        self._emit_total(Stage.DONE, 1.0, "安装完成")
        self.reporter.stage(Stage.DONE, "done", "安装完成")
        return InstallOutcome(
            ok=True,
            cancelled=False,
            stage=Stage.DONE,
            message="安装完成",
            journal=journal,
        )

    # ------------------------------------------------------------- 阶段调度
    def _should_skip(self, stage: Stage) -> bool:
        if stage is Stage.TORCH:
            return not self.options.install_torch
        if stage is Stage.YOLO:
            return not self.options.weights
        if stage is Stage.FFMPEG:
            return not self.options.install_ffmpeg
        return False

    def _skip_reason(self, stage: Stage) -> str:
        if stage is Stage.TORCH:
            return "已按用户选择跳过（复用本机已有的 PyTorch）"
        if stage is Stage.YOLO:
            return "未选择下载 YOLO 权重"
        if stage is Stage.FFMPEG:
            return "未选择随包安装 ffmpeg"
        return "已跳过"

    def _run_stage(self, stage: Stage) -> None:
        info = stage_info(stage)
        self._current_stage = stage
        self._check_stop()

        journal = self.journal
        assert journal is not None
        journal.mark_running(stage)
        journal.save()
        self.reporter.stage(stage, "running", info.title)
        self.reporter.progress(0.0, info.title)
        self._emit_total(stage, 0.0, info.title)
        self.reporter.log(f"—— {info.title} ——")

        handler = getattr(self, f"_stage_{stage.value}")
        try:
            result = handler()
        except InstallCancelled:
            raise
        except InstallError:
            raise
        except Exception as exc:  # noqa: BLE001 - 转成可读的安装错误
            raise InstallError(f"{info.title}失败：{exc}") from exc

        # 阶段处理函数返回 (说明, 明细)，或 (说明, 明细, 状态)；
        # 状态只用于"可选组件没装上"这类情况（记成 skipped 而不是 done）
        message, detail = result[0], result[1]
        status = result[2] if len(result) > 2 else "done"

        if status == "skipped":
            journal.mark_skipped(stage, message)
        else:
            journal.mark_done(stage, message, detail)
        journal.save()
        self.reporter.stage(stage, status, message)
        self.reporter.progress(1.0, message)
        self._emit_total(stage, 1.0, message)
        if message:
            self.reporter.log(f"{info.title}：{message}")

    def _emit_total(self, stage: Stage, fraction: float, text: str = "") -> None:
        value = weight_before(stage) + weight_span(stage) * max(0.0, min(fraction, 1.0))
        self.reporter.total(value, text)

    # ------------------------------------------------------------- 各阶段
    def _stage_prepare(self) -> tuple[str, str]:
        root = self.install_dir
        journal = self.journal
        assert journal is not None

        for directory in (
            root,
            paths.python_dir(root),
            paths.app_dir(root),
            paths.models_dir(root),
            paths.ffmpeg_bin_dir(root),
        ):
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise InstallError(f"无法创建目录 {directory}：{exc}") from exc

        # 目录可写探测：写状态文件本身就是最好的探针
        if not journal.save():
            raise InstallError(
                f"没有写入权限：{root}。请换一个目录，或用管理员身份重新运行安装程序。"
            )

        free = windows.free_space(root)
        needed = estimate_install_bytes(self.options)
        if free is not None and free < needed:
            self.reporter.log(
                f"警告：磁盘剩余 {paths.format_size(free)}，"
                f"本次安装预计需要约 {paths.format_size(needed)}。"
            )
            if free < MIN_FREE_SPACE_BYTES:
                raise InstallError(
                    f"磁盘空间不足：{paths.format_size(free)}（至少需要 "
                    f"{paths.format_size(MIN_FREE_SPACE_BYTES)}）。请清理磁盘或换一个盘安装。"
                )

        cached = self.cache_state()
        if cached.has_partials:
            self.reporter.log(
                f"发现上次的断点：{cached.partial_files} 个文件已下 "
                f"{paths.format_size(cached.partial_bytes)}，本次会接着下。"
            )
        if cached.ready_files:
            self.reporter.log(
                f"已有 {cached.ready_files} 个安装包在缓存里，本次只校验不重复下载。"
            )
        self.reporter.log("安装缓存目录：" + str(paths.cache_dir()))
        return f"安装目录 {root}", "已创建目录并写入安装状态"

    def _stage_runtime(self) -> tuple[str, str]:
        cache = paths.downloads_dir()
        item = runtime_download_item(cache, version=self.options.python_version)

        if runtime_present(paths.python_dir(self.install_dir), self.options.python_version):
            self.reporter.log(f"内嵌 Python 已存在：{paths.python_dir(self.install_dir)}")
        else:
            batch = self._download([item], Stage.RUNTIME, "下载内嵌 Python 运行时")
            self._require_success(batch, "下载内嵌 Python 运行时")
            try:
                extract_runtime(item.path, paths.python_dir(self.install_dir))
            except (OSError, zipfile.BadZipFile) as exc:
                # 压缩包坏了：删掉重下的机会留给下一次运行
                raise InstallError(f"解压内嵌 Python 失败（压缩包可能损坏）：{exc}") from exc

        write_pth(paths.python_dir(self.install_dir), self.options.python_version)
        python = paths.python_exe(self.install_dir)
        if not python.is_file():
            raise InstallError(f"内嵌解释器不存在：{python}")
        return f"Python {self.options.python_version}", f"解释器：{python}"

    def _stage_pip(self) -> tuple[str, str]:
        cache = paths.downloads_dir()
        item = get_pip_download_item(cache)
        batch = self._download([item], Stage.PIP, "下载 pip 引导脚本")
        self._require_success(batch, "下载 pip 引导脚本")

        python = str(paths.python_exe(self.install_dir))
        code, output = self._run_process(
            [python, str(item.path), "--no-warn-script-location", "--disable-pip-version-check"],
            stage=Stage.PIP,
            description="引导 pip",
        )
        if code != 0 and "pip" not in output.lower():
            raise InstallError(f"引导 pip 失败：{_tail(output)}")

        self._pip(
            ["install", "--upgrade", "pip", "setuptools", "wheel"],
            stage=Stage.PIP,
            description="升级 pip / setuptools / wheel",
        )
        return "pip 已就绪", "内嵌环境已可安装第三方包"

    def _stage_torch(self) -> tuple[str, str]:
        backend = backend_by_key(self.options.backend_key)
        self.reporter.log(f"PyTorch 构建：{backend.display}")
        self.reporter.log(f"官方索引：{backend.index_url}")

        tags = platform_tags(self.profile) if self.profile is not None else None
        # 版本跟随随包 requirements.txt：装出来的 PyTorch 与仓库锁定值一致
        pins = payload.torch_pins(self._payload_root)
        plan = self._plan_builder(
            backend,
            profile=self.profile,
            versions=pins or None,
            directory=paths.wheels_dir(),
            with_sizes=True,
            python_version=self.options.python_tuple,
            tags=tags,
        )
        self._plan = plan
        for note in plan.notes:
            self.reporter.log(note)
        for problem in plan.problems:
            self.reporter.log(f"提示：{problem}")
        if not plan.items:
            detail = "；".join(plan.problems) or "官方索引里没有匹配的 wheel"
            raise InstallError(f"无法确定要下载的 PyTorch 安装包：{detail}")

        batch = self._download(plan.items, Stage.TORCH, f"下载 PyTorch · {backend.label}")
        self._require_success(batch, "下载 PyTorch 安装包")

        files = [str(item.path) for item in plan.items if item.path.exists()]
        if not files:
            raise InstallError("PyTorch 安装包不存在，无法安装")
        self._pip(
            ["install", "--no-deps", "--upgrade", *files],
            stage=Stage.TORCH,
            description="安装 PyTorch",
        )
        version = self._installed_version("torch") or "未知版本"
        return f"PyTorch {version}（{backend.label}）", f"{len(files)} 个 wheel 已安装"

    def _stage_deps(self) -> tuple[str, str]:
        entries, source = payload.resolve_requirements(self._payload_root)
        if not entries:
            raise InstallError("没有可用的依赖清单（随包文件缺失）")
        # 写成临时 requirements 文件：避免命令行过长，也方便用户事后查看
        target = paths.cache_dir() / "requirements-install.txt"
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("\n".join(entries) + "\n", encoding="utf-8")
        except OSError as exc:
            raise InstallError(f"无法写入依赖清单：{exc}") from exc

        self.reporter.log(f"依赖来源：{source}，共 {len(entries)} 项（不含 PyTorch，避免覆盖）")
        self._pip(
            ["install", "--upgrade", "--requirement", str(target)],
            stage=Stage.DEPS,
            description="安装运行依赖",
        )
        return f"已安装 {len(entries)} 项依赖", f"清单：{target}"

    def _stage_app(self) -> tuple[str, str]:
        root = self._payload_root
        copied = payload.copy_static_files(root, self.install_dir)
        if copied:
            self.reporter.log(f"已复制 {len(copied)} 个随包文件（配置 / 文档 / 图标）")

        wheels = payload.bundled_wheels(root)
        if wheels:
            wheel = wheels[0]
            self.reporter.log(f"安装程序本体（wheel）：{wheel.name}")
            self._pip(
                ["install", "--no-deps", "--upgrade", str(wheel)],
                stage=Stage.APP,
                description="安装程序本体",
            )
            return f"程序本体 {wheel.name}", str(wheel)

        source = payload.source_root(root)
        if source is None:
            raise InstallError("随安装器分发的程序文件缺失（既没有 wheel 也没有源码目录）")

        self.reporter.log(f"安装程序本体（源码目录）：{source}")
        self._pip(
            ["install", "--no-deps", "--no-build-isolation", "--upgrade", str(source)],
            stage=Stage.APP,
            description="安装程序本体",
        )
        return "程序本体已安装（源码目录）", str(source)

    def _stage_yolo(self) -> tuple[str, str]:
        items = self._yolo_download_items()
        if not items:
            return "未选择权重", ""
        batch = self._download(items, Stage.YOLO, "下载 YOLO 权重")
        self._require_success(batch, "下载 YOLO 权重")
        names = "、".join(item.path.name for item in batch.items)
        return f"权重已就绪：{names}", f"目录：{paths.models_dir(self.install_dir)}"

    def _stage_ffmpeg(self) -> tuple[str, str] | tuple[str, str, str]:
        item = self._ffmpeg_item()
        batch = self._download([item], Stage.FFMPEG, "下载 ffmpeg")
        if batch.failed or not item.path.exists():
            # 可选组件：失败不阻断安装，只是少一项能力（记成"已跳过"，不算失败）
            reason = batch.failed[0].error if batch.failed else "文件不存在"
            self.reporter.log(f"ffmpeg 下载失败（可选组件，不影响使用）：{reason}")
            return "已跳过 ffmpeg（可选组件下载失败）", reason, "skipped"

        try:
            extracted = self._extract_ffmpeg(item.path)
        except (OSError, zipfile.BadZipFile) as exc:
            self.reporter.log(f"解压 ffmpeg 失败（可选组件）：{exc}")
            return "已跳过 ffmpeg（压缩包解压失败）", str(exc), "skipped"
        if not extracted:
            return "已跳过 ffmpeg（压缩包里没找到可执行文件）", "", "skipped"
        self.reporter.log("ffmpeg 已就绪：" + "、".join(path.name for path in extracted))
        return (
            f"ffmpeg 已就绪（{len(extracted)} 个可执行文件）",
            str(paths.ffmpeg_bin_dir(self.install_dir)),
        )

    def _stage_finalize(self) -> tuple[str, str]:
        root = self.install_dir
        scripts = windows.write_launcher_scripts(root, version=self.app_version)
        if scripts:
            self.reporter.log("已生成入口脚本：" + "、".join(path.name for path in scripts))
        self._write_uninstall_script()

        shortcuts = windows.create_shortcuts(
            root,
            desktop=self.options.desktop_shortcut,
            start_menu=self.options.start_menu_shortcut,
        )
        for path in shortcuts:
            self.reporter.log(f"已创建快捷方式：{path}")

        details: list[str] = []
        try:
            size = windows.directory_size(root)
            details.append(f"占用约 {paths.format_size(size)}")
        except OSError:  # pragma: no cover
            size = 0
        registered = windows.register_install(root, version=self.app_version, estimated_bytes=size)
        details.append("已登记卸载信息" if registered else "未登记卸载信息（非 Windows 或注册表不可写）")

        verified = self._verify_installation()
        if verified:
            details.append("自检通过：" + "、".join(verified))
        else:
            self.reporter.log("自检未确认模块齐全，但不影响安装结果")

        if self.options.delete_downloads:
            removed = self._cleanup_downloads()
            if removed:
                details.append(f"已清理下载缓存 {paths.format_size(removed)}")

        return "安装完成", "；".join(details)

    # ------------------------------------------------------------- 工具
    def _download(
        self,
        items: Sequence[DownloadItem],
        stage: Stage,
        title: str,
    ) -> DownloadBatchResult:
        """断点续传下载一批文件，并把进度折算进阶段 / 总进度。"""
        self._check_stop()
        if not items:
            return DownloadBatchResult()

        total_expected = sum(item.size or 0 for item in items)
        self.total_bytes = (self.total_bytes or 0) + total_expected
        received = 0
        span = weight_span(stage)
        count = len(items)

        def on_progress(event: DownloadProgress) -> None:
            nonlocal received
            fraction_in_file = event.fraction
            self.downloaded_bytes = self.downloaded_bytes - received + event.downloaded
            received = event.downloaded
            local = ((event.index - 1) + fraction_in_file) / max(count, 1)
            label = (
                f"{title}：{event.path.stem} "
                f"{event.downloaded / 1024 / 1024:.0f} MB"
                + (f" / {event.total / 1024 / 1024:.0f} MB" if event.total else "")
            )
            self.reporter.progress(local, label)
            self.reporter.total(weight_before(stage) + span * local, label)
            self.reporter.download(event)

        self.reporter.log(f"开始下载 {title}（{count} 个文件）")
        batch = self._downloader(
            list(items),
            progress=on_progress,
            stop=self.cancelled,
            resume=True,
        )
        if batch.cancelled:
            raise InstallCancelled()
        self._check_stop()
        return batch

    def _require_success(self, batch: DownloadBatchResult, title: str) -> None:
        if not batch.failed:
            for item in batch.items:
                if item.skipped:
                    self.reporter.log(f"已存在，跳过下载：{item.path.name}")
            return
        first = batch.failed[0]
        raise InstallError(f"{title}失败：{first.path.name}（{first.error}）")

    def _pip(self, args: Sequence[str], *, stage: Stage, description: str) -> None:
        python = str(paths.python_exe(self.install_dir))
        code, output = self._run_process(
            [python, "-m", "pip", *args], stage=stage, description=description
        )
        if code != 0:
            raise InstallError(f"{description}失败（pip 退出码 {code}）：{_tail(output)}")

    def _run_process(
        self,
        command: Sequence[str],
        *,
        stage: Stage,
        description: str,
    ) -> tuple[int, str]:
        """跑一次子进程：输出实时转给界面，进度条在阶段内做"伪进度"。"""
        self._check_stop()
        self.reporter.log("执行：" + " ".join(str(part) for part in command))
        span = weight_span(stage)
        lines = 0

        def on_line(text: str) -> None:
            nonlocal lines
            lines += 1
            self.reporter.log(text)
            # pip 不报百分比，用输出行数做一个"有在动"的进度（最多到 0.95）
            self.reporter.progress(min(lines / 400.0, 0.95), text[:80])
            self.reporter.total(
                weight_before(stage) + span * min(lines / 400.0, 0.95), f"{description}…"
            )

        code, output = self._process_runner(
            list(command),
            on_line=on_line,
            stop=self.cancelled,
            cwd=self.install_dir,
        )
        if code == -1 and self.cancelled():
            raise InstallCancelled()
        return code, output

    def _run_python(self, code: str) -> tuple[int, str]:
        """在内嵌环境里跑一小段 Python 代码（静默执行，不进界面日志）。"""
        python = paths.python_exe(self.install_dir)
        if not python.is_file():
            return 1, ""
        try:
            return self._process_runner(
                [str(python), "-c", code],
                on_line=None,
                stop=self.cancelled,
                cwd=self.install_dir,
            )
        except OSError as exc:  # pragma: no cover
            logger.debug("执行内嵌脚本失败：%s", exc)
            return 1, ""

    def _installed_version(self, package: str) -> str | None:
        """问内嵌环境：某个包装的是什么版本（用 metadata，不 import 它）。"""
        code, output = self._run_python(
            f"import importlib.metadata as m;print(m.version('{package}'))"
        )
        if code != 0:
            return None
        lines = [line.strip() for line in (output or "").splitlines() if line.strip()]
        return lines[-1] if lines else None

    def _verify_installation(self) -> list[str]:
        """安装完自检：内嵌环境里该有的模块是不是都在。"""
        script = (
            "import importlib.util as u;"
            "mods=[m for m in ('torch','ultralytics','cv2','yaml','PySide6')"
            " if u.find_spec(m) is None];"
            "print('MISSING:'+','.join(mods) if mods else 'OK')"
        )
        code, output = self._run_python(script)
        text = (output or "").strip()
        if code != 0:
            return []
        if text == "OK":
            return ["torch / ultralytics / cv2 / PySide6 均可导入"]
        if "MISSING:" in text:
            missing = text.split("MISSING:", 1)[-1].strip()
            self.reporter.log(f"自检：以下模块未能导入 → {missing}（程序仍可启动，缺哪个补哪个）")
            return []
        return []

    def _yolo_download_items(self) -> list[DownloadItem]:
        if not self.options.weights:
            return []
        target = paths.models_dir(self.install_dir)
        wanted = {name for name in self.options.weights}
        return [item for item in yolo_items(target) if item.path.name in wanted]

    def _ffmpeg_item(self) -> DownloadItem:
        return DownloadItem(
            url=FFMPEG_URL,
            path=paths.downloads_dir() / "ffmpeg-release-essentials.zip",
            label="ffmpeg（可选）",
            size=FFMPEG_APPROX_BYTES,
        )

    def _extract_ffmpeg(self, archive: Path) -> list[Path]:
        target = paths.ffmpeg_bin_dir(self.install_dir)
        target.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        with zipfile.ZipFile(archive) as bundle:
            for info in bundle.infolist():
                name = info.filename.replace("\\", "/")
                for wanted in FFMPEG_MEMBERS:
                    if name.endswith("/" + wanted) or name == wanted:
                        destination = target / Path(wanted).name
                        with bundle.open(info) as source, open(destination, "wb") as handle:
                            shutil.copyfileobj(source, handle)
                        written.append(destination)
        return written

    def _write_uninstall_script(self) -> Path | None:
        """写一个给"开始菜单 → 卸载"用的入口脚本。"""
        root = self.install_dir
        pythonw = paths.pythonw_exe(root)
        script = root / "Uninstall.cmd"
        body = (
            "@echo off\r\n"
            "rem 由 Video2PersonVideo 安装器生成：调用内嵌环境里的卸载程序（图形界面）。\r\n"
            f'"{pythonw}" -m video2personvideo.installer --uninstall --install-dir "{root}"\r\n'
            "if errorlevel 1 pause\r\n"
        )
        try:
            script.write_text(body, encoding="utf-8", newline="")
            return script
        except OSError as exc:  # pragma: no cover
            logger.warning("写入卸载脚本失败：%s", exc)
            return None

    def _cleanup_downloads(self) -> int:
        """删掉下载缓存，返回释放的字节数（默认不删，保留下来方便重装）。"""
        freed = 0
        for root in (paths.downloads_dir(), paths.wheels_dir()):
            try:
                freed += windows.directory_size(root)
                shutil.rmtree(root, ignore_errors=True)
            except OSError as exc:  # pragma: no cover
                logger.debug("清理 %s 失败：%s", root, exc)
        return freed


def _tail(text: str, *, limit: int = 12) -> str:
    """取输出的最后几行（错误提示里不能糊一屏 pip 日志）。"""
    lines = [line for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return "（无输出）"
    return "\n".join(lines[-limit:])


# --------------------------------------------------------------------- 回滚 / 卸载
def rollback_install(
    install_dir: str | Path,
    *,
    remove_cache: bool = False,
    reporter: Reporter | None = None,
    schedule_delete: bool = True,
) -> RollbackReport:
    """把"已经装上去的东西"清干净：快捷方式 → 注册表 → 安装目录（→ 缓存）。

    * 安装目录里可能有**正在运行的解释器**（卸载器自己就跑在里面），
      Windows 不允许删除正在使用的 exe，所以删除失败时会安排一个后台脚本稍后清理；
    * ``remove_cache=False``（默认）保留已经下好的 wheel：
      用户重装 / 换目录安装时不用再下几个 GB。
    """
    report = RollbackReport()
    report.kept_cache = not remove_cache
    report.cache_path = paths.cache_dir()

    root = Path(install_dir)
    reporter = reporter or Reporter()

    notes: list[str] = []

    try:
        removed_links = windows.remove_shortcuts()
        for path in removed_links:
            reporter.log(f"已删除快捷方式：{path}")
            report.removed_paths.append(path)
        if removed_links:
            notes.append(f"已删除 {len(removed_links)} 个快捷方式")
    except Exception as exc:  # noqa: BLE001 - 回滚要尽量做完
        logger.warning("删除快捷方式失败：%s", exc)

    try:
        if windows.unregister_install():
            reporter.log("已清除注册表卸载信息")
            notes.append("已清除卸载登记")
    except Exception as exc:  # noqa: BLE001
        logger.warning("清除注册表失败：%s", exc)

    if root.exists():
        try:
            shutil.rmtree(root)
            report.removed_paths.append(root)
            reporter.log(f"已删除安装目录：{root}")
            notes.append("已删除安装目录")
        except OSError as exc:
            logger.info("安装目录暂时无法删除（%s），改为退出后清理", exc)
            if schedule_delete:
                script = windows.schedule_directory_removal(root)
                if script is not None:
                    report.scheduled.append(root)
                    reporter.log("安装目录将在安装器退出后自动清理。")
                    notes.append("安装目录被占用，将在退出后自动删除")
            else:
                notes.append(f"部分文件未能删除：{exc}")

    if remove_cache:
        try:
            freed = windows.directory_size(report.cache_path)
            shutil.rmtree(report.cache_path, ignore_errors=True)
            reporter.log(f"已删除下载缓存：{report.cache_path}")
            notes.append(f"已同时删除下载缓存（{paths.format_size(freed)}）")
        except OSError as exc:  # pragma: no cover
            logger.warning("删除缓存失败：%s", exc)
    elif report.cache_path is not None and report.cache_path.exists():
        notes.append("已下载的安装包保留在缓存目录，重装可直接复用")

    # 状态镜像也要抹掉，否则下次启动会以为"还有未完成的安装"
    try:
        paths.mirrored_journal_path().unlink(missing_ok=True)
    except OSError:  # pragma: no cover
        pass

    report.message = "；".join(notes) + "。" if notes else "没有需要清理的内容。"
    return report


def uninstall(
    install_dir: str | Path,
    *,
    remove_cache: bool = True,
    reporter: Reporter | None = None,
) -> RollbackReport:
    """卸载：默认连下载缓存一起删（用户主动卸载，不再需要那几个 GB）。"""
    return rollback_install(
        install_dir, remove_cache=remove_cache, reporter=reporter, schedule_delete=True
    )


def mark_uninstalled(install_dir: str | Path) -> None:
    """在状态镜像里留下"已卸载"的痕迹（安装目录马上要被删掉，写不进主副本）。"""
    journal = InstallJournal.load_for(install_dir)
    if journal is None:
        return
    journal.mark_rolled_back("用户卸载")
    journal.save()


__all__ = [
    "MIN_FREE_SPACE_BYTES",
    "TORCH_APPROX_BYTES",
    "InstallCancelled",
    "InstallEngine",
    "InstallError",
    "InstallOutcome",
    "Reporter",
    "RollbackReport",
    "backend_approx_bytes",
    "describe_exit",
    "estimate_download_bytes",
    "estimate_install_bytes",
    "mark_uninstalled",
    "rollback_install",
    "uninstall",
]
