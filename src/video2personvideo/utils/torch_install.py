"""把「自检结论」变成「具体下哪些文件、怎么装」。

PyTorch 官方每个构建都有独立的 simple index（``/whl/cu126/torch/`` 这样的页面），
页面里列着该构建下全部 wheel 的下载地址与 sha256。本模块负责：

1. 抓取并解析索引页（纯正则，不依赖 bs4）；
2. 按「Python 版本 + 操作系统 + 目标版本」挑出唯一那个 wheel；
3. 组成 :class:`InstallPlan`（含断点续传需要的 sha256 与体积）；
4. 可选地把下载好的 wheel 用 ``pip --no-deps`` 装进当前环境，再补齐其余依赖。

版本优先跟随 ``requirements.txt`` 里的锁定值（保证装完与仓库一致），
读不到锁定值时才取索引里的最新版。
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from . import app_paths
from .downloader import DownloadItem, remote_size
from .gpu_probe import HardwareProfile, probe_hardware
from .logger import get_logger
from .torch_backends import TORCH_PACKAGES, TorchBackend

logger = get_logger(__name__)

#: PyTorch CPU wheel 也会用到 ``+cpu`` 这样的本地标签
_LOCAL_TAG = re.compile(r"\+(?P<tag>[A-Za-z0-9_.]+)")

#: 索引页里的 ``<a href="URL#sha256=HASH">文件名</a>``
_INDEX_LINK = re.compile(
    r'<a[^>]+href="(?P<url>[^"#]+)(?:#sha256=(?P<sha>[0-9a-fA-F]+))?"[^>]*>(?P<name>[^<]+)</a>'
)

#: ultralytics 官方权重（YOLO 模型）
YOLO_ASSETS_BASE = "https://github.com/ultralytics/assets/releases/download/v8.3.0"
#: 文件名 → （说明, 大致体积）。体积写在这里是为了让界面不必联网 HEAD 就能显示"还要下多少"
YOLO_WEIGHTS: dict[str, tuple[str, int]] = {
    "yolo11n.pt": ("检测模型（默认）", 5_400_000),
    "yolo11n-pose.pt": ("姿态模型（半身构图更准）", 6_100_000),
}

#: 各操作系统的 wheel 平台标签（**越靠前越优先**）
PLATFORM_TAGS: dict[str, tuple[str, ...]] = {
    "Windows": ("win_amd64", "win_arm64"),
    "Linux": (
        "manylinux_2_28_x86_64",
        "manylinux_2_27_x86_64",
        "manylinux_2_24_x86_64",
        "manylinux2014_x86_64",
        "manylinux1_x86_64",
        "linux_x86_64",
        "manylinux_2_28_aarch64",
        "manylinux2014_aarch64",
    ),
    "Darwin": (
        "macosx_11_0_arm64",
        "macosx_14_0_arm64",
        "macosx_10_9_x86_64",
        "macosx_10_13_x86_64",
        "macosx_10_9_universal2",
    ),
}

#: torch 与 torchvision 的小版本对应关系：``torchvision_minor = torch_minor + 15``
TORCHVISION_MINOR_OFFSET = 15


# ------------------------------------------------------------------ 数据模型
@dataclass(frozen=True, slots=True)
class WheelEntry:
    """索引页里的一个 wheel。"""

    filename: str
    url: str
    sha256: str | None = None

    @property
    def version(self) -> str:
        """文件名里的完整版本号（含 ``+cu126`` 这样的本地标签）。"""
        match = re.match(r"^[^-]+-(?P<version>[^-]+)-", self.filename)
        return match.group("version") if match else ""

    @property
    def base_version(self) -> str:
        return _LOCAL_TAG.sub("", self.version)


@dataclass(slots=True)
class InstallPlan:
    """一次「该下什么」的完整计划。"""

    backend: TorchBackend
    items: list[DownloadItem] = field(default_factory=list)
    #: 计划里缺的东西（索引里没找到对应 wheel 等）
    problems: list[str] = field(default_factory=list)
    #: 说明文字（版本来源、体积等）
    notes: list[str] = field(default_factory=list)
    #: 目标目录（wheel 落地位置）
    directory: Path | None = None

    @property
    def total_bytes(self) -> int | None:
        sizes = [item.size for item in self.items]
        if not sizes or any(size is None for size in sizes):
            return None
        return sum(size for size in sizes if size)

    @property
    def ok(self) -> bool:
        return bool(self.items) and not self.problems

    def pip_command(self, python: str | None = None) -> str:
        files = " ".join(f'"{item.path}"' for item in self.items)
        interpreter = python or sys.executable
        return f'"{interpreter}" -m pip install --no-deps --upgrade {files}'


# ------------------------------------------------------------------ 索引解析
def parse_index(html: str) -> list[WheelEntry]:
    """解析 simple index 页面，返回全部 wheel。"""
    entries: list[WheelEntry] = []
    for match in _INDEX_LINK.finditer(html or ""):
        name = (match.group("name") or "").strip()
        url = (match.group("url") or "").strip()
        if not name.endswith(".whl") or not url:
            continue
        entries.append(
            WheelEntry(filename=name, url=url, sha256=(match.group("sha") or None))
        )
    return entries


def fetch_index(
    index_url: str,
    package: str,
    *,
    timeout: float = 30.0,
) -> tuple[list[WheelEntry], str | None]:
    """抓取某个包的索引页；失败时返回 ``([], 错误说明)``。"""
    import urllib.error
    import urllib.request

    url = f"{index_url.rstrip('/')}/{package}/"
    request = urllib.request.Request(  # noqa: S310 - 地址来自官方索引
        url, headers={"User-Agent": "Video2PersonVideo/1.0 (torch-plan)"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            html = response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, OSError) as exc:
        logger.warning("读取索引失败 %s：%s", url, exc)
        return [], f"无法访问 {url}（{exc}）"
    return parse_index(html), None


# ------------------------------------------------------------------ 选择 wheel
def python_tag(python_version: tuple[int, int] | None = None) -> str:
    """当前解释器的 wheel 标签，如 ``cp311``。"""
    major, minor = python_version or (sys.version_info.major, sys.version_info.minor)
    return f"cp{major}{minor}"


def platform_tags(profile: HardwareProfile | None = None) -> tuple[str, ...]:
    """当前机器可接受的平台标签（按优先级排序）。"""
    profile = profile or probe_hardware()
    tags = list(PLATFORM_TAGS.get(profile.os_name, ()))
    machine = (profile.arch or platform.machine()).lower()
    if "arm" not in machine:  # 只保留与当前架构匹配的标签
        tags = [tag for tag in tags if "aarch64" not in tag and "arm64" not in tag]
    else:
        tags = [tag for tag in tags if tag in {"win_arm64"} or "aarch64" in tag or "arm64" in tag]
    if not tags:  # 未知平台：放宽到不限制
        return ()
    return tuple(tags)


def _version_key(version: str) -> tuple[int, ...]:
    parts = re.findall(r"\d+", version)
    return tuple(int(part) for part in parts[:4]) if parts else (0,)


def filter_entries(
    entries: Iterable[WheelEntry],
    *,
    tag: str,
    tags: Sequence[str],
) -> list[WheelEntry]:
    """按 Python 标签 + 平台标签筛出可用 wheel（已排除 free-threaded 的 ``cp313t``）。"""
    result: list[WheelEntry] = []
    for entry in entries:
        name = entry.filename
        if tag not in name:
            continue
        if f"{tag}t" in name:  # 3.13 自由线程构建，普通解释器装不上
            continue
        if tags and not any(name.endswith(f"-{item}.whl") for item in tags):
            continue
        result.append(entry)
    return result


def pick_version(entries: Sequence[WheelEntry], version: str | None = None) -> WheelEntry | None:
    """在候选里挑版本：给了 ``version`` 就精确匹配基础版本，否则取最新。"""
    if not entries:
        return None
    if version:
        for entry in entries:
            if entry.base_version == version:
                return entry
    ordered = sorted(entries, key=lambda item: _version_key(item.base_version))
    return ordered[-1] if ordered else None


def _pick_torchvision(
    entries: Sequence[WheelEntry],
    torch_version: str,
    pinned: str | None,
) -> WheelEntry | None:
    """给 torch 版本配上对应的 torchvision（找不齐时退到最近的旧版本）。"""
    if pinned:
        found = pick_version(entries, pinned)
        if found is not None:
            return found
    torch_parts = _version_key(torch_version)
    expected_minor = (torch_parts[1] if len(torch_parts) > 1 else 0) + TORCHVISION_MINOR_OFFSET

    candidates = sorted(entries, key=lambda item: _version_key(item.base_version))
    exact = [
        entry
        for entry in candidates
        if _version_key(entry.base_version)[:2] == (0, expected_minor)
    ]
    if exact:
        return exact[-1]

    older = [
        entry
        for entry in candidates
        if _version_key(entry.base_version)[:2] < (0, expected_minor)
    ]
    if older:
        return older[-1]
    return candidates[-1] if candidates else None


# ------------------------------------------------------------------ 版本锁定
def pinned_versions(requirements: Path | None = None) -> dict[str, str]:
    """从 ``requirements.txt`` 读取 torch / torchvision 的锁定版本。"""
    path = requirements or default_requirements_path()
    if path is None or not path.exists():
        return {}
    pins: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        text = line.split("#", 1)[0].strip()
        if not text:
            continue
        match = re.match(r"^(?P<name>[A-Za-z0-9_.\-]+)\s*==\s*(?P<version>[0-9][^\s;]*)", text)
        if not match:
            continue
        name = match.group("name").lower().replace("_", "-")
        if name in TORCH_PACKAGES:
            pins[name] = match.group("version")
    return pins


def default_requirements_path() -> Path | None:
    """定位 ``requirements.txt``（打包 / 安装器形态下可能不存在）。

    顺序：当前工作目录 → 安装根 / 仓库根（:func:`app_paths.app_root`）。
    """
    candidates = [
        Path.cwd() / "requirements.txt",
        app_paths.app_root() / "requirements.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def wheel_directory(root: Path | None = None) -> Path:
    """wheel 缓存目录（断点文件也在这里，重开程序可以接着下）。"""
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
        return root
    return app_paths.wheels_cache_dir()


def model_directory(root: Path | None = None) -> Path:
    """YOLO 权重目录：安装形态在安装根下，源码形态在仓库里，否则退回用户目录。"""
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
        return root
    return app_paths.models_dir(create=True)


# ------------------------------------------------------------------ 计划
def build_plan(
    backend: TorchBackend,
    *,
    profile: HardwareProfile | None = None,
    versions: dict[str, str] | None = None,
    with_sizes: bool = True,
    directory: Path | None = None,
    timeout: float = 30.0,
    python_version: tuple[int, int] | None = None,
    tags: Sequence[str] | None = None,
) -> InstallPlan:
    """为某个 PyTorch 构建生成下载计划（联网抓索引，但只读页面不下载大文件）。

    ``python_version`` 与 ``tags`` 用来指定**目标解释器**而不是当前解释器：
    安装器要把 wheel 装进随包的内嵌 Python（版本可能与安装器自己不同），
    所以必须能把「给谁下」和「谁来下」分开。
    """
    profile = profile or probe_hardware()
    pins = versions if versions is not None else pinned_versions()
    target_dir = directory or wheel_directory()

    plan = InstallPlan(backend=backend, directory=target_dir)
    tag = python_tag(python_version)
    tags = tuple(tags) if tags is not None else platform_tags(profile)

    html_by_package: dict[str, list[WheelEntry]] = {}
    for package in TORCH_PACKAGES:
        entries, error = fetch_index(backend.index_url, package, timeout=timeout)
        if error:
            plan.problems.append(error)
        html_by_package[package] = entries

    torch_entries = filter_entries(html_by_package.get("torch", []), tag=tag, tags=tags)
    torch_entry = pick_version(torch_entries, pins.get("torch"))
    if torch_entry is None:
        plan.problems.append(
            f"{backend.display} 里找不到适配 Python {sys.version_info.major}."
            f"{sys.version_info.minor} / {profile.os_name} 的 torch wheel"
        )
        return plan

    vision_entries = filter_entries(html_by_package.get("torchvision", []), tag=tag, tags=tags)
    vision_entry = _pick_torchvision(
        vision_entries, torch_entry.base_version, pins.get("torchvision")
    )
    if vision_entry is None:
        plan.problems.append("找不到与所选 torch 配套的 torchvision wheel")

    selected = [entry for entry in (torch_entry, vision_entry) if entry is not None]
    for entry in selected:
        size = remote_size(entry.url, timeout=timeout) if with_sizes else None
        plan.items.append(
            DownloadItem(
                url=entry.url,
                path=target_dir / entry.filename,
                sha256=entry.sha256,
                label=f"{entry.filename.split('-')[0]} {entry.base_version}",
                size=size,
            )
        )

    source = "requirements.txt 锁定版本" if pins else "官方索引最新版本"
    plan.notes.append(f"来源：{backend.index_url}（{source}）")
    plan.notes.append(f"共 {len(plan.items)} 个文件，支持断点续传，中断后重开可继续。")
    logger.info(
        "安装计划：%s → %s", backend.key, [item.path.name for item in plan.items]
    )
    return plan


def yolo_items(directory: Path | None = None, with_sizes: bool = False) -> list[DownloadItem]:
    """YOLO 权重下载项（可选步骤，体积只有几 MB）。

    体积默认用官方发布页的近似值（够界面算"还要下多少"），
    ``with_sizes=True`` 时才真的去 HEAD 一次拿准确值。
    """
    target = directory or model_directory()
    items: list[DownloadItem] = []
    for name, (label, approx) in YOLO_WEIGHTS.items():
        url = f"{YOLO_ASSETS_BASE}/{name}"
        size = remote_size(url) if with_sizes else approx
        items.append(DownloadItem(url=url, path=target / name, label=label, size=size))
    return items


# ------------------------------------------------------------------ 安装
def run_pip(
    args: Sequence[str],
    *,
    python: str | None = None,
    timeout: float = 1800.0,
    on_output=None,
) -> tuple[int, str]:
    """执行 pip，返回 ``(退出码, 输出)``；输出会逐行回调给界面。"""
    interpreter = python or sys.executable
    command = [interpreter, "-m", "pip", *args]
    logger.info("执行：%s", " ".join(command))
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
    try:
        completed = subprocess.run(  # noqa: S603 - 参数由本模块拼接，无 shell
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
            creationflags=flags,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return 1, f"无法执行 pip：{exc}"

    output = f"{completed.stdout or ''}{completed.stderr or ''}".strip()
    if on_output is not None:
        for line in output.splitlines():
            on_output(line)
    return completed.returncode, output


def pip_env() -> dict[str, str]:
    """pip 子进程环境：静音版本检查 / 交互提示，输出不缓冲便于实时显示。"""
    env = dict(os.environ)
    env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    env.setdefault("PIP_NO_INPUT", "1")
    env.setdefault("PIP_PROGRESS_BAR", "off")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONUTF8", "1")
    return env


def run_process_streaming(
    command: Sequence[str],
    *,
    on_line=None,
    stop=None,
    timeout: float | None = None,
    env: dict[str, str] | None = None,
    cwd: str | Path | None = None,
) -> tuple[int, str]:
    """执行子进程，**逐行**把输出交给调用方；``stop()`` 为真时终止它。

    返回 ``(退出码, 全部输出)``；被中止时退出码为 ``-1``
    （调用方据此区分"失败"与"用户暂停"）。安装器用它跑 pip 与内嵌解释器脚本。
    """
    import queue
    import threading

    logger.info("执行：%s", " ".join(str(item) for item in command))
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
    try:
        process = subprocess.Popen(  # noqa: S603 - 参数由调用方拼好，无 shell
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env if env is not None else pip_env(),
            cwd=str(cwd) if cwd is not None else None,
            creationflags=flags,
        )
    except OSError as exc:
        return 1, f"无法执行命令：{exc}"

    stream = process.stdout
    lines: "queue.Queue[str | None]" = queue.Queue()
    if stream is not None:
        def _pump() -> None:
            try:
                for line in stream:
                    lines.put(line)
            finally:
                lines.put(None)

        threading.Thread(target=_pump, daemon=True, name="process-output").start()

    started = time.monotonic()
    collected: list[str] = []
    cancelled = False
    drained = False
    while not drained:
        try:
            line = lines.get(timeout=0.2)
        except queue.Empty:
            line = ""
        if line is None:
            drained = True
            break
        if line:
            text = line.rstrip()
            collected.append(text)
            if on_line is not None:
                try:
                    on_line(text)
                except Exception as exc:  # noqa: BLE001 - 界面回调出错不该影响安装
                    logger.debug("子进程输出回调异常：%s", exc)
        if stop is not None and stop():
            cancelled = True
            logger.info("收到中止请求，正在结束 pip…")
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:  # pragma: no cover - 依赖真实进程
                process.kill()
            break
        if timeout is not None and time.monotonic() - started > timeout:  # pragma: no cover
            logger.warning("pip 执行超时（%.0f 秒），强制结束", timeout)
            process.terminate()
            break

    try:
        code = process.wait(timeout=30)
    except subprocess.TimeoutExpired:  # pragma: no cover
        process.kill()
        code = -1
    output = "\n".join(collected)
    if cancelled:
        return -1, output
    return code, output


def run_pip_streaming(
    args: Sequence[str],
    *,
    python: str | None = None,
    on_line=None,
    stop=None,
    timeout: float | None = None,
    env: dict[str, str] | None = None,
    cwd: str | Path | None = None,
) -> tuple[int, str]:
    """执行 ``python -m pip <args>``，逐行回显且可中止（见 :func:`run_process_streaming`）。"""
    interpreter = python or sys.executable
    return run_process_streaming(
        [interpreter, "-m", "pip", *args],
        on_line=on_line,
        stop=stop,
        timeout=timeout,
        env=env,
        cwd=cwd,
    )


def install_plan(
    plan: InstallPlan,
    *,
    python: str | None = None,
    on_output=None,
    install_requirements: bool = True,
) -> tuple[bool, str]:
    """安装计划里的 wheel，并按需补齐其余依赖。"""
    files = [str(item.path) for item in plan.items if item.path.exists()]
    if not files:
        return False, "没有可安装的 wheel 文件"

    code, output = run_pip(
        ["install", "--no-deps", "--upgrade", *files], python=python, on_output=on_output
    )
    if code != 0:
        return False, output

    requirements = default_requirements_path() if install_requirements else None
    if requirements is not None:
        code, more = run_pip(
            ["install", "--upgrade", "-r", str(requirements)], python=python, on_output=on_output
        )
        output = f"{output}\n{more}".strip()
        if code != 0:
            return False, output
    return True, output


def is_frozen() -> bool:
    """是否运行在 PyInstaller 打包出来的 exe 里（此时不能 pip 装进自身）。"""
    return bool(getattr(sys, "frozen", False))


def is_virtual_env() -> bool:
    """当前是否在虚拟环境里（打包环境一律视为不是）。"""
    if is_frozen():
        return False
    if sys.prefix != sys.base_prefix:
        return True
    return bool(os.environ.get("VIRTUAL_ENV") or os.environ.get("CONDA_PREFIX"))


__all__ = [
    "YOLO_ASSETS_BASE",
    "YOLO_WEIGHTS",
    "InstallPlan",
    "WheelEntry",
    "build_plan",
    "default_requirements_path",
    "fetch_index",
    "filter_entries",
    "install_plan",
    "is_frozen",
    "is_virtual_env",
    "model_directory",
    "parse_index",
    "pick_version",
    "pinned_versions",
    "pip_env",
    "platform_tags",
    "python_tag",
    "run_pip",
    "run_pip_streaming",
    "run_process_streaming",
    "wheel_directory",
    "yolo_items",
]
