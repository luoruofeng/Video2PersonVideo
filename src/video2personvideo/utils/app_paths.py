"""安装布局与随包资源定位。

同一份代码会在三种形态下运行，本模块负责让它们都能找到自己的东西：

1. **源码运行**（开发）：资源在仓库里（``configs/`` / ``assets/models/``）；
2. **PyInstaller 打包**（免安装 exe）：资源在 ``sys._MEIPASS`` 与 exe 同级目录；
3. **安装器装出来的环境**（推荐的分发方式）：安装根目录下放着
   ``python/``（内嵌解释器）、``app/``（程序本体）、``assets/models/``（权重）、
   ``ffmpeg/bin/``（随包 ffmpeg）与 ``install.json``（安装状态，本模块用它认目录）。

认安装根目录靠 ``install.json``（由 ``video2personvideo.installer`` 写入）这个标记文件，
因此 ``sys.prefix`` 是内嵌解释器的 ``python/`` 时，上一级就是安装根。

另外提供 :func:`register_bundled_tools`：把随包的可执行目录挂到 ``PATH``
（以及 Windows 的 DLL 搜索路径）上，这样 ``shutil.which("ffmpeg")`` 与
子进程调用都能直接生效，不必再去改每个调用点。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from .logger import get_logger

logger = get_logger(__name__)

#: 安装根目录的标记文件（由安装器写入，内容即安装状态日志）
INSTALL_MARKER = "install.json"

#: 环境变量：显式指定安装根 / 应用资源根（便携版或调试时用）
HOME_ENV_VAR = "V2PV_HOME"

#: 环境变量：显式指定 ffmpeg 可执行文件
FFMPEG_ENV_VAR = "V2PV_FFMPEG"

#: 随包可执行文件所在的相对目录（安装器把 ffmpeg 解压到 ``<安装根>/ffmpeg/bin``）
BUNDLED_BIN_DIRS = (
    Path("ffmpeg") / "bin",
    Path("bin"),
    Path("tools") / "bin",
)


def _is_install_root(candidate: Path) -> bool:
    """某个目录是不是"安装器装出来的安装根"。"""
    try:
        return candidate.is_dir() and (candidate / INSTALL_MARKER).is_file()
    except OSError:  # pragma: no cover - 依赖真实文件系统
        return False


def install_root() -> Path | None:
    """当前运行的安装根目录；不是"安装器装出来的环境"时返回 ``None``。

    查找顺序：``V2PV_HOME`` → 解释器所在目录的上一级（内嵌 Python 形态）→
    PyInstaller 运行时目录 → 从本文件往上找带标记的目录。
    """
    candidates: list[Path] = []

    override = os.environ.get(HOME_ENV_VAR)
    if override:
        candidates.append(Path(override))

    candidates.append(Path(sys.prefix).parent)
    candidates.append(Path(sys.executable).parent)
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            candidates.append(Path(meipass))

    here = Path(__file__).resolve()
    candidates.extend(here.parents)

    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:  # pragma: no cover
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        if _is_install_root(resolved):
            return resolved
    return None


def repo_root() -> Path | None:
    """源码运行时的仓库根目录（含 ``pyproject.toml`` 的那一级）。"""
    for parent in Path(__file__).resolve().parents:
        try:
            if (parent / "pyproject.toml").is_file():
                return parent
        except OSError:  # pragma: no cover
            continue
    return None


def app_root() -> Path:
    """应用资源根目录：安装根 → 仓库根 → 当前工作目录。"""
    root = install_root()
    if root is not None:
        return root
    repo = repo_root()
    if repo is not None:
        return repo
    return Path.cwd()


def app_payload_dir() -> Path | None:
    """程序本体所在目录（安装形态下是 ``<安装根>/app``）。"""
    root = install_root()
    if root is None:
        return repo_root()
    for name in ("app", ""):
        candidate = root / name if name else root
        if (candidate / "video2personvideo").is_dir() or (candidate / "src").is_dir():
            return candidate
    return root


def models_dir(*, create: bool = False) -> Path:
    """YOLO 权重目录。

    安装形态用 ``<安装根>/assets/models``（卸载时随安装目录一起清掉，
    也不会污染用户的当前目录）；源码形态沿用仓库的 ``assets/models``；
    两处都写不进去时退回 ``~/.video2personvideo/models``。
    """
    candidates = [app_root() / "assets" / "models", Path.cwd() / "assets" / "models"]
    for candidate in candidates:
        if not create:
            if candidate.is_dir():
                return candidate
            continue
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        except OSError as exc:  # pragma: no cover - 只读目录
            logger.debug("无法创建权重目录 %s：%s", candidate, exc)

    fallback = Path.home() / ".video2personvideo" / "models"
    if create:
        fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def user_state_dir() -> Path:
    """每个用户一份的程序状态目录（``%LOCALAPPDATA%/Video2PersonVideo``）。

    日志、安装状态镜像、下载缓存都放这里 —— 放在安装目录外，
    卸载时可以选择"保留安装包"，重装 / 换目录安装时也能直接复用已经下好的 wheel。
    """
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        root = Path(base) if base else Path.home() / "AppData" / "Local"
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support"
    else:
        base = os.environ.get("XDG_STATE_HOME")
        root = Path(base) if base else Path.home() / ".local" / "state"
    return root / "Video2PersonVideo"


def cache_dir() -> Path:
    """下载缓存根目录。"""
    base = user_state_dir() / "cache"
    base.mkdir(parents=True, exist_ok=True)
    return base


def wheels_cache_dir() -> Path:
    """PyTorch wheel 缓存目录（断点文件也在这里，重开安装器可以接着下）。"""
    base = cache_dir() / "wheels"
    base.mkdir(parents=True, exist_ok=True)
    return base


def downloads_cache_dir() -> Path:
    """除 wheel 之外的下载缓存（内嵌解释器、get-pip、ffmpeg 压缩包）。"""
    base = cache_dir() / "downloads"
    base.mkdir(parents=True, exist_ok=True)
    return base


def bundled_bin_dirs() -> list[Path]:
    """随包可执行文件目录（已存在的才算，按优先级排序）。"""
    roots: list[Path] = []
    root = install_root()
    if root is not None:
        roots.append(root)
    extra = os.environ.get("V2PV_TOOL_DIRS")
    if extra:
        roots.extend(Path(item) for item in extra.split(os.pathsep) if item.strip())
    if getattr(sys, "frozen", False):
        roots.append(Path(sys.executable).parent)
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            roots.append(Path(meipass))
    repo = repo_root()
    if repo is not None:
        roots.append(repo)

    found: list[Path] = []
    seen: set[Path] = set()
    for base in roots:
        for relative in BUNDLED_BIN_DIRS:
            candidate = base / relative
            try:
                resolved = candidate.resolve()
                if resolved in seen or not candidate.is_dir():
                    continue
            except OSError:  # pragma: no cover
                continue
            seen.add(resolved)
            found.append(candidate)
    return found


def register_bundled_tools() -> list[str]:
    """把随包可执行目录挂到 ``PATH`` / DLL 搜索路径上，返回本次新增的目录。

    幂等：重复调用不会把同一个目录重复塞进 ``PATH``。
    """
    added: list[str] = []
    current = os.environ.get("PATH", "")
    parts = current.split(os.pathsep) if current else []
    lowered = {item.rstrip("\\/").lower() for item in parts if item}

    for directory in bundled_bin_dirs():
        text = str(directory)
        if text.rstrip("\\/").lower() in lowered:
            continue
        parts.insert(0, text)
        lowered.add(text.rstrip("\\/").lower())
        added.append(text)
        if sys.platform == "win32":
            try:  # 让随包 DLL（如 ffmpeg 的依赖）也能被找到
                os.add_dll_directory(text)  # type: ignore[attr-defined]
            except (OSError, AttributeError) as exc:  # pragma: no cover
                logger.debug("添加 DLL 搜索目录失败 %s：%s", text, exc)

    if added:
        os.environ["PATH"] = os.pathsep.join(parts)
        logger.debug("已注册随包工具目录：%s", added)
    return added


def find_bundled_executable(*names: str) -> str | None:
    """在随包目录里找可执行文件（找不到返回 ``None``）。"""
    for directory in bundled_bin_dirs():
        for name in names:
            candidate = directory / name
            try:
                if candidate.is_file():
                    return str(candidate)
            except OSError:  # pragma: no cover
                continue
    return None


__all__ = [
    "BUNDLED_BIN_DIRS",
    "FFMPEG_ENV_VAR",
    "HOME_ENV_VAR",
    "INSTALL_MARKER",
    "app_payload_dir",
    "app_root",
    "bundled_bin_dirs",
    "cache_dir",
    "downloads_cache_dir",
    "find_bundled_executable",
    "install_root",
    "models_dir",
    "register_bundled_tools",
    "repo_root",
    "user_state_dir",
    "wheels_cache_dir",
]
