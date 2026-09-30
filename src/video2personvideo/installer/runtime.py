"""内嵌 Python 运行时：下载官方 embeddable 包、解压、补 ``._pth``。

安装器**不依赖目标机器上有没有 Python**：它自己带一份独立的解释器
（Python 官网的 "embeddable package"，约 10 MB 的 zip），解压到
``<安装目录>/python`` 后就是一套完全独立的运行环境 ——
主程序、PyTorch、PySide6 全部装在它自己里面，删掉安装目录就等于清干净，
不会和用户已有的 Python / conda 环境互相污染。

两个容易踩的坑，这里都处理掉：

1. embeddable 包里没有 pip，需要先跑一次 ``get-pip.py`` 引导；
2. embeddable 包用 ``python3xx._pth`` 代替 ``site``，默认**屏蔽** ``site-packages``
   也屏蔽 ``import site``，不改成“标准布局”的话 pip 装进去的包 import 不到。
"""

from __future__ import annotations

import platform
import zipfile
from pathlib import Path

from ..utils.downloader import DownloadItem
from ..utils.logger import get_logger

logger = get_logger(__name__)

#: 内嵌运行时版本。选 3.12：PyTorch / ultralytics / PySide6 的 wheel 都齐全，
#: 也是本仓库实测过的版本区间（``requires-python >= 3.11``）。
EMBEDDED_PYTHON_VERSION = "3.12.10"

#: 官方 embeddable 包地址模板
EMBEDDED_PYTHON_URL = (
    "https://www.python.org/ftp/python/{version}/python-{version}-embed-{arch}.zip"
)

#: pip 引导脚本（官方推荐的一次性脚本）
GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"

#: 内嵌包的架构标签
_ARCH_TAGS = {
    "x86_64": "amd64",
    "amd64": "amd64",
    "AMD64": "amd64",
    "aarch64": "arm64",
    "arm64": "arm64",
    "ARM64": "arm64",
}


def embedded_arch() -> str:
    """当前机器该下哪个架构的 embeddable 包（拿不到就按 amd64）。"""
    machine = platform.machine() or ""
    return _ARCH_TAGS.get(machine, _ARCH_TAGS.get(machine.lower(), "amd64"))


def runtime_archive_name(version: str | None = None, arch: str | None = None) -> str:
    resolved = version or EMBEDDED_PYTHON_VERSION
    return f"python-{resolved}-embed-{arch or embedded_arch()}.zip"


def runtime_url(version: str | None = None, arch: str | None = None) -> str:
    resolved = version or EMBEDDED_PYTHON_VERSION
    return EMBEDDED_PYTHON_URL.format(version=resolved, arch=arch or embedded_arch())


def runtime_download_item(
    cache_dir: str | Path,
    *,
    version: str | None = None,
    arch: str | None = None,
) -> DownloadItem:
    """内嵌运行时的下载项（放进缓存目录，支持断点续传）。"""
    name = runtime_archive_name(version, arch)
    return DownloadItem(
        url=runtime_url(version, arch),
        path=Path(cache_dir) / name,
        label=f"内嵌 Python {version or EMBEDDED_PYTHON_VERSION}",
    )


def get_pip_download_item(cache_dir: str | Path) -> DownloadItem:
    """pip 引导脚本的下载项（只有几 MB，同样走断点续传）。"""
    return DownloadItem(
        url=GET_PIP_URL,
        path=Path(cache_dir) / "get-pip.py",
        label="pip 引导脚本",
    )


def version_tuple(version: str | None = None) -> tuple[int, int]:
    """``"3.12.10"`` → ``(3, 12)``（挑 wheel 时用）。"""
    text = version or EMBEDDED_PYTHON_VERSION
    parts = [part for part in text.split(".") if part.isdigit()]
    values = [int(part) for part in parts[:2]]
    while len(values) < 2:
        values.append(0)
    return values[0], values[1]


def pth_name(version: str | None = None) -> str:
    """embeddable 包里的 ``._pth`` 文件名（如 ``python312._pth``）。"""
    major, minor = version_tuple(version)
    return f"python{major}{minor}._pth"


def runtime_present(python_dir: str | Path, version: str | None = None) -> bool:
    """解释器是否已经部署好（只看关键文件在不在）。"""
    root = Path(python_dir)
    for name in ("python.exe", "python", "pythonw.exe"):
        candidate = root / name
        try:
            if candidate.is_file() and candidate.stat().st_size > 0:
                return True
        except OSError:  # pragma: no cover
            continue
    return False


def extract_runtime(archive: str | Path, python_dir: str | Path) -> Path:
    """把 embeddable 包解压到 ``python_dir``（幂等：已解压过就直接返回）。"""
    target = Path(python_dir)
    target.mkdir(parents=True, exist_ok=True)
    archive_path = Path(archive)
    with zipfile.ZipFile(archive_path) as bundle:
        bundle.extractall(target)
    logger.info("已解压内嵌 Python 运行时到 %s", target)
    return target


def write_pth(python_dir: str | Path, version: str | None = None) -> Path:
    """把 ``._pth`` 改写成"带 site-packages 的标准布局"，返回该文件路径。

    内容含义（``._pth`` 存在时 Python 会**忽略** ``PYTHONPATH`` 与 ``site``）：

    * ``python312.zip`` —— 标准库压缩包；
    * ``.`` —— 解释器所在目录；
    * ``Lib\\site-packages`` —— pip 装包的位置（默认不在搜索路径里！）；
    * ``import site`` —— 打开 ``site`` 处理，这样 ``.pth`` 文件、``sitecustomize`` 才生效。
    """
    root = Path(python_dir)
    root.mkdir(parents=True, exist_ok=True)
    major, minor = version_tuple(version)
    target = root / pth_name(version)
    content = "\n".join(
        [
            f"python{major}{minor}.zip",
            ".",
            str(Path("Lib") / "site-packages"),
            "import site",
            "",
        ]
    )
    target.write_text(content, encoding="utf-8")
    logger.debug("已写入 %s", target)
    return target


def runtime_paths(python_dir: str | Path, version: str | None = None) -> dict[str, Path]:
    """运行时相关的关键路径（供界面显示与排错）。"""
    root = Path(python_dir)
    major, minor = version_tuple(version)
    return {
        "python_dir": root,
        "python_exe": root / ("python.exe" if platform.system() == "Windows" else "python"),
        "stdlib_zip": root / f"python{major}{minor}.zip",
        "pth_file": root / pth_name(version),
        "site_packages": root / "Lib" / "site-packages",
    }


__all__ = [
    "EMBEDDED_PYTHON_URL",
    "EMBEDDED_PYTHON_VERSION",
    "GET_PIP_URL",
    "embedded_arch",
    "extract_runtime",
    "get_pip_download_item",
    "pth_name",
    "runtime_archive_name",
    "runtime_download_item",
    "runtime_paths",
    "runtime_present",
    "runtime_url",
    "version_tuple",
    "write_pth",
]
