"""本机环境检查：判断"还需要不需要下载 PyTorch / YOLO 权重"。

只做**秒回**的本地判断（``find_spec`` 找包、读 dist-info 取版本号、``Path.exists``
找权重），既不 ``import torch``（那要好几秒）也不联网，所以可以放心地在启动路径上调。

权重会在两个地方找：

1. 仓库的 ``assets/models``（自检页下载的落地位置）；
2. **当前工作目录**——ultralytics 自己在找不到权重时会下到这里，
   ``yolo11n.pt`` 这样的相对路径也按 cwd 解析，所以用户"直接跑通"多半就是靠它。
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
from pathlib import Path

from ..config import DEFAULT_MODEL
from .logger import get_logger
from .torch_install import YOLO_WEIGHTS, model_directory

logger = get_logger(__name__)


def torch_installed() -> bool:
    """环境里有没有 PyTorch（只看装没装，不 import torch）。"""
    if torch_version() is not None:
        return True
    try:
        return importlib.util.find_spec("torch") is not None
    except Exception as exc:  # noqa: BLE001 - 拿不到就当没装
        logger.debug("探测 torch 失败：%s", exc)
        return False


def torch_version() -> str | None:
    """已安装 PyTorch 的版本号（没装 / 读不到时为 ``None``）。"""
    try:
        return importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        return None
    except Exception as exc:  # noqa: BLE001 - 元数据损坏时不该让界面崩
        logger.debug("读取 torch 版本失败：%s", exc)
        return None


def weight_search_dirs() -> list[Path]:
    """可能放着 YOLO 权重的目录（仓库 ``assets/models`` + 当前工作目录）。"""
    directories: list[Path] = []
    for candidate in (model_directory(), Path.cwd()):
        if candidate not in directories:
            directories.append(candidate)
    return directories


def find_weight(name: str, *, directories: list[Path] | None = None) -> Path | None:
    """在候选目录里找某个权重文件（存在且非空才算数）。"""
    for directory in directories if directories is not None else weight_search_dirs():
        candidate = directory / name
        try:
            if candidate.is_file() and candidate.stat().st_size > 0:
                return candidate
        except OSError as exc:  # pragma: no cover - 依赖真实文件系统
            logger.debug("检查权重 %s 失败：%s", candidate, exc)
    return None


def installed_weight_names(*, directories: list[Path] | None = None) -> list[str]:
    """本地已经有的 YOLO 权重文件名。"""
    return [
        name for name in YOLO_WEIGHTS if find_weight(name, directories=directories) is not None
    ]


def missing_weight_names(*, directories: list[Path] | None = None) -> list[str]:
    """本地还缺哪些 YOLO 权重文件名。"""
    return [
        name for name in YOLO_WEIGHTS if find_weight(name, directories=directories) is None
    ]


def environment_ready() -> bool:
    """PyTorch 装好了、默认权重也在本地 → 不需要再走一遍下载页。

    只看"默认检测模型"（``DEFAULT_MODEL``）：姿态模型是可选的，
    没下也不影响正常出片，不该为它弹一次下载页。
    """
    return torch_installed() and find_weight(DEFAULT_MODEL) is not None


def environment_summary() -> str | None:
    """一句话说明本机已经具备什么（什么都没有时返回 ``None``）。"""
    parts: list[str] = []
    version = torch_version()
    if version:
        parts.append(f"已安装 PyTorch {version}")
    names = installed_weight_names()
    if names:
        parts.append("本地已有 YOLO 权重 " + "、".join(names))
    if not parts:
        return None
    return "本机环境：" + " · ".join(parts) + "（已有的内容不会重复下载）"


__all__ = [
    "environment_ready",
    "environment_summary",
    "find_weight",
    "installed_weight_names",
    "missing_weight_names",
    "torch_installed",
    "torch_version",
    "weight_search_dirs",
]
