"""用户在安装向导里做的选择（组件清单）。

这份选择会原样写进安装状态文件（``install.json``），
所以"上次选了 CUDA 12.6、没装 ffmpeg"这类信息在下次继续安装时还在，
界面能直接把上次的选择摆回来，不用用户再挑一遍。
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..config import DEFAULT_MODEL
from ..utils.torch_backends import TORCH_BACKENDS, backend_by_key, recommend_backend
from ..utils.torch_install import YOLO_WEIGHTS
from . import paths
from .runtime import EMBEDDED_PYTHON_VERSION, version_tuple

#: 默认勾选的 YOLO 权重（检测模型必须有；姿态模型是可选增强）
DEFAULT_WEIGHTS: tuple[str, ...] = (DEFAULT_MODEL,)

#: 可选勾选的权重（除默认检测模型之外，界面上作为"可选增强"列出）
OPTIONAL_WEIGHTS: tuple[str, ...] = tuple(
    name for name in YOLO_WEIGHTS if name not in DEFAULT_WEIGHTS
)

#: 随包 ffmpeg 的下载地址（gyan.dev 的 release essentials 构建，长期有效）
FFMPEG_URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
#: ffmpeg 解压后需要的那两个可执行文件（在压缩包的 ``bin/`` 下）
FFMPEG_MEMBERS = ("bin/ffmpeg.exe", "bin/ffprobe.exe")
#: 界面显示的估计体积（压缩包）
FFMPEG_APPROX_BYTES = 90_000_000


@dataclass(slots=True)
class InstallOptions:
    """一次安装的完整选择。"""

    install_dir: Path = field(default_factory=paths.default_install_dir)
    #: PyTorch 构建 key（``cu126`` / ``cpu`` …），见 ``utils.torch_backends``
    backend_key: str = "cpu"
    #: 是否下载并安装 PyTorch（本机已有可用版本且用户想复用时可以关掉）
    install_torch: bool = True
    #: 要下载的 YOLO 权重文件名
    weights: list[str] = field(default_factory=lambda: list(DEFAULT_WEIGHTS))
    #: 是否随包部署 ffmpeg
    install_ffmpeg: bool = True
    #: 快捷方式
    desktop_shortcut: bool = True
    start_menu_shortcut: bool = True
    #: 装完之后是否删掉下载缓存（默认保留：重装 / 修复时不用重下几个 GB）
    delete_downloads: bool = False
    #: 装完是否直接启动程序
    launch_after_install: bool = True
    #: 内嵌运行时版本
    python_version: str = EMBEDDED_PYTHON_VERSION

    # ------------------------------------------------------------- 派生信息
    @property
    def python_tuple(self) -> tuple[int, int]:
        return version_tuple(self.python_version)

    @property
    def backend(self):
        return backend_by_key(self.backend_key)

    def resolved_install_dir(self) -> Path:
        return Path(self.install_dir).expanduser()

    # ------------------------------------------------------------- 序列化
    def to_dict(self) -> dict:
        data = asdict(self)
        data["install_dir"] = str(self.install_dir)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> InstallOptions:
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        clean = {key: value for key, value in (data or {}).items() if key in known}
        if clean.get("install_dir"):
            clean["install_dir"] = Path(str(clean["install_dir"]))
        weights = clean.get("weights")
        if isinstance(weights, (list, tuple)):
            clean["weights"] = [str(item) for item in weights if str(item) in YOLO_WEIGHTS]
        return cls(**clean)

    def describe(self) -> str:
        """一行摘要（日志 / 状态页用）。"""
        parts = [f"安装到 {self.resolved_install_dir()}"]
        parts.append(
            f"PyTorch：{self.backend.display}" if self.install_torch else "PyTorch：跳过（复用本机）"
        )
        if self.weights:
            parts.append("YOLO 权重：" + "、".join(self.weights))
        if self.install_ffmpeg:
            parts.append("随包 ffmpeg")
        return " · ".join(parts)


def default_options(
    *,
    install_dir: str | os.PathLike[str] | None = None,
    profile=None,
) -> InstallOptions:
    """给当前机器推荐一套默认选择（有显卡就默认 CUDA）。"""
    options = InstallOptions()
    if install_dir is not None:
        options.install_dir = Path(install_dir)
    if profile is not None:
        try:
            options.backend_key = recommend_backend(profile).backend.key
        except Exception:  # noqa: BLE001 - 探测不到就退回 CPU，绝不让界面崩
            options.backend_key = "cpu"
    return options


def available_backends():
    """全部可选的 PyTorch 构建（界面下拉用）。"""
    return TORCH_BACKENDS


__all__ = [
    "DEFAULT_WEIGHTS",
    "FFMPEG_APPROX_BYTES",
    "FFMPEG_MEMBERS",
    "FFMPEG_URL",
    "OPTIONAL_WEIGHTS",
    "InstallOptions",
    "available_backends",
    "default_options",
]
