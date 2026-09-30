"""安装器自带的"待安装内容"（payload）。

安装器自己是个独立的 exe，它要装的东西（程序本体 wheel、默认配置、说明文档）
在打包时就塞进它的 ``payload/`` 目录里；源码运行时直接在仓库里找。
这样安装过程**只需要网络下 PyTorch / YOLO**，程序本体不需要联网拉取。
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from ..utils.logger import get_logger

logger = get_logger(__name__)

#: 打包时 payload 的目录名
PAYLOAD_DIRNAME = "payload"

#: 随安装器分发的静态文件（相对 payload 根目录 → 安装目录里的相对路径）
STATIC_FILES: tuple[tuple[str, str], ...] = (
    ("requirements.txt", "requirements.txt"),
    ("requirements-gui.txt", "requirements-gui.txt"),
    ("README.md", "README.md"),
    ("LICENSE", "LICENSE"),
    ("assets/app.ico", "assets/app.ico"),
)
#: 打包时放程序本体 wheel 的目录
WHEEL_DIRNAME = "wheels"
#: 默认配置文件所在目录
CONFIGS_DIRNAME = "configs"

#: 找不到随包 requirements 时退回这份内置清单（与 pyproject 的 dependencies 一致）。
#: 注意：**故意不含 torch / torchvision** —— 它们由 PyTorch 阶段按显卡单独安装，
#: 混在一起装会把用户的 CUDA 版覆盖成 PyPI 的 CPU 版。
FALLBACK_REQUIREMENTS: tuple[str, ...] = (
    "ultralytics==8.4.163",
    "opencv-python==5.0.0.93",
    "numpy==2.5.3",
    "tqdm==4.70.1",
    "pyyaml==6.0.3",
    "ffmpeg-python==0.2.0",
)
#: 图形界面依赖（装完可以直接用界面）
FALLBACK_GUI_REQUIREMENTS: tuple[str, ...] = ("PySide6-Essentials==6.11.2",)

#: 需要从依赖清单里剔除的包（交给 PyTorch 阶段按硬件安装）
TORCH_PACKAGES = ("torch", "torchvision")


def payload_root() -> Path | None:
    """payload 根目录（打包形态 ``sys._MEIPASS/payload``，源码形态仓库根）。"""
    candidates: list[Path] = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / PAYLOAD_DIRNAME)
        candidates.append(Path(meipass))
    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).parent / PAYLOAD_DIRNAME)
        candidates.append(Path(sys.executable).parent)
    # 源码运行：从本文件往上找带 pyproject.toml 的仓库根
    for parent in Path(__file__).resolve().parents:
        candidates.append(parent)
        candidates.append(parent / PAYLOAD_DIRNAME)

    for candidate in candidates:
        try:
            if (candidate / "pyproject.toml").is_file() or (candidate / "src").is_dir():
                return candidate
        except OSError:  # pragma: no cover
            continue
    logger.warning("未找到安装内容（payload）目录")
    return None


def bundled_wheels(root: Path | None = None) -> list[Path]:
    """随安装器分发的 wheel（优先用它安装程序本体，免去构建步骤）。"""
    base = root or payload_root()
    if base is None:
        return []
    found: list[Path] = []
    for directory in (base / WHEEL_DIRNAME, base / "build" / PAYLOAD_DIRNAME, base):
        try:
            if not directory.is_dir():
                continue
            found.extend(sorted(directory.glob("video2personvideo-*.whl")))
        except OSError:  # pragma: no cover
            continue
    # 去重（build/ 与根目录可能指向同一个文件）
    unique: dict[str, Path] = {}
    for path in found:
        unique.setdefault(path.name, path)
    return list(unique.values())


def source_root(root: Path | None = None) -> Path | None:
    """源码形态的程序本体目录（含 ``src/`` 与 ``pyproject.toml``）。"""
    base = root or payload_root()
    if base is None:
        return None
    try:
        if (base / "src" / "video2personvideo").is_dir() and (base / "pyproject.toml").is_file():
            return base
    except OSError:  # pragma: no cover
        return None
    return None


def requirement_files(root: Path | None = None) -> list[Path]:
    """随包的依赖清单文件（运行期 + 图形界面）。"""
    base = root or payload_root()
    if base is None:
        return []
    found: list[Path] = []
    for name in ("requirements.txt", "requirements-gui.txt"):
        candidate = base / name
        try:
            if candidate.is_file():
                found.append(candidate)
        except OSError:  # pragma: no cover
            continue
    return found


def parse_requirement_lines(text: str) -> list[str]:
    """把 requirements 文本拆成"一行一个包"，顺手去掉注释与 ``-r`` 递归。"""
    lines: list[str] = []
    for raw in (text or "").splitlines():
        entry = raw.split("#", 1)[0].strip()
        if not entry:
            continue
        if entry.startswith("-r ") or entry.startswith("--requirement"):
            continue
        lines.append(entry)
    return lines


def package_name(entry: str) -> str:
    """从 ``torch==2.14.0`` / ``torch >= 2`` 里取出包名（小写、连字符归一）。"""
    text = entry.strip()
    for separator in ("==", ">=", "<=", "~=", "!=", ">", "<", "[", ";", " "):
        index = text.find(separator)
        if index > 0:
            text = text[:index]
    return text.strip().lower().replace("_", "-")


def strip_torch(entries: list[str]) -> list[str]:
    """剔除 torch / torchvision（由 PyTorch 阶段按显卡安装，不能被 pip 覆盖）。"""
    return [item for item in entries if package_name(item) not in TORCH_PACKAGES]


def resolve_requirements(root: Path | None = None) -> tuple[list[str], str]:
    """解析出要安装的依赖清单，返回 ``(包列表, 来源说明)``。"""
    files = requirement_files(root)
    if files:
        merged: list[str] = []
        for path in files:
            try:
                merged.extend(parse_requirement_lines(path.read_text(encoding="utf-8")))
            except OSError as exc:  # pragma: no cover
                logger.warning("读取依赖清单失败 %s：%s", path, exc)
        if merged:
            return strip_torch(merged), f"随包依赖清单（{len(files)} 个文件）"
    return (
        strip_torch([*FALLBACK_REQUIREMENTS, *FALLBACK_GUI_REQUIREMENTS]),
        "内置依赖清单",
    )


def torch_pins(root: Path | None = None) -> dict[str, str]:
    """从随包依赖清单里取出 torch / torchvision 的锁定版本。

    安装器跑在打包好的 exe 里，工作目录是用户双击的位置，
    ``torch_install.default_requirements_path()`` 找不到仓库里的 ``requirements.txt``，
    于是"按最新版装"会和仓库锁定的版本不一致。这里显式从 payload 读，
    保证装出来的 PyTorch 与 ``requirements.txt`` 一致。
    """
    pins: dict[str, str] = {}
    for path in requirement_files(root):
        try:
            lines = parse_requirement_lines(path.read_text(encoding="utf-8"))
        except OSError as exc:  # pragma: no cover
            logger.warning("读取依赖清单失败 %s：%s", path, exc)
            continue
        for line in lines:
            name = package_name(line)
            if name not in TORCH_PACKAGES:
                continue
            _name, separator, version = line.partition("==")
            if separator and version.strip():
                pins[name] = version.split(";", 1)[0].split()[0].strip()
    return pins


def copy_static_files(root: Path | None, install_dir: str | Path) -> list[Path]:
    """把默认配置、说明文档等随包文件复制进安装目录。"""
    base = root or payload_root()
    if base is None:
        return []
    target_root = Path(install_dir)
    copied: list[Path] = []

    for source_rel, target_rel in STATIC_FILES:
        source = base / source_rel
        try:
            if not source.is_file():
                continue
            destination = target_root / target_rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied.append(destination)
        except OSError as exc:  # pragma: no cover
            logger.warning("复制 %s 失败：%s", source, exc)

    # 默认配置：整个 configs 目录（用户之后可以直接改这些 YAML）
    configs = base / CONFIGS_DIRNAME
    try:
        if configs.is_dir():
            destination = target_root / CONFIGS_DIRNAME
            destination.mkdir(parents=True, exist_ok=True)
            for item in sorted(configs.glob("*.yaml")):
                shutil.copy2(item, destination / item.name)
                copied.append(destination / item.name)
    except OSError as exc:  # pragma: no cover
        logger.warning("复制配置目录失败：%s", exc)

    # 随包的 YOLO 权重（打包时若已经放进 assets/models，就一起装上）
    models = base / "assets" / "models"
    try:
        if models.is_dir():
            destination = target_root / "assets" / "models"
            destination.mkdir(parents=True, exist_ok=True)
            for weight in sorted(models.glob("*.pt")):
                shutil.copy2(weight, destination / weight.name)
                copied.append(destination / weight.name)
    except OSError as exc:  # pragma: no cover
        logger.warning("复制随包权重失败：%s", exc)

    return copied


__all__ = [
    "CONFIGS_DIRNAME",
    "FALLBACK_GUI_REQUIREMENTS",
    "FALLBACK_REQUIREMENTS",
    "PAYLOAD_DIRNAME",
    "STATIC_FILES",
    "TORCH_PACKAGES",
    "WHEEL_DIRNAME",
    "bundled_wheels",
    "copy_static_files",
    "package_name",
    "parse_requirement_lines",
    "payload_root",
    "requirement_files",
    "resolve_requirements",
    "source_root",
    "strip_torch",
    "torch_pins",
]
