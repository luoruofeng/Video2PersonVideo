"""推理设备探测（CUDA / MPS / CPU）与环境自检。"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .logger import get_logger

logger = get_logger(__name__)

AUTO_DEVICE_TOKENS = {"", "auto", "none", "null", "default"}


def torch_available() -> bool:
    """torch 是否可导入。"""
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def cuda_available() -> bool:
    """是否有可用的 NVIDIA CUDA 设备。"""
    try:
        import torch
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception as exc:  # pragma: no cover - 驱动异常等罕见情况
        logger.debug("cuda 探测失败：%s", exc)
        return False


def mps_available() -> bool:
    """是否有可用的 Apple Metal（MPS）设备。"""
    try:
        import torch
    except ImportError:
        return False
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def resolve_device(preferred: str | None = None) -> str:
    """把配置里的 device 解析为 ultralytics 能识别的设备字符串。

    ``None`` / ``"auto"`` 时自动选择：CUDA > MPS > CPU。
    """
    if preferred is None or str(preferred).strip().lower() in AUTO_DEVICE_TOKENS:
        if cuda_available():
            return "cuda:0"
        if mps_available():
            return "mps"
        return "cpu"
    return str(preferred).strip()


def package_version(name: str) -> str:
    """安全地读取已安装包版本，未安装返回 ``"未安装"``。"""
    try:
        return version(name)
    except PackageNotFoundError:
        return "未安装"
    except Exception:  # pragma: no cover
        return "未知"


def environment_report() -> dict[str, str]:
    """收集运行环境信息，供 ``v2pv --check`` 与出错提示使用。"""
    import platform

    from .ffmpeg_tools import find_ffmpeg, find_ffprobe

    report = {
        "Python": platform.python_version(),
        "操作系统": f"{platform.system()} {platform.release()}",
        "torch": package_version("torch"),
        "torchvision": package_version("torchvision"),
        "ultralytics": package_version("ultralytics"),
        "opencv-python": package_version("opencv-python"),
        "numpy": package_version("numpy"),
        "PySide6": package_version("PySide6-Essentials"),
        "PyInstaller": package_version("pyinstaller"),
        "CUDA 可用": "是" if cuda_available() else "否",
        "推理设备": resolve_device(),
    }

    if cuda_available():
        import torch

        report["CUDA 版本"] = str(torch.version.cuda)
        report["GPU"] = torch.cuda.get_device_name(0)
    elif mps_available():
        report["GPU"] = "Apple MPS"

    # 硬件型号 + "该装哪一套 PyTorch"：与界面上的首次运行自检同一套结论
    try:
        from .gpu_probe import probe_hardware
        from .torch_backends import recommend_backend

        profile = probe_hardware()
        if profile.gpus:
            report["显卡型号"] = "；".join(gpu.describe() for gpu in profile.gpus[:3])
        if profile.cuda_driver_version:
            report["驱动支持的 CUDA"] = f"最高 {profile.cuda_driver_version}"
        report["推荐 PyTorch"] = recommend_backend(profile).backend.display
    except Exception as exc:  # noqa: BLE001 - 自检失败不该影响 --check
        logger.debug("硬件自检失败：%s", exc)

    ffmpeg = find_ffmpeg()
    report["ffmpeg"] = ffmpeg or "未找到（音轨保留/重编码不可用，请安装并加入 PATH）"
    if ffmpeg:
        report["ffprobe"] = find_ffprobe() or "未找到（音量检测将退化为无音轨）"
    return report


def format_report(report: dict[str, str] | None = None) -> str:
    """把环境信息格式化成对齐的多行文本。"""
    data = report if report is not None else environment_report()
    width = max(len(key) for key in data)
    lines = ["环境自检："]
    lines.extend(f"  {key.ljust(width)} : {value}" for key, value in data.items())
    return "\n".join(lines)
