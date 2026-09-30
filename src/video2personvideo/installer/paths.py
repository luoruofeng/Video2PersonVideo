"""安装器用到的全部路径与系统位置（集中在这里，避免魔法字符串散落）。

默认安装位置选 ``%LOCALAPPDATA%\\Programs\\Video2PersonVideo``：
和 VS Code、GitHub Desktop 这类"每个用户装一份"的软件一致 ——
**不需要管理员权限**（不弹 UAC），卸载时也不会影响其他用户。
"""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

from ..utils import app_paths

#: 显示名（开始菜单、控制面板里的名字）
APP_DISPLAY_NAME = "Video2PersonVideo"
#: 安装目录名
APP_DIR_NAME = "Video2PersonVideo"
#: 发布者（写进卸载信息）
APP_PUBLISHER = "Video2PersonVideo"
#: 项目主页
APP_URL = "https://github.com/your-name/Video2PersonVideo"

#: 卸载信息注册表位置（当前用户，配合"无需管理员"的安装方式）
UNINSTALL_REGISTRY_KEY = (
    r"Software\Microsoft\Windows\CurrentVersion\Uninstall\Video2PersonVideo"
)
#: 程序自己的注册表位置（记录安装目录与版本，便于二次安装时探测）
APP_REGISTRY_KEY = r"Software\Video2PersonVideo"

#: 开始菜单 / 桌面的快捷方式名
SHORTCUT_NAME = "Video2PersonVideo"
UNINSTALL_SHORTCUT_NAME = "卸载 Video2PersonVideo"

#: 主界面启动器的控制台 / 无窗口参数（内嵌解释器 + 运行程序的参数）
GUI_ARGUMENTS = "-m video2personvideo --gui"
CLI_ARGUMENTS = "-m video2personvideo"

WINDOWS = platform.system() == "Windows"


def _env_path(name: str, fallback: Path) -> Path:
    value = os.environ.get(name)
    return Path(value) if value else fallback


def user_home() -> Path:
    return Path.home()


def local_app_data() -> Path:
    """``%LOCALAPPDATA%``（非 Windows 上退回 ``~/.local/share``）。"""
    return _env_path("LOCALAPPDATA", user_home() / ".local" / "share")


def roaming_app_data() -> Path:
    return _env_path("APPDATA", user_home() / ".config")


def default_install_dir() -> Path:
    """默认安装目录。"""
    if WINDOWS:
        return local_app_data() / "Programs" / APP_DIR_NAME
    return local_app_data() / APP_DIR_NAME


def install_dir_env_var() -> str:
    return "V2PV_INSTALL_DIR"


def configured_install_dir() -> Path | None:
    """用户在命令行 / 环境变量里指定的安装目录（没有则为 ``None``）。"""
    value = os.environ.get(install_dir_env_var())
    return Path(value) if value else None


def user_state_dir() -> Path:
    """安装器自己的状态目录（状态文件镜像、日志、下载缓存）。"""
    return app_paths.user_state_dir()


def cache_dir() -> Path:
    """下载缓存根目录（放在安装目录之外，卸载时可以选择保留）。"""
    return app_paths.cache_dir()


def wheels_dir() -> Path:
    """PyTorch wheel 缓存（断点也在里面，重装 / 修复直接复用）。"""
    return app_paths.wheels_cache_dir()


def downloads_dir() -> Path:
    """其它下载缓存（内嵌解释器压缩包、get-pip、ffmpeg 压缩包）。"""
    return app_paths.downloads_cache_dir()


def journal_path(install_dir: str | os.PathLike[str]) -> Path:
    """安装状态文件：既记录阶段进度，也是"这是安装目录"的标记（见 ``utils.app_paths``）。"""
    return Path(install_dir) / "install.json"


def mirrored_journal_path() -> Path:
    """状态文件的镜像（安装目录被手工删掉时，还能知道"曾经装过、装到哪儿"）。"""
    return user_state_dir() / "install.json"


def log_path() -> Path:
    return user_state_dir() / "installer.log"


# ----------------------------------------------------------------- 安装目录布局
def python_dir(install_dir: str | os.PathLike[str]) -> Path:
    return Path(install_dir) / "python"


def python_exe(install_dir: str | os.PathLike[str]) -> Path:
    """内嵌解释器（Windows 是 ``python.exe``，其它平台退回解释器名）。"""
    root = python_dir(install_dir)
    return root / "python.exe" if WINDOWS else root / "python3"


def pythonw_exe(install_dir: str | os.PathLike[str]) -> Path:
    """无窗口解释器（Windows 上启动图形界面用，避免弹黑框）。"""
    root = python_dir(install_dir)
    target = root / "pythonw.exe"
    return target if target.exists() else python_exe(install_dir)


def app_dir(install_dir: str | os.PathLike[str]) -> Path:
    """程序本体目录（安装进来的 ``video2personvideo`` 包 / 源码）。"""
    return Path(install_dir) / "app"


def assets_dir(install_dir: str | os.PathLike[str]) -> Path:
    return Path(install_dir) / "assets"


def models_dir(install_dir: str | os.PathLike[str]) -> Path:
    """YOLO 权重落地目录。"""
    return assets_dir(install_dir) / "models"


def ffmpeg_bin_dir(install_dir: str | os.PathLike[str]) -> Path:
    """随包 ffmpeg 目录（会被 :func:`utils.app_paths.register_bundled_tools` 挂到 PATH）。"""
    return Path(install_dir) / "ffmpeg" / "bin"


def configs_dir(install_dir: str | os.PathLike[str]) -> Path:
    return Path(install_dir) / "configs"


def launcher_cmd(install_dir: str | os.PathLike[str]) -> Path:
    """命令行入口脚本（``v2pv`` 的等价物，免开 PowerShell 也能用）。"""
    return Path(install_dir) / "v2pv.cmd"


def gui_launcher_cmd(install_dir: str | os.PathLike[str]) -> Path:
    """图形界面入口脚本（双击即用，不弹控制台窗口）。"""
    return Path(install_dir) / "Video2PersonVideo.cmd"


# ----------------------------------------------------------------- 系统位置
def start_menu_dir() -> Path:
    """当前用户的开始菜单「程序」目录。"""
    if WINDOWS:
        return roaming_app_data() / "Microsoft" / "Windows" / "Start Menu" / "Programs"
    return roaming_app_data() / "applications"


def desktop_dir() -> Path:
    """桌面目录（拿系统导出的路径，OneDrive 重定向过的机器也对）。"""
    if not WINDOWS:
        return user_home() / "Desktop"
    return _env_path("USERPROFILE", user_home()) / "Desktop"


def user_program_files() -> Path:
    return local_app_data() / "Programs"


def format_size(num_bytes: float | None) -> str:
    """人类可读的体积（界面 / 日志共用）。"""
    if num_bytes is None:
        return "未知"
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def is_windows() -> bool:
    return WINDOWS or sys.platform == "win32"


__all__ = [
    "APP_DIR_NAME",
    "APP_DISPLAY_NAME",
    "APP_PUBLISHER",
    "APP_REGISTRY_KEY",
    "APP_URL",
    "CLI_ARGUMENTS",
    "GUI_ARGUMENTS",
    "SHORTCUT_NAME",
    "UNINSTALL_REGISTRY_KEY",
    "UNINSTALL_SHORTCUT_NAME",
    "app_dir",
    "assets_dir",
    "cache_dir",
    "configs_dir",
    "configured_install_dir",
    "default_install_dir",
    "desktop_dir",
    "downloads_dir",
    "ffmpeg_bin_dir",
    "format_size",
    "gui_launcher_cmd",
    "install_dir_env_var",
    "is_windows",
    "journal_path",
    "launcher_cmd",
    "local_app_data",
    "log_path",
    "mirrored_journal_path",
    "models_dir",
    "python_dir",
    "python_exe",
    "pythonw_exe",
    "start_menu_dir",
    "user_state_dir",
    "wheels_dir",
]
