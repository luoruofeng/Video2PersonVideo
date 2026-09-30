"""Windows 平台集成：快捷方式、卸载信息注册表、目录体积与"自删"。

非 Windows 上所有函数都变成安全的空操作（返回 ``None`` / ``False``），
这样安装引擎可以在任意平台上被单测覆盖。
"""

from __future__ import annotations

import base64
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from ..utils.logger import get_logger
from . import paths

logger = get_logger(__name__)

#: 子进程超时（快捷方式 / 注册表操作都很快）
POWERSHELL_TIMEOUT = 60.0

#: 生成 .cmd / .lnk 时用的文件编码
_SCRIPT_ENCODING = "utf-8"


def _windows_only() -> bool:
    return paths.is_windows()


def _creation_flags(*extra: int) -> int:
    if not _windows_only():
        return 0
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    for item in extra:
        flags |= item
    return flags


def _ps_quote(text: str) -> str:
    """PowerShell 单引号字符串里的转义（'' 表示一个单引号）。"""
    return str(text).replace("'", "''")


def run_powershell(script: str, *, timeout: float = POWERSHELL_TIMEOUT) -> tuple[int, str]:
    """执行一段 PowerShell（用 ``-EncodedCommand``，绕开引号与编码问题）。"""
    if not _windows_only():  # pragma: no cover - 非 Windows 不会走到这
        return 1, "仅支持 Windows"
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    command = [
        "powershell",
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-EncodedCommand",
        encoded,
    ]
    try:
        completed = subprocess.run(  # noqa: S603 - 参数由本模块拼接，无 shell
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
            creationflags=_creation_flags(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("执行 PowerShell 失败：%s", exc)
        return 1, str(exc)
    output = f"{completed.stdout or ''}{completed.stderr or ''}".strip()
    if completed.returncode != 0:
        logger.warning("PowerShell 返回 %s：%s", completed.returncode, output)
    return completed.returncode, output


# --------------------------------------------------------------------- 快捷方式
def create_shortcut(
    link: str | os.PathLike[str],
    *,
    target: str | os.PathLike[str],
    arguments: str = "",
    working_dir: str | os.PathLike[str] = "",
    icon: str | os.PathLike[str] = "",
    description: str = "",
) -> bool:
    """创建一个 ``.lnk`` 快捷方式（用系统自带的 WScript.Shell，无需 pywin32）。"""
    if not _windows_only():
        return False
    destination = Path(link)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("无法创建快捷方式目录 %s：%s", destination.parent, exc)
        return False

    script = "; ".join(
        [
            "$ErrorActionPreference = 'Stop'",
            "$ws = New-Object -ComObject WScript.Shell",
            f"$sc = $ws.CreateShortcut('{_ps_quote(destination)}')",
            f"$sc.TargetPath = '{_ps_quote(target)}'",
            f"$sc.Arguments = '{_ps_quote(arguments)}'",
            f"$sc.WorkingDirectory = '{_ps_quote(working_dir)}'",
            f"$sc.Description = '{_ps_quote(description)}'",
            f"$sc.IconLocation = '{_ps_quote(icon)}'" if icon else "$sc.IconLocation = ''",
            "$sc.Save()",
        ]
    )
    code, output = run_powershell(script)
    if code != 0:
        logger.warning("创建快捷方式失败 %s：%s", destination, output)
        return False
    logger.info("已创建快捷方式：%s", destination)
    return True


def remove_shortcut(link: str | os.PathLike[str]) -> bool:
    """删除快捷方式（不存在也算成功）。"""
    target = Path(link)
    if not target.exists():
        return True
    try:
        target.unlink()
        return True
    except OSError as exc:
        logger.warning("删除快捷方式失败 %s：%s", target, exc)
        return False


# --------------------------------------------------------------------- 启动脚本
def _cmd_header() -> str:
    lines = [
        "@echo off",
        "rem 本文件由 Video2PersonVideo 安装器生成，重装 / 卸载时会被覆盖或删除。",
        "setlocal",
    ]
    return "\r\n".join(lines) + "\r\n"


def write_launcher_scripts(
    install_dir: str | os.PathLike[str],
    *,
    version: str = "",
) -> list[Path]:
    """写两个免安装目录之外也能用的入口脚本：``Video2PersonVideo.cmd`` / ``v2pv.cmd``。

    ``.cmd`` 里显式设置下载缓存与工作目录，双击就能用；
    图形界面入口用 ``pythonw.exe``（无控制台窗口）。
    """
    root = Path(install_dir)
    python = paths.python_exe(root)
    pythonw = paths.pythonw_exe(root)
    written: list[Path] = []

    gui = paths.gui_launcher_cmd(root)
    gui_body = (
        _cmd_header()
        + f'cd /d "{root}"\r\n'
        + f'start "" "{pythonw}" -m video2personvideo --gui\r\n'
        + "endlocal\r\n"
    )
    cli = paths.launcher_cmd(root)
    cli_body = (
        _cmd_header()
        + f'cd /d "{root}"\r\n'
        + f'"{python}" -m video2personvideo %*\r\n'
        + "endlocal\r\n"
    )
    for path, body in ((gui, gui_body), (cli, cli_body)):
        try:
            path.write_text(body, encoding=_SCRIPT_ENCODING, newline="")
            written.append(path)
        except OSError as exc:  # pragma: no cover
            logger.warning("写入入口脚本失败 %s：%s", path, exc)
    return written


def desktop_shortcut_path() -> Path:
    return paths.desktop_dir() / f"{paths.SHORTCUT_NAME}.lnk"


def start_menu_folder() -> Path:
    return paths.start_menu_dir() / paths.APP_DIR_NAME


def start_menu_shortcut_path() -> Path:
    return start_menu_folder() / f"{paths.SHORTCUT_NAME}.lnk"


def start_menu_uninstall_path() -> Path:
    return start_menu_folder() / f"{paths.UNINSTALL_SHORTCUT_NAME}.lnk"


def icon_path(install_dir: str | os.PathLike[str]) -> Path | None:
    """安装目录里的图标（没有就交给系统用 exe 自带的）。"""
    for candidate in (
        Path(install_dir) / "assets" / "app.ico",
        Path(install_dir) / "app.ico",
    ):
        try:
            if candidate.is_file():
                return candidate
        except OSError:  # pragma: no cover
            continue
    return None


def create_shortcuts(
    install_dir: str | os.PathLike[str],
    *,
    desktop: bool = True,
    start_menu: bool = True,
) -> list[Path]:
    """按用户勾选创建快捷方式，返回真正建好的那几个。"""
    root = Path(install_dir)
    pythonw = paths.pythonw_exe(root)
    icon = icon_path(root)
    created: list[Path] = []

    if start_menu:
        target = start_menu_shortcut_path()
        if create_shortcut(
            target,
            target=pythonw,
            arguments=paths.GUI_ARGUMENTS,
            working_dir=root,
            icon=icon or pythonw,
            description=f"{paths.APP_DISPLAY_NAME} 图形界面",
        ):
            created.append(target)
        # 卸载入口也放进开始菜单，符合 Windows 用户习惯
        uninstall_script = root / "Uninstall.cmd"
        if uninstall_script.is_file() and create_shortcut(
            start_menu_uninstall_path(),
            target=uninstall_script,
            arguments="",
            working_dir=root,
            icon=icon or pythonw,
            description=f"卸载 {paths.APP_DISPLAY_NAME}",
        ):
            created.append(start_menu_uninstall_path())

    if desktop:
        target = desktop_shortcut_path()
        if create_shortcut(
            target,
            target=pythonw,
            arguments=paths.GUI_ARGUMENTS,
            working_dir=root,
            icon=icon or pythonw,
            description=f"{paths.APP_DISPLAY_NAME} 图形界面",
        ):
            created.append(target)
    return created


def remove_shortcuts() -> list[Path]:
    """删掉所有由安装器创建的快捷方式（含开始菜单文件夹）。"""
    removed: list[Path] = []
    for candidate in (
        start_menu_shortcut_path(),
        start_menu_uninstall_path(),
        desktop_shortcut_path(),
    ):
        if remove_shortcut(candidate):
            removed.append(candidate)
    folder = start_menu_folder()
    try:
        if folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()
            removed.append(folder)
    except OSError as exc:  # pragma: no cover - 目录非空 / 被占用
        logger.debug("删除开始菜单文件夹失败 %s：%s", folder, exc)
    return removed


# --------------------------------------------------------------------- 注册表
def _uninstall_command(install_dir: str | os.PathLike[str]) -> str:
    """"应用和功能"里点"卸载"时执行的命令（走图形界面卸载器）。"""
    root = Path(install_dir)
    pythonw = paths.pythonw_exe(root)
    return f'"{pythonw}" -m video2personvideo.installer --uninstall --install-dir "{root}"'


def register_install(
    install_dir: str | os.PathLike[str],
    *,
    version: str,
    estimated_bytes: int | None = None,
) -> bool:
    """写"应用和功能"（控制面板）里的卸载信息。"""
    if not _windows_only():
        return False
    import winreg  # noqa: PLC0415 - 仅在 Windows 上可用

    root = Path(install_dir)
    pythonw = paths.pythonw_exe(root)
    icon = icon_path(root) or pythonw
    values: dict[str, tuple[object, int]] = {
        "DisplayName": (paths.APP_DISPLAY_NAME, winreg.REG_SZ),
        "DisplayVersion": (version or "0", winreg.REG_SZ),
        "Publisher": (paths.APP_PUBLISHER, winreg.REG_SZ),
        "InstallLocation": (str(root), winreg.REG_SZ),
        "DisplayIcon": (f"{icon}", winreg.REG_SZ),
        "UninstallString": (_uninstall_command(root), winreg.REG_SZ),
        "QuietUninstallString": (
            f'"{pythonw}" -m video2personvideo.installer --uninstall --yes --quiet '
            f'--install-dir "{root}"',
            winreg.REG_SZ,
        ),
        "URLInfoAbout": (paths.APP_URL, winreg.REG_SZ),
        "NoModify": (1, winreg.REG_DWORD),
        "NoRepair": (1, winreg.REG_DWORD),
    }
    if estimated_bytes:
        values["EstimatedSize"] = (max(int(estimated_bytes) // 1024, 1), winreg.REG_DWORD)

    try:
        with winreg.CreateKeyEx(
            winreg.HKEY_CURRENT_USER, paths.UNINSTALL_REGISTRY_KEY, 0, winreg.KEY_WRITE
        ) as key:
            for name, (value, kind) in values.items():
                winreg.SetValueEx(key, name, 0, kind, value)
            winreg.SetValueEx(key, "InstallDate", 0, winreg.REG_SZ, _today())
        with winreg.CreateKeyEx(
            winreg.HKEY_CURRENT_USER, paths.APP_REGISTRY_KEY, 0, winreg.KEY_WRITE
        ) as key:
            winreg.SetValueEx(key, "InstallLocation", 0, winreg.REG_SZ, str(root))
            winreg.SetValueEx(key, "Version", 0, winreg.REG_SZ, version or "0")
    except OSError as exc:
        logger.warning("写入注册表失败：%s", exc)
        return False
    logger.info("已登记卸载信息：%s", paths.UNINSTALL_REGISTRY_KEY)
    return True


def unregister_install() -> bool:
    """清掉注册表项（不存在也算成功）。"""
    if not _windows_only():
        return False
    import winreg  # noqa: PLC0415

    ok = True
    for key_path in (paths.UNINSTALL_REGISTRY_KEY, paths.APP_REGISTRY_KEY):
        try:
            winreg.DeleteKey(winreg.HKEY_CURRENT_USER, key_path)
        except FileNotFoundError:
            continue
        except OSError as exc:
            logger.warning("删除注册表项失败 %s：%s", key_path, exc)
            ok = False
    return ok


def _read_registry_value(key_path: str, name: str) -> str | None:
    if not _windows_only():
        return None
    import winreg  # noqa: PLC0415

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_READ) as key:
            value, _ = winreg.QueryValueEx(key, name)
            return str(value)
    except (FileNotFoundError, OSError):
        return None


def installed_location() -> Path | None:
    """注册表里登记的安装位置（没登记返回 ``None``）。"""
    for key_path, name in (
        (paths.APP_REGISTRY_KEY, "InstallLocation"),
        (paths.UNINSTALL_REGISTRY_KEY, "InstallLocation"),
    ):
        value = _read_registry_value(key_path, name)
        if value:
            return Path(value)
    return None


def installed_version() -> str | None:
    """注册表里登记的版本号。"""
    for key_path, name in (
        (paths.APP_REGISTRY_KEY, "Version"),
        (paths.UNINSTALL_REGISTRY_KEY, "DisplayVersion"),
    ):
        value = _read_registry_value(key_path, name)
        if value:
            return value
    return None


def _today() -> str:
    from datetime import date  # noqa: PLC0415

    return date.today().strftime("%Y%m%d")


# --------------------------------------------------------------------- 杂项
def directory_size(root: str | os.PathLike[str], *, limit: int = 200_000) -> int:
    """目录占用字节数（最多统计 ``limit`` 个文件，避免超大目录拖住界面）。"""
    total = 0
    count = 0
    for base, _dirs, files in os.walk(str(root)):
        for name in files:
            try:
                total += (Path(base) / name).stat().st_size
            except OSError:
                continue
            count += 1
            if count >= limit:
                return total
    return total


def free_space(directory: str | os.PathLike[str]) -> int | None:
    """目标磁盘剩余空间（探测失败返回 ``None``）。"""
    import shutil

    target = Path(directory)
    for candidate in (target, *target.parents):
        try:
            if candidate.exists():
                return shutil.disk_usage(candidate).free
        except OSError:  # pragma: no cover
            continue
    return None


def schedule_directory_removal(directory: str | os.PathLike[str], *, delay: int = 2) -> Path | None:
    """安排"等本进程退出后删除整个目录"。

    卸载时安装目录里有一个**正在运行的解释器**（就是卸载器自己），
    Windows 不允许删除正在使用的 exe，所以生成一个临时批处理：
    等几秒 → ``rmdir /s /q`` → 自删。用户看到的是"卸载完成"，
    目录会在后台被清干净。
    """
    if not _windows_only():  # pragma: no cover
        return None
    target = Path(directory)
    try:
        handle, temp_name = tempfile.mkstemp(prefix="v2pv-uninstall-", suffix=".cmd")
        os.close(handle)
        script = Path(temp_name)
        body = (
            "@echo off\r\n"
            f"ping -n {max(int(delay), 1) + 1} 127.0.0.1 >nul\r\n"
            f'rmdir /s /q "{target}"\r\n'
            'del "%~f0"\r\n'
        )
        script.write_text(body, encoding=_SCRIPT_ENCODING, newline="")
    except OSError as exc:
        logger.warning("无法创建清理脚本：%s", exc)
        return None

    flags = 0
    if _windows_only():
        flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0
        )
    try:
        subprocess.Popen(  # noqa: S603 - 执行自己刚写的清理脚本
            ["cmd", "/c", str(script)],
            creationflags=flags,
            close_fds=True,
            cwd=str(Path(tempfile.gettempdir())),
        )
    except OSError as exc:  # pragma: no cover
        logger.warning("启动清理脚本失败：%s", exc)
        return None
    logger.info("已安排后台删除：%s", target)
    return script


def open_directory(directory: str | os.PathLike[str]) -> bool:
    """在资源管理器里打开目录。"""
    target = Path(directory)
    if not target.exists():
        return False
    if sys.platform == "win32":
        try:
            os.startfile(str(target))  # noqa: S606 - 打开目录，参数由调用方给出
            return True
        except OSError as exc:  # pragma: no cover
            logger.warning("打开目录失败：%s", exc)
            return False
    return False


def launch_application(install_dir: str | os.PathLike[str]) -> bool:
    """用内嵌解释器启动主程序（装完之后"立即运行"）。"""
    root = Path(install_dir)
    pythonw = paths.pythonw_exe(root)
    if not pythonw.is_file():
        return False
    command = [str(pythonw), "-m", "video2personvideo", "--gui"]
    try:
        subprocess.Popen(  # noqa: S603 - 启动自己的程序
            command,
            cwd=str(root),
            creationflags=_creation_flags(getattr(subprocess, "DETACHED_PROCESS", 0)),
            close_fds=True,
        )
        return True
    except OSError as exc:  # pragma: no cover
        logger.warning("启动程序失败：%s", exc)
        return False


__all__ = [
    "POWERSHELL_TIMEOUT",
    "create_shortcut",
    "create_shortcuts",
    "desktop_shortcut_path",
    "directory_size",
    "free_space",
    "icon_path",
    "installed_location",
    "installed_version",
    "launch_application",
    "open_directory",
    "register_install",
    "remove_shortcut",
    "remove_shortcuts",
    "run_powershell",
    "schedule_directory_removal",
    "start_menu_folder",
    "start_menu_shortcut_path",
    "start_menu_uninstall_path",
    "unregister_install",
    "write_launcher_scripts",
]
