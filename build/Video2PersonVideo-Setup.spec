# -*- mode: python ; coding: utf-8 -*-
"""安装器（``Video2PersonVideo-Setup.exe``）的 PyInstaller 打包配置。

这个 exe 只做一件事：把主程序**装**到用户机器上。它自己**不含**
PyTorch / ultralytics / OpenCV —— 那些由安装过程按显卡下载并装进目标目录里的
内嵌 Python 环境，所以 Setup.exe 只有几十 MB，而不是近一个 GB。

用法（在项目根目录执行，推荐用 scripts\\build_installer.ps1 一键完成）：

    .venv\\Scripts\\python.exe -m PyInstaller --noconfirm --clean build/Video2PersonVideo-Setup.spec

产物：dist/Video2PersonVideo-Setup.exe
把这个 exe 发给用户，双击即可安装（不需要管理员权限、不需要预装 Python）。
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(SPECPATH).resolve().parent  # noqa: F821 - SPECPATH 由 PyInstaller 注入
SRC_DIR = PROJECT_ROOT / "src"
ENTRY = PROJECT_ROOT / "scripts" / "installer_entrypoint.py"
ICON = PROJECT_ROOT / "assets" / "app.ico"
WHEEL_DIR = PROJECT_ROOT / "build" / "installer_payload"

APP_NAME = "Video2PersonVideo-Setup"
#: 单文件（默认，便于分发）还是目录版（启动更快，便于排错）
ONE_FILE = os.environ.get("V2PV_SETUP_ONEFILE", "1") == "1"
#: 需要控制台时（调试用）：set V2PV_INSTALLER_CONSOLE=1
WITH_CONSOLE = os.environ.get("V2PV_INSTALLER_CONSOLE", "0") == "1"

# ------------------------------------------------------------------ payload
# 安装器要装的东西：程序本体（源码 + 可选 wheel）、默认配置、依赖清单、文档、图标。
# 统一塞进 exe 里的 ``payload/`` 目录，安装时由 installer.payload 读取。
datas: list[tuple[str, str]] = []


def _add(source: Path, target: str) -> None:
    if source.is_file():
        datas.append((str(source), target))
    elif source.is_dir():
        datas.append((str(source), target))


for name in ("pyproject.toml", "README.md", "LICENSE", "requirements.txt", "requirements-gui.txt"):
    _add(PROJECT_ROOT / name, "payload")

# 源码树：让安装器在"没有随包 wheel"时也能直接 pip install 这份源码
_add(PROJECT_ROOT / "configs", "payload/configs")
_add(PROJECT_ROOT / "assets", "payload/assets")
_add(SRC_DIR / "video2personvideo", "payload/src/video2personvideo")

# 打包前用 `pip wheel` 产出的程序本体 wheel（有就用它装，快且不依赖 setuptools）
if WHEEL_DIR.is_dir():
    for wheel in sorted(WHEEL_DIR.glob("*.whl")):
        datas.append((str(wheel), "payload/wheels"))

# ------------------------------------------------------------------ 依赖
hiddenimports = [
    # 安装器本体的全部模块（GUI 在函数里延迟导入，显式列出更稳）
    "video2personvideo.installer.app",
    "video2personvideo.installer.cli",
    "video2personvideo.installer.engine",
    "video2personvideo.installer.journal",
    "video2personvideo.installer.options",
    "video2personvideo.installer.paths",
    "video2personvideo.installer.payload",
    "video2personvideo.installer.runtime",
    "video2personvideo.installer.stages",
    "video2personvideo.installer.windows",
    "video2personvideo.installer.gui.pages",
    "video2personvideo.installer.gui.uninstall_dialog",
    "video2personvideo.installer.gui.wizard",
    "video2personvideo.installer.gui.worker",
    # 复用的工具模块（硬件自检 / 构建表 / 下载器 / 路径）
    "video2personvideo.utils.app_paths",
    "video2personvideo.utils.downloader",
    "video2personvideo.utils.gpu_probe",
    "video2personvideo.utils.logger",
    "video2personvideo.utils.torch_backends",
    "video2personvideo.utils.torch_install",
    # 界面主题与配置常量
    "video2personvideo.gui.theme",
    "video2personvideo.config",
    "winreg",
]

# 明显用不到的大件全部排除：安装器不该把 PyTorch / YOLO / OpenCV 也打进去
excludes = [
    "torch",
    "torchvision",
    "torchaudio",
    "ultralytics",
    "cv2",
    "numpy",
    "pandas",
    "matplotlib",
    "scipy",
    "PIL",
    "IPython",
    "jupyter",
    "notebook",
    "PyQt5",
    "PyQt6",
    "PySide2",
    "tests",
]

a = Analysis(
    [str(ENTRY)],
    pathex=[str(SRC_DIR)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe_kwargs = {
    "name": APP_NAME,
    "debug": False,
    "bootloader_ignore_signals": False,
    "strip": False,
    "upx": False,
    # 安装向导是图形界面：默认不要黑窗口；需要看输出时用 V2PV_INSTALLER_CONSOLE=1
    "console": WITH_CONSOLE,
    "icon": str(ICON) if ICON.exists() else None,
}

if ONE_FILE:
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.datas,
        [],
        runtime_tmpdir=None,
        **exe_kwargs,
    )
else:
    exe = EXE(pyz, a.scripts, [], exclude_binaries=True, **exe_kwargs)
    coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=False,
        name=APP_NAME,
    )
