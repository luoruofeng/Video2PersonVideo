# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置。

用法（在项目根目录执行）：
    .venv\\Scripts\\python.exe -m PyInstaller --noconfirm --clean build/Video2PersonVideo.spec

产物：dist/Video2PersonVideo/Video2PersonVideo.exe
把整个 dist/Video2PersonVideo 文件夹一起发给别人即可（对方无需安装 Python）。
"""

import os
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
    get_package_paths,
)

PROJECT_ROOT = Path(SPECPATH).resolve().parent  # noqa: F821 - SPECPATH 由 PyInstaller 注入
SRC_DIR = PROJECT_ROOT / "src"
ENTRY = PROJECT_ROOT / "scripts" / "entrypoint.py"
ICON = PROJECT_ROOT / "assets" / "app.ico"

APP_NAME = "Video2PersonVideo"
ONE_FILE = os.environ.get("V2PV_ONEFILE", "0") == "1"

# ---------------------------------------------------------------- 资源收集
datas = [
    # 默认配置文件随包分发
    (str(PROJECT_ROOT / "configs" / "default.yaml"), "configs"),
]
datas += collect_data_files("ultralytics")  # YOLO 的 cfg/yaml 等数据文件

# 内置权重（可选）：把 yolo11n.pt 等放到 assets/models/ 即可离线运行
MODELS_DIR = PROJECT_ROOT / "assets" / "models"
if MODELS_DIR.is_dir():
    for weight in sorted(MODELS_DIR.glob("*.pt")):
        datas.append((str(weight), "assets/models"))

binaries = collect_dynamic_libs("ultralytics")

# torchvision >=0.21 改用 "stable ABI" 命名（_C_stable.pyd / image_stable.pyd），
# 而 PyInstaller 自带 hook 只收集 _C.pyd，会漏掉这两个扩展。
# 漏掉的后果是运行时 torchvision/extension.py 静默加载失败，
# 推理时报 `operator torchvision::nms does not exist`（NMS 用不了）。
# 这里显式把 torchvision 目录下的扩展与依赖 DLL 一起打进包里。
try:
    _tv_dir = Path(get_package_paths("torchvision")[1])
    for _pattern in ("*.pyd", "*.dll"):
        for _lib in sorted(_tv_dir.glob(_pattern)):
            binaries.append((str(_lib), "torchvision"))
except Exception as _exc:  # pragma: no cover - 没装 torchvision 时跳过
    print(f"[spec] 收集 torchvision 扩展失败（可忽略）：{_exc}")

hiddenimports = [
    "PIL",
    "PIL.Image",
    "pandas",
    "yaml",
    "tqdm",
    "cv2",
    "torch",
    "torchvision",
    "torchvision.extension",
    "torchvision.ops",
    "torchvision.transforms",
    "ultralytics.models",
    "ultralytics.nn.modules",
    # 本项目模块（GUI 在函数内延迟导入，显式列出更稳）
    "video2personvideo.cli",
    "video2personvideo.config",
    "video2personvideo.core.batch",
    "video2personvideo.core.crop",
    "video2personvideo.core.detector",
    "video2personvideo.core.framing",
    "video2personvideo.core.layout",
    "video2personvideo.core.multi",
    "video2personvideo.core.noperson",
    "video2personvideo.core.pipeline",
    "video2personvideo.core.processor",
    "video2personvideo.core.ratio",
    "video2personvideo.core.smoothing",
    "video2personvideo.core.subject",
    "video2personvideo.core.video_io",
    "video2personvideo.gui.app",
    "video2personvideo.gui.controller",
    "video2personvideo.gui.preview_dialog",
    "video2personvideo.gui.ratio_dialog",
    "video2personvideo.gui.ratio_grid",
    "video2personvideo.gui.theme",
    "video2personvideo.gui.wizard",
    "video2personvideo.gui.pages",
]
hiddenimports += collect_submodules("ultralytics")

# 明确排除用不到的大体积库，减小产物体积
excludes = [
    "PyQt5",
    "PyQt6",
    "PySide2",
    "IPython",
    "jupyter",
    "notebook",
    "matplotlib.tests",
    "tests",
]

a = Analysis(
    [str(ENTRY)],
    pathex=[str(SRC_DIR)],
    binaries=binaries,
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
    "console": True,  # 需要看日志/进度条；想做纯界面版改成 False
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
