"""安装器 / 卸载器的命令行入口。

图形界面是默认形态（双击 ``Video2PersonVideo-Setup.exe`` 走这里），
但下面几个场景必须是命令行的：

* 卸载：开始菜单与「应用和功能」里的入口就是
  ``python -m video2personvideo.installer --uninstall --install-dir <目录>``，
  这样即使用户把最初下载的 Setup.exe 删了也能卸载；
* 静默安装：批量部署 / 自动化测试用 ``--silent``；
* 排错：``--list-backends`` 看这台机器能装哪几套 PyTorch。
"""

from __future__ import annotations

import argparse
import sys

from .. import __version__
from ..utils.downloader import DownloadProgress
from ..utils.gpu_probe import probe_hardware
from ..utils.logger import get_logger, setup_logging
from ..utils.torch_backends import TORCH_BACKENDS, recommend_backend
from ..utils.torch_install import YOLO_WEIGHTS
from . import paths
from .engine import (
    InstallEngine,
    InstallOutcome,
    Reporter,
    mark_uninstalled,
    rollback_install,
    estimate_download_bytes,
    estimate_install_bytes,
)
from .journal import InstallJournal, probe_install
from .options import InstallOptions, default_options
from .stages import Stage

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="Video2PersonVideo-Setup",
        description=(
            "Video2PersonVideo 安装器：把程序、内嵌 Python 运行时、按显卡挑选的 PyTorch "
            "与 YOLO 权重一次性装好，并登记卸载信息。不带参数时打开图形界面。"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--install-dir",
        help="安装目录（默认 %(default)s）",
    )
    parser.add_argument(
        "--backend",
        help=f"PyTorch 构建：{' / '.join(item.key for item in TORCH_BACKENDS)}（默认按显卡自动推荐）",
    )
    parser.add_argument("--python-version", help="内嵌 Python 版本")
    parser.add_argument(
        "--weights",
        help="要下载的 YOLO 权重，逗号分隔（默认 "
        f"{'、'.join(YOLO_WEIGHTS)}；传空字符串表示都不下）",
    )
    parser.add_argument("--no-torch", action="store_true", help="跳过 PyTorch（复用本机已装版本）")
    parser.add_argument("--no-ffmpeg", action="store_true", help="不随包安装 ffmpeg")
    parser.add_argument("--no-desktop-shortcut", action="store_true", help="不创建桌面快捷方式")
    parser.add_argument("--no-start-menu-shortcut", action="store_true", help="不创建开始菜单快捷方式")
    parser.add_argument(
        "--delete-downloads",
        action="store_true",
        help="安装完成后删除下载的安装包（默认保留，便于修复 / 重装）",
    )
    parser.add_argument("--no-launch", action="store_true", help="安装完成后不自动启动程序")

    actions = parser.add_argument_group("动作")
    actions.add_argument("--uninstall", action="store_true", help="卸载已安装的 Video2PersonVideo")
    actions.add_argument(
        "--rollback", action="store_true", help="回滚未完成的安装（删除已写入的文件）"
    )
    actions.add_argument(
        "--repair", action="store_true", help="重新安装（保留已下好的安装包，重跑未完成的步骤）"
    )
    actions.add_argument("--silent", action="store_true", help="命令行静默安装（不打开界面）")
    actions.add_argument(
        "--quiet",
        action="store_true",
        help="不询问、不打开界面（配合 --uninstall / --rollback 使用）",
    )
    actions.add_argument("--yes", action="store_true", help="跳过二次确认")
    actions.add_argument(
        "--keep-cache",
        action="store_true",
        help="卸载时保留已经下载的安装包（默认卸载会一并删除）",
    )
    actions.add_argument("--list-backends", action="store_true", help="列出这台机器可用的 PyTorch 构建")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def options_from_args(args: argparse.Namespace) -> InstallOptions:
    """把命令行参数整理成 :class:`InstallOptions`。"""
    base = default_options(install_dir=args.install_dir or paths.configured_install_dir())
    if args.backend:
        base.backend_key = args.backend
    if args.python_version:
        base.python_version = args.python_version
    if args.no_torch:
        base.install_torch = False
    if args.no_ffmpeg:
        base.install_ffmpeg = False
    if args.no_desktop_shortcut:
        base.desktop_shortcut = False
    if args.no_start_menu_shortcut:
        base.start_menu_shortcut = False
    if args.delete_downloads:
        base.delete_downloads = True
    if args.no_launch:
        base.launch_after_install = False
    if args.weights is not None:
        base.weights = [
            name.strip() for name in args.weights.split(",") if name.strip() in YOLO_WEIGHTS
        ]
    return base


def _console_reporter(*, quiet: bool = False) -> Reporter:
    """把安装进度打到控制台（静默安装 / 排错用）。"""

    def log(message: str) -> None:
        if not quiet:
            print(message, flush=True)

    def stage(stage: Stage, status: str, message: str) -> None:
        if quiet:
            return
        marks = {"running": "..", "done": "[OK]", "skipped": "[--]", "failed": "[!!]"}
        print(f"{marks.get(status, '  ')} {stage.value}: {message}", flush=True)

    def total(fraction: float, text: str) -> None:
        if quiet:
            return
        bar = "#" * int(fraction * 30)
        print(f"\r[{bar:<30}] {fraction * 100:5.1f}% {text[:40]:<40}", end="", flush=True)
        if fraction >= 1.0:
            print(flush=True)

    def download(event: DownloadProgress) -> None:
        return

    return Reporter(on_log=log, on_stage=stage, on_total_progress=total, on_download=download)


def run_install(options: InstallOptions, *, reporter: Reporter | None = None) -> InstallOutcome:
    """执行一次安装（不依赖界面）。"""
    engine = InstallEngine(options, reporter=reporter or Reporter())
    return engine.run()


def run_headless_install(args: argparse.Namespace) -> int:
    """``--silent`` 时的安装路径。"""
    setup_logging("INFO")
    options = options_from_args(args)

    if args.repair:
        journal = InstallJournal.load_for(options.resolved_install_dir())
        if journal is not None:
            journal.reset_all()
            journal.save()
            print(f"已重置安装状态，将重新执行全部步骤：{options.resolved_install_dir()}")

    reporter = _console_reporter(quiet=args.quiet)
    print(f"安装目录：{options.resolved_install_dir()}")
    print(f"预计下载：{paths.format_size(estimate_download_bytes(options))}")
    print(f"预计占用：{paths.format_size(estimate_install_bytes(options))}")

    outcome = run_install(options, reporter=reporter)
    if outcome.cancelled:
        print("\n安装已暂停（下次可继续，已下载内容不会重复下载）。")
        return 130
    if not outcome.ok:
        print(f"\n安装失败：{outcome.message}", file=sys.stderr)
        return 1
    print(f"\n安装完成：{options.resolved_install_dir()}")
    from . import windows  # noqa: PLC0415 - 只在需要时导入

    if options.launch_after_install:
        windows.launch_application(options.resolved_install_dir())
    return 0


def run_headless_uninstall(args: argparse.Namespace) -> int:
    """``--quiet`` / ``--yes`` 时的卸载路径（不打开界面）。"""
    setup_logging("INFO")
    target = args.install_dir or paths.configured_install_dir() or paths.default_install_dir()
    reporter = _console_reporter(quiet=False)
    mark_uninstalled(target)
    report = rollback_install(
        target, remove_cache=not args.keep_cache, reporter=reporter, schedule_delete=True
    )
    print(report.message)
    return 0


def list_backends() -> int:
    """列出这台机器能装哪几套 PyTorch（标出推荐项与不可用原因）。"""
    profile = probe_hardware()
    recommendation = recommend_backend(profile)
    for title, value in profile.summary_lines():
        print(f"{title}：{value}")
    print(f"\n推荐：{recommendation.backend.display} —— {recommendation.reason}")
    if recommendation.warnings:
        for warning in recommendation.warnings:
            print(f"提示：{warning}")

    by_key = {option.backend.key: option for option in recommendation.options}
    print("\n全部可选项：")
    for item in TORCH_BACKENDS:
        option = by_key.get(item.key)
        if option is None:
            mark, note = "?", "当前系统不提供该构建"
        elif option.usable:
            mark, note = "*", ""
        else:
            mark, note = "-", option.reason
        recommended = "（推荐）" if item.key == recommendation.backend.key else ""
        print(f"  [{mark}] {item.key:<10} {item.label}{recommended} {note}")
    print("\n（* = 可用，- = 不可用，? = 当前系统不提供）")
    return 0


def main(argv: list[str] | None = None) -> int:
    """命令行主函数（图形界面入口在 :mod:`video2personvideo.installer.app`）。"""
    args = parse_args(argv)

    if args.list_backends:
        return list_backends()

    if args.uninstall or args.rollback:
        if args.quiet or args.yes:
            return run_headless_uninstall(args)
        from .gui.uninstall_dialog import run_uninstall_gui  # noqa: PLC0415

        target = args.install_dir or paths.configured_install_dir() or paths.default_install_dir()
        return 0 if run_uninstall_gui(target, quiet=args.quiet) else 1

    if args.silent:
        return run_headless_install(args)

    from .gui.wizard import run_installer_gui  # noqa: PLC0415

    options = options_from_args(args)
    return run_installer_gui(options, repair=args.repair)


def describe_install_state(install_dir=None) -> str:
    """给排错用的一句话（``--status`` 之类场景）。"""
    probe = probe_install(install_dir)
    return probe.describe()


def backend_options() -> list[str]:
    """全部可用的构建 key（供界面下拉）。"""
    return [item.key for item in TORCH_BACKENDS]


__all__ = [
    "backend_options",
    "build_parser",
    "describe_install_state",
    "list_backends",
    "main",
    "options_from_args",
    "parse_args",
    "run_headless_install",
    "run_headless_uninstall",
    "run_install",
]
