"""命令行入口：``v2pv`` / ``python -m video2personvideo``。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .config import AppConfig, load_config
from .core.batch import discover_videos, run_batch
from .core.layout import MAX_WINDOWS
from .core.processor import process_video
from .core.ratio import preset_names
from .core.smoothing import CAMERA_PRESETS, NO_PERSON_MODES
from .utils.app_paths import register_bundled_tools
from .utils.device import environment_report, format_report
from .utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="v2pv",
        description=(
            "视频人像构图工具：用 YOLO 找到画面中的主要人物，把视频裁剪成指定长宽比，"
            "输出以人物为中心的近似「主播半身」构图。"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "示例：\n"
            "  v2pv -i data/input/demo.mp4\n"
            "  v2pv -i demo.mp4 --ratio 9:16 --target-size 1080x1920\n"
            "  v2pv -i data/input --recursive --ratio 16:9 --overwrite\n"
            "  v2pv -i demo.mp4 --no-crop --save-frames frames\n"
            "  v2pv --check\n"
            "  v2pv --gui\n\n"
            f"预置比例：{'、'.join(preset_names())}"
        ),
    )
    parser.add_argument("-i", "--source", type=Path, help="输入视频文件或视频文件夹")
    parser.add_argument(
        "-o", "--output", type=Path, help="输出视频路径（默认 <原视频同目录>/<原名>_person<原后缀>）"
    )
    parser.add_argument("-c", "--config", type=Path, help="YAML 配置文件路径")
    parser.add_argument(
        "-m", "--model", help="模型名或权重文件路径，如 yolo11n.pt / yolo11s.pt"
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        help="只保留的类别索引，默认 [0]（person）；传 -1 表示输出所有类别",
    )
    parser.add_argument("--conf", type=float, help="置信度阈值")
    parser.add_argument("--iou", type=float, help="NMS 的 IoU 阈值")
    parser.add_argument("--imgsz", type=int, help="推理分辨率（建议 32 的倍数）")
    parser.add_argument("--max-det", type=int, dest="max_det", help="单帧最多保留的目标数")
    parser.add_argument("--device", help="推理设备：auto / cpu / cuda:0 / mps")

    crop_group = parser.add_argument_group("裁剪构图")
    crop_group.add_argument(
        "--ratio", help=f"目标长宽比，如 9:16 / 4:5 / 1080x1920（预置：{'、'.join(preset_names())}）"
    )
    crop_group.add_argument(
        "--target-size", dest="target_size", help="目标像素尺寸，如 1080x1920（宽高需为偶数）"
    )
    crop_group.add_argument(
        "--crop",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="启用裁剪构图（--no-crop = 保留旧的逐帧画框标注模式）",
    )
    crop_group.add_argument(
        "--annotate", action="store_true", default=None, help="在输出帧上绘制检测框"
    )
    crop_group.add_argument("--detect-interval", type=int, dest="detect_interval", help="每 N 帧推理一次，中间帧线性插值")
    crop_group.add_argument("--infer-batch", type=int, dest="infer_batch", help="一次推理塞几帧（批量推理提速，默认 1）")
    crop_group.add_argument(
        "--no-keypoints",
        action="store_true",
        dest="no_keypoints",
        help="即使模型自带姿态关键点也不用于精修构图",
    )
    crop_group.add_argument(
        "--camera-follow",
        choices=list(CAMERA_PRESETS),
        help="镜头跟随档位：锁定 / 舒缓 / 标准 / 跟手（默认跟手），一次设置一组平滑参数（更稳 = 更不易看晕）",
    )
    crop_group.add_argument(
        "--smoothing-alpha",
        type=float,
        dest="smoothing_alpha",
        help="取景框平移平滑系数 (0,1]，越大越跟手、越小越稳",
    )
    crop_group.add_argument(
        "--deadzone",
        type=float,
        dest="smoothing_deadzone",
        help="自由活动区（相对框长）：人物在框内走动小于它时镜头完全不动，越大越稳",
    )
    crop_group.add_argument(
        "--zoom-alpha",
        type=float,
        dest="smoothing_zoom_alpha",
        help="缩放（推拉）平滑系数，越小越不易晕",
    )
    crop_group.add_argument(
        "--zoom-deadzone",
        type=float,
        dest="smoothing_zoom_deadzone",
        help="缩放死区（相对框高），越大越少推拉",
    )
    crop_group.add_argument(
        "--pan-speed",
        type=float,
        dest="smoothing_max_speed",
        help="镜头平移速度上限（相对框长/帧），防止甩镜；0 = 不限制",
    )
    crop_group.add_argument(
        "--hold-frames",
        type=int,
        dest="hold_frames",
        help="丢失目标后保持取景框的帧数（分屏时也是「漏检的人保留窗口 / 撤窗口」的帧数）",
    )
    crop_group.add_argument(
        "--no-person-mode",
        choices=list(NO_PERSON_MODES),
        dest="no_person_mode",
        help=(
            "画面里没有主要人物时怎么显示（默认 fit）："
            "fit = 全画面适配（整幅画面缩小完整放入，镜头不动、一帧看全）；"
            "tiles = 全景 + 特写（整幅画面收进上方通栏，下面给远景小人物开半身特写窗口）；"
            "scan = 空镜巡视（取景框缓慢扫过整幅画面）；"
            "center = 只取画面正中"
        ),
    )
    crop_group.add_argument(
        "--no-person-seconds",
        type=float,
        dest="no_person_seconds",
        help="从人物取景框过渡到无人物显示的时长（秒），越大越舒缓；0 = 立即切换",
    )
    crop_group.add_argument(
        "--blur-background",
        action=argparse.BooleanOptionalAction,
        default=None,
        dest="no_person_blur",
        help="无人物画面的底色用「同一帧的模糊放大版」而不是纯色（--no-blur-background 关闭）",
    )
    crop_group.add_argument(
        "--no-person-tiles-max",
        type=int,
        dest="no_person_tiles_max",
        help="「全景 + 特写」最多给几个远景小人物开特写窗口（1~4，默认 2）",
    )
    crop_group.add_argument(
        "--no-person-secondary-ratio",
        type=float,
        dest="no_person_secondary_ratio",
        help="「全景 + 特写」里小人物的高度下限（占画面比例）：低于它的不给特写（默认 0.12）",
    )
    crop_group.add_argument(
        "--scan-seconds",
        type=float,
        dest="smoothing_scan_seconds",
        help=(
            "「空镜巡视」的单程时长（秒）：取景框缓慢平移扫过整幅画面，"
            "只在 --no-person-mode scan 时用得上；0 = 关闭巡视、退回画面居中"
        ),
    )
    crop_group.add_argument("--min-person-ratio", type=float, dest="min_person_height_ratio", help="人物过小阈值（bbox 高 / 画面高）")
    crop_group.add_argument(
        "--min-sharpness",
        type=float,
        dest="min_person_sharpness",
        help=(
            "主要人物的清晰度下限（0~1 比率，与「够大」并列）：低于它的人算背景人物"
            "（被虚化的路人 / 远处海报里的人），默认 0.30；0 = 关闭清晰度判定"
        ),
    )
    crop_group.add_argument("--headroom", type=float, help="半身构图的头顶留白比例")
    crop_group.add_argument(
        "--speaker-tracking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "多个主要人物时优先对准正在说话的人"
            "（--no-speaker-tracking 关闭；需要 ffmpeg 抽音轨，缺失时自动跳过）"
        ),
    )
    crop_group.add_argument(
        "--speaker-weight",
        type=float,
        dest="speaker_weight",
        help="说话人偏置（≥1.0 时正在说话的人必定胜出；越小越接近原主角打分规则）",
    )
    crop_group.add_argument(
        "--speaker-switch-margin",
        type=float,
        dest="speaker_switch_margin",
        help="换人所需的最低领先分差，越大越不容易换人",
    )
    crop_group.add_argument(
        "--speaker-switch-hold",
        type=int,
        dest="speaker_switch_hold",
        help="换人前挑战者需连续领先的判定回合数，越大镜头越稳",
    )
    crop_group.add_argument(
        "--speaker-window",
        type=int,
        dest="speaker_window_frames",
        help="语音-嘴动的相关性滑窗长度（帧），约 1~1.5 秒",
    )
    crop_group.add_argument(
        "--multi-person",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "多人分屏：画面里有多个主要人物时，每人一个上半身小窗口同时显示"
            "（--no-multi-person 关闭 = 整段视频同时只显示一个人；默认开启）"
        ),
    )
    crop_group.add_argument(
        "--multi-person-max",
        type=int,
        dest="multi_person_max",
        help=f"多人分屏最多同时显示几个人（2~{MAX_WINDOWS}，超出的按主角 / 说话人优先级取舍）",
    )
    crop_group.add_argument(
        "--multi-person-order",
        choices=["spatial", "score"],
        dest="multi_person_order",
        help=(
            "多人分屏的人物排列（决定新人入座顺序）：spatial = 先说话人、其余按画面位置；"
            "score = 完全按主角打分。默认还开着 --multi-person-seat-lock，人物入座后不再换窗口"
        ),
    )
    crop_group.add_argument(
        "--layout-config",
        type=Path,
        dest="multi_person_layout_file",
        help="多人分屏的布局表配置（默认 configs/multi_person_layout.yaml，缺失时用内置布局）",
    )
    crop_group.add_argument(
        "--multi-person-stable-camera",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "分屏小窗口用「稳镜头」：平移 / 缩放按秒封顶、死区更大、换人就位不横扫"
            "（--no-multi-person-stable-camera 关闭 = 沿用镜头跟随档位，小窗口里会比较急；默认开启）"
        ),
    )
    crop_group.add_argument(
        "--multi-person-pan-speed",
        type=float,
        dest="multi_person_pan_speed",
        help="稳镜头的平移上限：每秒最多平移「窗口自身的几倍」（0.05~5.0，越小越稳）",
    )
    crop_group.add_argument(
        "--multi-person-seat-lock",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "分屏窗口「认人」：人物一旦入座就固定待在自己的窗口，"
            "不因说话人变化 / 位置微动而跟别人换窗口"
            "（--no-multi-person-seat-lock 关闭 = 每个关键帧按说话人 / 位置重排；默认开启）"
        ),
    )

    output_group = parser.add_argument_group("输出")
    output_group.add_argument("--no-audio", action="store_true", help="不保留原视频音轨")
    output_group.add_argument("--reencode", action="store_true", help="用 ffmpeg 重编码为 H.264")
    output_group.add_argument(
        "--normalize-vfr",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="源视频为可变帧率时先归一时间轴，避免画面忽快忽慢（--no-normalize-vfr 关闭）",
    )
    output_group.add_argument("--crf", type=int, help="重编码质量，越小画质越好（0-51）")
    output_group.add_argument(
        "--size-guard",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="输出体积超过原视频时自动压缩（--no-size-guard 关闭）",
    )
    output_group.add_argument(
        "--max-size-ratio",
        type=float,
        dest="max_size_ratio",
        help="输出体积上限相对原视频的倍数（1.0 = 不超过原视频）",
    )
    output_group.add_argument(
        "--min-bitrate",
        type=int,
        dest="min_video_bitrate_kbps",
        help="自动压缩时的视频码率下限（kbps），防止压得过糊",
    )
    output_group.add_argument(
        "--output-suffix", dest="output_suffix", help="强制输出后缀，如 .mp4（默认沿用原后缀）"
    )
    output_group.add_argument(
        "--same-dir",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="输出到原视频同目录（--no-same-dir = 统一 data/output/）",
    )
    output_group.add_argument("--overwrite", action="store_true", default=None, help="同名输出存在时覆盖（默认跳过）")
    output_group.add_argument("--metadata", action="store_true", default=None, help="写入比例/时间/模型等元数据标记")
    output_group.add_argument("--write-preview", action="store_true", dest="write_preview", default=None, help="额外输出首帧构图预览图")
    output_group.add_argument(
        "--save-frames", type=Path, dest="save_frames_dir", help="把含人的帧另存为图片的目录"
    )
    output_group.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="输入为文件夹时递归遍历子目录",
    )

    parser.add_argument("--show", action="store_true", help="处理时弹窗实时预览（按 q 提前结束）")
    parser.add_argument("--log-level", help="日志级别：DEBUG / INFO / WARNING / ERROR")
    parser.add_argument("--check", action="store_true", help="打印环境自检信息（可与处理同时使用）")
    parser.add_argument("--gui", action="store_true", help="启动图形界面客户端")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    """命令行主函数，返回进程退出码。"""
    # 安装器会把 ffmpeg 之类的随包可执行文件放在安装目录下；先挂到 PATH 上，
    # 这样 shutil.which / 子进程调用都能直接命中，不必再去改各调用点。
    register_bundled_tools()

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.gui:
        return _run_gui()

    try:
        cfg = load_config(args.config).with_overrides(**_override_kwargs(args))
    except (FileNotFoundError, ValueError) as exc:
        print(f"配置加载失败：{exc}", file=sys.stderr)
        return 2

    setup_logging(args.log_level or cfg.log_level)

    if args.check:
        print(format_report(environment_report()))

    if cfg.source is None:
        if args.check:
            return 0
        parser.print_help()
        print(
            "\n错误：请通过 -i/--source 指定输入视频或文件夹，或使用 --gui 打开图形界面。",
            file=sys.stderr,
        )
        return 2

    try:
        if Path(cfg.source).is_dir():
            return _run_batch(cfg)
        result = process_video(cfg, show_progress=True)
    except KeyboardInterrupt:
        print("\n已手动中断。", file=sys.stderr)
        return 130
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001 - 兜底，禁止裸异常崩溃
        logger.exception("处理过程中出现未预期的错误")
        print(f"处理失败：{exc}", file=sys.stderr)
        return 1

    print()
    print(result.summary())
    return 0


def _run_batch(cfg: AppConfig) -> int:
    """文件夹输入时的批量处理路径。"""
    sources = discover_videos(cfg.source, recursive=cfg.batch_recursive)
    if not sources:
        print("未在文件夹中发现可处理的视频（已跳过 *_person.* 产物）。", file=sys.stderr)
        return 1

    print(f"共发现 {len(sources)} 个视频，开始批量处理…")
    try:
        result = run_batch(sources, cfg, show_progress=True)
    except KeyboardInterrupt:
        print("\n已手动中断。", file=sys.stderr)
        return 130
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
        return 1

    print()
    print(result.summary())
    return 1 if result.failed else 0


def _smoothing_kwargs(args: argparse.Namespace) -> dict[str, float | None]:
    """镜头跟随档位 + 逐项微调：显式给出的逐项参数优先于档位预设。

    没给档位、也没给逐项参数时全部返回 ``None``（交给配置文件 / 内置默认值）。
    """
    preset = CAMERA_PRESETS.get(args.camera_follow or "")
    if preset is None:
        return {
            "smoothing_alpha": args.smoothing_alpha,
            "smoothing_deadzone": args.smoothing_deadzone,
            "smoothing_zoom_alpha": args.smoothing_zoom_alpha,
            "smoothing_zoom_deadzone": args.smoothing_zoom_deadzone,
            "smoothing_max_speed": args.smoothing_max_speed,
        }
    params = preset.params
    return {
        "smoothing_alpha": args.smoothing_alpha if args.smoothing_alpha is not None else params.alpha,
        "smoothing_deadzone": (
            args.smoothing_deadzone if args.smoothing_deadzone is not None else params.deadzone
        ),
        "smoothing_zoom_alpha": (
            args.smoothing_zoom_alpha if args.smoothing_zoom_alpha is not None else params.zoom_alpha
        ),
        "smoothing_zoom_deadzone": (
            args.smoothing_zoom_deadzone
            if args.smoothing_zoom_deadzone is not None
            else params.zoom_deadzone
        ),
        "smoothing_max_speed": (
            args.smoothing_max_speed if args.smoothing_max_speed is not None else params.max_speed
        ),
    }


def _override_kwargs(args: argparse.Namespace) -> dict[str, object]:
    """把命令行参数整理成 :meth:`AppConfig.with_overrides` 需要的字段。"""
    classes = None
    if args.classes is not None:
        classes = [] if -1 in args.classes else [int(c) for c in args.classes if c >= 0]

    target_width, target_height = _parse_target_size(args.target_size)

    return {
        "source": args.source,
        "output": args.output,
        "model": args.model,
        "classes": classes,
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "max_det": args.max_det,
        "device": args.device,
        "save_audio": False if args.no_audio else None,
        "reencode": True if args.reencode else None,
        "normalize_vfr": args.normalize_vfr,
        "crf": args.crf,
        "size_guard": args.size_guard,
        "max_size_ratio": args.max_size_ratio,
        "min_video_bitrate_kbps": args.min_video_bitrate_kbps,
        "save_frames_dir": args.save_frames_dir,
        "show": True if args.show else None,
        "log_level": args.log_level,
        "crop": args.crop,
        "annotate": args.annotate,
        "aspect_ratio": args.ratio,
        "target_width": target_width,
        "target_height": target_height,
        "detect_interval": args.detect_interval,
        "infer_batch": args.infer_batch,
        # 只有显式传 --no-keypoints 时才关闭（默认交给模型自身能力决定）
        "use_keypoints": False if args.no_keypoints else None,
        **_smoothing_kwargs(args),
        "hold_frames": args.hold_frames,
        "smoothing_scan_seconds": args.smoothing_scan_seconds,
        "no_person_mode": args.no_person_mode,
        "no_person_seconds": args.no_person_seconds,
        "no_person_blur": args.no_person_blur,
        "no_person_tiles_max": args.no_person_tiles_max,
        "no_person_secondary_ratio": args.no_person_secondary_ratio,
        "min_person_height_ratio": args.min_person_height_ratio,
        "min_person_sharpness": args.min_person_sharpness,
        "headroom": args.headroom,
        "speaker_tracking": args.speaker_tracking,
        "speaker_weight": args.speaker_weight,
        "speaker_switch_margin": args.speaker_switch_margin,
        "speaker_switch_hold": args.speaker_switch_hold,
        "speaker_window_frames": args.speaker_window_frames,
        "multi_person": args.multi_person,
        "multi_person_max": args.multi_person_max,
        "multi_person_order": args.multi_person_order,
        "multi_person_layout_file": args.multi_person_layout_file,
        "multi_person_stable_camera": args.multi_person_stable_camera,
        "multi_person_pan_speed": args.multi_person_pan_speed,
        "multi_person_seat_lock": args.multi_person_seat_lock,
        "output_suffix": args.output_suffix,
        "output_same_dir": args.same_dir,
        "overwrite": args.overwrite,
        "write_metadata": args.metadata,
        "write_preview": args.write_preview,
        "batch_recursive": args.recursive,
    }


def _parse_target_size(text: str | None) -> tuple[int | None, int | None]:
    """把 ``1080x1920`` 解析成 ``(1080, 1920)``；未给出时返回 ``(None, None)``。"""
    if not text:
        return None, None
    cleaned = text.strip().lower().replace("*", "x").replace("×", "x")
    parts = [item for item in cleaned.split("x") if item.strip()]
    if len(parts) != 2:
        raise ValueError(f"--target-size 格式应为 宽x高（如 1080x1920），当前为 {text!r}")
    return int(parts[0]), int(parts[1])


def _run_gui() -> int:
    """启动图形界面（未安装 PySide6 时给出安装提示）。"""
    try:
        from .gui.app import main as gui_main
    except ImportError as exc:
        print(
            f"无法启动图形界面：{exc}\n"
            "请先安装 GUI 依赖：pip install -r requirements-gui.txt",
            file=sys.stderr,
        )
        return 1
    return gui_main([])


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
