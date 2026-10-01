"""配置定义：默认值、YAML 加载、命令行覆盖。

这里同时承载 M0 的需求决策（输出位置 / 命名 / 分辨率 / 兜底 / 主角规则），
所有可调参数集中在此，``core`` 与 ``gui`` 都从这里取默认值。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import yaml

from .core.framing import DEFAULT_FRAMING_PARAMS, FramingParams
from .core.layout import (
    MAX_WINDOWS,
    MultiPersonPolicy,
    load_multi_person_policy,
)
from .core.multi import WindowCamera
from .core.noperson import (
    DEFAULT_SECONDARY_PERSON_RATIO,
    DEFAULT_TILES_MAX,
    NoPersonDisplay,
)
from .core.ratio import DEFAULT_RATIO_NAME, AspectRatio, resolve_ratio
from .core.smoothing import (
    DEFAULT_NO_PERSON_SECONDS,
    DEFAULT_WINDOW_PAN_SPEED,
    NO_PERSON_MODES,
    SmoothingParams,
)
from .core.speaker import DEFAULT_SPEAKER_PARAMS, SpeakerParams
from .core.subject import DEFAULT_MIN_PERSON_SHARPNESS, DEFAULT_WEIGHTS, SubjectWeights
from .utils.logger import get_logger

logger = get_logger(__name__)

#: COCO 数据集中 ``person`` 的类别索引
PERSON_CLASS_ID = 0

#: 默认模型：n 权重体积小、速度快，适合先跑通流程
DEFAULT_MODEL = "yolo11n.pt"

#: 仓库自带的默认配置文件
DEFAULT_CONFIG_PATH = Path("configs/default.yaml")

#: 统一输出目录（``output_same_dir=False`` 时使用）
DEFAULT_OUTPUT_DIR = Path("data") / "output"

#: 输出文件名后缀标记
PERSON_SUFFIX = "_person"

_PATH_FIELDS = ("source", "output", "save_frames_dir", "multi_person_layout_file")
_NULL_TOKENS = {"", "auto", "none", "null", "~"}

_INT_FIELDS = (
    "imgsz",
    "max_det",
    "crf",
    "min_video_bitrate_kbps",
    "target_width",
    "target_height",
    "min_box_px",
    "hold_frames",
    "detect_interval",
    "infer_batch",
    "speaker_window_frames",
    "speaker_switch_hold",
    "multi_person_max",
    "no_person_tiles_max",
)
_FLOAT_FIELDS = (
    "conf",
    "iou",
    "min_person_height_ratio",
    "min_person_sharpness",
    "closeup_ratio",
    "halfbody_ratio",
    "closeup_fill",
    "halfbody_fill",
    "fullbody_fill",
    "incomplete_fill",
    "headroom",
    "smoothing_alpha",
    "smoothing_deadzone",
    "smoothing_zoom_alpha",
    "smoothing_zoom_deadzone",
    "smoothing_max_speed",
    "smoothing_accel",
    "smoothing_scan_seconds",
    "no_person_seconds",
    "no_person_secondary_ratio",
    "max_size_ratio",
    "subject_area_weight",
    "subject_center_weight",
    "subject_continuity_weight",
    "subject_confidence_weight",
    "speaker_weight",
    "speaker_switch_margin",
    "speaker_min_mouth_motion",
    "multi_person_pan_speed",
)
_BOOL_FIELDS = (
    "save_audio",
    "reencode",
    "normalize_vfr",
    "size_guard",
    "show",
    "crop",
    "annotate",
    "overwrite",
    "output_same_dir",
    "write_preview",
    "write_metadata",
    "batch_recursive",
    "use_keypoints",
    "speaker_tracking",
    "multi_person",
    "multi_person_stable_camera",
    "multi_person_seat_lock",
    "no_person_blur",
)
_BOOL_TRUE = {"1", "true", "yes", "y", "on", "是", "开"}
_BOOL_FALSE = {"0", "false", "no", "n", "off", "否", "关"}


@dataclass(slots=True)
class AppConfig:
    """一次视频处理任务的全部参数。"""

    source: Path | None = None
    output: Path | None = None
    model: str = DEFAULT_MODEL
    classes: list[int] = field(default_factory=lambda: [PERSON_CLASS_ID])
    conf: float = 0.25
    iou: float = 0.45
    imgsz: int = 640
    max_det: int = 300
    device: str | None = None
    save_audio: bool = True
    reencode: bool = False
    #: 开始处理前先探测源视频的帧率 / 时长信息；若源是可变帧率（VFR，手机录像 /
    #: 录屏很常见），先用 ffmpeg 把画面归一到恒定帧率再逐帧处理，避免输出出现
    #: "开头画面很慢、随后突然变快"（声音正常）的音画不同步。无 ffmpeg 时自动跳过。
    normalize_vfr: bool = True
    #: 重编码质量（reencode 转 H.264 与音画同步校正重编码共用）
    crf: int = 23
    #: 体积守护：输出体积超过「原视频 × max_size_ratio」时自动重压缩到限制内
    size_guard: bool = True
    #: 体积守护的倍数上限（1.0 = 不超过原视频体积）
    max_size_ratio: float = 1.0
    #: 体积守护压缩时的视频码率下限（kbps），防止压到看不出人形
    min_video_bitrate_kbps: int = 250
    save_frames_dir: Path | None = None
    show: bool = False
    log_level: str = "INFO"

    # ------------------------------------------------------ M0：输出策略
    #: 是否启用"裁剪 + 构图"（关闭 = 保留旧的逐帧画框标注模式）
    crop: bool = True
    #: 是否在输出帧上绘制检测框（画框标注模式恒为 True）
    annotate: bool = False
    #: 输出到原视频同目录（False = 统一 data/output/）
    output_same_dir: bool = True
    #: 强制输出后缀（如 ".mp4"）；null = 沿用原视频后缀
    output_suffix: str | None = None
    #: 同名输出已存在时是否覆盖（False = 跳过并在日志标注）
    overwrite: bool = False
    #: 是否写入元数据标记（比例 / 处理时间 / 模型版本）
    write_metadata: bool = False
    #: 是否额外输出一张首帧构图预览图 ``<名称>_preview.jpg``
    write_preview: bool = False

    # ------------------------------------------------- M1-1：比例与像素
    #: 目标长宽比：支持 "9:16" / "3:5" / "1080x1920" / "1.7778"
    aspect_ratio: str = DEFAULT_RATIO_NAME
    #: 自定义目标像素；两者同时给出才生效，否则按短边 1080 推导
    target_width: int | None = None
    target_height: int | None = None

    # ------------------------------------------------- M1-2：主角选择
    #: 人物过小阈值：bbox 高度 < 画面高度该比例时视为无人物。
    #: 默认 0.73 只把"画面里占比很大的人"当主要人物，背景里的路人 / 小人被忽略。
    min_person_height_ratio: float = 0.73
    #: 人物清晰度下限（0~1 的比率，算法见 core/sharpness）：低于它的人算"背景人物" ——
    #: 被镜头虚化的路人、远处海报 / 屏幕里的人即便"看着很大"也不会被当成主要人物。
    #: 与"够大"是并列条件：两者都达标才算主要人物；0 = 关闭清晰度判定。
    min_person_sharpness: float = DEFAULT_MIN_PERSON_SHARPNESS
    subject_area_weight: float = DEFAULT_WEIGHTS.area
    subject_center_weight: float = DEFAULT_WEIGHTS.center
    subject_continuity_weight: float = DEFAULT_WEIGHTS.continuity
    subject_confidence_weight: float = DEFAULT_WEIGHTS.confidence

    # --------------------------------- 说话人跟随（多人物时对准正在说话的人）
    #: 是否启用"跟随正在说话的人"。仅当画面里有**两个及以上主要人物**且音轨里
    #: 有人在说话时才会真正介入，其余情况与旧行为完全一致；需要 ffmpeg 抽音轨，
    #: 缺失时安静跳过（YOLO 本身判断不了谁在说话，见 core/speaker.py 的说明）。
    speaker_tracking: bool = True
    #: 说话人偏置（与主角打分同量纲，0~1）：≥1.0 时"正在说话的人"必定胜出，
    #: 更小的值则是"软优先"，越小越接近原来的主角打分规则
    speaker_weight: float = 1.8
    #: 语音-嘴动的相关性滑窗（帧），约 1~1.5 秒
    speaker_window_frames: int = 45
    #: 换人所需的最低领先分差
    speaker_switch_margin: float = 0.15
    #: 换人前挑战者需连续领先的判定回合数（每次判定 = 一个关键帧）
    speaker_switch_hold: int = 8
    #: 嘴动强度的绝对下限：低于它认为"读不出嘴型"，不参与判定
    speaker_min_mouth_motion: float = 0.004

    # ------------------------------------------------- M1-8：多人分屏（多个上半身窗口）
    #: 画面里出现**两个及以上主要人物**时，每人给一个上半身小窗口同时显示，
    #: 而不是只跟一个人。窗口怎么排由布局配置文件决定（见 multi_person_layout_file）。
    #: 关掉它 = 回到"整段视频同时只显示一个人"的旧行为；默认开启。
    #: 只有出现多个主要人物时才生效：单人画面、以及过滤掉的背景小人都不受影响。
    multi_person: bool = True
    #: 最多同时显示几个主要人物（超出的按主角打分 / 说话人优先级取舍）
    multi_person_max: int = 4
    #: 人物排列顺序：``spatial`` = 先说话人，其余按画面位置（左→右、上→下）；
    #: ``score`` = 完全按主角打分（谁最抢镜谁坐 1 号窗口）
    multi_person_order: str = "spatial"
    #: 多人布局表（YAML）；``None`` = 用默认路径 configs/multi_person_layout.yaml，
    #: 找不到该文件时使用内置默认布局，不会影响正常出片
    multi_person_layout_file: Path | None = None
    #: 分屏窗口是否用"稳镜头"：每个小窗口只有一格大、放大倍数高，整幅画面里
    #: "跟得紧"的镜头放到小窗口里就变成了飞快地平移 / 推拉（看着像快进）。
    #: 打开后窗口镜头改为"按秒封顶 + 死区更大 + 更柔和"，换人时直接切镜头不横扫。
    multi_person_stable_camera: bool = True
    #: 稳镜头的平移上限：**每秒最多平移"窗口自身的几倍"**（越小越稳，0.1~2.0）
    multi_person_pan_speed: float = DEFAULT_WINDOW_PAN_SPEED
    #: 分屏窗口是否"认人"：人物一旦入座就固定待在自己的窗口，不因为说话人变化 /
    #: 位置微动而跟别人换窗口（关掉 = 每个关键帧按说话人 / 空间顺序重排窗口）
    multi_person_seat_lock: bool = True

    # ------------------------------------------------- M1-3：构图分档
    closeup_ratio: float = DEFAULT_FRAMING_PARAMS.closeup_ratio
    halfbody_ratio: float = DEFAULT_FRAMING_PARAMS.halfbody_ratio
    closeup_fill: float = DEFAULT_FRAMING_PARAMS.closeup_fill
    halfbody_fill: float = DEFAULT_FRAMING_PARAMS.halfbody_fill
    fullbody_fill: float = DEFAULT_FRAMING_PARAMS.fullbody_fill
    incomplete_fill: float = DEFAULT_FRAMING_PARAMS.incomplete_fill
    headroom: float = DEFAULT_FRAMING_PARAMS.headroom
    min_box_px: int = DEFAULT_FRAMING_PARAMS.min_box_px

    # ------------------------------------------------- M1-4：时序平滑
    #: 取景框中心点的 EMA 系数（越大越跟手、越小越稳）
    #: 默认值与「跟手」档一致
    smoothing_alpha: float = 0.45
    #: 自由活动区（相对取景框边长）：人物在框内走动时镜头**完全不动**，
    #: 超过它之后镜头也只把人物推回边界，不会拽回画面正中 —— 这是缓解
    #: "人动镜头就动"（画面整体流动导致头晕）最有效的一项
    smoothing_deadzone: float = 0.06
    #: 缩放（推拉）的 EMA 系数：比平移更慢，抑制"呼吸式变焦"
    smoothing_zoom_alpha: float = 0.15
    #: 缩放死区（相对取景框高度）
    smoothing_zoom_deadzone: float = 0.05
    #: 平移速度上限（相对框长 / 帧）：避免甩镜；0 = 不限制
    smoothing_max_speed: float = 0.05
    #: 加速平滑 (0, 1]：越小起步越柔和，1 = 不限制加速度
    smoothing_accel: float = 0.5
    #: 丢失目标后保持上一帧取景框的最大帧数。
    #: 多人分屏时它还有第二个作用：漏检的人**保留窗口**多少帧（这几帧窗口不动、
    #: 画面继续播放），以及撤掉一个分屏窗口 / 退出分屏要等多少帧才生效 ——
    #: 调大 = 人数抖动时更不容易闪；调小 = 人一少就立刻改布局。
    hold_frames: int = 30
    #: 空镜巡视的单程时长（秒）：只有显示方式为 ``scan`` 时才用得上。
    #: 取景框从中线性地平移扫过整幅画面，而不是直接居中 —— 直接居中会一次性丢掉
    #: 左右（或上下）约三分之二的内容。越大越慢越舒缓；0 = 关闭巡视，退回画面居中。
    smoothing_scan_seconds: float = 12.0

    # ------------------------------------------------- 无人物帧怎么显示
    #: 画面里没有主要人物（且超出保持帧数）时怎么显示：
    #:   ``fit``    全画面适配（默认）：整幅画面等比缩小完整放进画布，一帧看全、
    #:              镜头完全不移动；空出来的部分用同一帧的模糊放大版填满；
    #:   ``tiles``  全景 + 特写：整幅画面收进上方通栏，下面纵向排列几个"次要小人物"
    #:              （被 min_person_height_ratio 过滤掉的远景人物）的上半身特写；
    #:   ``scan``   空镜巡视：取景框缓慢平移扫过整幅画面（任一时刻只看得到局部）；
    #:   ``center`` 画面居中裁剪：最保守的一档。
    #: 无论哪一种，从"人物取景框"切换过去都是平滑过渡的（见 no_person_seconds）。
    no_person_mode: str = "fit"
    #: 从"人物取景框"过渡到无人物显示的时长（秒）：越大越舒缓，0 = 立即切换
    no_person_seconds: float = DEFAULT_NO_PERSON_SECONDS
    #: 无人物画面的底色是否用"同一帧的模糊放大版"（比纯色 / 黑边自然得多）
    no_person_blur: bool = True
    #: 「全景 + 特写」最多给几个小人物开特写窗口（1~4）
    no_person_tiles_max: int = DEFAULT_TILES_MAX
    #: 「次要小人物」的高度下限（占画面比例）：低于它的不给特写（放大了也只是马赛克）
    no_person_secondary_ratio: float = DEFAULT_SECONDARY_PERSON_RATIO

    # ------------------------------------------------- M1-7：性能
    #: 每 N 帧做一次 YOLO 推理，中间帧线性插值（默认 1 = 逐帧推理）
    detect_interval: int = 1
    #: 一次推理塞几帧（攒够这么多关键帧再批量送进模型；默认 1 = 逐帧推理）
    infer_batch: int = 1
    #: 使用姿态关键点精修构图（仅当模型自带关键点，即 ``*-pose.pt`` 时生效）
    use_keypoints: bool = True

    # ---------------------------------------------------------- M2：批量
    #: 输入为文件夹时是否递归遍历
    batch_recursive: bool = True

    # ---------------------------------------------------------------- 构造
    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> AppConfig:
        """从 dict（例如 YAML 内容）构造配置，忽略未知键并做类型规整。"""
        if not data:
            return cls()

        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            logger.warning("配置文件中存在未知字段，已忽略：%s", ", ".join(unknown))

        kwargs: dict[str, Any] = {k: v for k, v in data.items() if k in known}

        for key in _PATH_FIELDS:
            value = kwargs.get(key)
            kwargs[key] = None if _is_null(value) else Path(str(value))

        if "classes" in kwargs:
            value = kwargs["classes"]
            if value is None:
                kwargs["classes"] = []
            elif isinstance(value, (list, tuple, set)):
                kwargs["classes"] = [int(item) for item in value]
            else:
                kwargs["classes"] = [int(value)]

        for key in _INT_FIELDS:
            value = kwargs.get(key)
            if key in kwargs and not _is_null(value):
                kwargs[key] = int(value)

        for key in _FLOAT_FIELDS:
            value = kwargs.get(key)
            if key in kwargs and not _is_null(value):
                kwargs[key] = float(value)

        for key in _BOOL_FIELDS:
            if key in kwargs:
                kwargs[key] = _to_bool(kwargs[key], key)

        if _is_null(kwargs.get("device")):
            kwargs["device"] = None
        if kwargs.get("model") is not None:
            kwargs["model"] = str(kwargs["model"])
        if kwargs.get("aspect_ratio") is not None:
            kwargs["aspect_ratio"] = str(kwargs["aspect_ratio"]).strip() or DEFAULT_RATIO_NAME
        if kwargs.get("no_person_mode") is not None:
            # 允许写成 Fit / FIT / " 全画面适配 " 之类的形式，统一成小写模式名
            kwargs["no_person_mode"] = str(kwargs["no_person_mode"]).strip().lower() or "fit"
        if "output_suffix" in kwargs:
            suffix = kwargs["output_suffix"]
            kwargs["output_suffix"] = None if _is_null(suffix) else _normalize_suffix(str(suffix))

        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> AppConfig:
        """从 YAML 文件加载配置。"""
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(f"配置文件不存在：{target}")
        with target.open("r", encoding="utf-8") as fp:
            data = yaml.safe_load(fp) or {}
        if not isinstance(data, Mapping):
            raise ValueError(f"配置文件格式错误，应为键值对：{target}")
        logger.debug("已加载配置文件 %s", target)
        return cls.from_mapping(data)

    # ---------------------------------------------------------------- 修改
    def with_overrides(self, **kwargs: Any) -> AppConfig:
        """返回一份新配置，仅覆盖值不为 ``None`` 的字段（供命令行参数使用）。"""
        updates = {k: v for k, v in kwargs.items() if v is not None and k in self.to_dict()}
        return replace(self, **updates) if updates else self

    def to_dict(self) -> dict[str, Any]:
        """导出为 dict（Path 转字符串，便于写 YAML / 打印）。"""
        data = asdict(self)
        return {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in data.items()
        }

    # ------------------------------------------------------------ 派生对象
    def resolve_ratio(self) -> AspectRatio:
        """把 ``aspect_ratio`` + 自定义目标像素解析成 :class:`AspectRatio`。"""
        return resolve_ratio(
            self.aspect_ratio,
            target_width=self.target_width,
            target_height=self.target_height,
        )

    def framing_params(self) -> FramingParams:
        """构造构图参数对象（供 ``compute_target_box`` 使用）。"""
        return FramingParams(
            closeup_ratio=float(self.closeup_ratio),
            halfbody_ratio=float(self.halfbody_ratio),
            closeup_fill=float(self.closeup_fill),
            halfbody_fill=float(self.halfbody_fill),
            fullbody_fill=float(self.fullbody_fill),
            incomplete_fill=float(self.incomplete_fill),
            headroom=float(self.headroom),
            min_box_px=int(self.min_box_px),
        )

    def smoothing_params(self) -> SmoothingParams:
        """构造取景框平滑参数对象（供 ``BoxSmoother`` 使用）。"""
        return SmoothingParams(
            alpha=float(self.smoothing_alpha),
            deadzone=float(self.smoothing_deadzone),
            zoom_alpha=float(self.smoothing_zoom_alpha),
            zoom_deadzone=float(self.smoothing_zoom_deadzone),
            max_speed=float(self.smoothing_max_speed),
            accel=float(self.smoothing_accel),
            hold_frames=int(self.hold_frames),
            scan_seconds=float(self.smoothing_scan_seconds),
            no_person_mode=str(self.no_person_mode),
            no_person_seconds=float(self.no_person_seconds),
        )

    def no_person_display(self) -> NoPersonDisplay:
        """构造"无人物帧长什么样"的外观参数（配合 ``smoothing_params().no_person_mode``）。"""
        return NoPersonDisplay(
            blur=bool(self.no_person_blur),
            tiles_max=int(self.no_person_tiles_max),
            secondary_ratio=float(self.no_person_secondary_ratio),
        )

    def subject_weights(self) -> SubjectWeights:
        """构造主角打分权重对象。"""
        return SubjectWeights(
            area=float(self.subject_area_weight),
            center=float(self.subject_center_weight),
            continuity=float(self.subject_continuity_weight),
            confidence=float(self.subject_confidence_weight),
        )

    def speaker_params(self) -> SpeakerParams:
        """构造说话人判定参数对象（供 ``ActiveSpeakerSelector`` 使用）。"""
        return SpeakerParams(
            window_frames=int(self.speaker_window_frames),
            min_score=DEFAULT_SPEAKER_PARAMS.min_score,
            switch_margin=float(self.speaker_switch_margin),
            switch_hold=int(self.speaker_switch_hold),
            track_iou=DEFAULT_SPEAKER_PARAMS.track_iou,
            max_missing_frames=DEFAULT_SPEAKER_PARAMS.max_missing_frames,
            min_mouth_motion=float(self.speaker_min_mouth_motion),
        )

    def multi_person_policy(self) -> MultiPersonPolicy:
        """构造多人分屏策略：布局表来自布局配置文件，开关 / 上限 / 排列来自本配置。

        布局文件缺失时自动退回内置默认布局（开箱可用，打包后也能跑）；
        但用户**显式指定**了布局文件却找不到时会直接报错，不做静默降级。
        """
        policy = load_multi_person_policy(self.multi_person_layout_file)
        order = str(self.multi_person_order).strip().lower() or policy.order
        return replace(
            policy,
            enabled=bool(self.multi_person),
            max_persons=int(self.multi_person_max),
            order=order,
        )

    def window_camera(self) -> WindowCamera:
        """分屏小窗口的镜头策略（"每个窗口都正常速度播放"，见 ``core.multi``）。"""
        return WindowCamera(
            stable=bool(self.multi_person_stable_camera),
            pan_speed=float(self.multi_person_pan_speed),
            seat_lock=bool(self.multi_person_seat_lock),
        )

    def output_path_for(self, source: str | Path) -> Path:
        """按 M0 决策推导某个源视频的输出路径。"""
        return default_output_path(
            source,
            suffix=self.output_suffix,
            same_dir=bool(self.output_same_dir),
        )

    # ---------------------------------------------------------------- 校验
    def validate(self) -> None:
        """做基本参数校验，发现问题直接抛 ``ValueError``。"""
        if not 0.0 < float(self.conf) <= 1.0:
            raise ValueError(f"conf 必须在 (0, 1] 区间内，当前为 {self.conf}")
        if not 0.0 < float(self.iou) <= 1.0:
            raise ValueError(f"iou 必须在 (0, 1] 区间内，当前为 {self.iou}")
        if self.imgsz <= 0:
            raise ValueError(f"imgsz 必须为正整数，当前为 {self.imgsz}")
        if self.imgsz % 32 != 0:
            logger.warning("imgsz=%s 不是 32 的倍数，YOLO 会自动调整", self.imgsz)
        if not 0 <= int(self.crf) <= 51:
            raise ValueError(f"crf 必须在 [0, 51] 区间内，当前为 {self.crf}")
        if float(self.max_size_ratio) <= 0.0:
            raise ValueError(f"max_size_ratio 必须为正数，当前为 {self.max_size_ratio}")
        if int(self.min_video_bitrate_kbps) < 0:
            raise ValueError(
                f"min_video_bitrate_kbps 不能为负，当前为 {self.min_video_bitrate_kbps}"
            )

        # 比例：解析即校验（非法比例在此抛 RatioError，它是 ValueError 的子类）
        self.resolve_ratio()
        self.framing_params()

        if not 0.0 < float(self.smoothing_alpha) <= 1.0:
            raise ValueError(f"smoothing_alpha 必须在 (0, 1] 区间内，当前为 {self.smoothing_alpha}")
        if float(self.smoothing_deadzone) < 0.0:
            raise ValueError(f"smoothing_deadzone 不能为负，当前为 {self.smoothing_deadzone}")
        # 其余平滑参数交给 SmoothingParams 统一校验（含 zoom / speed / accel）
        self.smoothing_params()
        if int(self.hold_frames) < 0:
            raise ValueError(f"hold_frames 不能为负，当前为 {self.hold_frames}")
        if int(self.detect_interval) < 1:
            raise ValueError(f"detect_interval 必须 ≥ 1，当前为 {self.detect_interval}")
        if int(self.infer_batch) < 1:
            raise ValueError(f"infer_batch 必须 ≥ 1，当前为 {self.infer_batch}")
        if int(self.infer_batch) > 64:
            raise ValueError(f"infer_batch 过大（会吃满显存），当前为 {self.infer_batch}")
        if int(self.min_box_px) < 2:
            raise ValueError(f"min_box_px 必须 ≥ 2，当前为 {self.min_box_px}")
        if not 0.0 <= float(self.min_person_height_ratio) <= 1.0:
            raise ValueError(
                f"min_person_height_ratio 必须在 [0, 1] 区间内，当前为 {self.min_person_height_ratio}"
            )
        if not 0.0 <= float(self.min_person_sharpness) <= 1.0:
            raise ValueError(
                f"min_person_sharpness 必须在 [0, 1] 区间内，当前为 {self.min_person_sharpness}"
            )
        if (self.target_width is None) != (self.target_height is None):
            raise ValueError("target_width 与 target_height 必须同时给出")
        if self.output_suffix is not None and not str(self.output_suffix).startswith("."):
            raise ValueError(f"output_suffix 必须以 . 开头，当前为 {self.output_suffix}")
        if float(self.speaker_weight) < 0.0:
            raise ValueError(f"speaker_weight 不能为负，当前为 {self.speaker_weight}")
        # 其余说话人判定参数交给 SpeakerParams 统一校验
        self.speaker_params()

        if not 2 <= int(self.multi_person_max) <= MAX_WINDOWS:
            raise ValueError(
                f"multi_person_max 必须在 [2, {MAX_WINDOWS}] 区间内（分屏至少 2 人、最多 {MAX_WINDOWS} 人），"
                f"当前为 {self.multi_person_max}"
            )
        if str(self.multi_person_order).strip().lower() not in {"spatial", "score"}:
            raise ValueError(
                f"multi_person_order 只能是 spatial 或 score，当前为 {self.multi_person_order!r}"
            )
        if not 0.05 <= float(self.multi_person_pan_speed) <= 5.0:
            raise ValueError(
                "multi_person_pan_speed 必须在 [0.05, 5.0] 区间内"
                f"（每秒最多平移几个窗口宽度），当前为 {self.multi_person_pan_speed}"
            )
        # 布局文件提前解析一次：配置写错（网格不合法 / 人数与格子数不符）在这里就报出来
        self.multi_person_policy()

        # 无人物帧的显示方式：模式由 SmoothingParams 统一校验，这里管其余几项
        mode = str(self.no_person_mode).strip().lower()
        if mode not in NO_PERSON_MODES:
            raise ValueError(
                f"no_person_mode 只能是 {' / '.join(NO_PERSON_MODES)}，当前为 {self.no_person_mode!r}"
            )
        if not 1 <= int(self.no_person_tiles_max) <= 4:
            raise ValueError(
                f"no_person_tiles_max 必须在 [1, 4] 区间内（再多，竖屏里每格就只剩一条），"
                f"当前为 {self.no_person_tiles_max}"
            )
        if not 0.0 <= float(self.no_person_secondary_ratio) < 1.0:
            raise ValueError(
                f"no_person_secondary_ratio 必须在 [0, 1) 区间内，"
                f"当前为 {self.no_person_secondary_ratio}"
            )
        if mode == "tiles" and float(self.no_person_secondary_ratio) >= float(
            self.min_person_height_ratio
        ):
            logger.warning(
                "no_person_secondary_ratio（%.2f）不小于 min_person_height_ratio（%.2f），"
                "「全景 + 特写」将没有可特写的人物，等价于全画面适配",
                float(self.no_person_secondary_ratio),
                float(self.min_person_height_ratio),
            )


def _is_null(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip().lower() in _NULL_TOKENS)


def _to_bool(value: Any, key: str) -> bool:
    """把 YAML / 命令行里的各种"真假写法"规整成 bool。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in _BOOL_TRUE:
        return True
    if text in _BOOL_FALSE:
        return False
    logger.warning("字段 %s 的值 %r 无法识别为布尔，按 False 处理", key, value)
    return False


def _normalize_suffix(suffix: str) -> str:
    text = suffix.strip()
    if not text:
        raise ValueError("output_suffix 不能为空字符串")
    return text if text.startswith(".") else f".{text}"


def load_config(path: str | Path | None = None) -> AppConfig:
    """加载配置文件；``path`` 为 ``None`` 时使用仓库默认配置（存在才加载）。"""
    if path is not None:
        return AppConfig.from_yaml(path)
    if DEFAULT_CONFIG_PATH.exists():
        return AppConfig.from_yaml(DEFAULT_CONFIG_PATH)
    logger.debug("未找到默认配置文件，使用内置默认值")
    return AppConfig()


def default_output_path(
    source: str | Path,
    suffix: str | None = None,
    *,
    same_dir: bool = True,
) -> Path:
    """根据输入视频推导默认输出路径。

    M0 决策：输出到**原视频同目录**（``same_dir=False`` 时回到 ``data/output/``），
    命名为 ``<原名>_person<原后缀>``。
    """
    src = Path(source)
    extension = _normalize_suffix(suffix) if suffix else (src.suffix or ".mp4")
    name = f"{src.stem}{PERSON_SUFFIX}{extension}"
    return (src.parent / name) if same_dir else (DEFAULT_OUTPUT_DIR / name)
