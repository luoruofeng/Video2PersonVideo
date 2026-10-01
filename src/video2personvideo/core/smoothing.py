"""取景框时序平滑：自由活动区 + 速度限幅 + 缩放慢速跟随 + 丢失保持。

逐帧独立裁剪会让画面剧烈抖动，这里把**每帧的目标取景框**串成一条连续轨迹。
最朴素的做法是对中心点做指数移动平均（EMA），但它只解决"抖"，不解决"晃"：
人物一走，镜头就一直跟着平移，整个背景在视野里流动，看久了容易头晕
（视觉输入与前庭平衡感冲突，即"晕动症 / vection"）。

因此这里采用"像真人摄像师一样"的策略：

* **自由活动区**（``deadzone``，软边界）：人物在取景框内走动、偏离量小于死区时
  镜头**完全不动**（背景静止）；超出死区后只推"多出来的那一部分"，于是镜头把
  人送到死区边界就停住，**不会把人拽回画面正中** —— 这消除了绝大部分持续运动。
* **速度限幅**（``max_speed``）：镜头平移速度有上限（相对框长/帧），避免"甩镜"；
  偏离越大上限越宽松（:data:`CATCHUP`），但放宽是**平滑饱和**到
  :data:`MAX_SPEED_FACTOR` 倍为止的 —— 人物快走时镜头会跟着变快，但快得可预期，
  不会"越走越快"地线性放大成一次急促的大幅横摇。
* **加速平滑**（``accel``）：起步与反向不瞬间到满速，而是逐帧逼近，消除急起急停。
* **帧率归一化**（``fps``）：上面这些系数都按 :data:`REFERENCE_FPS` 的"每秒感受"标定，
  运行期给出真实帧率后换算成等效的每帧值 —— 24 / 30 / 60fps 的素材，镜头运动的
  **真实速度**一致；否则 60fps 素材的镜头会比 30fps 快一倍，还没看出"跟得紧"就先晕了。
* **缩放（推拉）比平移更慢更少**（``zoom_alpha`` / ``zoom_deadzone``）：缩放引起的
  不适感比平移更强，单独一套系数才能压住"呼吸式变焦"。

平滑之后仍由目标比例反推框宽，因此三条硬性约定不变：宽高比恒定（不变量 3）、
切片不越界（不变量 4）、输出尺寸恒定（不变量 2）。

丢失保持（兜底第一级）：短暂丢失目标（遮挡 / 转头）时沿用上一帧取景框，
最多 ``hold_frames`` 帧。

超时后仍然没有主要人物，就按 :data:`NO_PERSON_MODES` 里选定的方式显示（兜底第二级）：

* ``fit``（默认）**全画面适配**：把整幅画面等比缩小、完整放进输出画布，空出来的
  部分由裁剪层用同一帧的模糊放大版填满 —— 一帧看全、镜头**完全不移动**。
  从"人物取景框"到"整幅画面"用 ``no_person_seconds`` 秒平缓拉远（缓入缓出）；
* ``tiles`` **全景 + 特写**：整幅画面收进上方通栏，下方纵向排列若干"次要小人物"
  的半身特写 —— 既要环境全貌，也要看清被主角阈值过滤掉的远景人物；
* ``scan`` **空镜巡视**：取景框沿"画面比取景框多出来的那一侧"缓慢往返，
  用几个来回把整幅画面看完（任一时刻只看到局部，且画面一直在动）；
* ``center`` **画面居中**：直接取画面正中的等比取景框（最保守的一档）。

巡视没有可移动的空间（源比例与目标比例一致）、或该方式被关闭时，
同样退回画面居中的等比取景框（兜底第三级）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from .framing import (
    MODE_CENTER,
    MODE_FIT,
    MODE_HOLD,
    MODE_LABELS,
    MODE_SCAN,
    MODE_TILES,
    CropBox,
    center_box,
)
from .layout import TransitionFrame, fit_rect, transition_frame
from .ratio import AspectRatio

#: 画面里没有主要人物时的显示方式（``SmoothingParams.no_person_mode``）：
#: ``fit`` 全画面适配 / ``tiles`` 全景+特写 / ``scan`` 空镜巡视 / ``center`` 画面居中
NO_PERSON_MODES: tuple[str, ...] = (MODE_FIT, MODE_TILES, MODE_SCAN, MODE_CENTER)

#: 显示方式 → 展示名（与 :data:`~video2personvideo.core.framing.MODE_LABELS` 同一份文案）
NO_PERSON_LABELS: dict[str, str] = {mode: MODE_LABELS[mode] for mode in NO_PERSON_MODES}

#: 显示方式 → 一句话说明（日志 / 汇总 / GUI 提示共用，说清"看不到什么、代价是什么"）
NO_PERSON_DESCRIPTIONS: dict[str, str] = {
    MODE_FIT: "整幅画面等比缩小完整放入，镜头不动、一帧看全",
    MODE_TILES: "整幅画面收进上方通栏，下方给远景小人物开半身特写窗口",
    MODE_SCAN: "取景框缓慢扫过整幅画面，任一时刻只看得到局部",
    MODE_CENTER: "只取画面正中，两侧 / 上下的内容被裁掉",
}

#: 默认的无人物显示方式（一帧看全、零运动，最不容易看晕）
DEFAULT_NO_PERSON_MODE = MODE_FIT

#: 无人物过渡的默认时长（秒）：从"人物取景框"平缓拉远到"全画面适配"用多久
DEFAULT_NO_PERSON_SECONDS = 0.8


def no_person_mode_label(mode: str) -> str:
    """显示方式 → 展示名（未知取值原样返回）。"""
    key = str(mode).strip().lower()
    return NO_PERSON_LABELS.get(key, str(mode))


def no_person_mode_description(mode: str) -> str:
    """显示方式 → 一句话说明（未知取值返回空串）。"""
    return NO_PERSON_DESCRIPTIONS.get(str(mode).strip().lower(), "")


def no_person_choices() -> list[tuple[str, str]]:
    """``[(模式, 展示名), ...]``，顺序即 GUI 下拉 / 文档的展示顺序。"""
    return [(mode, NO_PERSON_LABELS[mode]) for mode in NO_PERSON_MODES]

#: 追赶系数：目标偏离每增加"一个框长"，平移速度上限就放宽这么多倍基准值。
#: 保证人物快走出画面时镜头跟得上；日常小幅移动受速度上限保护，不会甩镜。
CATCHUP = 10.0

#: 平移速度上限相对基准值的最大倍数（防止极端偏离时镜头飞出去）。
#: 放宽按 tanh **平滑饱和**到这个倍数：小偏离处的斜率仍等于 :data:`CATCHUP`
#: （与过去的线性放宽一致），但越走越快时不会线性放大成一次急促的大幅横摇。
MAX_SPEED_FACTOR = 4.0

#: 参数标定的基准帧率：``alpha`` / ``zoom_alpha`` / ``max_speed`` / ``hold_frames``
#: 都按它的"每秒感受"定义，给出真实 ``fps`` 后换算成等效的每帧值，于是不同帧率的
#: 素材（24 / 25 / 30 / 50 / 60fps）看到的是**同样的镜头运动速度**。
REFERENCE_FPS = 30.0

#: 空镜巡视时，可平移空间小于这么多个像素就认为"源比例已与目标比例一致"，
#: 不值得巡视（否则取景框会在几像素内来回抖，比静止更难看）。
SCAN_MIN_SPAN_PX = 8.0

# ------------------------------------------------------------------ 分屏小窗口
#: 多人分屏每个小窗口的镜头平移上限：**屏幕**上每秒最多平移"窗口自身的几倍"。
#:
#: 单窗口模式的速度上限是按"相对框长 / 帧"标定的（跟手档 = 5%/帧 = 1.5 个框宽/秒，
#: 人物越界时还会放大到 4 倍 = 6 个框宽/秒）：整幅画面里这只是"跟得紧"，但分屏窗口
#: 只有一格大、放大倍数高，同样的相对速度在屏幕上非常显眼 —— 看上去就是"画面在快进"。
#: 因此分屏窗口一律改用**按秒**标定，并且几乎不因人物跑远而加速（见
#: :func:`window_smoothing_params`）。
DEFAULT_WINDOW_PAN_SPEED = 0.5

#: 分屏窗口的自由活动区相对用户档位的倍数与下限（更"懒"：人物在窗口里晃动时镜头不动）
WINDOW_DEADZONE_SCALE = 2.0
WINDOW_MIN_DEADZONE = 0.15
#: 分屏窗口的缩放死区倍数与下限（压住小窗口里的"呼吸式变焦"）
WINDOW_ZOOM_DEADZONE_SCALE = 2.0
WINDOW_MIN_ZOOM_DEADZONE = 0.12
#: 分屏窗口的平滑 / 缩放 / 加速度系数相对用户档位的倍数（比单窗口更柔和）
WINDOW_ALPHA_SCALE = 0.55
WINDOW_ZOOM_ALPHA_SCALE = 0.5
WINDOW_ACCEL_SCALE = 0.7
#: 分屏窗口的"越界追赶"上限：人物快走出窗口时镜头最多快到这个倍数
#: （单窗口是 :data:`MAX_SPEED_FACTOR` = 4 倍，这里只到 1.6 倍，不会甩镜）
WINDOW_MAX_SPEED_FACTOR = 1.6
#: 分屏窗口的追赶系数（越小越"无动于衷"，只把人物推回窗口内）
WINDOW_CATCHUP = 2.0

#: 空镜巡视的默认单程时长（秒）：从画面一端平缓移到另一端用多久。
#: 12 秒 ≈ 6px/帧（1080p 横屏转竖屏实测），慢到看不出"被裁"，又能在
#: 一段十几秒的空镜里看完整个画面宽度。
DEFAULT_SCAN_SECONDS = 12.0

#: 巡视相位的"尚未开始"哨兵值（正常相位在 ``[0, 1)``）
_SCAN_IDLE = -1.0


@dataclass(frozen=True, slots=True)
class SmoothingParams:
    """镜头运动的稳定性参数（对应 ``AppConfig`` 的 ``smoothing_*`` 字段）。"""

    #: 取景框中心点的 EMA 系数，(0, 1]，越大越跟手（默认值即「跟手」档）
    alpha: float = 0.45
    #: 自由活动区（相对取景框边长）：人物在框内的位移小于它时镜头完全不动
    deadzone: float = 0.06
    #: 缩放（推拉）的 EMA 系数，通常明显小于 ``alpha``，避免呼吸式变焦
    zoom_alpha: float = 0.15
    #: 缩放死区（相对取景框高度）
    zoom_deadzone: float = 0.05
    #: 平移速度上限（相对框长 / 帧），0 = 不限制
    max_speed: float = 0.05
    #: 加速平滑，(0, 1]：越小起步越柔和，1 = 不限制加速度
    accel: float = 0.5
    #: 丢失目标后保持上一帧取景框的最大帧数（按 :data:`REFERENCE_FPS` 的时长标定）
    hold_frames: int = 30
    #: 空镜巡视的单程时长（秒）：画面里没有主要人物、且显示方式为 ``scan`` 时，
    #: 取景框从一端平缓移到另一端所需的时间；``0`` = 关闭巡视（退回画面居中）。
    scan_seconds: float = DEFAULT_SCAN_SECONDS
    #: 画面里没有主要人物（且超出保持帧数）时怎么显示，取值见 :data:`NO_PERSON_MODES`：
    #: ``fit`` = 整幅画面等比缩小完整放入（默认）；``tiles`` = 全景 + 特写；
    #: ``scan`` = 空镜巡视；``center`` = 画面居中裁剪
    no_person_mode: str = DEFAULT_NO_PERSON_MODE
    #: 从"人物取景框"过渡到无人物显示的时长（秒）：越大越舒缓，0 = 立即切换
    no_person_seconds: float = DEFAULT_NO_PERSON_SECONDS
    #: 源视频帧率；给出时把上面这些"每帧"参数换算成与帧率无关的等效值
    #: （``None`` = 不换算，即把源视频当作 :data:`REFERENCE_FPS`）
    fps: float | None = None
    #: "追赶系数"：人物偏离越大，速度上限放宽得越快（见 :func:`_speed_limit`）。
    #: 分屏小窗口用更小的值（不轻易加速），单人模式保持 :data:`CATCHUP`。
    catchup: float = CATCHUP
    #: 速度上限相对 ``max_speed`` 的最大倍数（``1.0`` = 无论偏多远都不加速）
    max_speed_factor: float = MAX_SPEED_FACTOR

    def __post_init__(self) -> None:
        if not 0.0 < float(self.alpha) <= 1.0:
            raise ValueError(f"alpha 必须在 (0, 1] 区间内，当前为 {self.alpha}")
        if float(self.deadzone) < 0.0:
            raise ValueError(f"deadzone 不能为负，当前为 {self.deadzone}")
        if not 0.0 < float(self.zoom_alpha) <= 1.0:
            raise ValueError(f"zoom_alpha 必须在 (0, 1] 区间内，当前为 {self.zoom_alpha}")
        if float(self.zoom_deadzone) < 0.0:
            raise ValueError(f"zoom_deadzone 不能为负，当前为 {self.zoom_deadzone}")
        if float(self.max_speed) < 0.0:
            raise ValueError(f"max_speed 不能为负，当前为 {self.max_speed}")
        if not 0.0 < float(self.accel) <= 1.0:
            raise ValueError(f"accel 必须在 (0, 1] 区间内，当前为 {self.accel}")
        if int(self.hold_frames) < 0:
            raise ValueError(f"hold_frames 不能为负，当前为 {self.hold_frames}")
        if float(self.catchup) < 0.0:
            raise ValueError(f"catchup 不能为负，当前为 {self.catchup}")
        if float(self.max_speed_factor) < 1.0:
            raise ValueError(
                f"max_speed_factor 不能小于 1（1 = 不加速），当前为 {self.max_speed_factor}"
            )
        if float(self.scan_seconds) < 0.0:
            raise ValueError(f"scan_seconds 不能为负（0 = 关闭空镜巡视），当前为 {self.scan_seconds}")
        mode = str(self.no_person_mode).strip().lower()
        if mode not in NO_PERSON_MODES:
            raise ValueError(
                f"no_person_mode 只能是 {' / '.join(NO_PERSON_MODES)}，当前为 {self.no_person_mode!r}"
            )
        if float(self.no_person_seconds) < 0.0:
            raise ValueError(
                f"no_person_seconds 不能为负（0 = 立即切换），当前为 {self.no_person_seconds}"
            )
        if self.fps is not None and float(self.fps) <= 0.0:
            raise ValueError(f"fps 必须为正（帧率未知时用 None），当前为 {self.fps}")


#: 默认平滑参数（= "跟手"档，与 :class:`~video2personvideo.config.AppConfig` 的默认值一致）
DEFAULT_SMOOTHING_PARAMS = SmoothingParams(
    alpha=0.45,
    deadzone=0.06,
    zoom_alpha=0.15,
    zoom_deadzone=0.05,
    max_speed=0.05,
)


@dataclass(frozen=True, slots=True)
class CameraPreset:
    """面向用户的"镜头跟随"档位（GUI 下拉 / CLI ``--camera-follow``）。"""

    key: str
    label: str
    description: str
    params: SmoothingParams


def _preset(key: str, label: str, description: str, **overrides: float) -> CameraPreset:
    return CameraPreset(key, label, description, replace(DEFAULT_SMOOTHING_PARAMS, **overrides))


#: 档位由稳到跟手：给"容易晕"的用户一个不需要调数值的选择
CAMERA_PRESETS: dict[str, CameraPreset] = {
    preset.key: preset
    for preset in (
        _preset(
            "lock",
            "锁定",
            "人物在画面内走动时镜头完全不动，最不容易看晕；只在人物快走出画面时缓慢平移",
            alpha=0.12,
            deadzone=0.32,
            zoom_alpha=0.05,
            zoom_deadzone=0.20,
            max_speed=0.015,
        ),
        _preset(
            "calm",
            "舒缓",
            "镜头偶尔缓慢平移，人物可在画面内较自由地走动",
            alpha=0.18,
            deadzone=0.20,
            zoom_alpha=0.07,
            zoom_deadzone=0.14,
            max_speed=0.02,
        ),
        _preset(
            "standard",
            "标准",
            "镜头较少移动，人物基本居中",
            alpha=0.25,
            deadzone=0.15,
            zoom_alpha=0.08,
            zoom_deadzone=0.10,
            max_speed=0.025,
        ),
        _preset(
            "active",
            "跟手",
            "默认：镜头跟得紧、画面运动感强，适合素材本身运动幅度小的情况",
            alpha=0.45,
            deadzone=0.06,
            zoom_alpha=0.15,
            zoom_deadzone=0.05,
            max_speed=0.05,
        ),
    )
}

#: 默认档位
DEFAULT_CAMERA_PRESET = "active"


def camera_preset(key: str | None) -> CameraPreset:
    """按档位名取预设；未知名回退到默认档位。"""
    if key is None:
        return CAMERA_PRESETS[DEFAULT_CAMERA_PRESET]
    return CAMERA_PRESETS.get(str(key).strip().lower(), CAMERA_PRESETS[DEFAULT_CAMERA_PRESET])


def preset_choices() -> list[tuple[str, str]]:
    """``[(档位名, 展示名), ...]``，顺序固定，供 GUI / CLI 使用。"""
    return [(preset.key, preset.label) for preset in CAMERA_PRESETS.values()]


#: 档位预设覆盖的 ``AppConfig`` 字段（逐项微调也走这几个键）
PRESET_FIELDS: tuple[str, ...] = (
    "smoothing_alpha",
    "smoothing_deadzone",
    "smoothing_zoom_alpha",
    "smoothing_zoom_deadzone",
    "smoothing_max_speed",
)


def preset_values(key: str | None) -> dict[str, float]:
    """把档位展开成 ``AppConfig`` 字段，供 GUI / CLI 一次写入一组参数。"""
    params = camera_preset(key).params
    return {
        "smoothing_alpha": params.alpha,
        "smoothing_deadzone": params.deadzone,
        "smoothing_zoom_alpha": params.zoom_alpha,
        "smoothing_zoom_deadzone": params.zoom_deadzone,
        "smoothing_max_speed": params.max_speed,
    }


def match_preset(alpha: float | None) -> str | None:
    """按平移平滑系数反查档位名；匹配不到（用户自定义过）返回 ``None``。"""
    if alpha is None:
        return None
    for key, preset in CAMERA_PRESETS.items():
        if abs(preset.params.alpha - float(alpha)) < 1e-6:
            return key
    return None


def _time_scale(fps: float | None) -> float:
    """每帧参数 → 等效值的缩放：帧率越高，每帧该走的越少（每秒感受不变）。"""
    if fps is None or float(fps) <= 0.0:
        return 1.0
    return REFERENCE_FPS / float(fps)


def _normalize_alpha(alpha: float, scale: float) -> float:
    """把按"每秒到达比例"标定的 EMA 系数换算到当前帧率（时间常数不变）。

    ``alpha`` 是每帧的到达比例，帧率提高一倍后每帧应到达得更少，
    才能让**每秒**的收敛速度维持原样；``scale == 1`` 时原样返回（不做无谓的浮点扰动）。
    """
    if scale == 1.0:
        return float(alpha)
    return 1.0 - (1.0 - float(alpha)) ** scale


def _soft_zone(offset: float, zone: float) -> float:
    """软死区：只保留超出死区的部分（位移在死区内时一动不动，且位移连续无跳变）。"""
    if zone <= 0.0:
        return offset
    if offset > zone:
        return offset - zone
    if offset < -zone:
        return offset + zone
    return 0.0


def _phase_for_offset(offset: float, span: float) -> float:
    """把"沿巡视轴偏了多少像素"换算成行程相位（取上升段，即向右 / 向下的那半程）。

    位移函数为 ``h(φ) = (1 - cos 2πφ) / 2 × span``，反解即得；“偏移超出行程”时
    夹到两端，于是取景框不会因为上一帧在画面外而算出一个奇怪的相位。
    """
    if span <= 0.0:
        return 0.0
    position = min(max(float(offset) / span, 0.0), 1.0)
    return math.acos(1.0 - 2.0 * position) / (2.0 * math.pi)


def _speed_limit(
    offset: float,
    size: float,
    max_speed: float,
    catchup: float = CATCHUP,
    max_factor: float = MAX_SPEED_FACTOR,
) -> float:
    """本帧允许的最大步长：基准速度上限 × 越界放大倍数（平滑饱和，不甩镜）。

    小偏离处斜率为 ``catchup``（和线性放宽一样"跟得住"），大偏离处平滑饱和到
    ``max_factor`` 倍 —— 人物跑起来时镜头变快但不"炸"，避免急促横摇。
    分屏小窗口用更小的 ``catchup`` / ``max_factor``（见 :func:`window_smoothing_params`）。
    """
    if max_speed <= 0.0 or size <= 0.0:
        return math.inf
    headroom = max(float(max_factor) - 1.0, 0.0)
    exceed = abs(offset) / size
    if headroom <= 0.0 or float(catchup) <= 0.0:
        return max_speed * size
    factor = 1.0 + headroom * math.tanh(catchup * exceed / headroom)
    return max_speed * size * factor


def window_smoothing_params(
    base: SmoothingParams,
    *,
    pan_speed: float = DEFAULT_WINDOW_PAN_SPEED,
) -> SmoothingParams:
    """多人分屏小窗口**专用**的镜头参数（纯函数，由用户档位推导）。

    单窗口的镜头是按"整幅画面"标定的：跟手档下平移上限是 **1.5 个框宽/秒**，人物偏离
    边界时还会一路放宽到 **4 倍（= 6 个框宽/秒）**。整幅画面里这只是"跟得紧"，但分屏
    窗口只有一格大、放大倍数高，同样的相对速度在屏幕上的位移被放得很好看 ——
    人物在窗口里稍有动作，镜头就飞快地平移 / 推拉，看着像"画面在快进"。

    因此分屏窗口换成一套"正常速度"的镜头：

    1. **按秒封顶**（``pan_speed``）：平移与缩放都统一成"每秒最多移动窗口自身的
       ``pan_speed`` 倍"，与帧率无关；并且几乎不因人物偏离而加速
       （``max_speed_factor = :data:`WINDOW_MAX_SPEED_FACTOR`，只有单窗口的一半不到），
       于是镜头永远是同一个慢速，不会甩；
    2. **更懒**：自由活动区放大到两倍以上（不低于 :data:`WINDOW_MIN_DEADZONE`）——
       人物在窗口里说话、晃动时镜头**完全不动**，只有快到窗口边缘才平移；
    3. **更柔和**：平滑 / 缩放 / 加速度系数按比例减小，起步与收尾都不急。

    只改镜头运动，不动构图：取景框的比例与尺寸算法与单窗口完全一致。
    """
    speed = float(pan_speed)
    if not speed > 0.0:
        speed = DEFAULT_WINDOW_PAN_SPEED
    return replace(
        base,
        alpha=float(base.alpha) * WINDOW_ALPHA_SCALE,
        deadzone=max(float(base.deadzone) * WINDOW_DEADZONE_SCALE, WINDOW_MIN_DEADZONE),
        zoom_alpha=float(base.zoom_alpha) * WINDOW_ZOOM_ALPHA_SCALE,
        zoom_deadzone=max(
            float(base.zoom_deadzone) * WINDOW_ZOOM_DEADZONE_SCALE, WINDOW_MIN_ZOOM_DEADZONE
        ),
        # "每秒 pan_speed 个窗口"换算成"每个基准帧多少"：_apply_fps 会再按真实帧率折算
        max_speed=min(float(base.max_speed), speed / REFERENCE_FPS),
        accel=float(base.accel) * WINDOW_ACCEL_SCALE,
        catchup=WINDOW_CATCHUP,
        max_speed_factor=WINDOW_MAX_SPEED_FACTOR,
    )


class BoxSmoother:
    """把所有目标取景框串成一条连续、稳定、不易看晕的轨迹。"""

    def __init__(
        self,
        ratio: AspectRatio,
        *,
        params: SmoothingParams | None = None,
        alpha: float | None = None,
        deadzone: float | None = None,
        zoom_alpha: float | None = None,
        zoom_deadzone: float | None = None,
        max_speed: float | None = None,
        accel: float | None = None,
        hold_frames: int | None = None,
        scan_seconds: float | None = None,
        no_person_mode: str | None = None,
        no_person_seconds: float | None = None,
        fps: float | None = None,
    ) -> None:
        """
        :param ratio: 目标比例，平滑后宽高比恒等于它（不变量 3）
        :param params: 一组平滑参数；下面的具名参数会覆盖其中对应字段
        :param alpha: 中心点 EMA 系数 (0, 1]
        :param deadzone: 自由活动区（相对取景框边长），目标位移小于它时不移动
        :param zoom_alpha: 缩放 EMA 系数 (0, 1]，通常比 ``alpha`` 小
        :param zoom_deadzone: 缩放死区（相对取景框高度）
        :param max_speed: 平移速度上限（相对框长 / 帧），0 = 不限制
        :param accel: 加速平滑 (0, 1]，越小起步越柔和
        :param hold_frames: 丢失目标后保持上一帧取景框的最大帧数
        :param scan_seconds: 空镜巡视的单程时长（秒）；``0`` = 关闭巡视
        :param no_person_mode: 没有主要人物时的显示方式（见 :data:`NO_PERSON_MODES`）
        :param no_person_seconds: 拉远过渡的时长（秒）；``0`` = 立即切换
        :param fps: 源视频帧率（换算"每秒感受"用）；不知道时留空，先用 :meth:`set_fps`
                    补上 —— 越早告知，镜头速度越准
        """
        base = params or DEFAULT_SMOOTHING_PARAMS
        overrides = {
            "alpha": alpha,
            "deadzone": deadzone,
            "zoom_alpha": zoom_alpha,
            "zoom_deadzone": zoom_deadzone,
            "max_speed": max_speed,
            "accel": accel,
            "hold_frames": hold_frames,
            "scan_seconds": scan_seconds,
            "no_person_mode": no_person_mode,
            "no_person_seconds": no_person_seconds,
            "fps": fps,
        }
        given = {key: value for key, value in overrides.items() if value is not None}
        self.params = replace(base, **given) if given else base

        self.ratio = ratio
        self._box: CropBox | None = None
        self._missing = 0
        self._ever_had_target = False
        #: 上一帧的实际步长（像素/帧），用于限制加速度
        self._vx = 0.0
        self._vy = 0.0
        self._vh = 0.0
        #: 空镜巡视的行程进度（0~1 一个来回）；``-1`` = 尚未开始巡视
        self._scan_phase = _SCAN_IDLE
        #: 无人物过渡：进度 0~1、起点（人物还在时的取景框）与最近一帧的几何
        self._fit_progress = 0.0
        self._fit_from: CropBox | None = None
        self._fit_transition: TransitionFrame | None = None
        self.frames = 0
        self.holds = 0
        self.centers = 0
        self.scans = 0
        #: 换算到当前帧率后的等效参数（``_apply_fps`` 维护）
        self._apply_fps()

    # ------------------------------------------------------------- 只读属性
    @property
    def alpha(self) -> float:
        return self.params.alpha

    @property
    def deadzone(self) -> float:
        return self.params.deadzone

    @property
    def zoom_alpha(self) -> float:
        return self.params.zoom_alpha

    @property
    def zoom_deadzone(self) -> float:
        return self.params.zoom_deadzone

    @property
    def max_speed(self) -> float:
        return self.params.max_speed

    @property
    def accel(self) -> float:
        return self.params.accel

    @property
    def hold_frames(self) -> int:
        return self.params.hold_frames

    @property
    def scan_seconds(self) -> float:
        """空镜巡视的单程时长（秒）；``0`` 表示关闭巡视。"""
        return self.params.scan_seconds

    @property
    def no_person_mode(self) -> str:
        """没有主要人物时的显示方式（见 :data:`NO_PERSON_MODES`）。"""
        return self.params.no_person_mode

    @property
    def no_person_seconds(self) -> float:
        """从"人物取景框"拉远到无人物显示的时长（秒）；``0`` = 立即切换。"""
        return self.params.no_person_seconds

    @property
    def fit_progress(self) -> float:
        """无人物过渡的进度：0 = 仍按人物取景框取景，1 = 已完全切到无人物显示。"""
        return self._fit_progress

    @property
    def transition(self) -> TransitionFrame | None:
        """最近一帧无人物过渡的几何（仅 ``fit`` / ``tiles`` 兜底帧有值）。"""
        return self._fit_transition

    # ------------------------------------------------- 只读属性（帧率换算后）
    @property
    def fps(self) -> float | None:
        """当前换算用的源帧率；``None`` 表示按 :data:`REFERENCE_FPS` 处理。"""
        return self.params.fps

    @property
    def effective_alpha(self) -> float:
        """换算到当前帧率后的**每帧**平滑系数（未给出 ``fps`` 时等于配置值）。"""
        return self._alpha

    @property
    def effective_zoom_alpha(self) -> float:
        """换算到当前帧率后的**每帧**缩放系数。"""
        return self._zoom_alpha

    @property
    def effective_max_speed(self) -> float:
        """换算到当前帧率后的**每帧**平移速度上限（相对框长）。"""
        return self._speed_cap

    @property
    def effective_hold_frames(self) -> int:
        """换算到当前帧率后的保持帧数（保证"保持多少秒"不随帧率变化）。"""
        return self._hold

    def set_fps(self, fps: float | None) -> None:
        """告知源视频的真实帧率：镜头跟随的"每秒感受"由此校准。

        典型用法是在探测到视频元信息之后、开始逐帧处理之前调用一次
        （``CropPipeline.set_fps`` 会代为转发）。``None`` 表示不换算；
        ``fps <= 0`` 这类无效值直接忽略 —— 探测失败时不能把镜头参数改坏。
        """
        if fps is not None and float(fps) <= 0.0:
            return
        self.params = replace(self.params, fps=None if fps is None else float(fps))
        self._apply_fps()

    def set_params(self, params: SmoothingParams) -> None:
        """整体替换平滑参数（分屏窗口换成"更稳"的一套时用）。

        只换镜头参数，**位置 / 速度 / 帧计数状态全部保留**：切换后画面不会跳。
        """
        self.params = params
        self._apply_fps()

    def _apply_fps(self) -> None:
        """把"每秒感受"标定的参数换算成当前帧率下的等效每帧值。"""
        scale = _time_scale(self.params.fps)
        self._alpha = _normalize_alpha(self.params.alpha, scale)
        self._zoom_alpha = _normalize_alpha(self.params.zoom_alpha, scale)
        self._speed_cap = float(self.params.max_speed) * scale
        if self.params.fps is None or self.params.hold_frames <= 0:
            self._hold = int(self.params.hold_frames)  # 0 = 不保持，不能被换算成 1
        else:
            # 保持时长按秒恒定：帧率翻倍，允许保持的帧数也翻倍
            self._hold = max(1, int(round(self.params.hold_frames / scale)))
        # 巡视同理按秒标定：每帧推进的相位 = 1 / (单程秒数 × 帧率 × 2)，帧率无关
        scan_seconds = float(self.params.scan_seconds)
        self._scan_step = (
            0.0
            if scan_seconds <= 0.0
            else scale / (2.0 * scan_seconds * REFERENCE_FPS)
        )
        # 无人物拉远过渡同样按秒标定：每帧推进"整段过渡的几分之一"
        no_person_seconds = float(self.params.no_person_seconds)
        self._fit_step = (
            0.0
            if no_person_seconds <= 0.0
            else scale / (no_person_seconds * REFERENCE_FPS)
        )

    @property
    def box(self) -> CropBox | None:
        """最近一次输出的取景框。"""
        return self._box

    @property
    def missing(self) -> int:
        """连续丢失目标的帧数。"""
        return self._missing

    def reset(self) -> None:
        """清空状态（处理下一段视频前调用）。"""
        self._box = None
        self._missing = 0
        self._ever_had_target = False
        self._vx = 0.0
        self._vy = 0.0
        self._vh = 0.0
        self._scan_phase = _SCAN_IDLE
        self._fit_progress = 0.0
        self._fit_from = None
        self._fit_transition = None
        self.frames = 0
        self.holds = 0
        self.centers = 0
        self.scans = 0

    # ----------------------------------------------------------------- 更新
    def update(
        self,
        target: CropBox | None,
        frame_size: tuple[float, float],
        *,
        fallback_dst: tuple[int, int, int, int] | None = None,
    ) -> CropBox:
        """喂入本帧的目标取景框（``None`` 表示本帧没找到人），返回平滑后的取景框。

        ``fallback_dst`` 只在"没有主要人物 + 显示方式为全景 + 特写"时用得上：
        由调用方给出全景窗口在画布上的格子，于是画面会**直接收进那一格**
        （而不是先拉远到画布正中再跳上去）。
        """
        frame_w, frame_h = float(frame_size[0]), float(frame_size[1])
        self.frames += 1

        if target is None:
            self._missing += 1
            if (
                self._box is not None
                and self._ever_had_target
                and self._missing <= self._hold
            ):
                self.holds += 1
                self._box = replace(self._box, mode=MODE_HOLD)
                # 保持期间速度归零：重新跟随时从静止柔和起步，不会"弹"一下
                self._vx = self._vy = self._vh = 0.0
                return self._box

            fallback = self.params.no_person_mode
            if fallback in (MODE_FIT, MODE_TILES):
                return self._fit_output(frame_w, frame_h, fallback, fallback_dst)

            scan = self._scan_target(frame_w, frame_h) if fallback == MODE_SCAN else None
            if scan is not None:
                target = scan
                self.scans += 1
                mode = MODE_SCAN
            else:
                target = center_box((frame_w, frame_h), self.ratio)
                self.centers += 1
                mode = MODE_CENTER
        else:
            self._missing = 0
            self._ever_had_target = True
            # 人物回来了：巡视行程与拉远过渡一并作废，从"整幅画面"平滑缩回人物身上
            self._scan_phase = _SCAN_IDLE
            self._fit_progress = 0.0
            self._fit_from = None
            self._fit_transition = None
            mode = target.mode

        fitted = target.fit_to(frame_w, frame_h)
        if self._box is None:
            # 首帧不插值，避免从"空"平滑过来
            self._box = replace(fitted, mode=mode)
            self._vx = self._vy = self._vh = 0.0
            return self._box

        current = self._box
        if mode in (MODE_CENTER, MODE_SCAN):
            # 兜底回中 / 空镜巡视：目标要么是画面正中、要么一直在缓慢移动，此时不能设
            # 死区 —— 否则取景框会停在离目标"一个死区"的位置上，等于把巡视卡死。
            # 速度限幅与加速平滑仍然生效，回中与巡视的起步依旧平缓。
            zone = zoom_zone = 0.0
        else:
            # 自由活动区按"框长边"取，横竖两个方向感受一致
            zone = self.deadzone * max(current.w, current.h)
            zoom_zone = self.zoom_deadzone * current.h

        center_x, self._vx = self._advance(
            current.cx, fitted.cx, current.w, zone, self._alpha, self._vx
        )
        center_y, self._vy = self._advance(
            current.cy, fitted.cy, current.h, zone, self._alpha, self._vy
        )
        # 缩放单独控制：更慢、死区更大，压住"呼吸式变焦"
        height, self._vh = self._advance(
            current.h, fitted.h, current.h, zoom_zone, self._zoom_alpha, self._vh
        )
        width = height * self.ratio.value

        box = CropBox(
            center_x - width / 2.0,
            center_y - height / 2.0,
            width,
            height,
            mode=mode,
        )
        self._box = box.fit_to(frame_w, frame_h)
        return self._box

    def snap_to(self, target: CropBox | None, frame_size: tuple[float, float]) -> CropBox:
        """把取景框**直接就位**：不做平滑、不横扫，速度归零（"换人就位"用）。

        平滑器会把"离目标很远"处理成一段较长的镜头运动 —— 分屏窗口换人时，那就是
        整块画面从旧人物快速扫到新人物，观众看到的就是"画面在快进"。换人时观众
        期待的是"切镜头"，所以这里直接把框放到目标位置（本帧即到位）。

        ``target`` 为空（本帧没有人物）时退回 :meth:`update` 的兜底逻辑。
        """
        if target is None:
            return self.update(None, frame_size)

        frame_w, frame_h = float(frame_size[0]), float(frame_size[1])
        self.frames += 1
        self._missing = 0
        self._ever_had_target = True
        self._vx = self._vy = self._vh = 0.0
        self._scan_phase = _SCAN_IDLE
        self._fit_progress = 0.0
        self._fit_from = None
        self._fit_transition = None
        self._box = target.fit_to(frame_w, frame_h)
        return self._box

    # ------------------------------------------------------------- 内部逻辑
    def fit_transition(
        self,
        frame_size: tuple[float, float],
        dst_to: tuple[int, int, int, int] | None = None,
    ) -> TransitionFrame:
        """无人物兜底的几何：从"人物还在时的取景框"过渡到"整幅画面"。

        :param dst_to: 画布上的终点矩形；默认是"整幅画面等比放入画布"的位置
                       （``tiles`` 模式下由调用方传入全景格子，画面直接收进上格）
        """
        frame_w, frame_h = float(frame_size[0]), float(frame_size[1])
        out_w, out_h = self.ratio.target_size
        start = (
            self._fit_from
            if self._fit_from is not None
            else center_box((frame_w, frame_h), self.ratio)
        )
        target = dst_to if dst_to is not None else fit_rect((frame_w, frame_h), (out_w, out_h))
        return transition_frame(
            (start.x, start.y, start.w, start.h),
            (0.0, 0.0, frame_w, frame_h),
            (0.0, 0.0, float(out_w), float(out_h)),
            (float(target[0]), float(target[1]), float(target[2]), float(target[3])),
            self._fit_progress,
        )

    def _fit_output(
        self,
        frame_w: float,
        frame_h: float,
        mode: str,
        dst_to: tuple[int, int, int, int] | None,
    ) -> CropBox:
        """无人物兜底帧：推进过渡进度，返回"正在取出的那一块画面"。

        返回的框比例会从目标比例一路过渡到源比例（:data:`MODE_FIT` 的说明里有交代），
        真正贴到画布上的合成由裁剪层按 :attr:`transition` 完成。
        """
        if self._box is None:
            # 视频一开始就没人：没有"从哪来"，直接落在最终状态，不做无谓的拉远
            self._fit_from = None
            self._fit_progress = 1.0
        else:
            if self._fit_from is None:
                self._fit_from = self._box
            self._fit_progress = (
                min(self._fit_progress + self._fit_step, 1.0)
                if self._fit_step > 0.0
                else 1.0
            )

        transition = self.fit_transition((frame_w, frame_h), dst_to)
        self._fit_transition = transition
        x, y, width, height = transition.src
        # 重新跟人时从静止柔和起步，不会"弹"一下
        self._vx = self._vy = self._vh = 0.0
        self._box = CropBox(x, y, width, height, mode=mode)
        return self._box

    def _scan_target(self, frame_w: float, frame_h: float) -> CropBox | None:
        """空镜巡视的目标取景框：沿"画面多出来的那一侧"缓慢往返。

        以"画面正中的最大等比取景框"为基准 —— 它的宽或高中必然有一边已经等于画面
        尺寸，另一边还差 ``span`` 像素，这段多出来的画面正是居中裁剪时被丢掉的内容
        （横屏素材转竖屏时就是左右两边各一大块）。让取景框在这段区间里按正弦缓动
        往返（一端正好贴住画面左 / 上边，另一端正好贴住右 / 下边），就能用几个来回
        把整幅画面看完。

        :returns: 本帧的巡视目标框；没有可巡视的空间（源比例已与目标一致）、
                  或巡视被关闭（``scan_seconds == 0``）时返回 ``None``，由调用方退回回中。
        """
        if self._scan_step <= 0.0:
            return None

        base = center_box((frame_w, frame_h), self.ratio)
        span_x = max(frame_w - base.w, 0.0)
        span_y = max(frame_h - base.h, 0.0)
        horizontal = span_x >= span_y
        span = span_x if horizontal else span_y
        if span < SCAN_MIN_SPAN_PX:
            return None

        if self._scan_phase < 0.0:
            # 首次巡视：从"当前取景框所在的位置"起步，免得先倒着跑到一端再出发
            current = 0.0 if self._box is None else (self._box.x if horizontal else self._box.y)
            self._scan_phase = _phase_for_offset(current, span)
        self._scan_phase = (self._scan_phase + self._scan_step) % 1.0

        # 正弦缓动：两端速度为 0、中间最快，于是折返是"减速 → 停 → 反向"，
        # 不会像匀速往返那样在端点出现一次生硬的急停急起。
        # 偏移量本身就是画面上合法的最左 / 最上位置（0 ~ span），因此全程不越界。
        offset = (0.5 - 0.5 * math.cos(2.0 * math.pi * self._scan_phase)) * span
        if horizontal:
            return replace(base, x=offset, mode=MODE_SCAN)
        return replace(base, y=offset, mode=MODE_SCAN)

    def _advance(
        self,
        current: float,
        target: float,
        size: float,
        zone: float,
        alpha: float,
        previous_step: float,
    ) -> tuple[float, float]:
        """单轴推进：软死区 → 速度限幅 → 加速平滑，返回 ``(新位置, 本帧步长)``。

        :param size: 该轴上的框尺寸（用于把速度上限换算成像素）
        :param zone: 自由活动区（像素），小于它时完全不动
        """
        offset = _soft_zone(target - current, zone)
        limit = _speed_limit(
            offset, size, self._speed_cap, self.params.catchup, self.params.max_speed_factor
        )
        step = min(max(alpha * offset, -limit), limit)

        if step * previous_step > 0.0 and abs(step) < abs(previous_step):
            # 同向减速：立即跟上（像阻尼一样收住），避免冲过目标后反复回摆
            return current + step, step

        # 加速 / 反向：限制每帧的速度变化量，起步不"窜"
        max_delta = limit if self.accel >= 1.0 else self.accel * limit
        delta = min(max(step - previous_step, -max_delta), max_delta)
        step = previous_step + delta
        return current + step, step
