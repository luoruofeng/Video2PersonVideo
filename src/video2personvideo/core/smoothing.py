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
最多 ``hold_frames`` 帧；超时后平滑过渡到画面中心的等比取景框。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from .framing import MODE_CENTER, MODE_HOLD, CropBox, center_box
from .ratio import AspectRatio

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


@dataclass(frozen=True, slots=True)
class SmoothingParams:
    """镜头运动的稳定性参数（对应 ``AppConfig`` 的 ``smoothing_*`` 字段）。"""

    #: 取景框中心点的 EMA 系数，(0, 1]，越大越跟手
    alpha: float = 0.25
    #: 自由活动区（相对取景框边长）：人物在框内的位移小于它时镜头完全不动
    deadzone: float = 0.15
    #: 缩放（推拉）的 EMA 系数，通常明显小于 ``alpha``，避免呼吸式变焦
    zoom_alpha: float = 0.08
    #: 缩放死区（相对取景框高度）
    zoom_deadzone: float = 0.10
    #: 平移速度上限（相对框长 / 帧），0 = 不限制
    max_speed: float = 0.025
    #: 加速平滑，(0, 1]：越小起步越柔和，1 = 不限制加速度
    accel: float = 0.5
    #: 丢失目标后保持上一帧取景框的最大帧数（按 :data:`REFERENCE_FPS` 的时长标定）
    hold_frames: int = 30
    #: 源视频帧率；给出时把上面这些"每帧"参数换算成与帧率无关的等效值
    #: （``None`` = 不换算，即把源视频当作 :data:`REFERENCE_FPS`）
    fps: float | None = None

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
        if self.fps is not None and float(self.fps) <= 0.0:
            raise ValueError(f"fps 必须为正（帧率未知时用 None），当前为 {self.fps}")


#: 默认平滑参数（与 :class:`~video2personvideo.config.AppConfig` 的默认值一致）
DEFAULT_SMOOTHING_PARAMS = SmoothingParams()


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
            "默认：镜头较少移动，人物基本居中",
            alpha=0.25,
            deadzone=0.15,
            zoom_alpha=0.08,
            zoom_deadzone=0.10,
            max_speed=0.025,
        ),
        _preset(
            "active",
            "跟手",
            "镜头跟得紧、画面运动感强，适合素材本身运动幅度小的情况",
            alpha=0.45,
            deadzone=0.06,
            zoom_alpha=0.15,
            zoom_deadzone=0.05,
            max_speed=0.05,
        ),
    )
}

#: 默认档位
DEFAULT_CAMERA_PRESET = "standard"


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


def _speed_limit(offset: float, size: float, max_speed: float) -> float:
    """本帧允许的最大步长：基准速度上限 × 越界放大倍数（平滑饱和，不甩镜）。

    小偏离处斜率为 :data:`CATCHUP`（和线性放宽一样"跟得住"），大偏离处平滑饱和到
    :data:`MAX_SPEED_FACTOR` 倍 —— 人物跑起来时镜头变快但不"炸"，避免急促横摇。
    """
    if max_speed <= 0.0 or size <= 0.0:
        return math.inf
    headroom = max(MAX_SPEED_FACTOR - 1.0, 0.0)
    exceed = abs(offset) / size
    factor = 1.0 if headroom <= 0.0 else 1.0 + headroom * math.tanh(CATCHUP * exceed / headroom)
    return max_speed * size * factor


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
        self.frames = 0
        self.holds = 0
        self.centers = 0
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
        self.frames = 0
        self.holds = 0
        self.centers = 0

    # ----------------------------------------------------------------- 更新
    def update(
        self, target: CropBox | None, frame_size: tuple[float, float]
    ) -> CropBox:
        """喂入本帧的目标取景框（``None`` 表示本帧没找到人），返回平滑后的取景框。"""
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
            target = center_box((frame_w, frame_h), self.ratio)
            self.centers += 1
            mode = MODE_CENTER
        else:
            self._missing = 0
            self._ever_had_target = True
            mode = target.mode

        fitted = target.fit_to(frame_w, frame_h)
        if self._box is None:
            # 首帧不插值，避免从"空"平滑过来
            self._box = replace(fitted, mode=mode)
            self._vx = self._vy = self._vh = 0.0
            return self._box

        current = self._box
        if mode == MODE_CENTER:
            # 兜底回中：目标是画面正中（不会抖），此时不能设死区，否则会停在
            # 离中心"一个死区"的位置上；速度限幅与加速平滑仍然生效，回中依旧平缓。
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

    # ------------------------------------------------------------- 内部逻辑
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
        limit = _speed_limit(offset, size, self._speed_cap)
        step = min(max(alpha * offset, -limit), limit)

        if step * previous_step > 0.0 and abs(step) < abs(previous_step):
            # 同向减速：立即跟上（像阻尼一样收住），避免冲过目标后反复回摆
            return current + step, step

        # 加速 / 反向：限制每帧的速度变化量，起步不"窜"
        max_delta = limit if self.accel >= 1.0 else self.accel * limit
        delta = min(max(step - previous_step, -max_delta), max_delta)
        step = previous_step + delta
        return current + step, step
