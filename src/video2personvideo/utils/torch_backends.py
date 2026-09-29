"""PyTorch 构建（CUDA / ROCm / XPU / CPU）选择表与推荐逻辑。

一块显卡能装哪一套 PyTorch，取决于三件事：

1. **架构算力**（NVIDIA 的 ``sm_xx``）：官方 wheel 只编译了部分算力，
   例如 Blackwell（``sm_120``）必须用 ``cu128`` 及以后的构建，
   而 Pascal（``sm_61``）在 ``cu128`` 之后被移除；
2. **驱动能力**：``nvidia-smi`` 报出的 ``CUDA Version`` 是驱动支持的最高运行时，
   装的 CUDA 构建不能超过它；
3. **操作系统**：ROCm 只有 Linux，XPU 目前是 Windows / Linux。

本模块把这些规则写成一张纯数据的表，:func:`recommend_backend` 据此挑出唯一答案，
并按需要给出理由与备选项。**纯函数、不联网、不落地**，方便单测。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .gpu_probe import (
    BACKEND_CPU,
    BACKEND_CUDA,
    BACKEND_ROCM,
    BACKEND_XPU,
    VENDOR_AMD,
    VENDOR_INTEL,
    VENDOR_NVIDIA,
    HardwareProfile,
)
from .logger import get_logger

logger = get_logger(__name__)

#: PyTorch 官方 wheel 索引根地址
TORCH_INDEX_BASE = "https://download.pytorch.org/whl"

#: 各后端必须安装的两个包
TORCH_PACKAGES = ("torch", "torchvision")


@dataclass(frozen=True, slots=True)
class TorchBackend:
    """一套可安装的 PyTorch 构建。"""

    key: str
    label: str
    #: 该构建在官方索引里的 tag（= ``key``），如 ``cu126`` / ``rocm6.4`` / ``cpu``
    tag: str
    kind: str
    #: CUDA 运行时版本（``None`` 表示不带 CUDA）
    cuda_version: str | None = None
    #: 官方要求的最低 NVIDIA 驱动（Windows / Linux），仅用于界面提示
    min_driver_windows: tuple[int, int] | None = None
    min_driver_linux: tuple[int, int] | None = None
    #: 支持的算力区间（下界、上界，闭区间）
    min_compute: tuple[int, int] | None = None
    max_compute: tuple[int, int] | None = None
    #: 支持的操作系统（``platform.system()`` 的取值）
    platforms: tuple[str, ...] = ("Windows", "Linux")
    #: 是否带一点实验性质（装不上要能退回 CPU）
    experimental: bool = False
    note: str = ""

    @property
    def index_url(self) -> str:
        return f"{TORCH_INDEX_BASE}/{self.tag}"

    @property
    def package_index_url(self) -> str:
        return f"{self.index_url}/torch/"

    @property
    def pip_command(self) -> str:
        """可直接复制粘贴的 pip 命令。"""
        return (
            f"pip install {' '.join(TORCH_PACKAGES)} "
            f"--index-url {self.index_url}"
        )

    @property
    def display(self) -> str:
        suffix = ""
        if self.experimental:
            suffix = "（实验性）"
        return f"{self.label}{suffix}"


def _v(text: str) -> tuple[int, ...]:
    """把 ``"12.6"`` 解析成 ``(12, 6)``，便于比较。"""
    return tuple(int(part) for part in str(text).split(".") if part.isdigit())


#: 全部可选的构建：**新版本在前**（推荐逻辑取第一个满足条件的）
TORCH_BACKENDS: tuple[TorchBackend, ...] = (
    TorchBackend(
        key="cu130",
        label="CUDA 13.0（NVIDIA，最新）",
        tag="cu130",
        kind=BACKEND_CUDA,
        cuda_version="13.0",
        min_driver_windows=(580, 65),
        min_driver_linux=(580, 65),
        min_compute=(7, 5),
        max_compute=(12, 0),
        note="最新 CUDA 运行时，需要较新的驱动。",
    ),
    TorchBackend(
        key="cu129",
        label="CUDA 12.9（NVIDIA）",
        tag="cu129",
        kind=BACKEND_CUDA,
        cuda_version="12.9",
        min_driver_windows=(575, 51),
        min_driver_linux=(575, 51),
        min_compute=(7, 5),
        max_compute=(12, 0),
        note="支持 Blackwell（RTX 50 系）与更新的驱动。",
    ),
    TorchBackend(
        key="cu128",
        label="CUDA 12.8（NVIDIA）",
        tag="cu128",
        kind=BACKEND_CUDA,
        cuda_version="12.8",
        min_driver_windows=(570, 65),
        min_driver_linux=(570, 26),
        min_compute=(7, 0),
        max_compute=(12, 0),
        note="支持 Blackwell（RTX 50 系）；不再包含 Pascal 及更早架构。",
    ),
    TorchBackend(
        key="cu126",
        label="CUDA 12.6（NVIDIA，兼容性最好）",
        tag="cu126",
        kind=BACKEND_CUDA,
        cuda_version="12.6",
        min_driver_windows=(560, 76),
        min_driver_linux=(560, 28),
        min_compute=(5, 0),
        max_compute=(9, 0),
        note="覆盖面最广的一档：从 Maxwell 到 Hopper 都能跑。",
    ),
    TorchBackend(
        key="cu124",
        label="CUDA 12.4（NVIDIA，较老驱动）",
        tag="cu124",
        kind=BACKEND_CUDA,
        cuda_version="12.4",
        min_driver_windows=(550, 54),
        min_driver_linux=(550, 54),
        min_compute=(5, 0),
        max_compute=(9, 0),
        note="适合驱动停在 550 / 551 左右的机器。",
    ),
    TorchBackend(
        key="cu121",
        label="CUDA 12.1（NVIDIA，老驱动）",
        tag="cu121",
        kind=BACKEND_CUDA,
        cuda_version="12.1",
        min_driver_windows=(527, 41),
        min_driver_linux=(525, 60),
        min_compute=(5, 0),
        max_compute=(9, 0),
        note="驱动较老（52x / 53x）时的稳妥选择。",
    ),
    TorchBackend(
        key="cu118",
        label="CUDA 11.8（NVIDIA，最老的兼容档）",
        tag="cu118",
        kind=BACKEND_CUDA,
        cuda_version="11.8",
        min_driver_windows=(452, 39),
        min_driver_linux=(450, 80),
        min_compute=(5, 0),
        max_compute=(9, 0),
        note="驱动很老、或者显卡是 Pascal / Maxwell 老架构时用这一档。",
    ),
    TorchBackend(
        key="rocm6.4",
        label="ROCm 6.4（AMD，Linux）",
        tag="rocm6.4",
        kind=BACKEND_ROCM,
        platforms=("Linux",),
        note="AMD 显卡在 Linux 上的官方支持路径。",
    ),
    TorchBackend(
        key="rocm6.2",
        label="ROCm 6.2（AMD，Linux）",
        tag="rocm6.2",
        kind=BACKEND_ROCM,
        platforms=("Linux",),
        note="适合 RDNA 2/3 等较早的 Radeon 卡。",
    ),
    TorchBackend(
        key="xpu",
        label="XPU（Intel Arc）",
        tag="xpu",
        kind=BACKEND_XPU,
        platforms=("Windows", "Linux"),
        experimental=True,
        note="Intel Arc 独显专用，需要较新的 Intel 驱动。",
    ),
    TorchBackend(
        key="cpu",
        label="CPU（通用，无需显卡驱动）",
        tag="cpu",
        kind=BACKEND_CPU,
        platforms=("Windows", "Linux", "Darwin"),
        note="任何机器都能用；搭配合适的显卡驱动时速度最慢但最稳。",
    ),
)

BACKENDS_BY_KEY: dict[str, TorchBackend] = {item.key: item for item in TORCH_BACKENDS}

#: Apple Silicon 用 CPU 版 wheel（PyTorch 自带 MPS 后端）
DEFAULT_BACKEND_KEY = "cpu"


@dataclass(slots=True)
class BackendOption:
    """某个后端在"这台机器"上的可用性。"""

    backend: TorchBackend
    usable: bool
    reason: str = ""

    @property
    def label(self) -> str:
        return self.backend.display


@dataclass(slots=True)
class BackendRecommendation:
    """自检给出的推荐结论。"""

    backend: TorchBackend
    reason: str
    #: 需要提醒用户的点（不影响推荐结论）
    warnings: list[str] = field(default_factory=list)
    #: 同平台上其它可选项（供手动覆盖）
    options: list[BackendOption] = field(default_factory=list)

    @property
    def key(self) -> str:
        return self.backend.key

    @property
    def is_gpu(self) -> bool:
        return self.backend.kind != BACKEND_CPU


# ------------------------------------------------------------------ 判定工具
def version_at_least(value: str | None, minimum: tuple[int, ...]) -> bool:
    """``value`` 是否 ≥ ``minimum``（``None`` 视为"未知"，一律通过）。"""
    if not value:
        return True
    current = _v(value)
    if len(current) < len(minimum):
        current = (*current, *(0 for _ in range(len(minimum) - len(current))))
    return current >= minimum


def driver_supports(backend: TorchBackend, profile: HardwareProfile) -> bool:
    """驱动能力是否够：优先用 ``nvidia-smi`` 报的 CUDA 版本比大小。"""
    if backend.kind != BACKEND_CUDA or not backend.cuda_version:
        return True

    driver_cuda = profile.cuda_driver_version
    if driver_cuda:
        return _v(driver_cuda) >= _v(backend.cuda_version)

    # 没有 CUDA 版本时退回比驱动号（Windows 才能这么比）
    drivers = profile.gpus_of(VENDOR_NVIDIA)
    versions = [gpu.driver_version for gpu in drivers if gpu.driver_version]
    if not versions:
        # 连驱动号都没读到：不拦，交给安装失败兜底
        return True
    minimum = (
        backend.min_driver_windows if profile.is_windows else backend.min_driver_linux
    )
    if minimum is None:
        return True
    return all(version_at_least(item, minimum) for item in versions)


def supports_compute(backend: TorchBackend, compute: tuple[int, int] | None) -> bool:
    """该构建是否包含这张卡的算力（算力未知时不拦）。"""
    if backend.kind != BACKEND_CUDA or compute is None:
        return True
    if backend.min_compute is not None and compute < backend.min_compute:
        return False
    return not (backend.max_compute is not None and compute > backend.max_compute)


def backends_for_platform(profile: HardwareProfile) -> list[TorchBackend]:
    """当前操作系统上理论上可安装的后端（新版本在前）。"""
    os_name = profile.os_name
    return [item for item in TORCH_BACKENDS if os_name in item.platforms]


def describe_option(backend: TorchBackend, profile: HardwareProfile) -> BackendOption:
    """给一个后端算出"在这台机器上能不能用、为什么"。"""
    if profile.os_name not in backend.platforms:
        return BackendOption(backend, False, f"当前系统（{profile.os_name}）不支持")

    if backend.kind == BACKEND_CUDA:
        nvidia = [gpu for gpu in profile.gpus_of(VENDOR_NVIDIA) if not gpu.integrated]
        if not nvidia:
            return BackendOption(backend, False, "未检测到 NVIDIA 显卡")
        computes = [gpu.compute for gpu in nvidia if gpu.compute]
        if computes and not any(supports_compute(backend, item) for item in computes):
            top = max(computes)
            return BackendOption(
                backend,
                False,
                f"该构建不含 sm_{top[0]}{top[1]} 的算力（显卡架构过新或过旧）",
            )
        if not driver_supports(backend, profile):
            limit = profile.cuda_driver_version or "较旧"
            return BackendOption(
                backend, False, f"驱动支持的 CUDA 只到 {limit}，装不了 CUDA {backend.cuda_version}"
            )
        return BackendOption(backend, True, "可用")

    if backend.kind == BACKEND_ROCM:
        amd = [gpu for gpu in profile.gpus_of(VENDOR_AMD) if not gpu.integrated]
        if not amd:
            return BackendOption(backend, False, "未检测到 AMD 独立显卡")
        return BackendOption(backend, True, "可用（需要系统已安装 ROCm 运行时）")

    if backend.kind == BACKEND_XPU:
        intel = [gpu for gpu in profile.gpus_of(VENDOR_INTEL) if gpu.backend == BACKEND_XPU]
        if not intel:
            return BackendOption(backend, False, "未检测到 Intel Arc 独显")
        return BackendOption(backend, True, "可用（实验性，需要较新的 Intel 驱动）")

    if backend.kind == BACKEND_CPU:
        if profile.is_macos and profile.arch in {"arm64", "aarch64"}:
            return BackendOption(backend, True, "可用（自带 Metal / MPS 加速）")
        return BackendOption(backend, True, "可用")

    return BackendOption(backend, False, "未知后端")  # pragma: no cover


# ------------------------------------------------------------------ 推荐
def recommend_backend(profile: HardwareProfile) -> BackendRecommendation:
    """给出这台机器"应该装哪一套 PyTorch"。"""
    options = [describe_option(item, profile) for item in backends_for_platform(profile)]
    warnings: list[str] = []

    cpu = BACKENDS_BY_KEY[DEFAULT_BACKEND_KEY]

    # ① Apple Silicon：CPU 版 wheel 自带 MPS，无需额外构建
    if profile.is_macos and profile.arch in {"arm64", "aarch64"}:
        return BackendRecommendation(
            backend=cpu,
            reason="Apple Silicon：PyTorch 官方 wheel 已内置 Metal(MPS) 加速，装 CPU 版即可。",
            warnings=warnings,
            options=options,
        )

    nvidia = [gpu for gpu in profile.gpus_of(VENDOR_NVIDIA) if not gpu.integrated]
    amd = [gpu for gpu in profile.gpus_of(VENDOR_AMD) if not gpu.integrated]
    intel = [gpu for gpu in profile.gpus_of(VENDOR_INTEL) if gpu.backend == BACKEND_XPU]

    # ② NVIDIA：算力 + 驱动双重过滤后取最新的一档
    if nvidia:
        computes = [gpu.compute for gpu in nvidia if gpu.compute]
        best_compute = max(computes) if computes else None
        legacy = [gpu for gpu in nvidia if gpu.legacy]
        if legacy:
            if best_compute is None or best_compute < (5, 0):
                warnings.append(
                    f"{legacy[0].name} 是 {legacy[0].architecture} 架构，"
                    "官方 wheel 已不含它的算力，建议直接用 CPU 模式。"
                )
            else:
                warnings.append(
                    f"{legacy[0].name}（{legacy[0].architecture} 架构）只能用最老的 CUDA 11.8 "
                    "构建；如果安装失败请改用 CPU 模式。"
                )

        for option in options:
            if option.backend.kind != BACKEND_CUDA:
                continue
            if not option.usable:
                continue
            return BackendRecommendation(
                backend=option.backend,
                reason=_cuda_reason(option.backend, profile, best_compute),
                warnings=warnings,
                options=options,
            )

        rejected = [item for item in options if item.backend.kind == BACKEND_CUDA]
        detail = "；".join(f"{item.backend.key}：{item.reason}" for item in rejected[:3])
        warnings.append(
            "检测到 NVIDIA 显卡，但没有可用的 CUDA 构建（"
            f"{detail or '驱动过旧或显卡架构过新'}）。"
            "可以更新显卡驱动后重新自检，或先用 CPU 模式。"
        )
        return BackendRecommendation(
            backend=cpu,
            reason="显卡驱动 / 架构组合暂时匹配不到官方 CUDA 构建，先按 CPU 安装。",
            warnings=warnings,
            options=options,
        )

    # ③ AMD 独显：Linux 走 ROCm，Windows 没有官方构建
    if amd:
        rocm = [item for item in options if item.backend.kind == BACKEND_ROCM and item.usable]
        if rocm:
            return BackendRecommendation(
                backend=rocm[0].backend,
                reason=f"检测到 AMD 独显 {amd[0].name}，Linux 上使用 {rocm[0].backend.display}。",
                warnings=warnings,
                options=options,
            )
        warnings.append(
            f"检测到 AMD 独显 {amd[0].name}，但当前系统没有官方 PyTorch 构建"
            "（ROCm 只支持 Linux）。已回退到 CPU 模式，速度会明显慢一些。"
        )
        return BackendRecommendation(
            backend=cpu,
            reason="AMD 显卡在 Windows 上暂无官方 PyTorch 构建，按 CPU 安装。",
            warnings=warnings,
            options=options,
        )

    # ④ Intel Arc：可选 XPU（实验性）
    if intel:
        xpu = next(
            (item for item in options if item.backend.kind == BACKEND_XPU and item.usable), None
        )
        if xpu is not None:
            warnings.append("Intel Arc 的 XPU 构建属于实验性支持，安装失败时请改用 CPU。")
            return BackendRecommendation(
                backend=xpu.backend,
                reason=f"检测到 {intel[0].name}，可使用 Intel XPU 构建加速。",
                warnings=warnings,
                options=options,
            )

    warnings.append("未检测到可加速的独立显卡，使用 CPU 模式。")
    return BackendRecommendation(
        backend=cpu,
        reason="没有可用的独立显卡（或驱动不支持），按 CPU 安装，稳定可靠。",
        warnings=warnings,
        options=options,
    )


def _cuda_reason(
    backend: TorchBackend,
    profile: HardwareProfile,
    compute: tuple[int, int] | None,
) -> str:
    nvidia = [gpu for gpu in profile.gpus_of(VENDOR_NVIDIA) if not gpu.integrated]
    name = nvidia[0].name if nvidia else "NVIDIA 显卡"
    bits = [f"检测到 {name}"]
    if compute is not None:
        bits.append(f"算力 sm_{compute[0]}{compute[1]}")
    if profile.cuda_driver_version:
        bits.append(f"驱动支持 CUDA {profile.cuda_driver_version}")
    bits.append(f"选择 {backend.label}")
    return "，".join(bits) + "。"


def backend_by_key(key: str | None) -> TorchBackend:
    """按 key 取构建，未知 key 回退到 CPU（界面下拉传参用）。"""
    return BACKENDS_BY_KEY.get(str(key or "").strip().lower(), BACKENDS_BY_KEY[DEFAULT_BACKEND_KEY])


def gpu_backend_labels() -> dict[str, str]:
    """``{key: label}``，供界面下拉与文档使用。"""
    return {item.key: item.label for item in TORCH_BACKENDS}


__all__ = [
    "BACKENDS_BY_KEY",
    "DEFAULT_BACKEND_KEY",
    "TORCH_BACKENDS",
    "TORCH_INDEX_BASE",
    "TORCH_PACKAGES",
    "BackendOption",
    "BackendRecommendation",
    "TorchBackend",
    "backend_by_key",
    "backends_for_platform",
    "describe_option",
    "driver_supports",
    "gpu_backend_labels",
    "recommend_backend",
    "supports_compute",
    "version_at_least",
]
