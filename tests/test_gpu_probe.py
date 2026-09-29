"""硬件自检与 PyTorch 构建推荐（纯函数，不联网、不依赖真实显卡）。"""

from __future__ import annotations

import pytest

from video2personvideo.utils.gpu_probe import (
    BACKEND_CPU,
    BACKEND_CUDA,
    BACKEND_MPS,
    BACKEND_ROCM,
    BACKEND_XPU,
    VENDOR_AMD,
    VENDOR_APPLE,
    VENDOR_INTEL,
    VENDOR_NVIDIA,
    GpuInfo,
    HardwareProfile,
    classify_gpu,
    is_virtual_gpu,
    normalize_gpu_name,
    probe_hardware,
)
from video2personvideo.utils.torch_backends import (
    backend_by_key,
    describe_option,
    gpu_backend_labels,
    recommend_backend,
)


# ------------------------------------------------------------------ 造数据
def make_profile(**kwargs) -> HardwareProfile:
    base: dict = {
        "os_name": "Windows",
        "arch": "AMD64",
        "is_windows": True,
        "cpu_name": "Test CPU",
        "cpu_cores": 8,
        "python_version": "3.13.0",
    }
    base.update(kwargs)
    return HardwareProfile(**base)


def nvidia_profile(
    name: str = "NVIDIA GeForce RTX 4090",
    compute: tuple[int, int] | None = (8, 9),
    cuda: str | None = "12.6",
    **kwargs,
):
    model = classify_gpu(name)
    gpu = GpuInfo(
        name=name,
        vendor=VENDOR_NVIDIA,
        compute=compute if compute is not None else model.compute,
        backend=BACKEND_CUDA,
        family=model.family,
        architecture=model.architecture,
        legacy=model.legacy,
        driver_version="560.94",
    )
    return make_profile(gpus=[gpu], cuda_driver_version=cuda, **kwargs)


# ------------------------------------------------------------------ 型号库
#: 市面上主流型号 → 期望的（厂商, 架构, 算力, 后端）
MODEL_CASES = [
    # NVIDIA 消费级
    ("NVIDIA GeForce RTX 5090", VENDOR_NVIDIA, "Blackwell", (12, 0), BACKEND_CUDA),
    ("NVIDIA GeForce RTX 5080", VENDOR_NVIDIA, "Blackwell", (12, 0), BACKEND_CUDA),
    ("NVIDIA GeForce RTX 4090", VENDOR_NVIDIA, "Ada Lovelace", (8, 9), BACKEND_CUDA),
    ("NVIDIA GeForce RTX 4060 Laptop GPU", VENDOR_NVIDIA, "Ada Lovelace", (8, 9), BACKEND_CUDA),
    ("NVIDIA GeForce RTX 3080 Ti", VENDOR_NVIDIA, "Ampere", (8, 6), BACKEND_CUDA),
    ("NVIDIA GeForce RTX 3050", VENDOR_NVIDIA, "Ampere", (8, 6), BACKEND_CUDA),
    ("NVIDIA GeForce RTX 2060 SUPER", VENDOR_NVIDIA, "Turing", (7, 5), BACKEND_CUDA),
    ("NVIDIA GeForce GTX 1660 Ti", VENDOR_NVIDIA, "Turing", (7, 5), BACKEND_CUDA),
    ("NVIDIA GeForce GTX 1080 Ti", VENDOR_NVIDIA, "Pascal", (6, 1), BACKEND_CUDA),
    ("NVIDIA GeForce GTX 950M", VENDOR_NVIDIA, "Maxwell", (5, 2), BACKEND_CUDA),
    ("NVIDIA GeForce GTX 750 Ti", VENDOR_NVIDIA, "Kepler", (3, 7), BACKEND_CUDA),
    ("NVIDIA GeForce MX250", VENDOR_NVIDIA, "Turing", (7, 5), BACKEND_CUDA),
    # NVIDIA 数据中心 / 工作站
    ("NVIDIA A100-SXM4-80GB", VENDOR_NVIDIA, "Ampere", (8, 0), BACKEND_CUDA),
    ("NVIDIA H100 PCIe", VENDOR_NVIDIA, "Hopper", (9, 0), BACKEND_CUDA),
    ("NVIDIA RTX 6000 Ada Generation", VENDOR_NVIDIA, "Ada Lovelace", (8, 9), BACKEND_CUDA),
    ("NVIDIA RTX A4000", VENDOR_NVIDIA, "Ampere", (8, 6), BACKEND_CUDA),
    ("NVIDIA L40S", VENDOR_NVIDIA, "Ada Lovelace", (8, 9), BACKEND_CUDA),
    # AMD
    ("AMD Radeon RX 9070 XT", VENDOR_AMD, "RDNA 4", None, BACKEND_ROCM),
    ("AMD Radeon RX 7900 XTX", VENDOR_AMD, "RDNA 3", None, BACKEND_ROCM),
    ("AMD Radeon RX 6800 XT", VENDOR_AMD, "RDNA 2", None, BACKEND_ROCM),
    ("AMD Radeon RX 5700 XT", VENDOR_AMD, "RDNA", None, BACKEND_ROCM),
    ("AMD Radeon RX 580", VENDOR_AMD, "GCN 4 (Polaris)", None, BACKEND_ROCM),
    ("AMD Radeon RX Vega 64", VENDOR_AMD, "Vega / GCN", None, BACKEND_ROCM),
    ("AMD Instinct MI250X", VENDOR_AMD, "CDNA", None, BACKEND_ROCM),
    # Intel
    ("Intel(R) Arc(TM) B580 Graphics", VENDOR_INTEL, "Battlemage", None, BACKEND_XPU),
    ("Intel(R) Arc(TM) A770 Graphics", VENDOR_INTEL, "Alchemist", None, BACKEND_XPU),
    ("Intel(R) Iris(R) Xe Graphics", VENDOR_INTEL, "集成显卡", None, BACKEND_CPU),
    ("Intel(R) UHD Graphics 630", VENDOR_INTEL, "集成显卡", None, BACKEND_CPU),
    # Apple / 其它
    ("Apple M3 Max", VENDOR_APPLE, "Apple GPU", None, BACKEND_MPS),
    ("Apple M1", VENDOR_APPLE, "Apple GPU", None, BACKEND_MPS),
]


@pytest.mark.parametrize("name,vendor,arch,compute,backend", MODEL_CASES)
def test_classify_gpu_covers_mainstream_models(
    name: str, vendor: str, arch: str, compute, backend: str
) -> None:
    model = classify_gpu(name)
    assert model.vendor == vendor, name
    assert model.architecture == arch, name
    assert model.compute == compute, name
    assert model.backend == backend, name


def test_classify_gpu_handles_brand_marks_and_case() -> None:
    assert classify_gpu("intel(r) arc(tm) a750 graphics").architecture == "Alchemist"
    assert normalize_gpu_name("Intel(R)  Arc(TM)   A770") == "Intel Arc A770"


def test_classify_gpu_unknown_falls_back() -> None:
    model = classify_gpu("Some Random Display Adapter")
    assert model.vendor == "unknown"
    assert model.backend == BACKEND_CPU


# ------------------------------------------------------------- 虚拟显卡过滤
@pytest.mark.parametrize(
    "name",
    [
        "GameViewer Virtual Display Adapter",
        "OrayIddDriver Device",
        "Microsoft Basic Display Adapter",
        "Parsec Virtual Display",
        "Sharing Monitor",
        "USB Display Device",
    ],
)
def test_virtual_gpus_are_ignored(name: str) -> None:
    assert is_virtual_gpu(name) is True


@pytest.mark.parametrize(
    "name",
    ["NVIDIA GeForce RTX 4090", "AMD Radeon RX 7900 XTX", "Intel(R) UHD Graphics 630"],
)
def test_real_gpus_are_not_treated_as_virtual(name: str) -> None:
    assert is_virtual_gpu(name) is False


# ------------------------------------------------------------------ 自检结果
def test_probe_hardware_returns_usable_profile() -> None:
    profile = probe_hardware(force=True)
    assert profile.os_name
    assert profile.python_version
    assert profile.cpu_cores > 0
    assert isinstance(profile.summary_lines(), list)
    # 结果里必须包含"显卡"这一项（没有独显也要明确说出来）
    keys = [key for key, _ in profile.summary_lines()]
    assert "显卡" in keys
    assert profile.primary_gpu is None or profile.primary_gpu.name


def test_primary_gpu_prefers_discrete_card() -> None:
    profile = make_profile(
        gpus=[
            GpuInfo(name="Intel(R) UHD Graphics 630", vendor=VENDOR_INTEL, integrated=True),
            GpuInfo(
                name="NVIDIA GeForce RTX 4090",
                vendor=VENDOR_NVIDIA,
                compute=(8, 9),
                memory_mb=24576,
            ),
        ]
    )
    assert profile.has_discrete_gpu
    assert profile.primary_gpu is not None and profile.primary_gpu.vendor == VENDOR_NVIDIA


def test_profile_describe_contains_key_facts() -> None:
    profile = nvidia_profile()
    text = profile.summary_lines()[5][1]
    assert "RTX 4090" in text
    assert "sm_89" in text


def test_probe_cache_can_be_cleared() -> None:
    from video2personvideo.utils.gpu_probe import clear_cache

    first = probe_hardware()
    assert probe_hardware() is first
    clear_cache()
    assert probe_hardware() is not first


# ------------------------------------------------------------------ 推荐逻辑
def test_recommend_cuda_matches_driver_capability() -> None:
    rec = recommend_backend(nvidia_profile(cuda="12.6"))
    assert rec.key == "cu126"
    assert rec.is_gpu
    assert "RTX 4090" in rec.reason


def test_recommend_newest_build_for_blackwell() -> None:
    rec = recommend_backend(
        nvidia_profile(name="NVIDIA GeForce RTX 5090", compute=(12, 0), cuda="12.9")
    )
    assert rec.key == "cu129"
    # cu126 不含 sm_120，不能选
    assert all(item.backend.key != "cu126" or not item.usable for item in rec.options)


def test_recommend_falls_back_to_cpu_when_driver_too_old_for_blackwell() -> None:
    rec = recommend_backend(
        nvidia_profile(name="NVIDIA GeForce RTX 5090", compute=(12, 0), cuda="12.4")
    )
    assert rec.key == "cpu"
    assert any("CUDA" in warning for warning in rec.warnings)


def test_recommend_keeps_pascal_on_older_cuda() -> None:
    """Pascal（sm_61）在 cu128 之后被移除，应落在 cu126 / cu124 一档。"""
    rec = recommend_backend(
        nvidia_profile(name="NVIDIA GeForce GTX 1080 Ti", compute=(6, 1), cuda="12.8")
    )
    assert rec.key == "cu126"


def test_recommend_uses_cuda118_for_very_old_driver() -> None:
    rec = recommend_backend(
        nvidia_profile(name="NVIDIA GeForce GTX 1060", compute=(6, 1), cuda="11.8")
    )
    assert rec.key == "cu118"


def test_recommend_kepler_falls_back_to_cpu() -> None:
    rec = recommend_backend(
        nvidia_profile(name="NVIDIA GeForce GTX 750 Ti", compute=(3, 7), cuda="11.8")
    )
    assert rec.key == "cpu"
    assert any("Maxwell" in warning or "Kepler" in warning for warning in rec.warnings)


def test_recommend_amd_on_linux_uses_rocm() -> None:
    profile = make_profile(
        os_name="Linux",
        is_linux=True,
        is_windows=False,
        gpus=[GpuInfo(name="AMD Radeon RX 7900 XTX", vendor=VENDOR_AMD, backend=BACKEND_ROCM)],
    )
    rec = recommend_backend(profile)
    assert rec.key.startswith("rocm")
    assert "AMD" in rec.reason


def test_recommend_amd_on_windows_uses_cpu_with_explanation() -> None:
    profile = make_profile(
        gpus=[GpuInfo(name="AMD Radeon RX 7900 XTX", vendor=VENDOR_AMD, backend=BACKEND_ROCM)]
    )
    rec = recommend_backend(profile)
    assert rec.key == "cpu"
    assert any("ROCm" in warning for warning in rec.warnings)


def test_recommend_intel_arc_uses_xpu() -> None:
    profile = make_profile(
        gpus=[
            GpuInfo(
                name="Intel(R) Arc(TM) A770 Graphics",
                vendor=VENDOR_INTEL,
                backend=BACKEND_XPU,
            )
        ]
    )
    rec = recommend_backend(profile)
    assert rec.key == "xpu"
    assert any("实验" in warning for warning in rec.warnings)


def test_recommend_apple_silicon_uses_cpu_wheel_with_mps() -> None:
    profile = make_profile(
        os_name="Darwin", is_macos=True, is_windows=False, arch="arm64",
        gpus=[GpuInfo(name="Apple M3", vendor=VENDOR_APPLE, backend=BACKEND_MPS)],
    )
    rec = recommend_backend(profile)
    assert rec.key == "cpu"
    assert "MPS" in rec.reason


def test_recommend_cpu_only_machine() -> None:
    rec = recommend_backend(make_profile())
    assert rec.key == "cpu"
    assert not rec.is_gpu
    assert any("独立显卡" in warning for warning in rec.warnings)


def test_options_mark_platform_support() -> None:
    """ROCm 只在 Linux 出现，XPU 在 mac 上不可用。"""
    linux = make_profile(
        os_name="Linux", is_linux=True, is_windows=False,
        gpus=[GpuInfo(name="AMD Radeon RX 7900 XTX", vendor=VENDOR_AMD)],
    )
    keys = {option.backend.key for option in recommend_backend(linux).options}
    assert "rocm6.4" in keys
    assert "cpu" in keys

    mac = make_profile(
        os_name="Darwin", is_macos=True, is_windows=False, arch="arm64",
        gpus=[GpuInfo(name="Apple M2", vendor=VENDOR_APPLE, backend=BACKEND_MPS)],
    )
    mac_keys = {option.backend.key for option in recommend_backend(mac).options}
    assert "xpu" not in mac_keys
    assert mac_keys == {"cpu"}


def test_describe_option_explains_rejections() -> None:
    profile = nvidia_profile(name="NVIDIA GeForce RTX 5090", compute=(12, 0), cuda="12.6")
    option = describe_option(backend_by_key("cu126"), profile)
    assert option.usable is False
    assert "sm_120" in option.reason

    driver = describe_option(
        backend_by_key("cu128"),
        nvidia_profile(name="NVIDIA GeForce RTX 5090", compute=(12, 0), cuda="12.6"),
    )
    assert driver.usable is False
    assert "12.6" in driver.reason


def test_backend_by_key_is_forgiving() -> None:
    assert backend_by_key("cu126").key == "cu126"
    assert backend_by_key("CU126").key == "cu126"
    assert backend_by_key("不存在").key == "cpu"
    assert backend_by_key(None).key == "cpu"


def test_gpu_backend_labels_cover_all_options() -> None:
    labels = gpu_backend_labels()
    assert labels["cpu"]
    assert labels["cu118"]
    assert len(labels) == 11
