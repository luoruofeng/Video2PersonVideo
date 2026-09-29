"""硬件自检：探测 CPU 与显卡型号，判断推理后端应该用哪一套 PyTorch。

设计要点：

* **只读不装**：本模块只做探测与归类，不下载、不安装（下载见 :mod:`.downloader`，
  挑选构建见 :mod:`.torch_backends`）。
* **多来源**：优先用已安装的 ``torch``（最准），其次 ``nvidia-smi``，
  再退回各平台自带的设备列举（Windows 的 CIM、Linux 的 ``lspci``、macOS 的
  ``system_profiler``）。任何一步失败都不会抛异常，只是少一条信息。
* **型号库**：:data:`GPU_MODEL_RULES` 覆盖市面主流型号（NVIDIA GeForce 7~50 系 /
  Quadro / Tesla / Hopper / Blackwell、AMD Radeon RX 5000~9000 / Vega / Instinct、
  Intel Arc A/B / Iris Xe、Apple M1~M4、Qualcomm Adreno），
  匹配出的 :class:`GpuModel` 给出架构与计算能力（compute capability），
  后端推荐再由 :mod:`.torch_backends` 结合驱动版本算出来。
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache

from .logger import get_logger

logger = get_logger(__name__)

# ------------------------------------------------------------------ 常量
VENDOR_NVIDIA = "nvidia"
VENDOR_AMD = "amd"
VENDOR_INTEL = "intel"
VENDOR_APPLE = "apple"
VENDOR_QUALCOMM = "qualcomm"
VENDOR_UNKNOWN = "unknown"

BACKEND_CUDA = "cuda"
BACKEND_ROCM = "rocm"
BACKEND_MPS = "mps"
BACKEND_XPU = "xpu"
BACKEND_CPU = "cpu"

VENDOR_LABELS = {
    VENDOR_NVIDIA: "NVIDIA",
    VENDOR_AMD: "AMD",
    VENDOR_INTEL: "Intel",
    VENDOR_APPLE: "Apple",
    VENDOR_QUALCOMM: "Qualcomm",
    VENDOR_UNKNOWN: "未知",
}

BACKEND_LABELS = {
    BACKEND_CUDA: "CUDA（NVIDIA）",
    BACKEND_ROCM: "ROCm（AMD）",
    BACKEND_MPS: "Metal / MPS（Apple）",
    BACKEND_XPU: "XPU（Intel Arc）",
    BACKEND_CPU: "CPU（纯处理器）",
}

#: 子进程超时（秒）：探测命令都很快，超时说明环境有问题，直接放弃
PROBE_TIMEOUT = 8.0


# ------------------------------------------------------------------ 数据模型
@dataclass(frozen=True, slots=True)
class GpuModel:
    """从显卡名字匹配出来的型号信息。"""

    vendor: str
    family: str
    architecture: str
    #: 计算能力（NVIDIA）；其他厂商为 ``None``
    compute: tuple[int, int] | None = None
    #: 可用后端
    backend: str = BACKEND_CUDA
    #: 集显 / 核显（性能有限，通常建议直接用 CPU）
    integrated: bool = False
    #: 是否已被现行 PyTorch 官方 wheel 放弃（例如 Kepler / Maxwell 老卡）
    legacy: bool = False

    @property
    def compute_text(self) -> str:
        if self.compute is None:
            return "—"
        return f"sm_{self.compute[0]}{self.compute[1]}"


@dataclass(slots=True)
class GpuInfo:
    """一张显卡（或核显）的探测结果。"""

    name: str
    vendor: str
    memory_mb: int | None = None
    driver_version: str | None = None
    compute: tuple[int, int] | None = None
    backend: str = BACKEND_CPU
    family: str = ""
    architecture: str = ""
    integrated: bool = False
    legacy: bool = False
    source: str = ""

    @property
    def vendor_label(self) -> str:
        return VENDOR_LABELS.get(self.vendor, self.vendor)

    @property
    def compute_text(self) -> str:
        if self.compute is None:
            return "—"
        return f"sm_{self.compute[0]}{self.compute[1]}"

    @property
    def memory_text(self) -> str:
        if not self.memory_mb:
            return "—"
        return f"{self.memory_mb / 1024:.1f} GB"

    def describe(self) -> str:
        """一行式描述，用于界面与 ``--check``。"""
        parts = [f"{self.vendor_label} {self.name}"]
        if self.family:
            parts.append(self.family)
        if self.memory_mb:
            parts.append(self.memory_text)
        if self.driver_version:
            parts.append(f"驱动 {self.driver_version}")
        if self.compute is not None:
            parts.append(self.compute_text)
        if self.integrated:
            parts.append("核显")
        if self.legacy:
            parts.append("架构过旧")
        return " · ".join(parts)


@dataclass(slots=True)
class HardwareProfile:
    """整机自检结果。"""

    os_name: str = ""
    os_release: str = ""
    arch: str = ""
    python_version: str = ""
    is_windows: bool = False
    is_linux: bool = False
    is_macos: bool = False
    cpu_name: str = ""
    cpu_cores: int = 0
    ram_gb: float | None = None
    gpus: list[GpuInfo] = field(default_factory=list)
    #: ``nvidia-smi`` 报告的驱动所支持的最高 CUDA 运行时版本（如 ``"12.6"``）
    cuda_driver_version: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def has_discrete_gpu(self) -> bool:
        return any(not gpu.integrated for gpu in self.gpus)

    def gpus_of(self, vendor: str) -> list[GpuInfo]:
        return [gpu for gpu in self.gpus if gpu.vendor == vendor]

    @property
    def primary_gpu(self) -> GpuInfo | None:
        """最有"算力"的那张卡：独显优先、显存大的优先。"""
        if not self.gpus:
            return None
        return max(
            self.gpus,
            key=lambda gpu: (
                not gpu.integrated,
                bool(gpu.compute),
                gpu.memory_mb or 0,
            ),
        )

    def summary_lines(self) -> list[tuple[str, str]]:
        """``(标题, 值)`` 列表，供界面表格 / 文本报告直接渲染。"""
        lines: list[tuple[str, str]] = [
            ("操作系统", f"{self.os_name} {self.os_release}".strip() or "未知"),
            ("Python", self.python_version or "未知"),
            ("处理器", self.cpu_name or "未知"),
            ("核心数", str(self.cpu_cores) if self.cpu_cores else "未知"),
            ("内存", f"{self.ram_gb:.1f} GB" if self.ram_gb else "未知"),
        ]
        if self.gpus:
            for index, gpu in enumerate(self.gpus, start=1):
                label = "显卡" if len(self.gpus) == 1 else f"显卡 {index}"
                lines.append((label, gpu.describe()))
        else:
            lines.append(("显卡", "未检测到独立显卡 / 核显信息（按 CPU 处理）"))
        if self.cuda_driver_version:
            lines.append(("驱动支持的 CUDA", f"最高 {self.cuda_driver_version}"))
        else:
            lines.append(("驱动支持的 CUDA", "未检测到 NVIDIA 驱动"))
        for note in self.notes:
            lines.append(("提示", note))
        return lines


# ------------------------------------------------------------------ 型号库
#: 显卡名字里的商标标记（``Intel(R) Arc(TM)`` 之类），匹配前先剥掉
_BRAND_MARKS = re.compile(r"\((?:R|TM|C)\)", flags=re.IGNORECASE)

#: 虚拟显示器 / 远程桌面注入的"假显卡"，不该参与自检
VIRTUAL_GPU_PATTERN = re.compile(
    r"virtual|indirect|idd|remote display|basic display|sharing monitor|mirror|"
    r"displaylink|spacedesk|parsec|gameviewer|oray|radmin|usb display|dameware|"
    r"splashtop|sunlogin|todesk|anydesk|teamviewer|dummy",
    flags=re.IGNORECASE,
)

#: (正则, 型号)——**顺序敏感**，越具体的规则越靠前
GPU_MODEL_RULES: tuple[tuple[str, GpuModel], ...] = (
    # ---------------- NVIDIA：数据中心 / 工作站 ----------------
    (
        r"RTX\s*PRO\s*\d{4}|B100|B200|GB200|GB300",
        GpuModel(VENDOR_NVIDIA, "Blackwell 工作站 / 计算卡", "Blackwell", (12, 0)),
    ),
    (
        r"\bH100\b|\bH200\b|\bH800\b|\bGH200\b|\bH20\b",
        GpuModel(VENDOR_NVIDIA, "Hopper 计算卡", "Hopper", (9, 0)),
    ),
    (
        r"\bA100\b|\bA800\b|\bA30\b|\bPG100\b",
        GpuModel(VENDOR_NVIDIA, "Ampere 计算卡", "Ampere", (8, 0)),
    ),
    (
        r"\bL4\b|\bL40S?\b|\bL20\b|\bL2\b|RTX\s*\d+\s*Ada|RTX\s*(A)?\d{4}\s*Ada",
        GpuModel(VENDOR_NVIDIA, "Ada 工作站卡", "Ada Lovelace", (8, 9)),
    ),
    (
        r"RTX\s*(A)?\d000\b|\bA10G?\b|\bA40\b|\bA16\b|\bA2\b|Quadro\s*RTX",
        GpuModel(VENDOR_NVIDIA, "Ampere / Turing 工作站卡", "Ampere", (8, 6)),
    ),
    (
        r"\bT4\b|\bT40\b|\bT10\b|Tesla\s*\w+",
        GpuModel(VENDOR_NVIDIA, "Turing 计算卡", "Turing", (7, 5)),
    ),
    (
        r"\bP100\b|\bP40\b|\bP4\b|Tesla\s*P",
        GpuModel(VENDOR_NVIDIA, "Pascal 计算卡", "Pascal", (6, 0)),
    ),
    # ---------------- NVIDIA：GeForce 消费级 ----------------
    (
        r"\bRTX\s*(PRO\s*)?50\d0\b|\bRTX\s*50[5-9]0\b|\bRTX\s*50\d0\s*(Ti|SUPER)\b",
        GpuModel(VENDOR_NVIDIA, "GeForce RTX 50 系", "Blackwell", (12, 0)),
    ),
    (
        r"\bRTX\s*40\d0\b|\bRTX\s*40\d0\s*(Ti|SUPER)\b",
        GpuModel(VENDOR_NVIDIA, "GeForce RTX 40 系", "Ada Lovelace", (8, 9)),
    ),
    (
        r"\bRTX\s*30\d0\b|\bRTX\s*30\d0\s*(Ti|SUPER)\b|\bRTX\s*3050\b",
        GpuModel(VENDOR_NVIDIA, "GeForce RTX 30 系", "Ampere", (8, 6)),
    ),
    (
        r"\bRTX\s*20\d0\b|\bRTX\s*20\d0\s*SUPER\b|\bTITAN\s*RTX\b",
        GpuModel(VENDOR_NVIDIA, "GeForce RTX 20 系", "Turing", (7, 5)),
    ),
    (
        r"\bGTX\s*16\d0\b|\bGTX\s*1650\b|\bGTX\s*1660\b",
        GpuModel(VENDOR_NVIDIA, "GeForce GTX 16 系", "Turing", (7, 5)),
    ),
    (
        r"\bMX\s*(2|3|4|5)\d0\b",
        GpuModel(VENDOR_NVIDIA, "GeForce MX 系列", "Turing", (7, 5)),
    ),
    (
        r"\bGTX\s*10\d0M?\b|\bGT\s*10\d0M?\b|\bMX\s*1[0-9]0\b|\bP\d{4}\b",
        GpuModel(VENDOR_NVIDIA, "GeForce GTX 10 系", "Pascal", (6, 1)),
    ),
    (
        r"\bGTX\s*9\d0M?\b|\bM\d000\b|\bMX\s*110\b|\bMX\s*130\b",
        GpuModel(VENDOR_NVIDIA, "GeForce GTX 9 系", "Maxwell", (5, 2), legacy=True),
    ),
    (
        r"\bGTX\s*7\d0M?\b|\bGT\s*7\d0M?\b|\bK\d0\b|\bK80\b|Kepler",
        GpuModel(VENDOR_NVIDIA, "GeForce GTX 7 系", "Kepler", (3, 7), legacy=True),
    ),
    (r"\bNVIDIA\b|\bGeForce\b|Quadro|Tesla", GpuModel(VENDOR_NVIDIA, "NVIDIA 显卡", "")),
    # ---------------- AMD ----------------
    (
        r"\bRX\s*9\d{3}\b|\bRadeon\s*9\d{3}\b",
        GpuModel(VENDOR_AMD, "Radeon RX 9000 系", "RDNA 4", backend=BACKEND_ROCM),
    ),
    (
        r"\bRX\s*7\d{3}\b|\bRadeon\s*7\d{3}\b",
        GpuModel(VENDOR_AMD, "Radeon RX 7000 系", "RDNA 3", backend=BACKEND_ROCM),
    ),
    (
        r"\bRX\s*6\d{3}\b|\bRadeon\s*6\d{3}\b",
        GpuModel(VENDOR_AMD, "Radeon RX 6000 系", "RDNA 2", backend=BACKEND_ROCM),
    ),
    (
        r"\bRX\s*5\d{3}\b|\bRadeon\s*5\d{3}\b",
        GpuModel(VENDOR_AMD, "Radeon RX 5000 系", "RDNA", backend=BACKEND_ROCM),
    ),
    (
        r"\bRX\s*5\d{2}M?\b|\bRX\s*4\d{2}M?\b|Polaris",
        GpuModel(VENDOR_AMD, "Radeon RX 500 / 400 系", "GCN 4 (Polaris)", backend=BACKEND_ROCM),
    ),
    (
        r"\bMI\s*\d{2,3}\b|Instinct",
        GpuModel(VENDOR_AMD, "Instinct 计算卡", "CDNA", backend=BACKEND_ROCM),
    ),
    (
        r"Vega|Radeon\s*VII|R9\s*Fury|\bR9\s*3\d0\b",
        GpuModel(VENDOR_AMD, "Radeon Vega / GCN", "Vega / GCN", backend=BACKEND_ROCM),
    ),
    (
        r"Radeon\s*Graphics|Radeon\s*\d{3}M?\b|Ryzen.*Graphics",
        GpuModel(
            VENDOR_AMD,
            "Radeon 核显",
            "集成显卡",
            backend=BACKEND_CPU,
            integrated=True,
        ),
    ),
    (r"\bAMD\b|\bRadeon\b|\bATI\b", GpuModel(VENDOR_AMD, "AMD 显卡", "")),
    # ---------------- Intel ----------------
    (
        r"Arc\s*B\d{3}|Arc\s*Pro\s*B\d{2}",
        GpuModel(VENDOR_INTEL, "Arc B 系列", "Battlemage", backend=BACKEND_XPU),
    ),
    (
        r"Arc\s*A\d{3}|Arc\s*Pro\s*A\d{2}|Arc\s*Graphics",
        GpuModel(VENDOR_INTEL, "Arc A 系列", "Alchemist", backend=BACKEND_XPU),
    ),
    (
        r"Data\s*Center\s*GPU|Ponte\s*Vecchio",
        GpuModel(VENDOR_INTEL, "数据中心 GPU", "Xe-HPC", backend=BACKEND_XPU),
    ),
    (
        r"Iris\s*Xe|Iris\s*Plus|UHD\s*Graphics|HD\s*Graphics|Intel.*Graphics",
        GpuModel(
            VENDOR_INTEL,
            "Intel 核显",
            "集成显卡",
            backend=BACKEND_CPU,
            integrated=True,
        ),
    ),
    (r"\bIntel\b", GpuModel(VENDOR_INTEL, "Intel 显卡", "")),
    # ---------------- Apple ----------------
    (
        r"Apple\s*M[1-9](\s*(Pro|Max|Ultra))?",
        GpuModel(VENDOR_APPLE, "Apple Silicon 统一内存", "Apple GPU", backend=BACKEND_MPS),
    ),
    # ---------------- 其他 ----------------
    (
        r"Adreno|Snapdragon|Mali|PowerVR",
        GpuModel(
            VENDOR_QUALCOMM,
            "移动 / 集成 GPU",
            "SoC GPU",
            backend=BACKEND_CPU,
            integrated=True,
        ),
    ),
)


def normalize_gpu_name(name: str) -> str:
    """去掉 ``(R)`` / ``(TM)`` / ``(C)`` 之类的商标标记，并压掉多余空格。"""
    text = _BRAND_MARKS.sub(" ", name or "")
    return re.sub(r"\s+", " ", text).strip()


def classify_gpu(name: str) -> GpuModel:
    """按名字匹配型号库；匹配不到时返回一个"未知厂商"的兜底型号。"""
    text = normalize_gpu_name(name)
    for pattern, model in GPU_MODEL_RULES:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return model
    # 认不出来的卡：不敢乱推荐 CUDA，退回 CPU 最稳
    return GpuModel(VENDOR_UNKNOWN, "", "", backend=BACKEND_CPU)


def is_virtual_gpu(name: str) -> bool:
    """是否是虚拟显示器 / 远程桌面注入的"假显卡"（自检时直接忽略）。"""
    text = normalize_gpu_name(name)
    if not VIRTUAL_GPU_PATTERN.search(text):
        return False
    # 名字里带 NVIDIA / AMD / Intel 的可能是真卡被远程软件改名，保守放行
    return classify_gpu(text).vendor == VENDOR_UNKNOWN


# ------------------------------------------------------------------ 子进程工具
def _run(command: list[str], timeout: float = PROBE_TIMEOUT) -> str | None:
    """安全执行探测命令，失败 / 超时返回 ``None``（绝不抛异常）。"""
    executable = shutil.which(command[0])
    if executable is None:
        return None
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    try:
        completed = subprocess.run(  # noqa: S603 - 命令由本模块硬编码
            [executable, *command[1:]],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
            creationflags=flags,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover - 环境相关
        logger.debug("探测命令 %s 失败：%s", command[0], exc)
        return None
    if completed.returncode != 0:
        logger.debug("探测命令 %s 返回 %s", command[0], completed.returncode)
        return None
    return completed.stdout or ""


def _power_shell(script: str) -> str | None:
    for shell in ("powershell", "pwsh"):
        output = _run(
            [shell, "-NoProfile", "-NonInteractive", "-Command", script],
        )
        if output:
            return output
    return None


def _to_int(text: str) -> int | None:
    digits = re.sub(r"[^\d]", "", text or "")
    return int(digits) if digits else None


# ------------------------------------------------------------------ 内存 / CPU
def _total_memory_gb() -> float | None:
    if os.name == "nt":
        try:
            import ctypes

            class _MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatus()
            status.dwLength = ctypes.sizeof(_MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return status.ullTotalPhys / 1024**3
        except Exception as exc:  # noqa: BLE001 - 探测失败不影响主流程
            logger.debug("读取内存失败：%s", exc)
        return None

    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return pages * page_size / 1024**3
    except (ValueError, OSError, AttributeError):
        return None


@lru_cache(maxsize=1)
def _cpu_name() -> str:
    # Windows：注册表里的名字最干净（platform.processor() 会给出 "Intel64 Family 6 …"）
    if os.name == "nt":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
                if value:
                    return str(value).strip()
        except OSError as exc:  # pragma: no cover - 取决于系统
            logger.debug("读取 CPU 型号失败：%s", exc)

    output = _power_shell("(Get-CimInstance Win32_Processor).Name") if os.name == "nt" else None
    if output and output.strip():
        return output.strip().splitlines()[0]

    name = platform.processor() or platform.machine()
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[-1].strip()
        except OSError:  # pragma: no cover
            pass
    return name.strip() or "未知"


# ------------------------------------------------------------------ GPU 探测
def _gpu_from_torch() -> list[GpuInfo]:
    """已装 torch 时最准的来源（能直接拿到算力）。"""
    try:
        import torch
    except ImportError:
        return []
    try:
        if not torch.cuda.is_available():
            return []
    except Exception as exc:  # noqa: BLE001 - 驱动异常
        logger.debug("torch CUDA 探测失败：%s", exc)
        return []

    devices: list[GpuInfo] = []
    try:
        count = torch.cuda.device_count()
        for index in range(count):
            props = torch.cuda.get_device_properties(index)
            capability = torch.cuda.get_device_capability(index)
            model = classify_gpu(props.name)
            devices.append(
                GpuInfo(
                    name=str(props.name).strip(),
                    vendor=VENDOR_NVIDIA,
                    memory_mb=int(props.total_memory // (1024 * 1024)) or None,
                    compute=(int(capability[0]), int(capability[1])),
                    backend=BACKEND_CUDA,
                    family=model.family,
                    architecture=model.architecture,
                    source="torch",
                )
            )
    except Exception as exc:  # noqa: BLE001 - 罕见驱动问题
        logger.debug("torch 读取显卡属性失败：%s", exc)
    return devices


def _gpu_from_nvidia_smi() -> tuple[list[GpuInfo], str | None]:
    """走 ``nvidia-smi``：一次拿到型号、显存、驱动版本与算力。"""
    query = "name,driver_version,memory.total,compute_cap"
    output = _run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"])
    if output is None:
        return [], None

    cuda_version: str | None = None
    header = _run(["nvidia-smi"])
    if header:
        match = re.search(r"CUDA Version:\s*([\d.]+)", header)
        if match:
            cuda_version = match.group(1)

    devices: list[GpuInfo] = []
    for line in output.splitlines():
        cells = [cell.strip() for cell in line.split(",")]
        if len(cells) < 1 or not cells[0]:
            continue
        name = cells[0]
        driver = cells[1] if len(cells) > 1 and cells[1] not in {"", "N/A"} else None
        memory = _to_int(cells[2]) if len(cells) > 2 else None
        compute: tuple[int, int] | None = None
        if len(cells) > 3 and cells[3] not in {"", "N/A"}:
            parts = re.findall(r"\d+", cells[3])
            if len(parts) >= 2:
                compute = (int(parts[0]), int(parts[1]))
        model = classify_gpu(name)
        devices.append(
            GpuInfo(
                name=name,
                vendor=VENDOR_NVIDIA,
                memory_mb=memory,
                driver_version=driver,
                compute=compute or model.compute,
                backend=BACKEND_CUDA,
                family=model.family,
                architecture=model.architecture,
                legacy=model.legacy,
                source="nvidia-smi",
            )
        )
    return devices, cuda_version


def _gpu_from_windows_cim() -> list[GpuInfo]:
    script = (
        "Get-CimInstance Win32_VideoController | "
        "Select-Object Name,DriverVersion,AdapterRAM | "
        "ForEach-Object { \"$($_.Name)|$($_.DriverVersion)|$($_.AdapterRAM)\" }"
    )
    output = _power_shell(script)
    if not output:
        return []

    devices: list[GpuInfo] = []
    for line in output.splitlines():
        cells = [cell.strip() for cell in line.split("|")]
        if not cells or not cells[0]:
            continue
        name = cells[0]
        driver = cells[1] if len(cells) > 1 and cells[1] not in {"", "N/A"} else None
        memory = _to_int(cells[2]) if len(cells) > 2 else None
        # WMI 的 AdapterRAM 是 32 位有符号，超过 4GB 会溢出，视作不可信
        if memory is not None and (memory <= 0 or memory > 8 * 1024):
            memory = None
        device = _gpu_from_name(name, memory_mb=memory, driver=driver, source="wmi")
        if device is not None:
            devices.append(device)
    return devices


def _gpu_from_lspci() -> list[GpuInfo]:
    output = _run(["lspci"])
    if not output:
        return []
    devices: list[GpuInfo] = []
    for line in output.splitlines():
        if not re.search(r"VGA compatible controller|3D controller|Display controller", line):
            continue
        name = line.split(":", 2)[-1].strip()
        device = _gpu_from_name(name, source="lspci")
        if device is not None:
            devices.append(device)
    return devices


def _gpu_from_system_profiler() -> list[GpuInfo]:
    output = _run(["system_profiler", "SPDisplaysDataType"], timeout=15.0)
    if not output:
        return []
    devices: list[GpuInfo] = []
    for line in output.splitlines():
        match = re.match(r"\s*Chipset Model:\s*(.+)", line)
        if match:
            device = _gpu_from_name(match.group(1).strip(), source="system_profiler")
            if device is not None:
                devices.append(device)
    return devices


def _gpu_from_name(
    name: str,
    *,
    memory_mb: int | None = None,
    driver: str | None = None,
    source: str = "",
) -> GpuInfo | None:
    """把一条"显卡名字"整理成 :class:`GpuInfo`；虚拟显示器返回 ``None``。"""
    if is_virtual_gpu(name):
        logger.debug("忽略虚拟 / 远程显示设备：%s", name)
        return None
    model = classify_gpu(name)
    return GpuInfo(
        name=name,
        vendor=model.vendor,
        memory_mb=memory_mb,
        driver_version=driver,
        compute=model.compute,
        backend=model.backend,
        family=model.family,
        architecture=model.architecture,
        integrated=model.integrated,
        legacy=model.legacy,
        source=source,
    )


def _mark_apple_silicon(profile: HardwareProfile) -> None:
    """macOS + arm64：即使拿不到显卡名字也要认得 MPS。"""
    if not profile.is_macos or profile.arch not in {"arm64", "aarch64"}:
        return
    if profile.gpus:
        return
    machine = platform.machine()
    profile.gpus.append(
        GpuInfo(
            name=f"Apple Silicon ({machine})",
            vendor=VENDOR_APPLE,
            backend=BACKEND_MPS,
            family="Apple Silicon 统一内存",
            architecture="Apple GPU",
            source="platform",
        )
    )


def probe_hardware(force: bool = False) -> HardwareProfile:
    """探测整机硬件，返回 :class:`HardwareProfile`。

    参数 ``force`` 为 ``True`` 时绕过缓存重新探测（界面上的「重新自检」用它）。
    """
    if not force:
        cached = _PROBE_CACHE.get("profile")
        if cached is not None:
            return cached

    system = platform.system()
    profile = HardwareProfile(
        os_name=system,
        os_release=platform.release(),
        arch=platform.machine(),
        python_version=platform.python_version(),
        is_windows=system == "Windows",
        is_linux=system == "Linux",
        is_macos=system == "Darwin",
        cpu_name=_cpu_name(),
        cpu_cores=os.cpu_count() or 0,
        ram_gb=_total_memory_gb(),
    )

    gpus: list[GpuInfo] = []
    cuda_version: str | None = None

    nvidia, cuda_version = _gpu_from_nvidia_smi()
    gpus.extend(nvidia)

    if not gpus or not any(gpu.vendor == VENDOR_NVIDIA for gpu in gpus):
        gpus.extend(_gpu_from_torch())

    # 兜底：去重前先按平台列举一次（Windows 的 CIM 能拿到核显与 AMD 卡）
    platform_gpus: list[GpuInfo] = []
    if profile.is_windows:
        platform_gpus = _gpu_from_windows_cim()
    elif profile.is_linux:
        platform_gpus = _gpu_from_lspci()
    elif profile.is_macos:
        platform_gpus = _gpu_from_system_profiler()

    seen = {_dedup_key(gpu) for gpu in gpus}
    for gpu in platform_gpus:
        key = _dedup_key(gpu)
        if key in seen:
            # 已有更权威的来源：只补它缺的字段
            for existing in gpus:
                if _dedup_key(existing) == key:
                    if existing.memory_mb is None and gpu.memory_mb:
                        existing.memory_mb = gpu.memory_mb
                    if existing.driver_version is None and gpu.driver_version:
                        existing.driver_version = gpu.driver_version
            continue
        seen.add(key)
        gpus.append(gpu)

    profile.gpus = gpus
    profile.cuda_driver_version = cuda_version
    _mark_apple_silicon(profile)
    _annotate_notes(profile)

    _PROBE_CACHE["profile"] = profile
    logger.debug("硬件自检完成：%s", profile.summary_lines())
    return profile


_PROBE_CACHE: dict[str, HardwareProfile] = {}


def clear_cache() -> None:
    """清空探测缓存（测试与「重新自检」用）。"""
    _PROBE_CACHE.clear()
    _cpu_name.cache_clear()


def _dedup_key(gpu: GpuInfo) -> str:
    """同一张卡可能被多个来源报出来，用"厂商 + 名字里的关键型号"去重。"""
    name = re.sub(r"[^a-z0-9]", "", (gpu.name or "").lower())
    name = re.sub(r"(nvidia|amd|intel|corporation|inc|llc|technologies|graphics)", "", name)
    return f"{gpu.vendor}:{name[:24]}"


def _annotate_notes(profile: HardwareProfile) -> None:
    """把自检结论里的人话提示挂到 profile 上（界面与 ``--check`` 共用）。"""
    if profile.is_macos and profile.arch in {"arm64", "aarch64"}:
        profile.notes.append("Apple Silicon：PyTorch 自带 Metal(MPS) 加速，安装 CPU 版即可。")

    nvidia = profile.gpus_of(VENDOR_NVIDIA)
    if nvidia:
        legacy = [gpu for gpu in nvidia if gpu.legacy]
        if legacy:
            profile.notes.append(
                f"{legacy[0].name}（{legacy[0].architecture}）架构较旧："
                "只能用最老的 CUDA 构建，Kepler 及更早的卡请改用 CPU 模式。"
            )
        if not profile.cuda_driver_version:
            profile.notes.append(
                "未读到 NVIDIA 驱动版本：请确认已安装显卡驱动，或改用 CPU 模式。"
            )

    amd = profile.gpus_of(VENDOR_AMD)
    if amd and any(not gpu.integrated for gpu in amd) and profile.is_windows:
        profile.notes.append(
            "AMD 独显在 Windows 上没有官方 PyTorch 构建（ROCm 仅 Linux），已回退到 CPU。"
        )

    intel = profile.gpus_of(VENDOR_INTEL)
    if intel and any(gpu.backend == BACKEND_XPU for gpu in intel):
        profile.notes.append("Intel Arc 独显可用 XPU 构建；若安装失败请改用 CPU 模式。")

    if not profile.gpus:
        profile.notes.append("未检测到可用显卡，将使用 CPU 模式（稳定但速度较慢）。")


__all__ = [
    "BACKEND_CPU",
    "BACKEND_CUDA",
    "BACKEND_LABELS",
    "BACKEND_MPS",
    "BACKEND_ROCM",
    "BACKEND_XPU",
    "GPU_MODEL_RULES",
    "GpuInfo",
    "GpuModel",
    "HardwareProfile",
    "VENDOR_AMD",
    "VENDOR_APPLE",
    "VENDOR_INTEL",
    "VENDOR_LABELS",
    "VENDOR_NVIDIA",
    "VENDOR_QUALCOMM",
    "VENDOR_UNKNOWN",
    "VIRTUAL_GPU_PATTERN",
    "classify_gpu",
    "clear_cache",
    "is_virtual_gpu",
    "normalize_gpu_name",
    "probe_hardware",
]
