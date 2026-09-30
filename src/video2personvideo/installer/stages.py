"""安装阶段模型：安装过程分几步、每步能不能中断、中断了会怎样。

把"步骤"单独建模有两个目的：

1. **进度可信**：总进度条按各阶段的权重（不是简单平均）推进，
   PyTorch 下载占大头，界面不会"20 秒就到 80%、然后卡 20 分钟"；
2. **退出可解释**：每个阶段都写清楚「在这一步退出会发生什么、怎么继续」，
   因为安装过程动辄几十 GB 的网络流量，用户中途关窗口是常态，
   不能只说一句"安装未完成"。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Stage(str, Enum):
    """安装阶段（顺序即执行顺序，``value`` 也是写进日志的键）。"""

    PREPARE = "prepare"
    RUNTIME = "runtime"
    PIP = "pip"
    TORCH = "torch"
    DEPS = "deps"
    APP = "app"
    YOLO = "yolo"
    FFMPEG = "ffmpeg"
    FINALIZE = "finalize"
    DONE = "done"

    def __str__(self) -> str:  # 便于日志里直接显示
        return self.value


@dataclass(frozen=True, slots=True)
class StageInfo:
    """一个阶段的元信息。"""

    stage: Stage
    title: str
    detail: str
    #: 进度条权重（相对值，总和为 100）
    weight: float
    #: 该阶段是否"可以安全地只做一半"：下载类会保留断点，重跑时只补差额
    resumable: bool
    #: 该阶段会往安装目录里写东西（退出后需要"回滚"才能干净）
    writes_disk: bool
    #: 该阶段是可跳过项（失败 / 跳过都不算安装失败）
    optional: bool = False
    #: 中断提示：在这一步退出会发生什么
    exit_note: str = ""

    @property
    def key(self) -> str:
        return self.stage.value


#: 各阶段的权重（总和 100）：PyTorch 下载是绝对大头
STAGES: tuple[StageInfo, ...] = (
    StageInfo(
        stage=Stage.PREPARE,
        title="准备安装目录",
        detail="检查磁盘空间与写权限，创建安装目录并记下安装状态。",
        weight=1.0,
        resumable=True,
        writes_disk=True,
        exit_note=(
            "此时只在安装目录里写了一个状态文件（install.json），"
            "退出后下次可以「继续安装」，也可以直接「回滚」把空目录删掉。"
        ),
    ),
    StageInfo(
        stage=Stage.RUNTIME,
        title="部署 Python 运行时",
        detail="下载官方内嵌版 Python（约 10 MB）并解压到安装目录，作为独立运行环境。",
        weight=8.0,
        resumable=True,
        writes_disk=True,
        exit_note=(
            "运行时压缩包的断点会保留，已经解压好的部分也不用重来，"
            "下次「继续安装」只补没下完的那一段。"
        ),
    ),
    StageInfo(
        stage=Stage.PIP,
        title="引导 pip",
        detail="下载 get-pip.py 并安装 pip / setuptools / wheel。",
        weight=4.0,
        resumable=True,
        writes_disk=True,
        exit_note="断点会保留；pip 本身没装完的话下次会重新引导一次（很快）。",
    ),
    StageInfo(
        stage=Stage.TORCH,
        title="下载并安装 PyTorch",
        detail="按显卡型号选一套官方构建（CUDA / CPU），断点续传下载 wheel 后安装。",
        weight=55.0,
        resumable=True,
        writes_disk=True,
        exit_note=(
            "这是体积最大的一步（CPU 版约 200 MB，CUDA 版 1~3 GB）。"
            "退出时断点会完整保留，下次「继续安装」从断点续传，已下好的文件只校验不重下。"
        ),
    ),
    StageInfo(
        stage=Stage.DEPS,
        title="安装运行依赖",
        detail="安装 ultralytics / OpenCV / PySide6 等其余依赖（不含 PyTorch，避免覆盖）。",
        weight=20.0,
        resumable=False,
        writes_disk=True,
        exit_note=(
            "pip 安装到一半被中断时，环境可能处于半成品状态；"
            "下次「继续安装」会重跑这一步（pip 是幂等的，已装好的包不会重复下载）。"
        ),
    ),
    StageInfo(
        stage=Stage.APP,
        title="安装程序本体",
        detail="把 Video2PersonVideo 安装进上面的 Python 环境。",
        weight=4.0,
        resumable=False,
        writes_disk=True,
        exit_note="与依赖安装同理：中断后重跑本步即可（很快）。",
    ),
    StageInfo(
        stage=Stage.YOLO,
        title="下载 YOLO 权重",
        detail="下载人物检测 / 姿态权重（每个几 MB）到安装目录，装完断网也能用。",
        weight=1.0,
        resumable=True,
        writes_disk=True,
        exit_note="只有几 MB，断点会保留，下次接着下。",
    ),
    StageInfo(
        stage=Stage.FFMPEG,
        title="部署 ffmpeg",
        detail="可选：随包放一份 ffmpeg，用于保留音轨、H.264 重编码与说话人判定。",
        weight=5.0,
        resumable=True,
        writes_disk=True,
        optional=True,
        exit_note="可选组件，断点会保留；不装也不影响基本裁剪功能。",
    ),
    StageInfo(
        stage=Stage.FINALIZE,
        title="创建快捷方式并登记卸载",
        detail="写入开始菜单 / 桌面快捷方式与卸载信息，之后可在「应用和功能」里卸载。",
        weight=2.0,
        resumable=False,
        writes_disk=True,
        exit_note="这一步很快；如果恰好中断，下次「继续安装」会补做。",
    ),
    StageInfo(
        stage=Stage.DONE,
        title="安装完成",
        detail="全部就绪，可以启动程序。",
        weight=0.0,
        resumable=True,
        writes_disk=False,
    ),
)

#: 真正的"执行步骤"（不含终态 ``DONE``）
ORDERED_STAGES: tuple[StageInfo, ...] = tuple(item for item in STAGES if item.stage is not Stage.DONE)

_BY_KEY: dict[str, StageInfo] = {item.key: item for item in STAGES}

#: 权重总和（归一化用）
TOTAL_WEIGHT = sum(item.weight for item in ORDERED_STAGES) or 1.0


def stage_info(stage: Stage | str) -> StageInfo:
    """按阶段（或它的字符串值）取元信息。"""
    key = stage.value if isinstance(stage, Stage) else str(stage)
    return _BY_KEY[key]


def ordered_stages(stages: tuple[StageInfo, ...] | None = None) -> tuple[StageInfo, ...]:
    return stages if stages is not None else ORDERED_STAGES


def weight_before(stage: Stage | str, stages: tuple[StageInfo, ...] | None = None) -> float:
    """某个阶段**开始前**已完成的权重占全部权重之比（0~1）。"""
    info = stage_info(stage)
    done = 0.0
    for item in ordered_stages(stages):
        if item.stage is info.stage:
            break
        done += item.weight
    return min(done / TOTAL_WEIGHT, 1.0)


def weight_span(stage: Stage | str) -> float:
    """某个阶段自身占全部权重的比例（0~1）。"""
    return stage_info(stage).weight / TOTAL_WEIGHT


__all__ = [
    "ORDERED_STAGES",
    "STAGES",
    "TOTAL_WEIGHT",
    "Stage",
    "StageInfo",
    "ordered_stages",
    "stage_info",
    "weight_before",
    "weight_span",
]
