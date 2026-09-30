"""把安装引擎接到 Qt 信号上（后台线程，界面不卡）。

三个 Worker 对应三类耗时操作：

* :class:`ProbeWorker` —— 硬件自检（会调 ``nvidia-smi`` 等子进程，可能要几秒）；
* :class:`InstallWorker` —— 一次完整的安装执行（下载 + pip + 建快捷方式）；
* :class:`RollbackWorker` —— 回滚 / 卸载（删目录、删快捷方式、清注册表）。

它们跑完都会发一个 ``completed`` 信号，由界面负责收线程（``thread.quit()``）。
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from ...utils.gpu_probe import HardwareProfile, probe_hardware
from ...utils.logger import get_logger
from ...utils.torch_backends import BackendRecommendation, recommend_backend
from ..engine import InstallEngine, InstallOutcome, Reporter, RollbackReport, rollback_install
from ..stages import Stage

logger = get_logger(__name__)


class ProbeWorker(QObject):
    """后台自检：CPU / 显卡 / 驱动 → 推荐该装哪一套 PyTorch。"""

    detected = Signal(object)  # HardwareProfile
    recommended = Signal(object)  # BackendRecommendation
    failed = Signal(str)
    completed = Signal()

    def __init__(self, *, force: bool = False) -> None:
        super().__init__()
        self.force = force

    @Slot()
    def run(self) -> None:
        try:
            profile: HardwareProfile = probe_hardware(force=self.force)
            self.detected.emit(profile)
            recommendation: BackendRecommendation = recommend_backend(profile)
            self.recommended.emit(recommendation)
        except Exception as exc:  # noqa: BLE001 - 自检失败不该挡住安装
            logger.exception("硬件自检失败")
            self.failed.emit(str(exc))
        finally:
            self.completed.emit()

    def cancel(self) -> None:  # 自检本身很快，留个空实现统一接口
        return


class InstallWorker(QObject):
    """在子线程里跑 :class:`InstallEngine`。"""

    stageChanged = Signal(object, str, str)  # Stage, status, message
    stageProgress = Signal(float, str)
    totalProgress = Signal(float, str)
    logMessage = Signal(str)
    downloadStats = Signal(object)  # DownloadProgress
    completed = Signal(object)  # InstallOutcome

    def __init__(self, engine: InstallEngine) -> None:
        super().__init__()
        self._engine = engine
        reporter = Reporter(
            on_log=self.logMessage.emit,
            on_stage=lambda stage, status, message: self.stageChanged.emit(stage, status, message),
            on_stage_progress=self.stageProgress.emit,
            on_total_progress=self.totalProgress.emit,
            on_download=self.downloadStats.emit,
        )
        engine.reporter = reporter

    @property
    def engine(self) -> InstallEngine:
        return self._engine

    @Slot()
    def run(self) -> None:
        try:
            outcome: InstallOutcome = self._engine.run()
        except Exception as exc:  # noqa: BLE001 - 引擎内部已兜底，这里再兜一层
            logger.exception("安装执行失败")
            self.logMessage.emit(f"安装执行失败：{exc}")
            return
        self.completed.emit(outcome)

    def cancel(self) -> None:
        """请求暂停（线程安全：只置一个标志位）。"""
        self._engine.cancel()

    @property
    def current_stage(self) -> Stage | None:
        return self._engine.current_stage

    @property
    def downloaded_bytes(self) -> int:
        return self._engine.downloaded_bytes


class RollbackWorker(QObject):
    """后台执行回滚 / 卸载。"""

    logMessage = Signal(str)
    completed = Signal(object)  # RollbackReport

    def __init__(self, install_dir: str | Path, *, remove_cache: bool = True) -> None:
        super().__init__()
        self._install_dir = Path(install_dir)
        self._remove_cache = remove_cache

    @Slot()
    def run(self) -> None:
        reporter = Reporter(on_log=self.logMessage.emit)
        try:
            report: RollbackReport = rollback_install(
                self._install_dir, remove_cache=self._remove_cache, reporter=reporter
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("回滚失败")
            report = RollbackReport(message=f"回滚过程中出现问题：{exc}")
        self.completed.emit(report)


__all__ = [
    "InstallWorker",
    "ProbeWorker",
    "RollbackWorker",
]
