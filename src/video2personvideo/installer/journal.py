"""安装状态日志（``install.json``）：安装走到哪一步了、下次该从哪儿接着走。

安装不是"一次成功"的操作：下载可能有几十 GB、用户随时可能关窗口、
pip 可能断在中间。所以每推进一步都落一次盘，并且**原子写入**
（先写 ``.tmp`` 再 ``os.replace``），保证任何时刻读到的都是一份完整状态。

状态文件有两份：

* ``<安装目录>/install.json`` —— 主副本，同时充当"这是安装目录"的标记
  （``utils.app_paths`` 据此认出安装根）；
* ``%LOCALAPPDATA%/Video2PersonVideo/install.json`` —— 镜像副本，
  安装目录被手工删掉、或安装目录在移动硬盘上暂时不可用时，还能知道"装过、装到哪"。

读的时候优先主副本，坏了就退回镜像；两份都不行就当"没装过"，
绝不让一个坏掉的状态文件把用户挡在门外。
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ..utils.logger import get_logger
from . import paths
from .stages import ORDERED_STAGES, Stage, TOTAL_WEIGHT, stage_info

logger = get_logger(__name__)

#: 状态文件格式版本（以后改结构时用来做兼容）
SCHEMA_VERSION = 1

#: 阶段状态
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"

#: 整体状态
STATE_IN_PROGRESS = "in_progress"
STATE_COMPLETE = "complete"
STATE_FAILED = "failed"
STATE_ROLLED_BACK = "rolled_back"

#: 事件日志最多保留多少条（够界面显示"上次在哪一步退出的"即可）
MAX_EVENTS = 60


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass(slots=True)
class StageRecord:
    """一个阶段的执行记录。"""

    stage: str
    status: str = STATUS_PENDING
    message: str = ""
    detail: str = ""
    attempts: int = 0
    started_at: str | None = None
    finished_at: str | None = None

    @property
    def done(self) -> bool:
        return self.status in (STATUS_DONE, STATUS_SKIPPED)

    @property
    def failed(self) -> bool:
        return self.status == STATUS_FAILED


@dataclass(slots=True)
class InstallJournal:
    """一份完整的安装状态。"""

    app_version: str
    install_dir: str
    python_version: str
    #: 选中的 PyTorch 构建（``cu126`` / ``cpu``…）
    backend_key: str = ""
    #: 用户在这次安装里做的选择（勾了哪些组件、要不要快捷方式…）
    options: dict[str, Any] = field(default_factory=dict)
    records: dict[str, StageRecord] = field(default_factory=dict)
    state: str = STATE_IN_PROGRESS
    schema: int = SCHEMA_VERSION
    started_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    finished_at: str | None = None
    #: 这份状态被"推进"过多少次（含中断后继续）
    runs: int = 1
    #: 最近发生的事（中断 / 失败 / 完成），给界面和排错用
    events: list[dict[str, Any]] = field(default_factory=list)

    # ------------------------------------------------------------- 构造
    @classmethod
    def create(
        cls,
        *,
        install_dir: str | os.PathLike[str],
        app_version: str,
        python_version: str,
        backend_key: str = "",
        options: dict[str, Any] | None = None,
    ) -> InstallJournal:
        journal = cls(
            app_version=app_version,
            install_dir=str(Path(install_dir)),
            python_version=python_version,
            backend_key=backend_key,
            options=dict(options or {}),
        )
        for info in ORDERED_STAGES:
            journal.records[info.key] = StageRecord(stage=info.key)
        return journal

    # ------------------------------------------------------------- 读写
    @property
    def install_path(self) -> Path:
        return Path(self.install_dir)

    def primary_path(self) -> Path:
        return paths.journal_path(self.install_dir)

    def save(self) -> bool:
        """原子写主副本 + 尽力写镜像副本；至少成功一处才算保存成功。"""
        self.updated_at = _now()
        payload = json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
        saved = self._atomic_write(self.primary_path(), payload)
        # 镜像副本只是"兜底记忆"，写不进去不影响安装
        self._atomic_write(paths.mirrored_journal_path(), payload)
        return saved

    @staticmethod
    def _atomic_write(target: Path, payload: str) -> bool:
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_name(f"{target.name}.tmp")
            temp.write_text(payload, encoding="utf-8")
            os.replace(temp, target)
            return True
        except OSError as exc:
            logger.debug("写入状态文件失败 %s：%s", target, exc)
            return False

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> InstallJournal | None:
        """读一份状态文件；文件不存在 / 内容坏掉时返回 ``None``。"""
        target = Path(path)
        try:
            raw = target.read_text(encoding="utf-8")
        except OSError:
            return None
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("安装状态文件已损坏（%s）：%s", target, exc)
            return None
        if not isinstance(data, dict):
            return None
        return cls.from_dict(data)

    @classmethod
    def load_for(cls, install_dir: str | os.PathLike[str]) -> InstallJournal | None:
        """按安装目录读状态：主副本优先，其次镜像副本。"""
        journal = cls.load(paths.journal_path(install_dir))
        if journal is not None:
            return journal
        primary = Path(install_dir).resolve()
        mirrored = cls.load(paths.mirrored_journal_path())
        if mirrored is None:
            return None
        try:
            if Path(mirrored.install_dir).resolve() == primary:
                return mirrored
        except OSError:  # pragma: no cover
            return None
        return None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["records"] = {key: asdict(value) for key, value in self.records.items()}
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InstallJournal:
        records: dict[str, StageRecord] = {}
        raw_records = data.get("records") or {}
        if isinstance(raw_records, dict):
            for key, value in raw_records.items():
                if not isinstance(value, dict):
                    continue
                records[str(key)] = StageRecord(
                    stage=str(value.get("stage") or key),
                    status=str(value.get("status") or STATUS_PENDING),
                    message=str(value.get("message") or ""),
                    detail=str(value.get("detail") or ""),
                    attempts=int(value.get("attempts") or 0),
                    started_at=value.get("started_at"),
                    finished_at=value.get("finished_at"),
                )
        journal = cls(
            app_version=str(data.get("app_version") or ""),
            install_dir=str(data.get("install_dir") or ""),
            python_version=str(data.get("python_version") or ""),
            backend_key=str(data.get("backend_key") or ""),
            options=dict(data.get("options") or {}),
            records=records,
            state=str(data.get("state") or STATE_IN_PROGRESS),
            schema=int(data.get("schema") or SCHEMA_VERSION),
            started_at=str(data.get("started_at") or _now()),
            updated_at=str(data.get("updated_at") or _now()),
            finished_at=data.get("finished_at"),
            runs=int(data.get("runs") or 1),
            events=list(data.get("events") or []),
        )
        # 补齐缺失的阶段（旧版本装过、后来加了新阶段时）
        for info in ORDERED_STAGES:
            journal.records.setdefault(info.key, StageRecord(stage=info.key))
        return journal

    # ------------------------------------------------------------- 查询
    def record(self, stage: Stage | str) -> StageRecord:
        key = stage.value if isinstance(stage, Stage) else str(stage)
        found = self.records.get(key)
        if found is None:
            found = StageRecord(stage=key)
            self.records[key] = found
        return found

    def status_of(self, stage: Stage | str) -> str:
        return self.record(stage).status

    def is_done(self, stage: Stage | str) -> bool:
        return self.record(stage).done

    @property
    def complete(self) -> bool:
        return self.state == STATE_COMPLETE

    @property
    def in_progress(self) -> bool:
        return self.state == STATE_IN_PROGRESS

    @property
    def failed(self) -> bool:
        return self.state == STATE_FAILED

    def next_stage(self) -> Stage | None:
        """下一个还没完成的阶段（全部完成返回 ``None``）。"""
        for info in ORDERED_STAGES:
            if info.optional:
                continue
            if not self.is_done(info.stage):
                return info.stage
        for info in ORDERED_STAGES:
            if not self.is_done(info.stage):
                return info.stage
        return None

    def completed_keys(self) -> list[str]:
        return [info.key for info in ORDERED_STAGES if self.is_done(info.stage)]

    def failed_stages(self) -> list[str]:
        return [info.key for info in ORDERED_STAGES if self.record(info.stage).failed]

    def interrupted_stage(self) -> Stage | None:
        """上次是在哪一步被打断的（``running`` 状态 = 进程没来得及收尾）。"""
        for info in ORDERED_STAGES:
            if self.record(info.stage).status == STATUS_RUNNING:
                return info.stage
        return self.next_stage()

    def overall_fraction(self) -> float:
        """已完成的阶段折算成的总进度（0~1）。"""
        done = sum(
            stage_info(info.stage).weight
            for info in ORDERED_STAGES
            if self.is_done(info.stage)
        )
        return min(done / TOTAL_WEIGHT, 1.0)

    def summary(self) -> str:
        """一句话说明当前进度（界面 / 日志共用）。"""
        done = len(self.completed_keys())
        total = len(ORDERED_STAGES)
        stage = self.next_stage()
        if self.complete:
            return f"安装已完成（{done}/{total} 步）"
        if self.failed:
            failed = "、".join(stage_info(key).title for key in self.failed_stages())
            return f"安装中断在失败状态（{failed}）"
        if stage is None:
            return f"安装进行中（{done}/{total} 步）"
        return f"安装进行中（{done}/{total} 步），下一步：{stage_info(stage).title}"

    def last_exit_note(self) -> str:
        """上次退出的说明。"""
        stage = self.interrupted_stage()
        if stage is None:
            return ""
        before = Path(self.install_dir)
        if not before.exists():
            return ""
        return stage_info(stage).exit_note

    # ------------------------------------------------------------- 推进
    def log_event(self, kind: str, stage: Stage | str | None, message: str = "") -> None:
        key = stage.value if isinstance(stage, Stage) else (str(stage) if stage else "")
        self.events.append(
            {"kind": kind, "stage": key, "message": message, "at": _now()}
        )
        if len(self.events) > MAX_EVENTS:
            del self.events[: len(self.events) - MAX_EVENTS]

    def begin_run(self) -> None:
        """开始（或继续）一次安装执行。"""
        self.runs += 1
        self.state = STATE_IN_PROGRESS
        self.finished_at = None
        self.log_event("run", None, f"第 {self.runs} 次安装执行")

    def mark_running(self, stage: Stage | str, message: str = "") -> None:
        record = self.record(stage)
        record.status = STATUS_RUNNING
        record.message = message
        record.started_at = _now()
        record.finished_at = None
        record.attempts += 1

    def mark_done(self, stage: Stage | str, message: str = "", detail: str = "") -> None:
        record = self.record(stage)
        record.status = STATUS_DONE
        record.message = message
        record.detail = detail
        record.finished_at = _now()

    def mark_skipped(self, stage: Stage | str, message: str = "") -> None:
        record = self.record(stage)
        record.status = STATUS_SKIPPED
        record.message = message
        record.finished_at = _now()

    def mark_failed(self, stage: Stage | str, message: str = "") -> None:
        record = self.record(stage)
        record.status = STATUS_FAILED
        record.message = message
        record.finished_at = _now()
        self.state = STATE_FAILED
        self.log_event("failed", stage, message)

    def mark_interrupted(self, stage: Stage | str | None, reason: str = "") -> None:
        """记录"用户在这一步退出了"（不是失败，是可以接着做的状态）。"""
        if stage is not None:
            record = self.record(stage)
            # 断在 running / pending 上都保持"未完成"，下次继续时重做这一步
            if record.status == STATUS_RUNNING:
                record.status = STATUS_PENDING
                record.message = reason or "上次退出时中断在这一步"
        self.state = STATE_IN_PROGRESS
        self.log_event("interrupted", stage, reason)

    def finish(self) -> None:
        self.state = STATE_COMPLETE
        self.finished_at = _now()
        for info in ORDERED_STAGES:
            if not self.is_done(info.stage):
                self.mark_skipped(info.stage, "已完成")
        self.mark_done(Stage.DONE, "安装完成")
        self.log_event("complete", Stage.DONE, "安装完成")

    def mark_rolled_back(self, reason: str = "") -> None:
        self.state = STATE_ROLLED_BACK
        self.log_event("rolled_back", None, reason)

    def reset_from(self, stage: Stage | str, *, include: bool = True) -> None:
        """把某个阶段（及其之后）重置成待执行 —— "重新安装 / 修复安装"用。"""
        target = stage.value if isinstance(stage, Stage) else str(stage)
        keys = [info.key for info in ORDERED_STAGES]
        if target not in keys:
            self.reset_all()
            return
        start = keys.index(target) if include else keys.index(target) + 1
        for key in keys[start:]:
            self.records[key] = StageRecord(stage=key)
        self.records[Stage.DONE.value] = StageRecord(stage=Stage.DONE.value)
        self.state = STATE_IN_PROGRESS
        self.finished_at = None
        self.log_event("reset", stage, f"重置阶段：{target} 及其之后")

    def reset_all(self) -> None:
        for info in ORDERED_STAGES:
            self.records[info.key] = StageRecord(stage=info.key)
        self.records[Stage.DONE.value] = StageRecord(stage=Stage.DONE.value)
        self.state = STATE_IN_PROGRESS
        self.finished_at = None
        self.log_event("reset", None, "全部阶段重置")


@dataclass(slots=True)
class InstallProbe:
    """一次"这台机器上装没装、装到哪"的探测结果（只读，不落盘）。"""

    install_dir: Path
    journal: InstallJournal | None = None
    #: 安装目录是否存在
    directory_exists: bool = False
    #: 登记的安装位置（来自注册表 / 状态镜像）
    registered_dir: Path | None = None

    @property
    def installed(self) -> bool:
        return self.journal is not None

    @property
    def complete(self) -> bool:
        return bool(self.journal and self.journal.complete)

    @property
    def incomplete(self) -> bool:
        return bool(self.journal and not self.journal.complete)

    @property
    def missing_files(self) -> bool:
        """状态说装完了，但目录已经不在了（用户手工删过 / 移动过）。"""
        return self.installed and not self.directory_exists

    def describe(self) -> str:
        """一句给界面用的话。"""
        if self.journal is None:
            if self.registered_dir is not None:
                return (
                    "检测到上一次的安装记录，但安装目录里没有状态文件"
                    f"（{self.registered_dir}）。可以重新安装或先卸载残留。"
                )
            return "未检测到已安装的 Video2PersonVideo。"
        if self.missing_files:
            return "安装记录存在，但安装目录已被删除，建议卸载后重新安装。"
        if self.complete:
            return f"已安装 Video2PersonVideo {self.journal.app_version}（{self.install_dir}）"
        return f"上次安装未完成：{self.journal.summary()}"


def probe_install(
    install_dir: str | os.PathLike[str] | None = None,
    *,
    registered_dir: str | os.PathLike[str] | None = None,
) -> InstallProbe:
    """探测安装现状：默认看命令行 / 默认安装目录，也可只按注册表记录看。"""
    target = Path(install_dir) if install_dir else (paths.configured_install_dir() or paths.default_install_dir())
    exists = False
    try:
        exists = target.is_dir()
    except OSError:  # pragma: no cover
        exists = False

    journal = InstallJournal.load_for(target) if exists else None
    if journal is None:
        # 目录不在（或没状态文件）时，试试镜像副本 / 注册表记的位置
        candidates: list[Path] = []
        if registered_dir:
            candidates.append(Path(registered_dir))
        candidates.append(target)
        for candidate in candidates:
            mirrored = InstallJournal.load_for(candidate)
            if mirrored is not None:
                journal = mirrored
                target = Path(mirrored.install_dir or candidate)
                break

    probe = InstallProbe(install_dir=Path(target), journal=journal, directory_exists=exists)
    if registered_dir:
        probe.registered_dir = Path(registered_dir)
    elif journal is not None:
        probe.registered_dir = Path(journal.install_dir)
    return probe


__all__ = [
    "MAX_EVENTS",
    "SCHEMA_VERSION",
    "STATE_COMPLETE",
    "STATE_FAILED",
    "STATE_IN_PROGRESS",
    "STATE_ROLLED_BACK",
    "STATUS_DONE",
    "STATUS_FAILED",
    "STATUS_PENDING",
    "STATUS_RUNNING",
    "STATUS_SKIPPED",
    "InstallJournal",
    "InstallProbe",
    "StageRecord",
    "probe_install",
]
