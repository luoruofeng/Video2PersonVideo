"""批量处理与任务编排：路径扫描、任务队列、失败隔离、汇总。

设计要点：

* 输入可以是单个文件，也可以是文件夹（递归 / 非递归可配）；
* 单个视频失败只记录错误并继续下一个，绝不中断整批；
* 开始前先预检输出目录写权限，避免处理一半才失败；
* 进度分两级：文件级（第 i / N 个）+ 帧级（当前视频 x / y 帧），
  合成出的整体进度单调不倒退。
"""

from __future__ import annotations

import contextlib
import os
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..config import PERSON_SUFFIX, AppConfig
from ..utils.logger import get_logger
from .processor import ProcessResult, ProgressInfo, build_detector, format_bytes, process_video

logger = get_logger(__name__)

#: 允许批量处理的视频扩展名白名单
VIDEO_EXTENSIONS: tuple[str, ...] = (".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm")

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_SKIPPED = "skipped"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"

_STATUS_LABELS = {
    STATUS_PENDING: "待处理",
    STATUS_RUNNING: "处理中",
    STATUS_DONE: "完成",
    STATUS_SKIPPED: "跳过",
    STATUS_FAILED: "失败",
    STATUS_CANCELLED: "已取消",
}

#: 中间产物 / 已生成产物的名字特征，遍历时直接忽略
_ARTIFACT_TOKENS = (".noaudio", ".encoded", ".sizing", ".meta")


def status_label(status: str) -> str:
    return _STATUS_LABELS.get(status, status)


@dataclass(slots=True)
class Task:
    """一个待处理的视频任务。"""

    source: Path
    output: Path
    status: str = STATUS_PENDING
    error: str = ""
    result: ProcessResult | None = None

    @property
    def ok(self) -> bool:
        return self.status == STATUS_DONE


@dataclass(slots=True)
class BatchProgress:
    """批量进度：同时给出文件级与整体进度。"""

    file_index: int
    file_count: int
    file_fraction: float
    overall: float
    message: str
    source: Path | None = None
    mode: str = ""
    eta: float = 0.0
    fps: float = 0.0
    elapsed: float = 0.0


@dataclass(slots=True)
class BatchResult:
    """整批处理的汇总。"""

    tasks: list[Task] = field(default_factory=list)
    elapsed: float = 0.0

    @property
    def succeeded(self) -> list[Task]:
        return [task for task in self.tasks if task.status == STATUS_DONE]

    @property
    def failed(self) -> list[Task]:
        return [task for task in self.tasks if task.status == STATUS_FAILED]

    @property
    def skipped(self) -> list[Task]:
        return [task for task in self.tasks if task.status == STATUS_SKIPPED]

    @property
    def cancelled(self) -> list[Task]:
        return [task for task in self.tasks if task.status == STATUS_CANCELLED]

    @property
    def volume(self) -> tuple[int, int]:
        """(原视频字节合计, 输出字节合计)，都只统计真正产出结果的任务。"""
        sources = sum(
            task.result.source_bytes for task in self.tasks if task.result is not None
        )
        outputs = sum(
            task.result.output_bytes for task in self.tasks if task.result is not None
        )
        return sources, outputs

    def summary(self) -> str:
        lines = [
            f"任务总数     : {len(self.tasks)}",
            f"成功 / 跳过 / 失败 / 取消 : {len(self.succeeded)} / "
            f"{len(self.skipped)} / {len(self.failed)} / {len(self.cancelled)}",
            f"总耗时       : {self.elapsed:.1f}s",
        ]
        source_bytes, output_bytes = self.volume
        if output_bytes:
            volume = format_bytes(output_bytes)
            if source_bytes:
                volume += (
                    f"（原视频合计 {format_bytes(source_bytes)}，"
                    f"{output_bytes / source_bytes:.2f}×）"
                )
            lines.append(f"输出体积     : {volume}")
        if self.failed:
            lines.append("失败清单：")
            lines += [f"  - {task.source.name}：{task.error}" for task in self.failed]
        if self.skipped:
            lines.append("跳过清单：")
            lines += [f"  - {task.source.name}：{task.error}" for task in self.skipped]
        return "\n".join(lines)


def is_person_artifact(path: str | Path) -> bool:
    """判断是否为已生成的产物（``*_person.*``）或中间文件，遍历时应忽略。"""
    target = Path(path)
    if target.stem.lower().endswith(PERSON_SUFFIX):
        return True
    return any(token in target.name.lower() for token in _ARTIFACT_TOKENS)


def discover_videos(
    target: str | Path,
    *,
    recursive: bool = True,
    extensions: Iterable[str] = VIDEO_EXTENSIONS,
) -> list[Path]:
    """扫描输入路径：文件 → 单任务；文件夹 → 按扩展名白名单遍历视频。

    显式传入的文件不做扩展名过滤（用户指定什么就处理什么），
    文件夹遍历时忽略 ``*_person.*`` 与中间产物，避免重复处理。
    """
    path = Path(target)
    if not path.exists():
        raise FileNotFoundError(f"输入路径不存在：{path}")
    if path.is_file():
        return [path]

    allowed = {ext.lower() for ext in extensions}
    iterator = path.rglob("*") if recursive else path.glob("*")
    found = [
        item
        for item in iterator
        if item.is_file()
        and item.suffix.lower() in allowed
        and not is_person_artifact(item)
    ]
    return sorted(found, key=lambda item: str(item).lower())


def check_writable(directory: str | Path) -> None:
    """预检输出目录可写；不可用时立即报错（避免处理一半才失败）。"""
    target = Path(directory)
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(f"输出目录不可创建：{target}（{exc}）") from exc

    probe = target / ".v2pv_write_test"
    try:
        probe.write_bytes(b"")
    except OSError as exc:
        raise RuntimeError(f"输出目录不可写：{target}（{exc}）") from exc
    finally:
        with contextlib.suppress(OSError):
            probe.unlink(missing_ok=True)
    if not os.access(target, os.W_OK):  # pragma: no cover - 兜底
        raise RuntimeError(f"输出目录不可写：{target}")


def plan_tasks(sources: Sequence[str | Path], cfg: AppConfig) -> list[Task]:
    """把源文件列表变成任务队列（输出路径按 M0 决策推导）。"""
    return [Task(source=Path(src), output=cfg.output_path_for(src)) for src in sources]


def run_batch(
    sources: Sequence[str | Path],
    cfg: AppConfig,
    *,
    progress_cb: Callable[[BatchProgress], None] | None = None,
    stop_cb: Callable[[], bool] | None = None,
    task_cb: Callable[[Task], None] | None = None,
    show_progress: bool = False,
    detector: object | None = None,
) -> BatchResult:
    """按顺序处理一批视频，失败隔离、可取消、可上报两级进度。"""
    if not sources:
        raise ValueError("没有待处理的视频")

    tasks = plan_tasks(sources, cfg)
    for directory in sorted({task.output.parent for task in tasks}, key=str):
        check_writable(directory)

    if detector is None:
        # 权重只加载一次，整批复用：模型加载 + 推理器初始化（GPU 上还有 CUDA 上下文 /
        # 卷积算法选择）要花几秒，而它对每个视频的检测结果毫无影响。
        # 加载失败时退回旧行为（交给 process_video 逐个文件重试并各自报错），绝不中断整批。
        try:
            detector = build_detector(cfg)
        except Exception as exc:  # noqa: BLE001 - 模型不可用时按单个文件失败处理
            logger.warning("检测器加载失败，改为逐个文件重试：%s", exc)
            detector = None

    started = time.perf_counter()
    total = len(tasks)
    last_overall = 0.0

    def notify(
        index: int,
        fraction: float,
        message: str,
        source: Path | None,
        *,
        mode: str = "",
        eta: float = 0.0,
        fps: float = 0.0,
    ) -> None:
        nonlocal last_overall
        overall = min(max((index + fraction) / total, last_overall), 1.0)
        last_overall = overall
        if progress_cb is None:
            return
        try:
            progress_cb(
                BatchProgress(
                    file_index=index + 1,
                    file_count=total,
                    file_fraction=min(max(fraction, 0.0), 1.0),
                    overall=overall,
                    message=message,
                    source=source,
                    mode=mode,
                    eta=eta,
                    fps=fps,
                    elapsed=time.perf_counter() - started,
                )
            )
        except Exception as exc:  # noqa: BLE001 - 回调异常不应中断处理
            logger.debug("批次进度回调异常：%s", exc)

    for index, task in enumerate(tasks):
        if stop_cb is not None and stop_cb():
            logger.warning("已取消，剩余 %d 个任务不再处理", total - index)
            for remaining in tasks[index:]:
                remaining.status = STATUS_CANCELLED
                if task_cb is not None:
                    task_cb(remaining)
            break

        task.status = STATUS_RUNNING
        if task_cb is not None:
            task_cb(task)
        logger.info("开始处理（%d/%d）：%s", index + 1, total, task.source.name)
        notify(index, 0.0, f"正在处理 {task.source.name}（{index + 1}/{total}）", task.source)

        def on_file_progress(
            info: ProgressInfo, _index: int = index, _task: Task = task
        ) -> None:
            notify(
                _index,
                info.fraction,
                info.message,
                _task.source,
                mode=info.mode,
                eta=info.eta,
                fps=info.fps,
            )

        try:
            result = process_video(
                cfg.with_overrides(source=task.source, output=task.output),
                progress_cb=on_file_progress,
                stop_cb=stop_cb,
                show_progress=show_progress,
                detector=detector,
            )
        except Exception as exc:  # noqa: BLE001 - 单文件失败必须隔离
            task.status = STATUS_FAILED
            task.error = str(exc)
            logger.error("处理失败：%s（%s）", task.source, exc)
        else:
            task.result = result
            if result.skipped:
                task.status = STATUS_SKIPPED
                task.error = result.skip_reason
            else:
                task.status = STATUS_DONE

        if task_cb is not None:
            task_cb(task)
        notify(
            index,
            1.0,
            f"{status_label(task.status)}：{task.source.name}",
            task.source,
        )

    return BatchResult(tasks=tasks, elapsed=time.perf_counter() - started)
