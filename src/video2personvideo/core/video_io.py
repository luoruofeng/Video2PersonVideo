"""视频/图像的底层读写封装（基于 OpenCV）。

读与写默认都跑在**后台线程**里：解码（:class:`VideoReader` 的预读）与编码
（:class:`VideoWriter` 的异步写帧）都是独立的 C 例程、会让出 GIL，因此能与主线程的
推理 / 裁剪重叠，把这两段等待从总耗时里去掉。帧的顺序与内容与逐帧同步读写
**逐像素一致**（队列按序进出、出错一样往上抛）；想退回单线程读写，
把环境变量 ``V2PV_SYNC_IO=1`` 设上即可。
"""

from __future__ import annotations

import contextlib
import os
import queue
import sys
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..utils.logger import get_logger

logger = get_logger(__name__)

#: 读不到 FPS 时的兜底帧率
FALLBACK_FPS = 25.0

#: 后台读 / 写队列的深度（帧数）：太小重叠不起来，太大白占内存
IO_QUEUE_SIZE = 4

#: 收尾时等待后台线程的上限（秒）；正常是毫秒级，超时说明卡在底层 I/O 上
IO_JOIN_TIMEOUT = 10.0

#: 环境变量：设为 1 时关闭后台读写，退回完全同步的旧行为（排错 / 兼容用）
SYNC_IO_ENV_VAR = "V2PV_SYNC_IO"


def sync_io_forced() -> bool:
    """是否强制走同步读写（``V2PV_SYNC_IO=1``）。"""
    value = str(os.environ.get(SYNC_IO_ENV_VAR, "")).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _put(target: queue.Queue, item: object, stop: threading.Event) -> bool:
    """带"停止检查"的入队：队列满时每 0.1s 看一眼是否已请求停止。

    返回是否真的放进去了。生产者线程因此不会因为消费者提前退出而永久卡住，
    ``close()`` 里的 ``join`` 也就一定能返回。
    """
    while not stop.is_set():
        try:
            target.put(item, timeout=0.1)
            return True
        except queue.Full:
            continue
    return False

#: 不同容器使用不同的编码器：mp4v 兼容性最好，avi 用 MJPG 更稳
_FOURCC_BY_SUFFIX = {
    ".mp4": "mp4v",
    ".m4v": "mp4v",
    ".mov": "mp4v",
    ".mkv": "mp4v",
    ".avi": "MJPG",
}


@dataclass(slots=True)
class VideoMeta:
    """视频元信息。"""

    path: Path
    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def size(self) -> tuple[int, int]:
        """OpenCV 的 (宽, 高)。"""
        return self.width, self.height

    @property
    def duration(self) -> float:
        """时长（秒）；未知帧数时返回 0。"""
        return self.frame_count / self.fps if self.fps > 0 else 0.0

    def describe(self) -> str:
        frames = str(self.frame_count) if self.frame_count > 0 else "未知"
        return (
            f"{self.path.name} | {self.width}x{self.height} | {self.fps:.2f} FPS | "
            f"{frames} 帧 | 约 {self.duration:.1f}s"
        )


class VideoReader:
    """逐帧读取视频，用法::

        with VideoReader("in.mp4") as reader:
            for index, frame in reader.frames():
                ...

    :meth:`frames` 默认在**后台线程里预读**（队列深度 :data:`IO_QUEUE_SIZE` 帧），
    解码与主线程的推理 / 裁剪重叠；帧序号与帧内容与同步读取完全一致。
    想要"读一帧、处理一帧"的严格同步行为，传 ``prefetch=False`` 或设 ``V2PV_SYNC_IO=1``。
    """

    def __init__(
        self,
        path: str | Path,
        *,
        prefetch: bool | None = None,
        queue_size: int = IO_QUEUE_SIZE,
    ) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"输入视频不存在：{self.path}")

        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开视频（编码不支持或文件损坏）：{self.path}")

        fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.meta = VideoMeta(
            path=self.path,
            width=int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            height=int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
            fps=fps if fps > 0 else FALLBACK_FPS,
            frame_count=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        )
        self._counted = 0
        #: 后台预读开关（``None`` = 跟随默认：开启，除非 ``V2PV_SYNC_IO=1``）
        self._prefetch = (not sync_io_forced()) if prefetch is None else bool(prefetch)
        self._queue_size = max(int(queue_size), 1)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._items: queue.Queue | None = None

    # ------------------------------------------------------------- 迭代读取
    def frames(self) -> Iterator[tuple[int, np.ndarray]]:
        """按 (帧序号, BGR 图像) 迭代，帧序号从 0 开始。

        默认后台预读：解码与主线程的计算重叠，顺序与内容完全一致。
        """
        if not self._prefetch:
            yield from self._frames_sync()
            return
        yield from self._frames_prefetch()

    def _frames_sync(self) -> Iterator[tuple[int, np.ndarray]]:
        """逐帧同步读取（原行为，``prefetch=False`` 时使用）。"""
        index = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            self._counted = index + 1
            yield index, frame
            index += 1

    def _frames_prefetch(self) -> Iterator[tuple[int, np.ndarray]]:
        """后台线程预读解码，主线程只从队列里取（帧序与内容与同步读取一致）。

        解码失败 / 读流异常都会**留给主线程抛出**，不会被吞掉；消费者提前退出
        （取消处理、只想读几帧）时生产者会在 0.1s 内看到停止信号并退出。
        """
        items: queue.Queue[tuple[int, np.ndarray] | None] = queue.Queue(
            maxsize=self._queue_size
        )
        self._items = items
        errors: list[BaseException] = []

        def produce() -> None:
            index = 0
            try:
                while not self._stop.is_set():
                    ok, frame = self.cap.read()
                    if not ok:
                        break
                    index += 1
                    self._counted = index
                    if not _put(items, (index - 1, frame), self._stop):
                        return
            except BaseException as exc:  # noqa: BLE001 - 交给主线程上报
                errors.append(exc)
            finally:
                # 消费者已经走了（停止请求）就不必再送结束标记，否则它会一直等
                _put(items, None, self._stop)

        self._thread = threading.Thread(target=produce, name="v2pv-reader", daemon=True)
        self._thread.start()
        try:
            while True:
                item = items.get()
                if item is None:
                    break
                yield item
        finally:
            self._stop.set()
            self._join_reader()

        if errors:
            raise errors[0]

    def _join_reader(self) -> None:
        """等预读线程退出（最多 :data:`IO_JOIN_TIMEOUT` 秒）。"""
        thread = self._thread
        self._thread = None
        self._items = None
        if thread is None:
            return
        thread.join(timeout=IO_JOIN_TIMEOUT)
        if thread.is_alive():  # pragma: no cover - 只有底层解码卡死才会走到
            logger.debug("预读线程未在 %.0f 秒内退出，跳过等待", IO_JOIN_TIMEOUT)

    def read(self) -> tuple[bool, np.ndarray | None]:
        """读取单帧（自行控制循环时使用）。"""
        ok, frame = self.cap.read()
        if ok:
            self._counted += 1
        return ok, frame if ok else None

    @property
    def frames_read(self) -> int:
        return self._counted

    # ------------------------------------------------------------- 资源释放
    def close(self) -> None:
        # 顺序很重要：先让预读线程停下（它可能正拿着 cap 在读），再释放解码器
        self._stop.set()
        items = self._items
        if items is not None:
            with contextlib.suppress(queue.Full):
                items.put_nowait(None)  # 唤醒仍阻塞在 get() 的消费者
        self._join_reader()
        if getattr(self, "cap", None) is not None:
            self.cap.release()
            self.cap = None  # type: ignore[assignment]

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - 兜底释放
        with contextlib.suppress(Exception):
            self.close()


class VideoWriter:
    """写视频。用法::

        with VideoWriter("out.mp4", fps=30, size=(1280, 720)) as writer:
            writer.write(frame)

    :meth:`write` 默认只把帧放进队列，由**后台线程**按入队顺序逐帧编码：
    编码与主线程的推理 / 裁剪重叠，写进文件的内容与同步写逐像素一致。
    队列满时 :meth:`write` 会短暂阻塞（背压），不会无限吃内存。
    传 ``async_write=False`` 或设 ``V2PV_SYNC_IO=1`` 可退回同步写。
    """

    def __init__(
        self,
        path: str | Path,
        fps: float,
        size: tuple[int, int],
        fourcc: str | None = None,
        *,
        async_write: bool | None = None,
        queue_size: int = IO_QUEUE_SIZE,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.size = (int(size[0]), int(size[1]))
        self.fps = float(fps) if fps and fps > 0 else FALLBACK_FPS
        self.fourcc = fourcc or _FOURCC_BY_SUFFIX.get(self.path.suffix.lower(), "mp4v")
        self._warned_shape = False

        code = cv2.VideoWriter_fourcc(*self.fourcc)
        self.writer = cv2.VideoWriter(str(self.path), code, self.fps, self.size)
        if not self.writer.isOpened():
            raise RuntimeError(
                f"无法创建输出视频：{self.path}（编码器 {self.fourcc} 不可用，"
                "请换用 .mp4 后缀或安装带编码器的 opencv 版本）"
            )

        #: 异步写：编码放后台线程（``None`` = 跟随默认：开启，除非 ``V2PV_SYNC_IO=1``）
        self._async = (not sync_io_forced()) if async_write is None else bool(async_write)
        self._queue: queue.Queue[np.ndarray | None] | None = None
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        if self._async:
            self._queue = queue.Queue(maxsize=max(int(queue_size), 1))
            self._thread = threading.Thread(
                target=self._write_loop, name="v2pv-writer", daemon=True
            )
            self._thread.start()

    def write(self, frame: np.ndarray) -> None:
        """写入一帧；尺寸与初始化不一致的帧会被跳过并只告警一次。

        异步模式下这里只入队（顺序即写出顺序）；写线程出错时在下一次调用（或
        :meth:`close`）抛出，绝不静默丢帧。
        """
        if self._queue is None:
            self._write_frame(frame)
            return
        if self._error is not None:
            raise RuntimeError(f"写入视频失败：{self._error}") from self._error
        # 队列满时短暂阻塞：写线程始终在消费，因此不会死锁，只会形成背压
        self._queue.put(frame)

    def _write_loop(self) -> None:
        """写线程：按入队顺序编码；出错后继续把队列排空（避免主线程卡在 put 上）。

        队列句柄在这里取本地引用：``close()`` 之后会摘掉 ``self._queue``，
        线程若在那一瞬间才开始取帧，读到 ``None`` 就会把剩下的帧丢掉。
        """
        queue_ = self._queue
        if queue_ is None:  # pragma: no cover - 只有构造异常才会走到
            return
        while True:
            frame = queue_.get()
            try:
                if frame is None:
                    return
                if self._error is None:
                    self._write_frame(frame)
            except BaseException as exc:  # noqa: BLE001 - 留给主线程上报
                self._error = exc

    def _write_frame(self, frame: np.ndarray) -> None:
        """真正写一帧（做尺寸校验，与旧行为逐像素一致）。"""
        if frame.shape[1] != self.size[0] or frame.shape[0] != self.size[1]:
            if not self._warned_shape:
                logger.warning(
                    "帧尺寸 %sx%s 与输出尺寸 %sx%s 不一致，已跳过该帧",
                    frame.shape[1],
                    frame.shape[0],
                    self.size[0],
                    self.size[1],
                )
                self._warned_shape = True
            return
        self.writer.write(frame)

    def close(self) -> None:
        """排空队列 → 等写线程结束 → 释放编码器；有未上报的写错误时抛出。"""
        error = self._error
        self._error = None
        queue_ = self._queue
        if queue_ is not None:
            queue_.put(None)  # 送结束标记；写线程一直在消费，不会阻塞
        thread = self._thread
        self._thread = None
        if thread is not None:
            # 先把队列排空（写线程按序取走全部帧），再等它退出
            thread.join(timeout=IO_JOIN_TIMEOUT)
            if thread.is_alive():  # pragma: no cover - 只有编码器卡死才会走到
                logger.debug("写线程未在 %.0f 秒内退出，跳过等待", IO_JOIN_TIMEOUT)
        self._queue = None
        if getattr(self, "writer", None) is not None:
            self.writer.release()
            self.writer = None  # type: ignore[assignment]
        if error is not None:
            if sys.exc_info()[0] is None:
                raise RuntimeError(f"写入视频失败：{error}") from error
            # 正有别的异常在往上抛（例如处理被中断）：不要抢戏，只留一条日志
            logger.debug("写线程还有未上报的错误：%s", error)

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def save_frame_image(
    frame: np.ndarray,
    directory: str | Path,
    index: int,
    *,
    prefix: str = "frame",
    ext: str = ".jpg",
    quality: int = 95,
) -> Path:
    """把单帧保存为图片（截图功能）。"""
    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{prefix}_{index:06d}{ext}"
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)] if ext.lower() in {".jpg", ".jpeg"} else []
    if not cv2.imwrite(str(target), frame, params):
        raise RuntimeError(f"截图保存失败：{target}")
    return target
