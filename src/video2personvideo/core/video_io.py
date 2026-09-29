"""视频/图像的底层读写封装（基于 OpenCV）。"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..utils.logger import get_logger

logger = get_logger(__name__)

#: 读不到 FPS 时的兜底帧率
FALLBACK_FPS = 25.0

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
    """

    def __init__(self, path: str | Path) -> None:
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

    # ------------------------------------------------------------- 迭代读取
    def frames(self) -> Iterator[tuple[int, np.ndarray]]:
        """按 (帧序号, BGR 图像) 迭代，帧序号从 0 开始。"""
        index = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            self._counted = index + 1
            yield index, frame
            index += 1

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
    """

    def __init__(
        self,
        path: str | Path,
        fps: float,
        size: tuple[int, int],
        fourcc: str | None = None,
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

    def write(self, frame: np.ndarray) -> None:
        """写入一帧；尺寸与初始化不一致的帧会被跳过并只告警一次。"""
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
        if getattr(self, "writer", None) is not None:
            self.writer.release()
            self.writer = None  # type: ignore[assignment]

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
