"""pytest 公共 fixture。"""

from __future__ import annotations

import contextlib
import logging
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pytest

from video2personvideo.core.subject import Detection


def _write_video(
    path: Path,
    *,
    size: tuple[int, int] = (64, 48),
    frames: int = 12,
    fps: float = 10.0,
    draw: Callable[[np.ndarray, int], None] | None = None,
    fill: int = 0,
) -> Path:
    """用 MJPG/AVI 写一个合成视频（兼容性最好）。"""
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    if not writer.isOpened():  # pragma: no cover - 取决于 OpenCV 构建
        pytest.skip("当前 OpenCV 构建缺少 MJPG 编码器，跳过视频相关测试")

    width, height = size
    for index in range(frames):
        frame = np.full((height, width, 3), fill, dtype=np.uint8)
        if draw is not None:
            draw(frame, index)
        writer.write(frame)
    writer.release()

    if not path.exists() or path.stat().st_size == 0:  # pragma: no cover
        pytest.skip("测试视频生成失败")
    return path


@pytest.fixture()
def synthetic_video(tmp_path: Path) -> Path:
    """生成一个 12 帧的小视频，供读写/流水线测试使用。"""

    def draw(frame: np.ndarray, index: int) -> None:
        frame[:] = (index * 20) % 255
        cv2.rectangle(frame, (5 + index, 5), (20 + index, 30), (255, 255, 255), -1)

    return _write_video(tmp_path / "sample.avi", size=(64, 48), frames=12, draw=draw)


@pytest.fixture()
def make_video(tmp_path: Path) -> Callable[..., Path]:
    """工厂 fixture：按需生成各种尺寸 / 长度的合成视频。"""
    counter = {"value": 0}

    def _factory(
        name: str | None = None,
        *,
        size: tuple[int, int] = (160, 120),
        frames: int = 20,
        fps: float = 10.0,
        draw: Callable[[np.ndarray, int], None] | None = None,
        fill: int = 0,
    ) -> Path:
        counter["value"] += 1
        target = tmp_path / (name or f"clip_{counter['value']}.avi")
        return _write_video(target, size=size, frames=frames, fps=fps, draw=draw, fill=fill)

    return _factory


@pytest.fixture()
def make_vfr_video(tmp_path: Path) -> Callable[..., Path]:
    """工厂 fixture：用 ffmpeg 生成"可变帧率（VFR）"视频。

    做法是把「30fps 一段」和「10fps 一段」拼接起来：每帧的真实显示时长并不
    相等（这正是手机录像 / 录屏常见的形态），逐帧按单一 fps 写回就会出现
    "开头慢、后面突然变快"。没有 ffmpeg 时跳过依赖它的用例。
    """
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:  # pragma: no cover - 取决于环境
        pytest.skip("需要 ffmpeg 才能生成可变帧率测试视频")

    def _factory(
        name: str = "vfr.mp4",
        *,
        fast: float = 1.0,
        slow: float = 1.0,
        size: tuple[int, int] = (160, 120),
    ) -> Path:
        width, height = size
        target = tmp_path / name
        args = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size={width}x{height}:rate=30:duration={fast}",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size={width}x{height}:rate=10:duration={slow}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={fast + slow}",
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1[v]",
            "-map",
            "[v]",
            "-map",
            "2:a",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-video_track_timescale",
            "90000",
            str(target),
        ]
        completed = subprocess.run(args, capture_output=True, text=True, check=False)
        if completed.returncode != 0 or not target.exists() or target.stat().st_size == 0:
            pytest.skip(f"无法生成可变帧率测试视频：{completed.stderr.strip()[-200:]}")
        return target

    return _factory


class FakeDetector:
    """不依赖 YOLO 的假检测器：用给定函数按帧产出检测框。"""

    def __init__(
        self,
        provider: Callable[[np.ndarray, int], list[tuple[float, float, float, float]]],
        keypoint_provider: Callable[[np.ndarray, int], list] | None = None,
    ) -> None:
        self._provider = provider
        self._keypoints = keypoint_provider
        self._index = 0
        self.calls = 0
        #: 每次批量推理实际塞了几帧（验证批量推理真的生效）
        self.batch_sizes: list[int] = []

    def detect_boxes(self, frame: np.ndarray) -> list[Detection]:
        index = self._index
        self._index += 1
        self.calls += 1
        boxes = self._provider(frame, index)
        keypoints = self._keypoints(frame, index) if self._keypoints is not None else None
        return [
            Detection(
                bbox=tuple(box),
                confidence=0.9,
                keypoints=(
                    keypoints[pos] if keypoints is not None and pos < len(keypoints) else None
                ),
            )
            for pos, box in enumerate(boxes)
        ]

    def detect_boxes_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        self.batch_sizes.append(len(frames))
        return [self.detect_boxes(frame) for frame in frames]

    def detect(self, frame: np.ndarray) -> tuple[np.ndarray, int]:
        """兼容旧的"推理 + 画框"接口。"""
        count = len(self.detect_boxes(frame))
        return frame, count


@pytest.fixture()
def fake_detector() -> Callable[..., FakeDetector]:
    """工厂 fixture：``fake_detector(provider)`` → :class:`FakeDetector`。"""

    def _factory(provider, keypoint_provider=None) -> FakeDetector:
        return FakeDetector(provider, keypoint_provider)

    return _factory


def make_pose_keypoints(
    bbox: tuple[float, float, float, float],
    *,
    confidence: float = 0.9,
) -> tuple[tuple[float, float, float], ...]:
    """按人体比例造一套 COCO-17 关键点（用于姿态路径的测试）。"""
    x1, y1, x2, y2 = bbox
    width, height = x2 - x1, y2 - y1
    cx = (x1 + x2) / 2.0

    def point(fx: float, fy: float) -> tuple[float, float, float]:
        return (cx + fx * width, y1 + fy * height, confidence)

    points: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)] * 17
    points[0] = point(0.0, 0.08)  # 鼻
    points[1] = point(-0.03, 0.06)  # 左眼
    points[2] = point(0.03, 0.06)  # 右眼
    points[3] = point(-0.06, 0.07)  # 左耳
    points[4] = point(0.06, 0.07)  # 右耳
    points[5] = point(-0.11, 0.20)  # 左肩
    points[6] = point(0.11, 0.20)  # 右肩
    points[11] = point(-0.07, 0.50)  # 左胯
    points[12] = point(0.07, 0.50)  # 右胯
    return tuple(points)


@pytest.fixture()
def pose_keypoints() -> Callable[..., tuple]:
    """工厂 fixture：``pose_keypoints(bbox)`` → 一套合成关键点。"""
    return make_pose_keypoints


@pytest.fixture()
def moving_person() -> Callable[[np.ndarray, int], list[tuple[float, float, float, float]]]:
    """人物在画面中左右游走（用来验证追踪与平滑）。"""

    def _provider(frame: np.ndarray, index: int) -> list[tuple[float, float, float, float]]:
        height, width = frame.shape[:2]
        center = width * 0.25 + (width * 0.5) * (index % 20) / 19.0
        half = width * 0.08
        return [(center - half, height * 0.15, center + half, height * 0.95)]

    return _provider


@contextlib.contextmanager
def capture_project_logs(level: int = logging.WARNING):
    """捕获 ``video2personvideo`` logger 的记录（它的 propagate=False，caplog 收不到）。"""
    records: list[logging.LogRecord] = []

    class _Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    target = logging.getLogger("video2personvideo")
    handler = _Collector(level=level)
    target.addHandler(handler)
    previous = target.level
    target.setLevel(min(previous or level, level))
    try:
        yield records
    finally:
        target.removeHandler(handler)
        target.setLevel(previous)


@pytest.fixture()
def project_logs():
    """测试内捕获项目日志的 fixture。"""
    with capture_project_logs() as records:
        yield records
