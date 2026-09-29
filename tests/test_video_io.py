"""视频读写模块测试。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from video2personvideo.core.video_io import (
    VideoReader,
    VideoWriter,
    save_frame_image,
)


def test_reader_meta(synthetic_video: Path) -> None:
    with VideoReader(synthetic_video) as reader:
        meta = reader.meta
        assert meta.width == 64
        assert meta.height == 48
        assert meta.fps == pytest.approx(10.0, abs=0.5)
        assert meta.size == (64, 48)
        assert "sample.avi" in meta.describe()


def test_reader_iterates_all_frames(synthetic_video: Path) -> None:
    with VideoReader(synthetic_video) as reader:
        frames = [(index, frame.shape) for index, frame in reader.frames()]
    assert len(frames) == 12
    assert frames[0][0] == 0
    assert frames[-1][1] == (48, 64, 3)
    assert reader.frames_read == 12


def test_reader_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        VideoReader(tmp_path / "missing.mp4")


def test_writer_roundtrip(tmp_path: Path) -> None:
    target = tmp_path / "out.avi"
    frames = [np.full((48, 64, 3), value, dtype=np.uint8) for value in (10, 120, 240)]
    with VideoWriter(target, fps=10.0, size=(64, 48)) as writer:
        for frame in frames:
            writer.write(frame)
    assert target.exists() and target.stat().st_size > 0

    with VideoReader(target) as reader:
        written = list(reader.frames())
    assert len(written) == len(frames)


def test_writer_skips_mismatched_frame(tmp_path: Path) -> None:
    target = tmp_path / "out.avi"
    with VideoWriter(target, fps=10.0, size=(64, 48)) as writer:
        writer.write(np.zeros((32, 32, 3), dtype=np.uint8))  # 尺寸不符，应被跳过
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))

    with VideoReader(target) as reader:
        assert len(list(reader.frames())) == 1


def test_save_frame_image(tmp_path: Path) -> None:
    frame = np.full((20, 20, 3), 200, dtype=np.uint8)
    saved = save_frame_image(frame, tmp_path / "frames", 7)
    assert saved.name == "frame_000007.jpg"
    assert cv2.imread(str(saved)) is not None
