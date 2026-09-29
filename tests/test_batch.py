"""批量处理测试：路径扫描、任务编排、失败隔离、进度语义（M2 / M5）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from video2personvideo.config import AppConfig
from video2personvideo.core.batch import (
    STATUS_CANCELLED,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_SKIPPED,
    VIDEO_EXTENSIONS,
    BatchProgress,
    check_writable,
    discover_videos,
    is_person_artifact,
    plan_tasks,
    run_batch,
)


def _touch(path: Path, content: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


# ------------------------------------------------------------------ 扫描
def test_discover_single_file(tmp_path: Path) -> None:
    video = _touch(tmp_path / "a.mp4")
    assert discover_videos(video) == [video]


def test_discover_file_ignores_extension_whitelist(tmp_path: Path) -> None:
    """显式指定的文件不做扩展名过滤。"""
    odd = _touch(tmp_path / "a.xyz")
    assert discover_videos(odd) == [odd]


def test_discover_folder_filters_by_extension(tmp_path: Path) -> None:
    folder = tmp_path / "videos"
    keep = [_touch(folder / name) for name in ("a.mp4", "b.mov", "c.mkv")]
    _touch(folder / "notes.txt")
    _touch(folder / "image.png")

    found = discover_videos(folder)
    assert found == sorted(keep, key=lambda item: str(item).lower())


def test_discover_folder_skips_person_artifacts(tmp_path: Path) -> None:
    folder = tmp_path / "videos"
    keep = _touch(folder / "a.mp4")
    _touch(folder / "a_person.mp4")
    _touch(folder / "b_person.mov")
    _touch(folder / "c.noaudio.mp4")
    _touch(folder / "d.encoded.mp4")
    _touch(folder / "e.sizing.mp4")

    assert discover_videos(folder) == [keep]


def test_discover_folder_recursive_flag(tmp_path: Path) -> None:
    folder = tmp_path / "videos"
    top = _touch(folder / "top.mp4")
    nested = _touch(folder / "sub" / "deep.mp4")

    assert discover_videos(folder, recursive=True) == sorted(
        [top, nested], key=lambda item: str(item).lower()
    )
    assert discover_videos(folder, recursive=False) == [top]


def test_discover_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        discover_videos(tmp_path / "nope")


def test_is_person_artifact() -> None:
    assert is_person_artifact("a_person.mp4")
    assert is_person_artifact("dir/a_person.mov")
    assert is_person_artifact("a.noaudio.mp4")
    assert is_person_artifact("a.sizing.mp4")
    assert is_person_artifact("a.meta.mp4")
    assert not is_person_artifact("a.mp4")
    assert not is_person_artifact("person.mp4")


def test_extensions_whitelist_covers_common_containers() -> None:
    assert {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"} <= set(VIDEO_EXTENSIONS)


# ------------------------------------------------------------- 写权限预检
def test_check_writable_creates_directory(tmp_path: Path) -> None:
    target = tmp_path / "out" / "deep"
    check_writable(target)
    assert target.is_dir()
    assert not list(target.iterdir())  # 探测文件必须被清理


def test_check_writable_rejects_file_as_directory(tmp_path: Path) -> None:
    blocker = _touch(tmp_path / "blocker")
    with pytest.raises(RuntimeError):
        check_writable(blocker / "sub")


# ------------------------------------------------------------------ 任务
def test_plan_tasks_uses_config_naming(tmp_path: Path) -> None:
    cfg = AppConfig(output_same_dir=True)
    tasks = plan_tasks([tmp_path / "a.mp4", tmp_path / "b.mov"], cfg)
    assert tasks[0].output == tmp_path / "a_person.mp4"
    assert tasks[1].output == tmp_path / "b_person.mov"


def test_run_batch_isolates_failures(tmp_path: Path, make_video, fake_detector, moving_person) -> None:
    good = make_video("good.avi", size=(160, 120), frames=8)
    missing = tmp_path / "missing.avi"
    cfg = AppConfig(save_audio=False, aspect_ratio="9:16")

    result = run_batch(
        [good, missing],
        cfg,
        detector=fake_detector(moving_person),
    )

    assert len(result.tasks) == 2
    assert result.tasks[0].status == STATUS_DONE
    assert result.tasks[1].status == STATUS_FAILED
    assert "不存在" in result.tasks[1].error
    assert good.with_name("good_person.avi").exists()
    assert "成功 / 跳过 / 失败" in result.summary()
    assert len(result.succeeded) == 1 and len(result.failed) == 1


def test_run_batch_skips_existing_output(tmp_path: Path, make_video, fake_detector, moving_person) -> None:
    video = make_video("clip.avi", size=(160, 120), frames=6)
    (video.parent / "clip_person.avi").write_bytes(b"already there")

    result = run_batch(
        [video], AppConfig(save_audio=False), detector=fake_detector(moving_person)
    )
    assert result.tasks[0].status == STATUS_SKIPPED
    assert result.tasks[0].error


def test_run_batch_overwrites_when_asked(tmp_path: Path, make_video, fake_detector) -> None:
    video = make_video("clip.avi", size=(160, 120), frames=6)
    existing = video.parent / "clip_person.avi"
    existing.write_bytes(b"stale")

    static = fake_detector(lambda frame, index: [(30.0, 20.0, 70.0, 110.0)])
    result = run_batch([video], AppConfig(save_audio=False, overwrite=True), detector=static)
    assert result.tasks[0].status == STATUS_DONE
    assert existing.stat().st_size > len(b"stale")


def test_run_batch_progress_is_monotonic(tmp_path: Path, make_video, fake_detector) -> None:
    videos = [make_video(f"c{index}.avi", size=(160, 120), frames=6) for index in range(3)]
    seen: list[BatchProgress] = []
    static = fake_detector(lambda frame, index: [(30.0, 20.0, 70.0, 110.0)])

    run_batch(videos, AppConfig(save_audio=False), progress_cb=seen.append, detector=static)

    assert seen
    overalls = [item.overall for item in seen]
    assert overalls == sorted(overalls)
    assert overalls[-1] == pytest.approx(1.0)
    assert all(0.0 <= item.file_fraction <= 1.0 for item in seen)
    assert {item.file_count for item in seen} == {3}
    assert seen[-1].file_index == 3


def test_run_batch_cancel_marks_remaining(tmp_path: Path, make_video, fake_detector) -> None:
    videos = [make_video(f"c{index}.avi", size=(160, 120), frames=6) for index in range(3)]
    static = fake_detector(lambda frame, index: [(30.0, 20.0, 70.0, 110.0)])
    state = {"cancel": False}

    def stop_cb() -> bool:
        return state["cancel"]

    def task_cb(task) -> None:
        # 第一个任务跑完后请求取消
        if task.status == STATUS_DONE:
            state["cancel"] = True

    result = run_batch(
        videos,
        AppConfig(save_audio=False),
        stop_cb=stop_cb,
        task_cb=task_cb,
        detector=static,
    )
    assert result.tasks[0].status == STATUS_DONE
    assert all(task.status == STATUS_CANCELLED for task in result.tasks[1:])
    assert len(result.cancelled) == 2
    assert result.tasks[0].result is not None
    assert result.tasks[0].result.frames_processed == 6


def test_run_batch_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        run_batch([], AppConfig())
