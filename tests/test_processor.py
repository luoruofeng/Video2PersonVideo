"""处理流水线测试。

默认只跑不依赖模型的轻量用例；需要真实推理的端到端用例用 ``-m slow`` 或
环境变量 ``RUN_SLOW=1`` 手动开启。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from video2personvideo.config import AppConfig
from video2personvideo.core.processor import process_video


def test_requires_source() -> None:
    with pytest.raises(ValueError):
        process_video(AppConfig())


def test_missing_source_file(tmp_path: Path) -> None:
    cfg = AppConfig(source=tmp_path / "nope.mp4", output=tmp_path / "out.mp4", save_audio=False)
    with pytest.raises(FileNotFoundError):
        process_video(cfg)


def test_invalid_config_rejected(synthetic_video: Path, tmp_path: Path) -> None:
    cfg = AppConfig(source=synthetic_video, output=tmp_path / "out.avi", conf=2.0, save_audio=False)
    with pytest.raises(ValueError):
        process_video(cfg)


def _two_people(frame, index):
    return [(20.0, 40.0, 100.0, 220.0), (220.0, 40.0, 300.0, 220.0)]


def _multi_config(source: Path, output: Path, **overrides) -> AppConfig:
    base = {
        "source": source,
        "output": output,
        "aspect_ratio": "9:16",
        "target_width": 90,
        "target_height": 160,
        "save_audio": False,
        "normalize_vfr": False,
        "size_guard": False,
    }
    base.update(overrides)
    return AppConfig(**base)


def test_processor_splits_multiple_people(make_video, fake_detector, tmp_path: Path) -> None:
    """多人画面：整段视频都走分屏，汇总里如实报出「多人分屏」与布局。"""
    source = make_video("two.avi", size=(320, 240), frames=8)
    cfg = _multi_config(source, tmp_path / "two_person.avi")

    result = process_video(cfg, show_progress=False, detector=fake_detector(_two_people))

    assert result.frames_processed == 8
    assert result.multi_frames == 8
    assert result.windows_peak == 2
    assert result.multi_layouts  # 用了哪些布局也要报出来
    summary = result.summary()
    assert "多人分屏" in summary
    assert "分屏布局" in summary
    assert (tmp_path / "two_person.avi").exists()


def test_processor_multi_person_off_keeps_single_window(
    make_video, fake_detector, tmp_path: Path
) -> None:
    """关掉开关：同一段多人视频不再分屏，汇总里说明"未启用"。"""
    source = make_video("two.avi", size=(320, 240), frames=6)
    cfg = _multi_config(source, tmp_path / "single.avi", multi_person=False)

    result = process_video(cfg, show_progress=False, detector=fake_detector(_two_people))

    assert result.multi_frames == 0
    assert result.windows_peak == 0
    assert result.multi_layouts == {}
    assert "未启用" in result.multi_note
    assert "多人分屏" in result.summary()


def test_processor_single_person_video_does_not_split(
    make_video, fake_detector, tmp_path: Path
) -> None:
    """单人视频：即使开关默认打开，输出与"同时只显示一个人"一致。"""
    source = make_video("one.avi", size=(320, 240), frames=6)
    cfg = _multi_config(source, tmp_path / "one_person.avi")

    def one_person(frame, index):
        return [(120.0, 40.0, 200.0, 220.0)]

    result = process_video(cfg, show_progress=False, detector=fake_detector(one_person))

    assert result.multi_frames == 0
    assert "本片未出现多个主要人物" in result.multi_note


@pytest.mark.slow
def test_end_to_end(synthetic_video: Path, tmp_path: Path) -> None:
    pytest.importorskip("ultralytics")
    if os.environ.get("RUN_SLOW") != "1":
        pytest.skip("慢速测试：设置 RUN_SLOW=1 后运行")

    output = tmp_path / "out.avi"
    cfg = AppConfig(
        source=synthetic_video,
        output=output,
        model="yolo11n.pt",
        imgsz=320,
        conf=0.5,
        save_audio=False,
    )
    result = process_video(cfg, show_progress=False)

    assert output.exists() and output.stat().st_size > 0
    assert result.frames_processed == 12
    assert result.frames_with_person == 0  # 合成视频里没有人
    assert "输出视频" in result.summary()
