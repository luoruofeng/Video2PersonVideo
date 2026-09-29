"""命令行入口测试。"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from video2personvideo.cli import _override_kwargs, build_parser, main
from video2personvideo.core.smoothing import CAMERA_PRESETS


def test_parser_defaults() -> None:
    args = build_parser().parse_args([])
    assert args.source is None
    assert args.no_audio is False
    assert args.check is False
    assert args.gui is False


def test_override_kwargs_maps_flags(tmp_path: Path) -> None:
    args = build_parser().parse_args(["-i", "in.mp4", "--no-audio", "--reencode", "--conf", "0.4"])
    kwargs = _override_kwargs(args)

    assert kwargs["save_audio"] is False
    assert kwargs["reencode"] is True
    assert kwargs["conf"] == pytest.approx(0.4)
    # 未显式指定 -o 时交给 AppConfig 按 M0 决策推导（原视频同目录）
    assert kwargs["output"] is None
    assert kwargs["show"] is None
    assert kwargs["crop"] is None


def test_override_kwargs_crop_options() -> None:
    args = build_parser().parse_args(
        [
            "-i", "in.mp4",
            "--ratio", "4:5",
            "--target-size", "1080x1350",
            "--no-crop",
            "--overwrite",
            "--detect-interval", "3",
            "--no-same-dir",
            "--output-suffix", ".mp4",
        ]
    )
    kwargs = _override_kwargs(args)

    assert kwargs["crop"] is False
    assert kwargs["aspect_ratio"] == "4:5"
    assert kwargs["target_width"] == 1080
    assert kwargs["target_height"] == 1350
    assert kwargs["overwrite"] is True
    assert kwargs["detect_interval"] == 3
    assert kwargs["output_same_dir"] is False
    assert kwargs["output_suffix"] == ".mp4"


def test_parse_target_size_rejects_bad_input() -> None:
    from video2personvideo.cli import _parse_target_size

    assert _parse_target_size("1080x1920") == (1080, 1920)
    assert _parse_target_size(None) == (None, None)
    with pytest.raises(ValueError):
        _parse_target_size("1080")


def test_main_batch_folder_without_videos(tmp_path: Path, capsys) -> None:
    folder = tmp_path / "empty"
    folder.mkdir()
    assert main(["-i", str(folder), "--no-audio"]) == 1
    assert "未在文件夹中发现" in capsys.readouterr().err


def test_override_kwargs_classes_all() -> None:
    args = build_parser().parse_args(["-i", "in.mp4", "--classes", "-1"])
    assert _override_kwargs(args)["classes"] == []


def test_override_kwargs_camera_follow_preset() -> None:
    """档位一次设置一组平滑参数。"""
    args = build_parser().parse_args(["-i", "in.mp4", "--camera-follow", "lock"])
    kwargs = _override_kwargs(args)
    lock = CAMERA_PRESETS["lock"].params

    assert kwargs["smoothing_alpha"] == pytest.approx(lock.alpha)
    assert kwargs["smoothing_deadzone"] == pytest.approx(lock.deadzone)
    assert kwargs["smoothing_zoom_alpha"] == pytest.approx(lock.zoom_alpha)
    assert kwargs["smoothing_max_speed"] == pytest.approx(lock.max_speed)


def test_override_kwargs_explicit_flags_beat_preset() -> None:
    args = build_parser().parse_args(
        ["-i", "in.mp4", "--camera-follow", "lock", "--deadzone", "0.05", "--pan-speed", "0.1"]
    )
    kwargs = _override_kwargs(args)

    assert kwargs["smoothing_deadzone"] == pytest.approx(0.05)
    assert kwargs["smoothing_max_speed"] == pytest.approx(0.1)
    # 未显式给出的项仍取档位值
    assert kwargs["smoothing_alpha"] == pytest.approx(CAMERA_PRESETS["lock"].params.alpha)


def test_override_kwargs_defaults_smoothing_to_none() -> None:
    """既不给档位也不给逐项参数时，交给配置文件 / 内置默认值。"""
    kwargs = _override_kwargs(build_parser().parse_args(["-i", "in.mp4"]))

    assert kwargs["smoothing_alpha"] is None
    assert kwargs["smoothing_deadzone"] is None
    assert kwargs["smoothing_max_speed"] is None



def test_override_kwargs_classes_subset() -> None:
    args = build_parser().parse_args(["-i", "in.mp4", "--classes", "0", "2"])
    assert _override_kwargs(args)["classes"] == [0, 2]


def test_main_check_without_source_returns_zero(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--check"]) == 0
    captured = capsys.readouterr()
    assert "环境自检" in captured.out


def test_main_without_source_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 2
    captured = capsys.readouterr()
    assert "用法" in captured.out or "usage" in captured.out


def test_main_missing_file_returns_one(tmp_path: Path) -> None:
    assert main(["-i", str(tmp_path / "nope.mp4"), "--no-audio"]) == 1


def test_main_version_exits(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert "v2pv" in capsys.readouterr().out


def test_namespace_shape() -> None:
    assert isinstance(build_parser().parse_args([]), argparse.Namespace)


# ------------------------------------------------------------------ 多人分屏
def test_override_kwargs_defaults_multi_person_to_none() -> None:
    """不传相关开关时交给配置文件 / 内置默认值（默认开启）。"""
    kwargs = _override_kwargs(build_parser().parse_args(["-i", "in.mp4"]))

    assert kwargs["multi_person"] is None
    assert kwargs["multi_person_max"] is None
    assert kwargs["multi_person_order"] is None
    assert kwargs["multi_person_layout_file"] is None


def test_override_kwargs_multi_person_flags(tmp_path: Path) -> None:
    layout = tmp_path / "layout.yaml"
    args = build_parser().parse_args(
        [
            "-i", "in.mp4",
            "--no-multi-person",
            "--multi-person-max", "3",
            "--multi-person-order", "score",
            "--layout-config", str(layout),
        ]
    )
    kwargs = _override_kwargs(args)

    assert kwargs["multi_person"] is False
    assert kwargs["multi_person_max"] == 3
    assert kwargs["multi_person_order"] == "score"
    assert kwargs["multi_person_layout_file"] == layout


def test_override_kwargs_multi_person_can_be_enabled_explicitly() -> None:
    args = build_parser().parse_args(["-i", "in.mp4", "--multi-person"])
    assert _override_kwargs(args)["multi_person"] is True


def test_multi_person_order_rejects_unknown_choice() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["-i", "in.mp4", "--multi-person-order", "random"])
