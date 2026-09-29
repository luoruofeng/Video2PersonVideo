"""配置模块测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from video2personvideo.config import (
    DEFAULT_MODEL,
    PERSON_CLASS_ID,
    AppConfig,
    default_output_path,
    load_config,
)
from video2personvideo.core.ratio import parse_ratio

#: 仓库自带配置文件（测试用绝对路径，不受工作目录影响）
DEFAULT_CONFIG_FILE = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def test_defaults() -> None:
    cfg = AppConfig()
    assert cfg.model == DEFAULT_MODEL
    assert cfg.classes == [PERSON_CLASS_ID]
    assert cfg.device is None
    assert cfg.save_audio is True
    assert cfg.reencode is False


def test_from_mapping_coerces_types() -> None:
    cfg = AppConfig.from_mapping(
        {
            "source": "data/input/a.mp4",
            "output": "out/a.mp4",
            "classes": ["0", "2"],
            "imgsz": "960",
            "conf": "0.4",
            "device": "auto",
            "save_frames_dir": None,
        }
    )
    assert isinstance(cfg.source, Path)
    assert cfg.classes == [0, 2]
    assert cfg.imgsz == 960
    assert cfg.conf == pytest.approx(0.4)
    assert cfg.device is None
    assert cfg.save_frames_dir is None


def test_from_mapping_ignores_unknown_fields() -> None:
    cfg = AppConfig.from_mapping({"model": "yolo11s.pt", "不存在的字段": 1})
    assert cfg.model == "yolo11s.pt"


def test_from_mapping_empty_classes_means_all() -> None:
    assert AppConfig.from_mapping({"classes": None}).classes == []


def test_yaml_roundtrip(tmp_path: Path) -> None:
    import yaml

    target = tmp_path / "custom.yaml"
    original = AppConfig(source=Path("in.mp4"), output=Path("out.mp4"), conf=0.35, imgsz=1280)
    target.write_text(yaml.safe_dump(original.to_dict(), allow_unicode=True), encoding="utf-8")

    loaded = AppConfig.from_yaml(target)
    assert loaded.source == Path("in.mp4")
    assert loaded.output == Path("out.mp4")
    assert loaded.conf == pytest.approx(0.35)
    assert loaded.imgsz == 1280


def test_from_yaml_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        AppConfig.from_yaml(tmp_path / "nope.yaml")


def test_load_config_missing_path_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_config("configs/__not_exists__.yaml")


def test_with_overrides_skips_none() -> None:
    cfg = AppConfig(conf=0.25, model="yolo11n.pt")
    updated = cfg.with_overrides(conf=0.5, model=None)
    assert updated.conf == pytest.approx(0.5)
    assert updated.model == "yolo11n.pt"
    assert cfg.conf == pytest.approx(0.25)  # 原对象不被修改


@pytest.mark.parametrize(
    ("field", "value"),
    [("conf", 0.0), ("conf", 1.5), ("iou", 0.0), ("imgsz", 0), ("crf", 99)],
)
def test_validate_rejects_invalid(field: str, value: float) -> None:
    cfg = AppConfig(**{field: value})
    with pytest.raises(ValueError):
        cfg.validate()


def test_validate_accepts_defaults() -> None:
    AppConfig().validate()


def test_default_output_path_same_dir() -> None:
    """M0 决策：默认输出到原视频同目录，命名为 <原名>_person<原后缀>。"""
    assert default_output_path("data/input/a.mp4") == Path("data/input/a_person.mp4")
    assert default_output_path("clip.mov") == Path("clip_person.mov")
    assert default_output_path("clip") == Path("clip_person.mp4")


def test_default_output_path_options() -> None:
    assert default_output_path("d/a.mov", ".mp4") == Path("d/a_person.mp4")
    assert default_output_path("d/a.mov", same_dir=False) == Path("data/output/a_person.mov")


def test_output_path_for_uses_config() -> None:
    cfg = AppConfig(output_same_dir=False, output_suffix="mp4")
    assert cfg.output_path_for("d/in.mov") == Path("data/output/in_person.mp4")
    cfg2 = AppConfig(output_same_dir=True)
    assert cfg2.output_path_for("d/in.mov") == Path("d/in_person.mov")


def test_defaults_include_crop_settings() -> None:
    cfg = AppConfig()
    assert cfg.crop is True
    assert cfg.annotate is False
    assert cfg.aspect_ratio == "9:16"
    assert cfg.overwrite is False
    assert cfg.detect_interval == 1
    assert cfg.output_same_dir is True


def test_from_mapping_coerces_crop_fields() -> None:
    cfg = AppConfig.from_mapping(
        {
            "crop": "false",
            "overwrite": "yes",
            "aspect_ratio": "4:5",
            "target_width": "1080",
            "target_height": "1350",
            "detect_interval": "3",
            "smoothing_alpha": "0.5",
            "output_suffix": "mp4",
        }
    )
    assert cfg.crop is False
    assert cfg.overwrite is True
    assert cfg.detect_interval == 3
    assert cfg.smoothing_alpha == pytest.approx(0.5)
    assert cfg.output_suffix == ".mp4"
    assert cfg.resolve_ratio().target_size == (1080, 1350)


def test_resolve_ratio_and_derived_objects() -> None:
    cfg = AppConfig(aspect_ratio="1:1")
    assert cfg.resolve_ratio().target_size == (1080, 1080)
    assert cfg.framing_params().halfbody_ratio == pytest.approx(0.25)
    assert cfg.subject_weights().continuity == pytest.approx(1.2)


def test_validate_rejects_bad_crop_params() -> None:
    with pytest.raises(ValueError):
        AppConfig(aspect_ratio="0:0").validate()
    with pytest.raises(ValueError):
        AppConfig(smoothing_alpha=0.0).validate()
    with pytest.raises(ValueError):
        AppConfig(detect_interval=0).validate()
    with pytest.raises(ValueError):
        AppConfig(target_width=1080).validate()
    with pytest.raises(ValueError):
        AppConfig(output_suffix="mp4").validate()


def test_validate_rejects_bad_smoothing_params() -> None:
    with pytest.raises(ValueError):
        AppConfig(smoothing_zoom_alpha=0.0).validate()
    with pytest.raises(ValueError):
        AppConfig(smoothing_zoom_deadzone=-0.01).validate()
    with pytest.raises(ValueError):
        AppConfig(smoothing_max_speed=-1.0).validate()
    with pytest.raises(ValueError):
        AppConfig(smoothing_accel=0.0).validate()


def test_smoothing_params_derived_from_config() -> None:
    cfg = AppConfig(smoothing_alpha=0.4, smoothing_deadzone=0.2, hold_frames=7)
    params = cfg.smoothing_params()
    assert params.alpha == pytest.approx(0.4)
    assert params.deadzone == pytest.approx(0.2)
    assert params.hold_frames == 7


# ---------------------------------------------------------------- 多人分屏
def test_multi_person_defaults_to_on() -> None:
    """需求：多人分屏默认打开；关掉才回到"同时只显示一个人"。"""
    cfg = AppConfig()
    assert cfg.multi_person is True
    assert cfg.multi_person_max == 4
    assert cfg.multi_person_order == "spatial"
    assert cfg.multi_person_layout_file is None


def test_from_mapping_coerces_multi_person_fields() -> None:
    cfg = AppConfig.from_mapping(
        {
            "multi_person": "false",
            "multi_person_max": "3",
            "multi_person_order": "score",
            "multi_person_layout_file": "configs/multi_person_layout.yaml",
        }
    )
    assert cfg.multi_person is False
    assert cfg.multi_person_max == 3
    assert cfg.multi_person_order == "score"
    assert cfg.multi_person_layout_file == Path("configs/multi_person_layout.yaml")


def test_multi_person_policy_comes_from_config_and_file() -> None:
    cfg = AppConfig(multi_person_max=3, multi_person_order="score")
    policy = cfg.multi_person_policy()

    assert policy.enabled is True
    assert policy.max_persons == 3
    assert policy.order == "score"
    assert policy.layout_for(cfg.resolve_ratio(), 3).capacity == 3


def test_multi_person_policy_reads_explicit_layout_file(tmp_path: Path) -> None:
    layout = tmp_path / "mini.yaml"
    layout.write_text("layouts:\n  portrait:\n    2: ['1', '2']\n", encoding="utf-8")

    policy = AppConfig(multi_person_layout_file=layout).multi_person_policy()

    assert policy.source.endswith("mini.yaml")
    # 只覆盖了 portrait 的 2 人档，其余档位仍沿用内置默认
    assert policy.layout_for(parse_ratio("9:16"), 3).capacity == 3


def test_repo_default_yaml_enables_multi_person() -> None:
    """仓库自带配置也必须与内置默认一致（默认开启）。"""
    cfg = load_config(DEFAULT_CONFIG_FILE)

    assert cfg.multi_person is True
    assert cfg.multi_person_max == 4
    assert cfg.multi_person_policy().enabled is True


def test_validate_rejects_bad_multi_person(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        AppConfig(multi_person_max=1).validate()
    with pytest.raises(ValueError):
        AppConfig(multi_person_max=99).validate()
    with pytest.raises(ValueError):
        AppConfig(multi_person_order="random").validate()
    with pytest.raises(FileNotFoundError):
        AppConfig(multi_person_layout_file=tmp_path / "nope.yaml").validate()
