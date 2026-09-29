"""本机环境检查：PyTorch 装没装、YOLO 权重在不在（不联网、不起界面）。"""

from __future__ import annotations

from pathlib import Path

from video2personvideo.utils import env_check


def _weight(directory: Path, name: str, *, size: int = 10) -> Path:
    target = directory / name
    target.write_bytes(b"x" * size)
    return target


# ------------------------------------------------------------------ 找权重
def test_find_weight_in_given_directories(tmp_path: Path) -> None:
    target = _weight(tmp_path, "yolo11n.pt")
    assert env_check.find_weight("yolo11n.pt", directories=[tmp_path]) == target
    assert env_check.find_weight("yolo11n-pose.pt", directories=[tmp_path]) is None


def test_empty_file_does_not_count_as_weight(tmp_path: Path) -> None:
    """下到一半留下的空文件不能被当成"已经有了"。"""
    _weight(tmp_path, "yolo11n.pt", size=0)
    assert env_check.find_weight("yolo11n.pt", directories=[tmp_path]) is None


def test_search_dirs_cover_model_dir_and_cwd(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(env_check, "model_directory", lambda root=None: tmp_path / "models")
    dirs = env_check.weight_search_dirs()
    assert dirs[0] == tmp_path / "models"
    assert Path.cwd() in dirs


def test_installed_and_missing_names(tmp_path: Path) -> None:
    _weight(tmp_path, "yolo11n.pt")
    assert env_check.installed_weight_names(directories=[tmp_path]) == ["yolo11n.pt"]
    assert env_check.missing_weight_names(directories=[tmp_path]) == ["yolo11n-pose.pt"]


# ------------------------------------------------------------------ 就绪判断
def test_environment_ready_needs_torch_and_default_weight(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(env_check, "weight_search_dirs", lambda: [tmp_path])

    # 权重在、torch 没装 → 还不算就绪
    _weight(tmp_path, "yolo11n.pt")
    monkeypatch.setattr(env_check, "torch_installed", lambda: False)
    assert env_check.environment_ready() is False

    # 两个都齐了 → 就绪
    monkeypatch.setattr(env_check, "torch_installed", lambda: True)
    assert env_check.environment_ready() is True


def test_environment_ready_ignores_optional_pose_weight(monkeypatch, tmp_path: Path) -> None:
    """姿态模型是可选项，缺它不该弹下载页。"""
    monkeypatch.setattr(env_check, "weight_search_dirs", lambda: [tmp_path])
    monkeypatch.setattr(env_check, "torch_installed", lambda: True)
    _weight(tmp_path, "yolo11n.pt")

    assert env_check.find_weight("yolo11n-pose.pt") is None
    assert env_check.environment_ready() is True


def test_environment_ready_without_weight(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(env_check, "weight_search_dirs", lambda: [tmp_path])
    monkeypatch.setattr(env_check, "torch_installed", lambda: True)
    assert env_check.environment_ready() is False


# ------------------------------------------------------------------ 一句话描述
def test_environment_summary_none_when_blank(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(env_check, "weight_search_dirs", lambda: [tmp_path])
    monkeypatch.setattr(env_check, "torch_version", lambda: None)
    assert env_check.environment_summary() is None


def test_environment_summary_lists_both(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(env_check, "weight_search_dirs", lambda: [tmp_path])
    monkeypatch.setattr(env_check, "torch_version", lambda: "2.14.0+cpu")
    _weight(tmp_path, "yolo11n.pt")

    text = env_check.environment_summary()
    assert text is not None
    assert "2.14.0+cpu" in text
    assert "yolo11n.pt" in text


def test_torch_version_reads_metadata() -> None:
    """在装了 torch 的环境里能读出真实版本号（没装则返回 None）。"""
    version = env_check.torch_version()
    assert version is None or version[0].isdigit()
