"""下载计划的解析与挑选（索引页用内联 HTML 模拟，不联网）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from video2personvideo.utils.gpu_probe import BACKEND_CUDA, VENDOR_NVIDIA, GpuInfo, HardwareProfile
from video2personvideo.utils.torch_backends import TorchBackend, backend_by_key
from video2personvideo.utils.torch_install import (
    InstallPlan,
    build_plan,
    filter_entries,
    parse_index,
    pick_version,
    pinned_versions,
    platform_tags,
    python_tag,
    wheel_directory,
    yolo_items,
)

INDEX_HTML = """
<!DOCTYPE html>
<html><body>
<h1>Links for torch</h1>
<a href="https://example.com/torch-2.9.1%2Bcu126-cp313-cp313-win_amd64.whl#sha256=aaa111">torch-2.9.1+cu126-cp313-cp313-win_amd64.whl</a><br/>
<a href="https://example.com/torch-2.9.1%2Bcu126-cp312-cp312-win_amd64.whl">torch-2.9.1+cu126-cp312-cp312-win_amd64.whl</a><br/>
<a href="https://example.com/torch-2.9.1%2Bcu126-cp313-cp313-linux_x86_64.whl">torch-2.9.1+cu126-cp313-cp313-linux_x86_64.whl</a><br/>
<a href="https://example.com/torch-2.10.0%2Bcu126-cp313-cp313-win_amd64.whl#sha256=bbb222">torch-2.10.0+cu126-cp313-cp313-win_amd64.whl</a><br/>
<a href="https://example.com/torch-2.11.0%2Bcu126-cp313t-cp313t-win_amd64.whl">torch-2.11.0+cu126-cp313t-cp313t-win_amd64.whl</a><br/>
</body></html>
"""


def _profile(**kwargs) -> HardwareProfile:
    base: dict = {"os_name": "Windows", "arch": "AMD64", "is_windows": True}
    base.update(kwargs)
    return HardwareProfile(**base)


def _nvidia_profile() -> HardwareProfile:
    return _profile(
        gpus=[
            GpuInfo(
                name="NVIDIA GeForce RTX 4070",
                vendor=VENDOR_NVIDIA,
                compute=(8, 9),
                backend=BACKEND_CUDA,
            )
        ],
        cuda_driver_version="12.6",
    )


# ------------------------------------------------------------------ 索引解析
def test_parse_index_reads_urls_and_hashes() -> None:
    entries = parse_index(INDEX_HTML)
    assert len(entries) == 5
    first = entries[0]
    assert first.filename == "torch-2.9.1+cu126-cp313-cp313-win_amd64.whl"
    assert first.sha256 == "aaa111"
    assert first.version == "2.9.1+cu126"
    assert first.base_version == "2.9.1"
    assert entries[1].sha256 is None  # 没有 hash 片段时不该乱猜


def test_parse_index_ignores_non_wheel_links() -> None:
    html = '<a href="https://x/y.tar.gz">y.tar.gz</a><a href="https://x/z.whl">z.whl</a>'
    assert [item.filename for item in parse_index(html)] == ["z.whl"]


# ------------------------------------------------------------------ 挑选 wheel
def test_python_and_platform_tags() -> None:
    assert python_tag((3, 11)) == "cp311"
    assert "win_amd64" in platform_tags(_profile())
    assert "manylinux_2_28_x86_64" in platform_tags(_profile(os_name="Linux", is_windows=False))
    arm64_mac = platform_tags(_profile(os_name="Darwin", arch="arm64", is_windows=False))
    assert "macosx_11_0_arm64" in arm64_mac
    assert all("x86_64" not in tag for tag in arm64_mac)


def test_filter_entries_respects_python_tag_and_platform() -> None:
    entries = parse_index(INDEX_HTML)
    picked = filter_entries(entries, tag="cp313", tags=("win_amd64",))
    names = [item.filename for item in picked]
    # cp312 与 linux 的被滤掉，自由线程的 cp313t 也必须滤掉
    assert all("cp312" not in name for name in names)
    assert all("linux" not in name for name in names)
    assert all("cp313t" not in name for name in names)
    assert len(names) == 2


def test_pick_version_prefers_pin_then_latest() -> None:
    entries = filter_entries(parse_index(INDEX_HTML), tag="cp313", tags=("win_amd64",))
    assert pick_version(entries, "2.9.1") is not None
    assert pick_version(entries, "2.9.1").base_version == "2.9.1"
    assert pick_version(entries).base_version == "2.10.0"  # 没锁定就取最新
    assert pick_version([], "2.9.1") is None


def test_pick_torchvision_matches_torch_minor() -> None:
    from video2personvideo.utils.torch_install import _pick_torchvision

    def entry(version: str):
        from video2personvideo.utils.torch_install import WheelEntry

        return WheelEntry(filename=f"torchvision-{version}-cp313-cp313-win_amd64.whl", url="u")

    entries = [entry("0.26.0"), entry("0.28.0"), entry("0.30.0")]
    # torch 2.11 → 期望 torchvision 0.26
    assert _pick_torchvision(entries, "2.11.0", None).base_version == "0.26.0"
    # torch 2.13 → 期望 0.28（少一档时退到最近的旧版本）
    assert _pick_torchvision(entries, "2.13.0", None).base_version == "0.28.0"
    # 有锁定值时以锁定为准
    assert _pick_torchvision(entries, "2.11.0", "0.30.0").base_version == "0.30.0"


# ------------------------------------------------------------------ 版本锁定
def test_pinned_versions_reads_requirements(tmp_path: Path) -> None:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text(
        "# 注释\n"
        "torch==2.14.0  # 框架\n"
        "torchvision==0.29.0\n"
        "numpy==2.5.3\n",
        encoding="utf-8",
    )
    pins = pinned_versions(requirements)
    assert pins == {"torch": "2.14.0", "torchvision": "0.29.0"}


def test_pinned_versions_missing_file() -> None:
    assert pinned_versions(Path("does-not-exist.txt")) == {}


def test_repo_requirements_is_parsable() -> None:
    pins = pinned_versions()
    assert pins.get("torch")
    assert pins.get("torchvision")


# ------------------------------------------------------------------ 生成计划
def test_build_plan_uses_index_and_pins(monkeypatch, tmp_path: Path) -> None:
    def fake_fetch(index_url: str, package: str, *, timeout: float = 30.0):
        html = (
            f'<a href="https://x/{package}-2.14.0-cp313-cp313-win_amd64.whl#sha256=ab">'
            f"{package}-2.14.0-cp313-cp313-win_amd64.whl</a>"
        )
        return parse_index(html), None

    monkeypatch.setattr("video2personvideo.utils.torch_install.fetch_index", fake_fetch)
    monkeypatch.setattr(
        "video2personvideo.utils.torch_install.remote_size", lambda url, timeout=30.0: 1024
    )

    plan = build_plan(
        backend_by_key("cu126"),
        profile=_nvidia_profile(),
        versions={"torch": "2.14.0", "torchvision": "0.29.0"},
        directory=tmp_path,
    )
    assert plan.ok
    assert [item.path.name.split("-")[0] for item in plan.items] == ["torch", "torchvision"]
    assert plan.total_bytes == 2048
    assert plan.items[0].sha256 == "ab"
    assert any("cu126" in note for note in plan.notes)
    assert "pip install" in plan.pip_command()


def test_build_plan_reports_problems_when_no_wheel(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "video2personvideo.utils.torch_install.fetch_index",
        lambda index_url, package, *, timeout=30.0: ([], None),
    )
    plan = build_plan(backend_by_key("cu126"), profile=_nvidia_profile(), directory=tmp_path)
    assert not plan.ok
    assert plan.problems
    assert not plan.items


def test_build_plan_reports_network_failure(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "video2personvideo.utils.torch_install.fetch_index",
        lambda index_url, package, *, timeout=30.0: ([], "无法访问索引"),
    )
    plan = build_plan(backend_by_key("cpu"), profile=_profile(), directory=tmp_path)
    assert not plan.ok
    assert any("无法访问" in problem for problem in plan.problems)


def test_install_plan_requires_files(tmp_path: Path) -> None:
    from video2personvideo.utils.torch_install import install_plan

    plan = InstallPlan(backend=backend_by_key("cpu"))
    ok, message = install_plan(plan)
    assert ok is False
    assert "没有可安装" in message


@pytest.mark.slow
def test_real_index_plan_is_still_valid(tmp_path: Path) -> None:
    """真连 PyTorch 官方索引：确认 URL 规则、平台标签与 sha256 都还对得上。

    需要联网，默认跳过（``RUN_SLOW=1 pytest -m slow``）。
    """
    plan = build_plan(
        backend_by_key("cpu"),
        profile=_profile(),
        with_sizes=False,
        directory=tmp_path,
    )
    assert plan.ok, plan.problems
    names = [item.path.name for item in plan.items]
    assert any(name.startswith("torch-") for name in names)
    assert any(name.startswith("torchvision-") for name in names)
    # 官方索引会混用 download.pytorch.org 与 download-r2.pytorch.org 两个 CDN
    assert all("pytorch.org/" in item.url for item in plan.items)
    assert all(item.sha256 for item in plan.items)


# ------------------------------------------------------------------ 其它
def test_wheel_directory_is_created(tmp_path: Path) -> None:
    target = wheel_directory(tmp_path / "wheels")
    assert target.is_dir()


def test_yolo_items_are_optional_and_named() -> None:
    items = yolo_items()
    names = {item.path.name for item in items}
    assert "yolo11n.pt" in names
    assert all(item.url.startswith("https://") for item in items)


def test_torch_backend_pip_command_and_urls() -> None:
    backend: TorchBackend = backend_by_key("cu126")
    assert backend.index_url == "https://download.pytorch.org/whl/cu126"
    assert backend.package_index_url.endswith("/cu126/torch/")
    assert "--index-url" in backend.pip_command


@pytest.mark.parametrize(
    "key,cuda,min_compute,max_compute",
    [
        ("cu118", "11.8", (5, 0), (9, 0)),
        ("cu126", "12.6", (5, 0), (9, 0)),
        ("cu128", "12.8", (7, 0), (12, 0)),
        ("cpu", None, None, None),
    ],
)
def test_backend_table_windows_are_sane(key, cuda, min_compute, max_compute) -> None:
    backend = backend_by_key(key)
    assert backend.cuda_version == cuda
    assert backend.min_compute == min_compute
    assert backend.max_compute == max_compute
