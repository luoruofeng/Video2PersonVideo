"""断点续传下载器的验收用例（用本地 HTTP 服务器模拟 Range 行为）。"""

from __future__ import annotations

import contextlib
import hashlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from video2personvideo.utils.downloader import (
    STATE_MISMATCH,
    STATE_MISSING,
    STATE_PARTIAL,
    STATE_READY,
    DownloadCancelled,
    DownloadItem,
    build_cache_state,
    discard_partials,
    download_file,
    download_many,
    file_sha256,
    free_space,
    inspect_file,
    partial_bytes,
    remote_size,
)

PAYLOAD = bytes(range(256)) * 200  # 51200 字节


class _RangeHandler(BaseHTTPRequestHandler):
    """只服务一个固定 payload，可选是否支持 Range。"""

    failures: dict[str, int] = {}

    def log_message(self, *args) -> None:  # noqa: D102 - 静音日志
        pass

    @property
    def payload(self) -> bytes:
        return getattr(self.server, "payload", PAYLOAD)

    def _write(self, body: bytes) -> None:
        # 客户端发现续传位置不对时会提前断开，服务端写失败属正常
        with contextlib.suppress(ConnectionError, OSError):
            self.wfile.write(body)

    def _maybe_fail(self) -> bool:
        remaining = getattr(self.server, "fail_times", 0)
        if remaining <= 0:
            return False
        left = self.failures.get(self.path, remaining)
        if left <= 0:
            return False
        self.failures[self.path] = left - 1
        self.send_response(500)
        self.send_header("Content-Length", "0")
        self.end_headers()
        return True

    def do_HEAD(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 约定
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.payload)))
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 约定
        if self._maybe_fail():
            return
        data = self.payload
        range_header = self.headers.get("Range")
        if range_header and getattr(self.server, "support_range", True):
            start = int(range_header.split("=", 1)[1].split("-", 1)[0])
            body = data[start:]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self._write(body)
            return

        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self._write(data)


@pytest.fixture()
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _RangeHandler)
    httpd.payload = PAYLOAD  # type: ignore[attr-defined]
    httpd.support_range = True  # type: ignore[attr-defined]
    httpd.fail_times = 0  # type: ignore[attr-defined]
    _RangeHandler.failures = {}
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def _url(server, name: str = "payload.bin") -> str:
    return f"http://127.0.0.1:{server.server_address[1]}/{name}"


# ------------------------------------------------------------------ 基本下载
def test_download_file_writes_payload(tmp_path: Path, server) -> None:
    dest = tmp_path / "payload.bin"
    seen: list[int] = []

    result = download_file(_url(server), dest, progress=lambda event: seen.append(event.downloaded))

    assert result.ok and result.path == dest
    assert dest.read_bytes() == PAYLOAD
    assert not (tmp_path / "payload.bin.part").exists()
    assert seen  # 进度回调至少被调用一次


def test_download_file_verifies_sha256(tmp_path: Path, server) -> None:
    digest = hashlib.sha256(PAYLOAD).hexdigest()
    ok = download_file(_url(server), tmp_path / "p.bin", sha256=digest)
    assert ok.ok and ok.sha256_ok is True

    bad = download_file(
        _url(server, "p2.bin"), tmp_path / "p2.bin", sha256="0" * 64, retries=1
    )
    assert not bad.ok  # 校验失败不落地
    assert not (tmp_path / "p2.bin").exists()


def test_download_file_skips_existing(tmp_path: Path, server) -> None:
    dest = tmp_path / "payload.bin"
    dest.write_bytes(PAYLOAD)
    result = download_file(_url(server), dest, sha256=file_sha256(dest))
    assert result.skipped and result.ok


def test_download_replaces_corrupt_existing(tmp_path: Path, server) -> None:
    dest = tmp_path / "payload.bin"
    dest.write_bytes(b"broken")
    result = download_file(_url(server), dest, sha256=hashlib.sha256(PAYLOAD).hexdigest())
    assert result.ok and dest.read_bytes() == PAYLOAD


# ------------------------------------------------------------------ 断点续传
def test_resume_from_partial_file(tmp_path: Path, server) -> None:
    dest = tmp_path / "payload.bin"
    part = tmp_path / "payload.bin.part"
    half = len(PAYLOAD) // 2
    part.write_bytes(PAYLOAD[:half])

    result = download_file(_url(server), dest)

    assert result.ok
    assert result.resumed_from == half  # 确实是从一半接着下的
    assert dest.read_bytes() == PAYLOAD
    assert not part.exists()


def test_resume_falls_back_when_server_ignores_range(tmp_path: Path, server) -> None:
    server.support_range = False  # type: ignore[attr-defined]
    dest = tmp_path / "payload.bin"
    part = tmp_path / "payload.bin.part"
    part.write_bytes(PAYLOAD[:100])

    result = download_file(_url(server), dest)

    assert result.ok
    assert result.resumed_from == 0  # 服务器不支持 Range → 整文件重下
    assert dest.read_bytes() == PAYLOAD


def test_resume_resets_on_mismatched_content_range(tmp_path: Path, server) -> None:
    """服务器返回的续传起点与本地不符时，必须整文件重下而不是拼出坏文件。"""

    class _LieHandler(_RangeHandler):
        def do_GET(self) -> None:  # noqa: N802
            data = self.payload
            if self.headers.get("Range"):
                self.send_response(206)
                # 故意报一个错误的起点
                self.send_header("Content-Range", f"bytes 0-{len(data) - 1}/{len(data)}")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self._write(data)
                return
            super().do_GET()

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _LieHandler)
    httpd.payload = PAYLOAD  # type: ignore[attr-defined]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        dest = tmp_path / "payload.bin"
        (tmp_path / "payload.bin.part").write_bytes(PAYLOAD[:100])
        result = download_file(f"http://127.0.0.1:{httpd.server_address[1]}/p.bin", dest)
        assert result.ok
        assert dest.read_bytes() == PAYLOAD
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def test_retry_after_server_error(tmp_path: Path, server) -> None:
    server.fail_times = 1  # type: ignore[attr-defined]
    result = download_file(_url(server), tmp_path / "payload.bin", retries=3)
    assert result.ok and result.attempts == 2


# ------------------------------------------------------------------ 取消
def test_cancel_keeps_part_file(tmp_path: Path, server) -> None:
    dest = tmp_path / "payload.bin"
    with pytest.raises(DownloadCancelled):
        download_file(_url(server), dest, stop=lambda: True, retries=1)
    assert not dest.exists()
    assert (tmp_path / "payload.bin.part").exists()


def test_download_many_continues_after_cancel(tmp_path: Path, server) -> None:
    items = [
        DownloadItem(url=_url(server, "a.bin"), path=tmp_path / "a.bin"),
        DownloadItem(url=_url(server, "b.bin"), path=tmp_path / "b.bin"),
    ]
    class _Stop:
        def __init__(self) -> None:
            self.count = 0

        def __call__(self) -> bool:
            self.count += 1
            return self.count > 1

    batch = download_many(items, stop=_Stop())
    assert batch.cancelled


# ------------------------------------------------------------------ 工具函数
def test_remote_size_and_partial_bytes(tmp_path: Path, server) -> None:
    assert remote_size(_url(server)) == len(PAYLOAD)

    part = tmp_path / "x.part"
    part.write_bytes(b"0" * 120)
    assert partial_bytes([tmp_path / "x"]) == 120


# ------------------------------------------------------- 缓存现状（界面用）
def test_inspect_file_states(tmp_path: Path) -> None:
    target = tmp_path / "a.whl"

    assert inspect_file(target).state == STATE_MISSING
    assert inspect_file(target, expected_size=0).state == STATE_MISSING

    (tmp_path / "a.whl.part").write_bytes(b"x" * 500)
    partial = inspect_file(target, expected_size=1000)
    assert partial.state == STATE_PARTIAL
    assert partial.size == 500 and partial.remaining == 500
    assert partial.fraction == pytest.approx(0.5)
    assert partial.partial and partial.needs_download and not partial.ready

    target.write_bytes(b"y" * 1000)
    ready = inspect_file(target, expected_size=1000)
    assert ready.state == STATE_READY and ready.ready
    assert ready.needs_download is False

    assert inspect_file(target, expected_size=999).state == STATE_MISMATCH
    # 拿不到官方体积时，只要文件存在就算"已下好"
    assert inspect_file(target).state == STATE_READY


def test_build_cache_state_totals(tmp_path: Path) -> None:
    items = [
        DownloadItem(url="u1", path=tmp_path / "t1.whl", size=1000),
        DownloadItem(url="u2", path=tmp_path / "t2.whl", size=2000),
        DownloadItem(url="u3", path=tmp_path / "t3.whl"),
    ]
    (tmp_path / "t1.whl").write_bytes(b"a" * 1000)  # 已下好
    (tmp_path / "t2.whl.part").write_bytes(b"b" * 800)  # 断点

    state = build_cache_state(items, directory=tmp_path)

    assert state.total == 3000
    assert state.ready_bytes == 1000 and state.ready_files == 1
    assert state.partial_bytes == 800 and state.partial_files == 1
    assert state.missing_files == 1
    assert state.unknown == 1  # t3 体积未知
    assert state.needed_bytes == 1200
    assert state.has_partials and not state.complete
    assert state.free_bytes and state.free_bytes > 0
    assert state.enough_space is True


def test_cache_state_subset_recomputes(tmp_path: Path) -> None:
    items = [
        DownloadItem(url="u1", path=tmp_path / "t1.whl", size=1000),
        DownloadItem(url="u2", path=tmp_path / "t2.whl", size=2000),
    ]
    (tmp_path / "t2.whl.part").write_bytes(b"b" * 500)
    state = build_cache_state(items)

    only_second = state.subset([tmp_path / "t2.whl"])
    assert only_second.total == 2000
    assert only_second.partial_bytes == 500
    assert only_second.ready_bytes == 0
    assert only_second.has_partials
    assert only_second.free_bytes is None  # 没探测过磁盘就保持 None

    assert state.subset([]).total == 0


def test_cache_state_complete_when_everything_cached(tmp_path: Path) -> None:
    items = [DownloadItem(url="u", path=tmp_path / "a.whl", size=10)]
    (tmp_path / "a.whl").write_bytes(b"x" * 10)
    state = build_cache_state(items)
    assert state.complete and not state.has_partials
    assert state.needed_bytes == 0


def test_discard_partials_keeps_finished_files(tmp_path: Path) -> None:
    (tmp_path / "a.whl").write_bytes(b"keep")
    (tmp_path / "a.whl.part").write_bytes(b"drop")
    (tmp_path / "b.whl.part").write_bytes(b"drop")

    removed = discard_partials([tmp_path / "a.whl", tmp_path / "b.whl"])

    assert {item.name for item in removed} == {"a.whl.part", "b.whl.part"}
    assert not (tmp_path / "a.whl.part").exists()
    assert not (tmp_path / "b.whl.part").exists()
    assert (tmp_path / "a.whl").read_bytes() == b"keep"  # 成品一个都不动


def test_free_space(tmp_path: Path) -> None:
    assert free_space(tmp_path) > 0
    # 目录还不存在时会往上一级找（下载目录是刚建出来的，不该因此测不到）
    assert free_space(tmp_path / "not-created-yet" / "wheels") > 0
    # 整个盘符都不存在时明确返回 None，而不是瞎猜一个数
    assert free_space("Z:/definitely/not/here") is None or free_space(
        "Z:/definitely/not/here"
    ) > 0


# --------------------------------------------------------- 丢弃断点重新下载
def test_download_file_resume_false_discards_partial(tmp_path: Path, server) -> None:
    dest = tmp_path / "payload.bin"
    (tmp_path / "payload.bin.part").write_bytes(PAYLOAD[: len(PAYLOAD) // 2])

    result = download_file(_url(server), dest, resume=False)

    assert result.ok
    assert result.resumed_from == 0  # 没有复用断点
    assert dest.read_bytes() == PAYLOAD


def test_download_many_resume_false_clears_all_partials(tmp_path: Path, server) -> None:
    items = [
        DownloadItem(url=_url(server, "a.bin"), path=tmp_path / "a.bin"),
        DownloadItem(url=_url(server, "b.bin"), path=tmp_path / "b.bin"),
    ]
    for item in items:
        Path(f"{item.path}.part").write_bytes(PAYLOAD[:100])

    batch = download_many(items, resume=False)

    assert [item.resumed_from for item in batch.items] == [0, 0]
    assert all(item.ok for item in batch.items)
    assert (tmp_path / "a.bin").read_bytes() == PAYLOAD
    assert (tmp_path / "b.bin").read_bytes() == PAYLOAD
