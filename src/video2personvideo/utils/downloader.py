"""支持**断点续传**的下载器（只用标准库，无第三方依赖）。

PyTorch 的 CPU / CUDA wheel 单个就有 200 MB ~ 2.5 GB，断个网就重头下谁也受不了。
所以这里的核心是：

* 先写 ``<目标文件名>.part``，下完再原子改名 —— 中途失败不会留下"看起来能用"的残缺文件；
* 重新开始时读 ``.part`` 的大小，带 ``Range`` 头续传；服务器不支持 Range（返回 200）
  时自动退回整文件重下；
* 校验 ``Content-Range`` 的起始位置，服务器换源 / 代理返回错内容也能识别；
* 可选按官方索引给出的 ``sha256`` 校验，校验失败自动重下一次；
* 网络抖动可重试，取消（停止）时保留 ``.part``，下次运行接着下。

界面要回答"现在能不能接着下、还要下多少、磁盘够不够"，所以这里还提供了一组
**只 stat 不读文件**的探测函数（:func:`inspect_file` / :func:`build_cache_state` /
:func:`free_space`），以及"丢弃断点重来"（``resume=False`` / :func:`discard_partials`）。
"""

from __future__ import annotations

import hashlib
import http.client
import os
import shutil
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .logger import get_logger

logger = get_logger(__name__)

#: 单次读取的块大小（1 MiB）
CHUNK_SIZE = 1024 * 1024
#: 进度回调的最小间隔（秒），避免刷爆界面
PROGRESS_INTERVAL = 0.15
#: 默认超时（连接 + 读取）
DEFAULT_TIMEOUT = 30.0
#: 默认重试次数
DEFAULT_RETRIES = 3
#: 未指定 UA 时部分 CDN 会拒绝请求
USER_AGENT = "Video2PersonVideo/1.0 (resumable-downloader)"


class DownloadCancelled(Exception):
    """用户主动取消下载（``.part`` 会保留，便于下次续传）。"""


class DownloadError(RuntimeError):
    """下载在重试后仍然失败。"""


@dataclass(slots=True)
class DownloadProgress:
    """一次进度回调携带的全部信息。"""

    path: Path
    url: str
    downloaded: int
    total: int | None
    resumed_from: int
    speed_bps: float = 0.0
    #: 第几个文件 / 共几个（批量下载时用）
    index: int = 1
    count: int = 1

    @property
    def fraction(self) -> float:
        if not self.total:
            return 0.0
        return min(max(self.downloaded / self.total, 0.0), 1.0)

    @property
    def eta(self) -> float:
        """预计剩余秒数（未知返回 0）。"""
        if not self.total or self.speed_bps <= 0:
            return 0.0
        return max(self.total - self.downloaded, 0) / self.speed_bps


@dataclass(slots=True)
class DownloadResult:
    """单个文件的结果。"""

    path: Path
    url: str
    bytes_total: int | None = None
    resumed_from: int = 0
    #: 目标文件已存在且校验通过，本次完全跳过
    skipped: bool = False
    sha256_ok: bool | None = None
    attempts: int = 1
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(slots=True)
class DownloadItem:
    """一个待下载的文件。"""

    url: str
    path: Path
    sha256: str | None = None
    label: str = ""
    size: int | None = None

    @property
    def display(self) -> str:
        return self.label or self.path.name


@dataclass(slots=True)
class DownloadBatchResult:
    """批量下载的结果汇总。"""

    items: list[DownloadResult] = field(default_factory=list)
    cancelled: bool = False

    @property
    def succeeded(self) -> list[DownloadResult]:
        return [item for item in self.items if item.ok and not item.error]

    @property
    def failed(self) -> list[DownloadResult]:
        return [item for item in self.items if item.error]


ProgressCallback = Callable[[DownloadProgress], None]
StopCallback = Callable[[], bool]

# --------------------------------------------------------------------- 缓存状态
#: 已下好（大小与官方索引一致，下载时仍会做一次 sha256 校验）
STATE_READY = "ready"
#: 有断点文件，可以接着下
STATE_PARTIAL = "partial"
#: 目标文件存在但大小与官方索引不符，会被重新下载
STATE_MISMATCH = "mismatch"
#: 还没开始下载
STATE_MISSING = "missing"


@dataclass(slots=True)
class FileState:
    """单个文件在磁盘上的现状（只看大小，不做 sha256，秒回）。"""

    path: Path
    state: str = STATE_MISSING
    #: 磁盘上已有多少字节（完整文件则是文件本身的大小）
    size: int = 0
    #: 官方索引给出的体积（拿不到时为 ``None``）
    expected: int | None = None

    @property
    def ready(self) -> bool:
        return self.state == STATE_READY

    @property
    def partial(self) -> bool:
        return self.state == STATE_PARTIAL

    @property
    def needs_download(self) -> bool:
        return self.state != STATE_READY

    @property
    def fraction(self) -> float:
        if not self.expected:
            return 0.0
        return min(max(self.size / self.expected, 0.0), 1.0)

    @property
    def remaining(self) -> int:
        if not self.expected:
            return 0
        return max(self.expected - self.size, 0)


@dataclass(slots=True)
class CacheState:
    """一批文件的缓存现状汇总（界面用它回答"还要下多少、磁盘够不够"）。"""

    files: list[FileState] = field(default_factory=list)
    #: 全部文件的总体积（体积已知的部分）
    total: int = 0
    #: 已下好的体积
    ready_bytes: int = 0
    #: 断点文件里已经下到的体积（可复用）
    partial_bytes: int = 0
    #: 体积未知的文件个数（官方索引没给大小）
    unknown: int = 0
    #: 目标磁盘剩余空间（探测失败时为 ``None``）
    free_bytes: int | None = None

    @property
    def ready_files(self) -> int:
        return sum(1 for item in self.files if item.ready)

    @property
    def partial_files(self) -> int:
        return sum(1 for item in self.files if item.partial)

    @property
    def missing_files(self) -> int:
        return sum(1 for item in self.files if item.state == STATE_MISSING)

    @property
    def has_partials(self) -> bool:
        """有没有可以续传的断点。"""
        return any(item.partial for item in self.files)

    @property
    def complete(self) -> bool:
        """所有文件都已经下好（本次只需校验，不用下载）。"""
        return bool(self.files) and all(item.ready for item in self.files)

    @property
    def needed_bytes(self) -> int:
        """还需要从网上下多少（已下好的与断点里已有的都算省下来了）。"""
        return max(self.total - self.ready_bytes - self.partial_bytes, 0)

    @property
    def enough_space(self) -> bool | None:
        """磁盘剩余空间够不够；探测不到剩余空间时返回 ``None``。"""
        if self.free_bytes is None:
            return None
        return self.free_bytes >= self.needed_bytes

    @property
    def unknown_sizes(self) -> bool:
        return self.unknown > 0

    def subset(self, paths: Iterable[str | os.PathLike[str]]) -> CacheState:
        """取出其中一部分文件重新汇总（界面只显示勾选的那些时用）。

        ``total`` 按被选中的文件重算；``free_bytes`` 原样保留。
        """
        wanted = {Path(item) for item in paths}
        chosen = [item for item in self.files if item.path in wanted]
        state = CacheState(files=chosen, free_bytes=self.free_bytes)
        for item in chosen:
            if item.expected:
                state.total += item.expected
            elif not item.ready:
                state.unknown += 1
            if item.ready:
                state.ready_bytes += item.size or (item.expected or 0)
            elif item.partial:
                state.partial_bytes += item.size
        return state


def inspect_file(path: str | os.PathLike[str], *, expected_size: int | None = None) -> FileState:
    """看一眼某个文件当前是什么状态（只 stat，不读内容）。"""
    target = Path(path)
    if target.exists():
        size = target.stat().st_size
        if size > 0:
            if expected_size is None or size == expected_size:
                return FileState(path=target, state=STATE_READY, size=size, expected=expected_size)
            return FileState(path=target, state=STATE_MISMATCH, size=size, expected=expected_size)

    part = partial_path(target)
    if part.exists():
        done = part.stat().st_size
        if done > 0:
            return FileState(path=target, state=STATE_PARTIAL, size=done, expected=expected_size)
    return FileState(path=target, state=STATE_MISSING, expected=expected_size)


def build_cache_state(
    items: Sequence[DownloadItem],
    *,
    directory: str | os.PathLike[str] | None = None,
) -> CacheState:
    """汇总一批待下载文件的缓存现状（给界面显示"已缓存 / 断点 / 待下载"）。"""
    state = CacheState()
    for item in items:
        found = inspect_file(item.path, expected_size=item.size)
        state.files.append(found)
        if item.size:
            state.total += item.size
        elif not found.ready:
            state.unknown += 1
        if found.ready:
            state.ready_bytes += found.size or (item.size or 0)
        elif found.partial:
            state.partial_bytes += found.size
    if directory is not None:
        state.free_bytes = free_space(directory)
    return state


def free_space(directory: str | os.PathLike[str]) -> int | None:
    """目标磁盘的剩余字节数（目录不存在时往上一级找，探测失败返回 ``None``）。"""
    target = Path(directory)
    for candidate in (target, *target.parents):
        try:
            if candidate.exists():
                return shutil.disk_usage(candidate).free
        except OSError as exc:  # pragma: no cover - 依赖真实文件系统
            logger.debug("探测磁盘剩余空间失败（%s）：%s", candidate, exc)
    return None


def discard_partials(paths: Iterable[str | os.PathLike[str]]) -> list[Path]:
    """删掉这些文件的断点（``.part``），返回真正删掉的列表。

    只动断点文件，**不碰**已经下好并校验过的成品。
    """
    removed: list[Path] = []
    for item in paths:
        part = partial_path(item)
        try:
            if part.exists():
                part.unlink()
                removed.append(part)
        except OSError as exc:  # pragma: no cover - 依赖真实文件系统
            logger.warning("删除断点失败：%s（%s）", part, exc)
    return removed


def _request(url: str, headers: dict[str, str] | None = None, method: str = "GET"):
    merged = {"User-Agent": USER_AGENT}
    if headers:
        merged.update(headers)
    return urllib.request.Request(url, headers=merged, method=method)


def remote_size(url: str, timeout: float = DEFAULT_TIMEOUT) -> int | None:
    """探测远端文件大小；不支持 HEAD 时退回 ``Range: bytes=0-0``。"""
    try:
        with urllib.request.urlopen(  # noqa: S310 - 地址来自官方索引
            _request(url, method="HEAD"), timeout=timeout
        ) as response:
            length = response.headers.get("Content-Length")
            if length and length.isdigit():
                return int(length)
    except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
        logger.debug("HEAD %s 失败：%s", url, exc)

    try:
        with urllib.request.urlopen(  # noqa: S310
            _request(url, {"Range": "bytes=0-0"}), timeout=timeout
        ) as response:
            content_range = response.headers.get("Content-Range", "")
            if "/" in content_range:
                tail = content_range.rsplit("/", 1)[-1].strip()
                if tail.isdigit():
                    return int(tail)
    except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
        logger.debug("Range 探测 %s 失败：%s", url, exc)
    return None


def file_sha256(path: Path, chunk_size: int = CHUNK_SIZE) -> str:
    """流式计算文件 sha256（大文件也不吃内存）。"""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _verify_sha256(path: Path, expected: str | None) -> bool | None:
    if not expected:
        return None
    return file_sha256(path).lower() == expected.lower()


def download_file(
    url: str,
    dest: str | os.PathLike[str],
    *,
    sha256: str | None = None,
    progress: ProgressCallback | None = None,
    stop: StopCallback | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    chunk_size: int = CHUNK_SIZE,
    index: int = 1,
    count: int = 1,
    resume: bool = True,
) -> DownloadResult:
    """下载 ``url`` 到 ``dest``，支持断点续传与 sha256 校验。

    * ``progress`` 会被高频调用（内部已限流），抛异常不影响下载；
    * ``stop`` 返回 ``True`` 时抛 :class:`DownloadCancelled`，``.part`` 保留；
    * 已存在且校验通过的目标文件会被直接跳过；
    * ``resume=False`` 表示"丢弃断点，从头下"（用户点了「重新下载」）。
    """
    target = Path(dest)
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(f"{target.name}.part")

    result = DownloadResult(path=target, url=url)

    if not resume and part.exists():
        logger.info("丢弃断点，重新下载：%s", part.name)
        part.unlink(missing_ok=True)

    if target.exists() and target.stat().st_size > 0:
        if target.stat().st_size >= 64 * CHUNK_SIZE:
            # 大文件校验要读全量，卡几秒很正常，先说清楚免得看起来像卡死
            logger.info("校验已下载的 %s（sha256，可能要几秒）…", target.name)
        ok = _verify_sha256(target, sha256)
        if ok is not False:
            result.skipped = True
            result.sha256_ok = ok
            result.bytes_total = target.stat().st_size
            _emit(
                progress,
                DownloadProgress(
                    path=target,
                    url=url,
                    downloaded=result.bytes_total,
                    total=result.bytes_total,
                    resumed_from=0,
                    index=index,
                    count=count,
                ),
            )
            logger.info("已存在，跳过下载：%s", target.name)
            return result
        logger.warning("%s 校验失败，重新下载", target.name)
        target.unlink(missing_ok=True)

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        result.attempts = attempt
        try:
            resumed_from = _download_once(
                url,
                part,
                sha256=sha256,
                progress=progress,
                stop=stop,
                timeout=timeout,
                chunk_size=chunk_size,
                index=index,
                count=count,
            )
        except DownloadCancelled:
            raise
        except (urllib.error.URLError, OSError, http.client.HTTPException, DownloadError) as exc:
            last_error = exc
            logger.warning("下载失败（第 %d/%d 次）：%s → %s", attempt, retries, url, exc)
            if stop is not None and stop():
                raise DownloadCancelled(url) from exc
            continue

        result.resumed_from = resumed_from
        result.bytes_total = part.stat().st_size
        os.replace(part, target)
        result.sha256_ok = _verify_sha256(target, sha256)
        if result.sha256_ok is False:
            logger.warning("校验和不匹配，删除后重下：%s", target.name)
            target.unlink(missing_ok=True)
            continue
        return result

    result.error = str(last_error) if last_error else "下载失败"
    return result


def _download_once(
    url: str,
    part: Path,
    *,
    sha256: str | None,
    progress: ProgressCallback | None,
    stop: StopCallback | None,
    timeout: float,
    chunk_size: int,
    index: int,
    count: int,
) -> int:
    """下载一轮（一次连接），返回"从第几字节开始"（即续传起点）。"""
    existing = part.stat().st_size if part.exists() else 0
    headers: dict[str, str] = {}
    if existing:
        headers["Range"] = f"bytes={existing}-"

    resumed_from = existing
    with urllib.request.urlopen(  # noqa: S310 - 地址来自官方索引
        _request(url, headers), timeout=timeout
    ) as response:
        status = getattr(response, "status", None) or response.getcode()
        content_range = response.headers.get("Content-Range", "")
        length = response.headers.get("Content-Length")

        if status == 206:
            start = _range_start(content_range)
            if start is None or start != existing:
                logger.warning(
                    "服务器返回的续传起点（%s）与本地不符，重置断点后重试", content_range
                )
                part.unlink(missing_ok=True)
                raise DownloadError("续传位置不匹配")
        elif status == 200:
            if existing:
                logger.info("服务器不支持断点续传，从头下载：%s", part.name)
                # 以 "wb" 打开会截断，这里只记录，不再往旧内容后面追加
                existing = 0
                resumed_from = 0
        else:
            raise DownloadError(f"HTTP {status}")

        total = existing + int(length) if length and str(length).isdigit() else None

        mode = "ab" if existing else "wb"
        downloaded = existing
        started = time.monotonic()
        last_emit = 0.0
        with open(part, mode) as handle:
            while True:
                if stop is not None and stop():
                    raise DownloadCancelled(url)
                block = response.read(chunk_size)
                if not block:
                    break
                handle.write(block)
                downloaded += len(block)

                now = time.monotonic()
                if progress is not None and now - last_emit >= PROGRESS_INTERVAL:
                    last_emit = now
                    elapsed = max(now - started, 1e-6)
                    speed = (downloaded - existing) / elapsed
                    _emit(
                        progress,
                        DownloadProgress(
                            path=part,
                            url=url,
                            downloaded=downloaded,
                            total=total,
                            resumed_from=resumed_from,
                            speed_bps=speed,
                            index=index,
                            count=count,
                        ),
                    )
            handle.flush()

    if progress is not None:
        elapsed_total = max(time.monotonic() - started, 1e-6)
        _emit(
            progress,
            DownloadProgress(
                path=part,
                url=url,
                downloaded=part.stat().st_size,
                total=total or part.stat().st_size,
                resumed_from=resumed_from,
                speed_bps=max(downloaded - existing, 0) / elapsed_total,
                index=index,
                count=count,
            ),
        )
    return resumed_from


def _range_start(content_range: str) -> int | None:
    # 形如 "bytes 1024-2048/4096"
    if not content_range:
        return None
    body = content_range.split(" ", 1)[-1]
    start = body.split("-", 1)[0].strip()
    return int(start) if start.isdigit() else None


def _emit(progress: ProgressCallback | None, event: DownloadProgress) -> None:
    if progress is None:
        return
    try:
        progress(event)
    except Exception as exc:  # noqa: BLE001 - 进度回调出错不该影响下载
        logger.debug("进度回调异常：%s", exc)


def download_many(
    items: Sequence[DownloadItem],
    *,
    progress: ProgressCallback | None = None,
    stop: StopCallback | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    resume: bool = True,
) -> DownloadBatchResult:
    """按顺序下载多个文件；单个失败不中断后面的。

    ``resume=False`` 时先丢弃全部断点，再从零开始下载。
    """
    batch = DownloadBatchResult()
    total = len(items)
    if not resume:
        discard_partials([item.path for item in items])
    for position, item in enumerate(items, start=1):
        try:
            batch.items.append(
                download_file(
                    item.url,
                    item.path,
                    sha256=item.sha256,
                    progress=progress,
                    stop=stop,
                    timeout=timeout,
                    retries=retries,
                    index=position,
                    count=total,
                    resume=resume,
                )
            )
        except DownloadCancelled:
            batch.cancelled = True
            logger.info("下载已取消，已保留断点（.part）")
            return batch
        except Exception as exc:  # noqa: BLE001 - 单文件失败继续下一个
            batch.items.append(
                DownloadResult(path=item.path, url=item.url, error=str(exc))
            )
    return batch


def partial_path(path: str | os.PathLike[str]) -> Path:
    """返回某个文件的断点文件路径（界面展示"已下载多少"用）。"""
    target = Path(path)
    return target.with_name(f"{target.name}.part")


def partial_bytes(paths: Iterable[str | os.PathLike[str]]) -> int:
    """累计断点文件已下载的字节数。"""
    total = 0
    for item in paths:
        part = partial_path(item)
        if part.exists():
            total += part.stat().st_size
    return total


__all__ = [
    "CHUNK_SIZE",
    "DEFAULT_RETRIES",
    "DEFAULT_TIMEOUT",
    "STATE_MISMATCH",
    "STATE_MISSING",
    "STATE_PARTIAL",
    "STATE_READY",
    "CacheState",
    "DownloadBatchResult",
    "DownloadCancelled",
    "DownloadError",
    "DownloadItem",
    "DownloadProgress",
    "DownloadResult",
    "FileState",
    "build_cache_state",
    "discard_partials",
    "download_file",
    "download_many",
    "file_sha256",
    "free_space",
    "inspect_file",
    "partial_bytes",
    "partial_path",
    "remote_size",
]
