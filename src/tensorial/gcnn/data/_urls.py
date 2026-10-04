"""Helpers for resolving URL path specs to local paths, downloading as needed.

A path spec here is a ``str`` (a single path or URL), a ``Sequence[str]``, or a
``dict[str, ...]`` mapping names to such specs.  The resolver is *shape-preserving*
-- it returns the same container type it was given, and only replaces URL leaves
with their cached local counterparts.  Local paths (and ``None`` leaves) pass
through untouched.
"""

from collections.abc import Sequence
import hashlib
import os
from pathlib import Path
import tempfile
from urllib.parse import urlparse

import requests
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

__all__ = ("is_url", "resolve_spec", "resolve_url")


def is_url(path: str) -> bool:
    """Return True if ``path`` looks like a URL rather than a local path."""
    scheme = urlparse(path).scheme
    return scheme in ("http", "https", "ftp")


def resolve_url(url: str, cache_dir: str | os.PathLike) -> str:
    """Return a local path for ``url``, downloading it into ``cache_dir`` if needed.

    The cache key is a hash of the URL, so re-fetching the same URL reuses the
    cached file.  The download is atomic: bytes are streamed to a temporary file in
    the same directory and renamed into place, so a partial download never shadows
    a complete one and concurrent fetches are safe.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    key = hashlib.sha256(url.encode()).hexdigest()[:16]
    suffix = Path(urlparse(url).path).suffix
    cached = cache_dir / f"{key}{suffix}"

    if cached.exists():
        return str(cached)

    _download(url, cached)
    return str(cached)


def resolve_spec(spec, cache_dir: str | os.PathLike):
    """Resolve a path spec, downloading any URLs it contains.

    Shape-preserving: a ``str`` goes in, a ``str`` comes out; a ``Sequence`` goes
    in, a ``list`` comes out (each element resolved); a ``dict`` goes in, a ``dict``
    comes out.  Only URL leaves change; local paths and ``None`` pass through.
    """
    if spec is None or isinstance(spec, Path):
        return spec
    if isinstance(spec, dict):
        return {name: resolve_spec(s, cache_dir) for name, s in spec.items()}
    if isinstance(spec, str):
        return resolve_url(spec, cache_dir) if is_url(spec) else spec
    if isinstance(spec, Sequence):
        return [resolve_spec(s, cache_dir) for s in spec]
    return spec


def _download(url: str, target: Path) -> None:
    """Stream ``url`` to ``target`` atomically, with a progress bar."""
    try:
        with requests.get(url, stream=True, timeout=30) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))

            # Write to a temp file in the target directory so the rename is atomic.
            with tempfile.NamedTemporaryFile(
                dir=target.parent, delete=False, suffix=target.suffix
            ) as tmp:
                try:
                    with Progress(
                        TextColumn("[progress.description]{task.description}"),
                        DownloadColumn(),
                        BarColumn(),
                        "[progress.percentage]{task.percentage:>3.0f}%",
                        TransferSpeedColumn(),
                        TimeRemainingColumn(),
                        refresh_per_second=10,
                    ) as progress:
                        task = progress.add_task(f"Downloading {target.name}", total=total or None)
                        for chunk in response.iter_content(chunk_size=1 << 16):
                            tmp.write(chunk)
                            progress.update(task, advance=len(chunk))
                    tmp.flush()
                    os.fsync(tmp.fileno())
                except BaseException:
                    os.unlink(tmp.name)
                    raise

            os.replace(tmp.name, target)
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Could not download {url}: {e}") from e
