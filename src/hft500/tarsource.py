"""Read Parquet files that live inside UNCOMPRESSED .tar archives, without extracting.

Why this works: a plain tar stores each member as one contiguous byte range, and
Parquet only needs random access (it reads its footer from the end of the file).
So a file-like "window" onto [offset, offset + size) of the tar is a valid
Parquet source. Virtual paths look like  "<tar path>::<member name>".
"""
from __future__ import annotations

import io
import re
import tarfile
from pathlib import Path

SEP = "::"
_REGISTRY: dict[str, tuple[str, int, int]] = {}   # virtual path -> (tar path, offset, size)
_MEMBER_RE = re.compile(r"(?:^|/)data/(?P<source>[^/]+)/(?:(?P<inst>[^/]+)/)?(?P<file>[^/]+\.parquet)$")


class TarSlice(io.RawIOBase):
    """Read-only, seekable view of bytes [offset, offset+size) of a file."""

    def __init__(self, path: str, offset: int, size: int):
        self._f = open(path, "rb")
        self._off, self._size, self._pos = offset, size, 0

    def readable(self) -> bool: return True
    def seekable(self) -> bool: return True
    def tell(self) -> int: return self._pos

    def seek(self, pos: int, whence: int = 0) -> int:
        base = {0: 0, 1: self._pos, 2: self._size}[whence]
        self._pos = max(0, min(self._size, base + pos))
        return self._pos

    def readinto(self, b) -> int:
        n = min(len(b), self._size - self._pos)
        if n <= 0:
            return 0
        self._f.seek(self._off + self._pos)
        data = self._f.read(n)
        b[: len(data)] = data
        self._pos += len(data)
        return len(data)

    def close(self) -> None:
        self._f.close()
        super().close()


def index_tar(tar_path: str | Path) -> list[dict]:
    """One dict per .parquet member: source, instrument, file name, virtual path, size."""
    rows = []
    with tarfile.open(tar_path, mode="r:") as tf:          # "r:" = refuse compressed tars
        for m in tf:
            if not m.isfile():
                continue
            mm = _MEMBER_RE.search(m.name)
            if not mm:
                continue
            vpath = f"{tar_path}{SEP}{m.name}"
            _REGISTRY[vpath] = (str(tar_path), m.offset_data, m.size)
            rows.append(dict(source=mm["source"], instrument=mm["inst"] or "(none)",
                             member=m.name, path=vpath, size_bytes=m.size))
    return rows


def open_any(path: str):
    """Return something pyarrow can read: a plain path, or a TarSlice for virtual paths."""
    if SEP in path:
        if path not in _REGISTRY:
            tar, member = path.split(SEP, 1)
            index_tar(tar)
        return io.BufferedReader(TarSlice(*_REGISTRY[path]), buffer_size=1 << 20)
    return path
