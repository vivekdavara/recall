"""Small archives shaped like GitHub's, for tests that must not use the network."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

COMMIT = "c" * 40
PAGE = "docs/en/docs/tutorial/a.md"
FILES = {
    PAGE: b"# A { #a }\n\n{* ../../docs_src/a/t1.py *}\n",
    "docs/en/docs/release-notes.md": b"# Release notes\n",
    "docs/en/docs/reference/fastapi.md": b"::: fastapi.FastAPI\n",
    "docs_src/a/t1.py": b"x = 1\n",
    "docs_src/a/unused.py": b"y = 2\n",
    "LICENSE": b"MIT\n",
}


def make_archive(
    path: Path,
    files: dict[str, bytes] = FILES,
    commit: str | None = COMMIT,
    top: str = "fastapi-x",
    extra: list[tarfile.TarInfo] | None = None,
) -> Path:
    headers = {"comment": commit} if commit is not None else {}
    with tarfile.open(path, "w:gz", format=tarfile.PAX_FORMAT, pax_headers=headers) as tar:
        root = tarfile.TarInfo(top)
        root.type = tarfile.DIRTYPE
        tar.addfile(root)
        for name, data in files.items():
            info = tarfile.TarInfo(f"{top}/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        for info in extra or []:
            tar.addfile(info, io.BytesIO(b"z" * info.size) if info.isreg() else None)
    return path
