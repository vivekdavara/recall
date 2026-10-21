"""Download the pinned archive, check it, and extract the files the manifest lists. DESIGN.md "The corpus".

The archive's own bytes aren't pinned (GitHub generates archives on request and has changed their compression
before); what is checked is the commit id the archive declares and the size and SHA-256 of every file used.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from importlib import resources
from pathlib import Path, PurePosixPath

from recall.corpus import normalize, source

MAX_ARCHIVE_BYTES = 100 * 1024 * 1024  # the pinned archive is about 18 MB


class CorpusError(Exception):
    pass


@dataclass(frozen=True)
class ManifestFile:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class Manifest:
    repository: str
    tag: str
    commit: str
    files: tuple[ManifestFile, ...]

    def to_json(self) -> str:
        body = {
            "repository": self.repository,
            "tag": self.tag,
            "commit": self.commit,
            "files": [{"path": f.path, "size": f.size, "sha256": f.sha256} for f in self.files],
        }
        return json.dumps(body, indent=1) + "\n"

    @staticmethod
    def from_json(text: str) -> Manifest:
        body = json.loads(text)
        files = tuple(ManifestFile(f["path"], int(f["size"]), f["sha256"]) for f in body["files"])
        manifest = Manifest(body["repository"], body["tag"], body["commit"], files)
        manifest.check()
        return manifest

    def check(self) -> None:
        """Paths are relative, normalized and unique, and listed in order."""
        paths = [f.path for f in self.files]
        for p in paths:
            if not _safe(p):
                raise CorpusError(f"manifest path not allowed: {p!r}")
        if paths != sorted(set(paths)):
            raise CorpusError("manifest paths must be unique and sorted")

    def pages(self) -> list[str]:
        return [page for f in self.files if (page := source.page_of(f.path)) is not None]


def load_manifest() -> Manifest:
    return Manifest.from_json(resources.files("recall.corpus").joinpath("manifest.json").read_text("utf-8"))


def _safe(path: str) -> bool:
    p = PurePosixPath(path)
    return (
        bool(path)
        and not p.is_absolute()
        and str(p) == path
        and all(part not in ("", ".", "..") for part in p.parts)
        and "\\" not in path
    )


def _members(tar: tarfile.TarFile, commit: str) -> dict[str, tarfile.TarInfo]:
    """The archive's members by path below its top directory, after checking the commit it declares."""
    declared = tar.pax_headers.get("comment")
    if declared != commit:
        raise CorpusError(f"the archive declares commit {declared!r}, not the pinned {commit}")
    members: dict[str, tarfile.TarInfo] = {}
    top = None
    for member in tar.getmembers():
        head, _, rest = member.name.partition("/")
        if top is None:
            top = head
        if head != top:
            raise CorpusError(f"archive member outside the top directory {top!r}: {member.name!r}")
        if not rest or member.isdir():
            continue
        if rest in members:
            raise CorpusError(f"archive has two members for {rest!r}")
        members[rest] = member
    return members


def _read_member(tar: tarfile.TarFile, members: dict[str, tarfile.TarInfo], path: str) -> bytes:
    member = members.get(path)
    if member is None:
        raise CorpusError(f"the archive has no {path}")
    if not member.isreg():
        raise CorpusError(f"{path} is not a regular file in the archive")
    f = tar.extractfile(member)
    if f is None:
        raise CorpusError(f"{path} can't be read from the archive")
    with f:
        return f.read()


def _describe(path: str, data: bytes) -> ManifestFile:
    return ManifestFile(path, len(data), hashlib.sha256(data).hexdigest())


def build_manifest(archive: Path, commit: str = source.COMMIT, tag: str = source.TAG) -> Manifest:
    """The manifest for an archive: the kept pages, every file they include, and the license."""
    with tarfile.open(archive, "r:*") as tar:
        members = _members(tar, commit)
        paths = {source.LICENSE_PATH}
        for path in sorted(members):
            if source.page_of(path) is None:
                continue
            paths.add(path)
            text = _read_member(tar, members, path).decode("utf-8")
            paths.update(normalize.includes(path, text))
        files = tuple(_describe(p, _read_member(tar, members, p)) for p in sorted(paths))
    manifest = Manifest(source.REPOSITORY, tag, commit, files)
    manifest.check()
    return manifest


def extract(archive: Path, manifest: Manifest, dest: Path) -> None:
    """Write the manifest's files from the archive to dest, all or nothing: dest is replaced only if all check out."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{dest.name}.", dir=dest.parent))
    try:
        with tarfile.open(archive, "r:*") as tar:
            members = _members(tar, manifest.commit)
            for expected in manifest.files:
                data = _read_member(tar, members, expected.path)
                if _describe(expected.path, data) != expected:
                    raise CorpusError(
                        f"{expected.path} differs from the manifest ({len(data)} bytes, sha256"
                        f" {hashlib.sha256(data).hexdigest()}; expected {expected.size} bytes, {expected.sha256})"
                    )
                target = staging.joinpath(*PurePosixPath(expected.path).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
        _replace(staging, dest)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _replace(staging: Path, dest: Path) -> None:
    old = None
    if dest.exists():
        old = dest.with_name(f".{dest.name}.old-{os.getpid()}")
        dest.rename(old)
    staging.rename(dest)
    if old is not None:
        shutil.rmtree(old)


def verify(dest: Path, manifest: Manifest) -> list[str]:
    """What is wrong with an extracted corpus: missing, changed or extra files. Empty when it matches."""
    problems: list[str] = []
    listed = {f.path for f in manifest.files}
    for expected in manifest.files:
        target = dest.joinpath(*PurePosixPath(expected.path).parts)
        if not target.is_file() or target.is_symlink():
            problems.append(f"missing: {expected.path}")
            continue
        if _describe(expected.path, target.read_bytes()) != expected:
            problems.append(f"changed: {expected.path}")
    if dest.is_dir():
        for found in sorted(p for p in dest.rglob("*") if not p.is_dir()):
            rel = found.relative_to(dest).as_posix()
            if rel not in listed:
                problems.append(f"not in the manifest: {rel}")
    return problems


def download(url: str, to: Path, max_bytes: int = MAX_ARCHIVE_BYTES) -> None:
    """Stream url to a file, refusing more than max_bytes."""
    total = 0
    with urllib.request.urlopen(url, timeout=60) as response, open(to, "wb") as out:
        while chunk := response.read(1 << 16):
            total += len(chunk)
            if total > max_bytes:
                raise CorpusError(f"{url} is larger than {max_bytes} bytes")
            out.write(chunk)


def fetch(dest: Path, manifest: Manifest, url: str = source.ARCHIVE_URL) -> bool:
    """Make dest hold exactly the manifest's files. Returns False if it already did (nothing downloaded)."""
    if dest.is_dir() and not verify(dest, manifest):
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".archive.", suffix=".tar.gz", dir=dest.parent)
    os.close(fd)
    archive = Path(name)
    try:
        download(url, archive)
        extract(archive, manifest, dest)
    finally:
        with contextlib.suppress(FileNotFoundError):
            archive.unlink()
    return True
