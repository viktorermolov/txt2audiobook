"""Atomic small state files shared by the pipeline and publisher."""

import json
import os
import tempfile
from pathlib import Path


def fsync_file(path: Path) -> None:
    """Flush a completed payload before making its name durable."""
    with Path(path).open("rb") as stream:
        os.fsync(stream.fileno())


def fsync_directory(path: Path) -> None:
    """Flush a directory entry created by a rename or replacement."""
    fd = os.open(Path(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def replace_and_fsync(source: Path, destination: Path) -> None:
    """Atomically replace a file and persist the resulting directory entry."""
    source = Path(source)
    destination = Path(destination)
    fsync_file(source)
    os.replace(source, destination)
    fsync_directory(destination.parent)


def rename_directory_and_fsync(source: Path, destination: Path) -> None:
    """Atomically publish a completed staging directory on one filesystem."""
    source = Path(source)
    destination = Path(destination)
    source_parent = source.parent
    os.rename(source, destination)
    fsync_directory(source_parent)
    if destination.parent != source_parent:
        fsync_directory(destination.parent)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        fsync_directory(path.parent)
    finally:
        Path(name).unlink(missing_ok=True)
