"""Process-scoped local locks; released by the OS even if a worker exits."""

import json
import os
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def exclusive(name):
    folder = ROOT / ".cache/research_locks"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (name + ".lock")
    with path.open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError(
                    f"Another robotics worker owns {name}; do not launch a competing run"
                ) from error
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        metadata = folder / (name + ".json")
        metadata.write_text(
            json.dumps({"pid": os.getpid(), "acquired_utc": datetime.now(UTC).isoformat(), "name": name}),
            encoding="utf8",
        )
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
