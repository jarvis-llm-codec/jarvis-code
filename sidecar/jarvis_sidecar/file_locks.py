from __future__ import annotations

import json
import os
import tempfile
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator

_DEFAULT_TIMEOUT_SECONDS = 120.0


def _lock_path_for(target_path: Path) -> Path:
    return target_path.with_name(f".{target_path.name}.lock")


@contextmanager
def cross_process_file_lock(
    target_path: str | Path,
    *,
    timeout: float = _DEFAULT_TIMEOUT_SECONDS,
) -> Iterator[Path]:
    """Acquire an OS file lock for a file path.

    The lock lives beside the protected file, so separate JARVIS windows that
    append/rewrite the same memory file serialize even when they are separate
    processes.
    """
    target = Path(target_path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = _lock_path_for(target)
    deadline = time.monotonic() + max(0.0, timeout)
    fh = lock_path.open("a+b")
    acquired = False
    try:
        while True:
            fh.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"timed out acquiring file lock: {lock_path}")
                time.sleep(0.02)

        owner = {
            "pid": os.getpid(),
            "target": str(target),
            "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        }
        try:
            fh.seek(0)
            fh.truncate()
            fh.write((json.dumps(owner, ensure_ascii=False) + "\n").encode("utf-8"))
            fh.flush()
        except OSError:
            pass
        yield lock_path
    finally:
        if acquired:
            try:
                fh.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
        fh.close()


def locked_append_text(path: str | Path, text: str, *, encoding: str = "utf-8") -> None:
    target = Path(path).expanduser()
    with cross_process_file_lock(target):
        with target.open("a", encoding=encoding, newline="") as fh:
            fh.write(text)
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass


def locked_atomic_write_text(path: str | Path, content: str, *, encoding: str = "utf-8") -> None:
    target = Path(path).expanduser()
    with cross_process_file_lock(target):
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp")
        try:
            try:
                os.fchmod(fd, 0o600)
            except (AttributeError, OSError):
                pass
            with os.fdopen(fd, "w", encoding=encoding, errors="replace", newline="") as fh:
                fh.write(content)
                fh.flush()
                try:
                    os.fsync(fh.fileno())
                except OSError:
                    pass
            os.replace(tmp_name, str(target))
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
