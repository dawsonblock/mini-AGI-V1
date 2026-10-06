"""Single-writer authority for mutable weight workspaces.

Immutable checkpoint generations may be read concurrently.  The mutable
``weights/`` workspace may have exactly one writer: corpus training, live
learning, reactivation, or any future mutation tool.  A thread lock is not
enough because the common failure mode is two different processes.

The lease uses the operating system's advisory file lock and writes diagnostic
metadata into ``.writer.lock``.  On Unix/macOS it uses ``flock``; on Windows it
uses ``msvcrt.locking``.  A small in-process registry also makes accidental
nested writer acquisition fail deterministically in tests and in one process.
"""
from __future__ import annotations

import json
import os
import socket
import time
import uuid
from pathlib import Path

_HELD: set[str] = set()


class WriterLease:
    def __init__(self, root: str | os.PathLike, purpose: str = "writer"):
        self.root = Path(root).resolve()
        self.path = self.root / ".writer.lock"
        self.purpose = str(purpose)
        self.id = uuid.uuid4().hex
        self._fh = None
        self._key = str(self.path)

    def acquire(self):
        self.root.mkdir(parents=True, exist_ok=True)
        if self._key in _HELD:
            raise RuntimeError(f"weights workspace already has a writer in this process: {self.root}")
        fh = open(self.path, "a+b", buffering=0)
        try:
            self._lock(fh)
        except Exception:
            fh.close()
            raise
        self._fh = fh
        _HELD.add(self._key)
        meta = {
            "lease_id": self.id,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "purpose": self.purpose,
            "started": time.time(),
        }
        payload = (json.dumps(meta, sort_keys=True) + "\n").encode("utf-8")
        fh.seek(0)
        fh.truncate()
        fh.write(payload)
        try:
            os.fsync(fh.fileno())
        except OSError:
            pass
        return self

    @staticmethod
    def _lock(fh):
        if os.name == "nt":
            import msvcrt
            fh.seek(0)
            if fh.read(1) == b"":
                fh.seek(0)
                fh.write(b"\0")
                fh.flush()
            fh.seek(0)
            try:
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as e:
                raise RuntimeError("weights workspace is already owned by another writer") from e
        else:
            import fcntl
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as e:
                raise RuntimeError("weights workspace is already owned by another writer") from e

    @staticmethod
    def _unlock(fh):
        try:
            if os.name == "nt":
                import msvcrt
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass

    def release(self):
        fh, self._fh = self._fh, None
        if fh is None:
            return
        try:
            # Keep the last owner metadata useful after release.
            try:
                fh.seek(0)
                raw = fh.read().decode("utf-8", errors="replace").strip()
                meta = json.loads(raw) if raw else {}
            except Exception:
                meta = {}
            meta.update({"released": time.time(), "released_pid": os.getpid()})
            payload = (json.dumps(meta, sort_keys=True) + "\n").encode("utf-8")
            fh.seek(0)
            fh.truncate()
            fh.write(payload)
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass
        finally:
            self._unlock(fh)
            fh.close()
            _HELD.discard(self._key)

    def __enter__(self):
        return self.acquire()

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False


def read_writer_metadata(root: str | os.PathLike) -> dict | None:
    p = Path(root) / ".writer.lock"
    try:
        text = p.read_text(encoding="utf-8").strip("\x00\n ")
        return json.loads(text) if text else None
    except (OSError, ValueError):
        return None
