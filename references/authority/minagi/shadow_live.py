"""Quarantined live-learning intake.

User text is useful experience immediately as episodic memory, but it is not
sufficient evidence for an irreversible production-weight update. Shadow mode
records an append-only candidate stream. v4.1 can *seal* the current stream into
an immutable content-addressed batch; the qualification ledger then binds a
candidate to that exact batch digest rather than to a file that keeps growing.
"""
from __future__ import annotations
import hashlib, json, os, shutil, time
from pathlib import Path


class ShadowLearningBuffer:
    def __init__(self, root, max_record_chars=200_000):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "pending.jsonl"
        self.max_record_chars = int(max_record_chars)
        self.records = 0
        self.tokens = 0
        if self.path.exists():
            try:
                for line in self.path.open(encoding="utf-8"):
                    d = json.loads(line)
                    self.records += 1
                    self.tokens += int(d.get("tokens", 0))
            except Exception as e:
                raise RuntimeError(f"malformed shadow-learning buffer: {self.path}") from e

    def feed(self, text, tok, note="chat"):
        text = str(text or "")
        if not text:
            return []
        if len(text) > self.max_record_chars:
            raise ValueError("shadow-learning record too large")
        ids = list(tok.encode(text).ids)
        rec = {"ts": time.time(), "note": str(note), "text": text,
               "tokens": len(ids),
               "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
               "status": "pending"}
        self.root.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.records += 1
        self.tokens += len(ids)
        return [rec]

    def seal(self):
        """Freeze the current pending stream into a content-addressed batch.

        The pending file is not truncated. Multiple candidates may reference
        the same sealed evidence, and later conversation appends cannot change
        the digest of an already-sealed batch.
        """
        from .provenance import sha256_file
        if not self.path.exists() or self.path.stat().st_size == 0:
            raise ValueError("cannot seal an empty shadow-learning buffer")
        digest = sha256_file(self.path)
        batches = self.root / "batches"
        batches.mkdir(parents=True, exist_ok=True)
        dst = batches / f"{digest}.jsonl"
        if not dst.exists():
            tmp = batches / f".{digest}.tmp"
            shutil.copy2(self.path, tmp)
            if sha256_file(tmp) != digest:
                tmp.unlink(missing_ok=True)
                raise RuntimeError("shadow batch changed while sealing")
            os.replace(tmp, dst)
            try:
                os.chmod(dst, 0o444)
            except OSError:
                pass
            try:
                fd = os.open(str(batches), os.O_RDONLY)
                try: os.fsync(fd)
                finally: os.close(fd)
            except OSError:
                pass
        return {"path": str(dst), "sha256": digest,
                "records": self.records, "tokens": self.tokens}

    @property
    def pending(self):
        return self.tokens

    @property
    def chunk(self):
        return 0

    @property
    def steps(self):
        return 0

    def state(self):
        return {"mode": "shadow", "records": self.records, "tokens": self.tokens,
                "pending": self.pending, "chunk": self.chunk, "steps": self.steps,
                "path": str(self.path)}
