"""Adversarial worker-side backends for IPC isolation tests.

These classes are intentionally hostile: they model what a
compromised worker could do — emit non-protocol bytes on the reply
channel, attempt to reach protected files, or wedge forever. They
live in an importable helper module (never in __main__) so spawned
worker interpreters can resolve them under multiprocessing spawn.
"""


class ProtocolAttacker:
    """On load(), writes raw non-protocol bytes (pickle + junk) into
    every writable fd — the supervisor must tear the channel down,
    not deserialize anything."""

    backend_id = "hf-peft"

    def __init__(self, **_kw):
        pass

    def load(self, snapshot):  # pragma: no cover - runs in child
        import os
        import pickle
        evil = pickle.dumps({"payload": {"__reduce__": "os.system"}})
        junk = b"\x00\xff" * 512 + evil
        for fd in range(3, 32):
            try:
                os.write(fd, junk)
            except OSError:
                continue
        return {"handle": True}

    def health_probe(self, handle):
        return None

    def infer(self, handle, request):
        return {"echo": request.get("prompt", "")}

    def unload(self, handle):
        return None


class SilentBackend:
    """Answers the boot handshake then never answers load — the
    supervisor's start deadline must expire and kill it."""

    backend_id = "hf-peft"

    def __init__(self, **_kw):
        pass

    def load(self, snapshot):  # pragma: no cover - runs in child
        import time
        time.sleep(3600)
        return {"handle": True}

    def health_probe(self, handle):
        return None

    def infer(self, handle, request):
        return {}

    def unload(self, handle):
        return None


class EnvDumpBackend:
    """Reports its own process environment through the protocol —
    lets the supervisor-side test verify the worker's environment was
    constructed (scrubbed), not inherited."""

    backend_id = "hf-peft"

    def __init__(self, **_kw):
        pass

    def load(self, snapshot):  # pragma: no cover - runs in child
        return {"handle": True}

    def health_probe(self, handle):
        return None

    def infer(self, handle, request):  # pragma: no cover - runs in child
        import os
        import tempfile
        return {"env_names": sorted(os.environ),
                "tmpdir": str(tempfile.gettempdir()),
                "uid": os.getuid() if hasattr(os, "getuid") else -1}

    def unload(self, handle):
        return None
