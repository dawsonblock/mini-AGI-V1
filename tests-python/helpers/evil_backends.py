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


class AccessProbeBackend:
    """v16.4.6 WP1 probe: the supervisor sends candidate paths; the
    worker reports what IT can actually read/write — the honest test
    of identity separation is what the worker's own syscalls return,
    not what the parent configured."""

    backend_id = "hf-peft"

    def __init__(self, **_kw):
        pass

    def load(self, snapshot):
        return {"handle": True}

    def health_probe(self, handle):
        return None

    def infer(self, handle, request):  # pragma: no cover - in child
        import os
        report = {"uid": os.getuid() if hasattr(os, "getuid") else -1,
                  "gid": os.getgid() if hasattr(os, "getgid") else -1,
                  "read": {}, "write": {}}
        for path in request.get("paths") or []:
            try:
                with open(path, "rb") as fh:
                    fh.read(8)
                report["read"][path] = "READABLE"
            except PermissionError:
                report["read"][path] = "denied"
            except OSError as exc:
                report["read"][path] = f"error:{exc.errno}"
        for path in request.get("write_paths") or []:
            try:
                with open(path, "ab") as fh:
                    fh.write(b"x")
                report["write"][path] = "WRITABLE"
            except PermissionError:
                report["write"][path] = "denied"
            except OSError as exc:
                report["write"][path] = f"error:{exc.errno}"
        probe = request.get("connect")
        if probe:
            import socket
            host, _, port = probe.partition(":")
            try:
                s = socket.create_connection(
                    (host, int(port)), timeout=3)
                s.close()
                report["connect"] = "CONNECTED"
            except OSError as exc:
                report["connect"] = f"denied:{exc.errno}"
        return report

    def unload(self, handle):
        return None


class SpawnBackend:
    """v16.4.6 WP2 probe: spawns children/grandchildren on demand —
    including a setsid() escapee and a SIGTERM-ignoring straggler —
    and reports their pids so the supervisor-side test can prove
    every owned process is dead after terminate()."""

    backend_id = "hf-peft"

    def __init__(self, **_kw):
        self.children = []

    def load(self, snapshot):  # pragma: no cover - runs in child
        return {"handle": True}

    def health_probe(self, handle):
        return None

    def infer(self, handle, request):  # pragma: no cover - in child
        import os
        import signal
        import subprocess
        import sys
        import time
        op = (request or {}).get("op")
        if op == "spawn":
            ignore = bool(request.get("ignore_term"))
            detach = bool(request.get("setsid"))
            code = (
                "import signal,time,os,sys\n"
                + ("os.setsid()\n" if detach else "")
                + ("signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
                   if ignore else "")
                + ("sys.stderr.write('ready\\n');sys.stderr.flush()\n"
                   "time.sleep(3600)\n"))
            for attempt in range(5):
                try:
                    p = subprocess.Popen(
                        [sys.executable, "-c", code],
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE)
                    break
                except BlockingIOError:
                    if attempt == 4:
                        raise
                    time.sleep(0.2)
            p.stderr.readline()  # the child is fully set up
            self.children.append(p)
            return {"pid": p.pid}
        if op == "pids":
            return {"pids": [c.pid for c in self.children],
                    "uid": os.getuid() if hasattr(os, "getuid") else -1}
        return {"ok": True}

    def unload(self, handle):
        return None
