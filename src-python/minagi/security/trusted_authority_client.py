"""Client for the supervised launch service (UPGRADE_PLAN §3.2).

The research plane holds no signing keys and no writable deployment
paths; it submits *documents* to the service and receives a signed
receipt digest or a refusal. This module is deliberately thin — all
authority lives behind the socket.
"""
from __future__ import annotations

import json
import socket
from pathlib import Path


class ServiceUnavailable(RuntimeError):
    """The supervised launch service cannot be reached."""


class RequestRefused(PermissionError):
    """The service refused the request."""


class SupervisorClient:
    """JSON-lines client for `minagi.runtime.service`."""

    def __init__(self, socket_path):
        self.socket_path = Path(socket_path)

    def _call(self, req: dict) -> dict:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.connect(str(self.socket_path))
        except OSError as exc:
            raise ServiceUnavailable(
                f"supervised launch service unreachable at "
                f"{self.socket_path}: {exc}") from exc
        try:
            f = sock.makefile("rwb")
            f.write((json.dumps(req) + "\n").encode())
            f.flush()
            line = f.readline()
            if not line:
                raise ServiceUnavailable("service closed the connection "
                                         "without a response")
            resp = json.loads(line)
        except (ServiceUnavailable, RequestRefused):
            raise
        except OSError as exc:
            # a refused connection can surface as a bare reset rather
            # than a refused frame (e.g. peer-uid rejection racing the
            # write path)
            raise ServiceUnavailable(
                f"supervised launch service refused the connection: "
                f"{exc}") from exc
        finally:
            sock.close()
        if not resp.get("ok"):
            if "refused" in resp:
                raise RequestRefused(resp["refused"])
            raise ServiceUnavailable(resp.get("error", "unknown error"))
        return resp

    def ping(self) -> dict:
        return self._call({"op": "ping"})

    def status(self) -> dict:
        return self._call({"op": "status"})

    def recover(self) -> dict:
        return self._call({"op": "recover"})

    def quarantine(self, reason: str = "") -> dict:
        return self._call({"op": "quarantine", "reason": reason})

    def rollback(self) -> dict:
        return self._call({"op": "rollback"})

    def launch(self, request_fields: dict, *, backend: str = "hf-peft",
               receipt_path=None) -> dict:
        return self._call({"op": "launch", "backend": backend,
                           "request": dict(request_fields),
                           "receipt_path": receipt_path})
