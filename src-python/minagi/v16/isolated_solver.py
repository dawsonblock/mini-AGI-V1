from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import resource
import subprocess
from typing import Mapping, Sequence


@dataclass(frozen=True)
class SolverSandboxLimitsV160:
    timeout_seconds: float = 30.0
    address_space_bytes: int = 2 * 1024 * 1024 * 1024
    cpu_seconds: int = 20
    open_files: int = 64
    output_bytes: int = 1024 * 1024


class IsolatedJSONSolverV160:
    """Run an untrusted candidate solver out-of-process using a narrow JSON protocol.

    The evaluator sends only case_id/input/task_kind.  Expected answers, hidden
    suite key and aggregate scoring state never enter the candidate process.

    This is a process boundary, not a complete hostile-code sandbox.  For a
    production secrecy boundary, configure a dedicated uid/gid and an OS-level
    sandbox/container/VM that denies access to evaluator storage.
    """

    def __init__(
        self,
        command: Sequence[str],
        *,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        uid: int | None = None,
        gid: int | None = None,
        limits: SolverSandboxLimitsV160 | None = None,
    ):
        command = tuple(str(x) for x in command)
        if not command:
            raise ValueError("solver command required")
        self.command = command
        self.cwd = str(cwd) if cwd is not None else None
        self.env = dict(env or {})
        self.uid = uid
        self.gid = gid
        self.limits = limits or SolverSandboxLimitsV160()

    def _preexec(self):
        lim = self.limits
        resource.setrlimit(resource.RLIMIT_CPU, (lim.cpu_seconds, lim.cpu_seconds))
        resource.setrlimit(resource.RLIMIT_NOFILE, (lim.open_files, lim.open_files))
        if lim.address_space_bytes > 0:
            resource.setrlimit(resource.RLIMIT_AS, (lim.address_space_bytes, lim.address_space_bytes))
        if self.gid is not None:
            os.setgid(int(self.gid))
        if self.uid is not None:
            os.setuid(int(self.uid))

    def solve(self, *, case_id: str, input_text: str, task_kind: str) -> str:
        request = json.dumps(
            {"case_id": str(case_id), "input": str(input_text), "task_kind": str(task_kind)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        base_env = {"PATH": os.environ.get("PATH", "")}
        base_env.update(self.env)
        kwargs = dict(
            args=self.command,
            input=request,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.cwd,
            env=base_env,
            timeout=self.limits.timeout_seconds,
            check=False,
        )
        if os.name == "posix":
            kwargs["preexec_fn"] = self._preexec
        proc = subprocess.run(**kwargs)
        if proc.returncode != 0:
            err = proc.stderr[:4096].decode("utf-8", "replace")
            raise RuntimeError(f"isolated solver failed with code {proc.returncode}: {err}")
        if len(proc.stdout) > self.limits.output_bytes:
            raise RuntimeError("isolated solver output exceeds configured limit")
        try:
            response = json.loads(proc.stdout.decode("utf-8"))
        except Exception as exc:
            raise RuntimeError("isolated solver emitted invalid JSON") from exc
        if not isinstance(response, dict) or "text" not in response or not isinstance(response["text"], str):
            raise RuntimeError("isolated solver response must be a JSON object containing string field 'text'")
        return response["text"]

    def as_candidate_solver(self):
        # FrozenTransferEvaluator calls candidate_solver(input, task_kind) and
        # intentionally does not expose case_id.  Use a deterministic digest as
        # the process-visible case id so no expected answer or split metadata leaks.
        import hashlib
        def solve(input_text: str, task_kind: str):
            cid = hashlib.sha256((str(task_kind) + "\0" + str(input_text)).encode()).hexdigest()
            return self.solve(case_id=cid, input_text=str(input_text), task_kind=str(task_kind))
        return solve
