from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator

from .models import ExecutionContext, require_digest


_current_execution_context: ContextVar[ExecutionContext | None] = ContextVar("rc14_execution_context", default=None)


@dataclass
class ExecutionContextGuard:
    """Process-local guard that makes request epoch pinning explicit and enforceable.

    Code performing request-scoped reads/writes can call ``require`` rather than silently
    falling back to the mutable serving head. This does not replace StateEpoch leases; it
    binds the lease-backed ExecutionContext to the current call context.
    """

    @contextmanager
    def bind(self, context: ExecutionContext) -> Iterator[ExecutionContext]:
        token = _current_execution_context.set(context)
        try:
            yield context
        finally:
            _current_execution_context.reset(token)

    def current(self) -> ExecutionContext | None:
        return _current_execution_context.get()

    def require(self, *, epoch_digest: str | None = None) -> ExecutionContext:
        context = self.current()
        if context is None:
            raise PermissionError("RC14 request-scoped operation requires a pinned ExecutionContext")
        if epoch_digest is not None:
            require_digest(epoch_digest, field_name="epoch_digest")
            if context.epoch_digest != epoch_digest:
                raise PermissionError("ExecutionContext epoch mismatch")
        return context

    def source_epoch_digest(self) -> str:
        return self.require().epoch_digest
