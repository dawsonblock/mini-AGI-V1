from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PolicySourceReport:
    safe: bool
    violations: tuple[str, ...]


_FORBIDDEN_IMPORTS = {"os", "subprocess", "socket", "shutil", "ctypes", "multiprocessing", "requests", "urllib", "pathlib"}
_FORBIDDEN_CALLS = {"eval", "exec", "compile", "open", "__import__", "input"}


def inspect_policy_source(source: str) -> PolicySourceReport:
    tree = ast.parse(source)
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in _FORBIDDEN_IMPORTS:
                    violations.append(f"forbidden import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] in _FORBIDDEN_IMPORTS:
                violations.append(f"forbidden import: {node.module}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_CALLS:
            violations.append(f"forbidden call: {node.func.id}")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            violations.append(f"dunder attribute access: {node.attr}")
    return PolicySourceReport(not violations, tuple(sorted(set(violations))))


def inspect_policy_file(path: str | Path) -> PolicySourceReport:
    return inspect_policy_source(Path(path).read_text())
