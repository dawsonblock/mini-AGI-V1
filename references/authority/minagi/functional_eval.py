"""Task-level capability evaluation.

Language-model loss is retained for training diagnostics, but capability gates
must measure outcomes.  In particular, Python tasks are executed against unit
tests instead of receiving credit merely for parsing.  The executor is a
small local harness with AST restrictions, an isolated interpreter, timeouts,
and Unix resource limits.  It is defense-in-depth for benchmark snippets, not
a replacement for an OS/container security boundary for hostile code.
"""
from __future__ import annotations

import ast
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

_NUM = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
_FENCE = re.compile(r"```(?:python)?\s*(.*?)```", re.I | re.S)
_ALLOWED_IMPORTS = {"math", "statistics", "collections", "itertools", "functools"}
_DENIED_CALLS = {"open", "exec", "eval", "compile", "input", "__import__", "breakpoint"}
_DENIED_ATTRS = {"system", "popen", "spawn", "fork", "socket", "connect", "unlink",
                 "remove", "rmtree", "chmod", "chown", "kill"}


@dataclass
class EvalTask:
    id: str
    prompt: str
    scorer: str
    expected: object | None = None
    tolerance: float = 1e-6
    timeout: float = 2.0


def load_tasks(path: str | Path) -> list[EvalTask]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            d = json.loads(line)
            out.append(EvalTask(**d))
    return out


def extract_python(text: str) -> str:
    m = _FENCE.search(text or "")
    if m:
        return m.group(1).strip()
    lines = (text or "").splitlines()
    for i, line in enumerate(lines):
        if line.lstrip().startswith(("def ", "class ", "import ", "from ")):
            return "\n".join(lines[i:]).strip()
    return (text or "").strip()


def _check_python_ast(code: str) -> tuple[bool, str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"syntax error: {e.msg} at line {e.lineno}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in _ALLOWED_IMPORTS:
                    return False, f"import not allowed in benchmark sandbox: {a.name}"
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".")[0]
            if mod not in _ALLOWED_IMPORTS:
                return False, f"import not allowed in benchmark sandbox: {node.module}"
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _DENIED_CALLS:
                return False, f"call not allowed in benchmark sandbox: {node.func.id}"
            if isinstance(node.func, ast.Attribute) and node.func.attr in _DENIED_ATTRS:
                return False, f"attribute call not allowed in benchmark sandbox: {node.func.attr}"
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            return False, "global/nonlocal declarations are not allowed in benchmark code"
    return True, "valid restricted Python AST"


def _preexec_limits():
    if os.name == "nt":
        return None
    def limit():
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
            resource.setrlimit(resource.RLIMIT_AS, (256 << 20, 256 << 20))
            resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
            resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
        except Exception:
            pass
    return limit


def _python_harness(code: str, spec: dict, tolerance: float) -> str:
    entry = str(spec.get("entrypoint") or "")
    cases = list(spec.get("cases") or [])
    if not entry or not cases:
        raise ValueError("python_tests expected requires entrypoint and non-empty cases")
    payload = json.dumps(cases, separators=(",", ":"), ensure_ascii=False)
    # Keep the comparison implementation outside the candidate function's
    # namespace conventions and report one compact JSON record to the parent.
    return code + "\n\n" + f'''\nimport json as _mj_json\nimport math as _mj_math\n_mj_cases = _mj_json.loads({payload!r})\n_mj_fn = globals().get({entry!r})\nif not callable(_mj_fn):\n    print(_mj_json.dumps({{"ok": False, "detail": "missing callable {entry}"}}))\n    raise SystemExit(0)\n\ndef _mj_equal(a, b):\n    if isinstance(a, (int,float)) and isinstance(b, (int,float)):\n        return _mj_math.isclose(float(a), float(b), rel_tol={float(tolerance)!r}, abs_tol={float(tolerance)!r})\n    if isinstance(a, (list,tuple)) and isinstance(b, (list,tuple)):\n        return len(a)==len(b) and all(_mj_equal(x,y) for x,y in zip(a,b))\n    if isinstance(a, dict) and isinstance(b, dict):\n        return a.keys()==b.keys() and all(_mj_equal(a[k], b[k]) for k in a)\n    return a == b\n\n_mj_fail = []\nfor _mj_i, _mj_case in enumerate(_mj_cases):\n    try:\n        _mj_args = _mj_case.get("args", [])\n        _mj_kwargs = _mj_case.get("kwargs", {{}})\n        _mj_got = _mj_fn(*_mj_args, **_mj_kwargs)\n        _mj_want = _mj_case.get("expected")\n        if not _mj_equal(_mj_got, _mj_want):\n            _mj_fail.append({{"case": _mj_i, "got": repr(_mj_got), "expected": repr(_mj_want)}})\n    except Exception as _mj_e:\n        _mj_fail.append({{"case": _mj_i, "error": type(_mj_e).__name__ + ": " + str(_mj_e)}})\nprint(_mj_json.dumps({{"ok": not _mj_fail, "failures": _mj_fail[:3], "cases": len(_mj_cases)}}))\n'''


def run_python_tests(code: str, spec: dict, tolerance: float = 1e-6,
                     timeout: float = 2.0) -> dict:
    ok_ast, detail = _check_python_ast(code)
    if not ok_ast:
        return {"ok": False, "detail": detail}
    try:
        script = _python_harness(code, spec, tolerance)
    except (TypeError, ValueError) as e:
        return {"ok": False, "detail": f"invalid test spec: {e}"}
    with tempfile.TemporaryDirectory(prefix="miniagi-eval-") as td:
        path = Path(td) / "candidate.py"
        path.write_text(script, encoding="utf-8")
        env = {"PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"}
        try:
            cp = subprocess.run(
                [sys.executable, "-I", "-S", str(path)], cwd=td,
                env=env, text=True, capture_output=True,
                timeout=max(0.1, float(timeout)), preexec_fn=_preexec_limits())
        except subprocess.TimeoutExpired:
            return {"ok": False, "detail": f"execution timed out after {timeout:g}s"}
        if cp.returncode != 0:
            err = (cp.stderr or cp.stdout or "").strip().splitlines()
            return {"ok": False, "detail": "runtime failure: " + (err[-1] if err else f"exit {cp.returncode}")}
        lines = [x for x in cp.stdout.splitlines() if x.strip()]
        if not lines:
            return {"ok": False, "detail": "test harness produced no result"}
        try:
            rec = json.loads(lines[-1])
        except json.JSONDecodeError:
            return {"ok": False, "detail": "candidate polluted test protocol output"}
        if rec.get("ok"):
            return {"ok": True, "detail": f"passed {rec.get('cases', 0)} executable cases"}
        return {"ok": False, "detail": "failed cases: " + json.dumps(rec.get("failures", []), ensure_ascii=False)}


def score(task: EvalTask, output: str) -> dict:
    kind = task.scorer
    ok = False
    detail = ""
    if kind == "numeric":
        nums = _NUM.findall(output or "")
        if nums:
            got = float(nums[-1]); want = float(task.expected)
            ok = math.isclose(got, want, rel_tol=task.tolerance, abs_tol=task.tolerance)
            detail = f"got={got:g} expected={want:g}"
        else:
            detail = "no numeric answer found"
    elif kind == "exact":
        got = " ".join((output or "").strip().split()).lower()
        want = " ".join(str(task.expected).strip().split()).lower()
        ok = got == want; detail = f"got={got!r} expected={want!r}"
    elif kind == "contains":
        wants = task.expected if isinstance(task.expected, list) else [task.expected]
        low = (output or "").lower()
        missing = [str(w) for w in wants if str(w).lower() not in low]
        ok = not missing; detail = "missing=" + repr(missing)
    elif kind == "python_ast":
        ok, detail = _check_python_ast(extract_python(output))
    elif kind == "python_tests":
        rec = run_python_tests(extract_python(output), task.expected or {},
                               tolerance=task.tolerance, timeout=task.timeout)
        ok, detail = bool(rec["ok"]), str(rec["detail"])
    else:
        raise ValueError(f"unknown scorer: {kind}")
    return {"ok": bool(ok), "detail": detail}


def summarise(results: list[dict]) -> dict:
    n = len(results)
    passed = sum(bool(r.get("ok")) for r in results)
    by_kind = {}
    for r in results:
        k = r.get("scorer", "unknown")
        d = by_kind.setdefault(k, {"passed": 0, "total": 0})
        d["total"] += 1; d["passed"] += int(bool(r.get("ok")))
    return {"passed": passed, "total": n,
            "accuracy": passed / n if n else 0.0, "by_scorer": by_kind}
