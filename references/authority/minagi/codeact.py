"""Capability-scoped executable actions without unrestricted Python ``exec``.

The model may emit Python-shaped code because control/data flow are useful, but
mini-AGI does not hand that code a Python interpreter. ``CapabilitySandbox``
walks a deliberately small AST. The only side effects possible are calls to
tool functions explicitly present in the capability registry.

v4.1 hardens the VM against resource-amplification attacks: input variables and
tool outputs must be plain bounded data, AST size/depth is bounded, integer
magnitude and collection growth are checked *before* expensive operations,
tool and print budgets are explicit, and call targets cannot be smuggled in via
variables. This is still an action VM, not a security boundary for arbitrary
native code inside a registered tool; registered capabilities remain trusted.
"""
from __future__ import annotations

import ast
import math
import operator
from dataclasses import dataclass, field


class ActionRejected(ValueError):
    pass


@dataclass
class ActionResult:
    result: object = None
    variables: dict = field(default_factory=dict)
    tool_calls: list = field(default_factory=list)
    prints: list[str] = field(default_factory=list)
    steps: int = 0


class _Break(Exception):
    pass


class _Continue(Exception):
    pass


class CapabilitySandbox:
    SAFE_FUNCS = {
        "len": len, "sum": sum, "min": min, "max": max, "sorted": sorted,
        "abs": abs, "round": round, "str": str, "int": int, "float": float,
        "bool": bool, "list": list, "tuple": tuple, "dict": dict,
    }
    BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
           ast.Mod: operator.mod, ast.Pow: operator.pow}
    UN = {ast.UAdd: operator.pos, ast.USub: operator.neg, ast.Not: operator.not_}
    CMP = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
           ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
           ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b}
    _SCALARS = (type(None), bool, int, float, str, bytes)

    def __init__(self, tools=None, max_steps=10_000, max_loop=1_000,
                 max_collection=10_000, max_ast_nodes=4_000,
                 max_data_depth=16, max_int_bits=4096, max_tool_calls=64,
                 max_print_chars=20_000):
        self.tools = dict(tools or {})
        overlap = set(self.tools) & set(self.SAFE_FUNCS)
        if overlap:
            raise ValueError(f"tool names shadow safe builtins: {sorted(overlap)}")
        if any(not isinstance(k, str) or not k.isidentifier() or k.startswith("__")
               for k in self.tools):
            raise ValueError("tool names must be non-dunder Python identifiers")
        if any(not callable(v) for v in self.tools.values()):
            raise ValueError("every tool capability must be callable")
        self.max_steps = int(max_steps)
        self.max_loop = int(max_loop)
        self.max_collection = int(max_collection)
        self.max_ast_nodes = int(max_ast_nodes)
        self.max_data_depth = int(max_data_depth)
        self.max_int_bits = int(max_int_bits)
        self.max_tool_calls = int(max_tool_calls)
        self.max_print_chars = int(max_print_chars)
        self._steps = 0
        self._calls = []
        self._prints = []
        self._printed_chars = 0
        self.env = {}

    def _tick(self, n=1):
        self._steps += n
        if self._steps > self.max_steps:
            raise ActionRejected("action exceeded instruction budget")

    def _plain(self, v, depth=0):
        """Validate/copy data crossing the VM boundary.

        Returning a copy prevents a tool from retaining a mutable object that
        the action later mutates behind the audit log's back.
        """
        if depth > self.max_data_depth:
            raise ActionRejected("value nesting exceeds data-depth budget")
        if isinstance(v, bool) or v is None:
            return v
        if isinstance(v, int):
            if v.bit_length() > self.max_int_bits:
                raise ActionRejected("integer exceeds magnitude budget")
            return int(v)
        if isinstance(v, float):
            if not math.isfinite(v):
                raise ActionRejected("non-finite floats are forbidden")
            return float(v)
        if isinstance(v, (str, bytes)):
            if len(v) > self.max_collection:
                raise ActionRejected("string/bytes value exceeds size budget")
            return v[:] if isinstance(v, bytes) else str(v)
        if isinstance(v, (list, tuple)):
            if len(v) > self.max_collection:
                raise ActionRejected("sequence exceeds size budget")
            vals = [self._plain(x, depth + 1) for x in v]
            return vals if isinstance(v, list) else tuple(vals)
        if isinstance(v, dict):
            if len(v) > self.max_collection:
                raise ActionRejected("mapping exceeds size budget")
            return {self._plain(k, depth + 1): self._plain(x, depth + 1)
                    for k, x in v.items()}
        raise ActionRejected(f"non-plain value crosses capability boundary: {type(v).__name__}")

    def _bounded(self, v):
        return self._plain(v)

    def _preflight_binop(self, op, a, b):
        if isinstance(op, ast.Pow):
            if not isinstance(b, (int, float)):
                raise ActionRejected("power exponent must be numeric")
            if abs(float(b)) > 32:
                raise ActionRejected("power exponent too large")
            if isinstance(a, int) and isinstance(b, int) and b >= 0:
                est = max(1, a.bit_length()) * max(1, b)
                if est > self.max_int_bits:
                    raise ActionRejected("power result would exceed integer budget")
        if isinstance(op, ast.Mult):
            seq, mult = None, None
            if isinstance(a, (str, bytes, list, tuple)) and isinstance(b, int):
                seq, mult = a, b
            elif isinstance(b, (str, bytes, list, tuple)) and isinstance(a, int):
                seq, mult = b, a
            if seq is not None and mult is not None:
                if mult < 0:
                    mult = 0
                if len(seq) * mult > self.max_collection:
                    raise ActionRejected("repetition would exceed collection budget")
            if isinstance(a, int) and isinstance(b, int):
                if a and b and a.bit_length() + b.bit_length() > self.max_int_bits + 1:
                    raise ActionRejected("integer product would exceed magnitude budget")
        if isinstance(op, ast.Add):
            if isinstance(a, (str, bytes, list, tuple)) and isinstance(b, type(a)):
                if len(a) + len(b) > self.max_collection:
                    raise ActionRejected("concatenation would exceed collection budget")

    def run(self, code, variables=None):
        try:
            tree = ast.parse(code, mode="exec")
        except SyntaxError as e:
            raise ActionRejected(f"syntax error: {e.msg} at line {e.lineno}") from e
        nodes = list(ast.walk(tree))
        if len(nodes) > self.max_ast_nodes:
            raise ActionRejected("action exceeds AST-node budget")
        self.env = {str(k): self._plain(v) for k, v in dict(variables or {}).items()}
        if any(not k.isidentifier() or k.startswith("__") for k in self.env):
            raise ActionRejected("input variable names must be non-dunder identifiers")
        self._steps = 0
        self._calls = []
        self._prints = []
        self._printed_chars = 0
        for st in tree.body:
            self._stmt(st)
        result = self.env.get("_result")
        return ActionResult(result=self._plain(result), variables=self._plain(dict(self.env)),
                            tool_calls=list(self._calls), prints=list(self._prints),
                            steps=self._steps)

    def _assign(self, target, value):
        self._tick()
        value = self._bounded(value)
        if isinstance(target, ast.Name):
            if target.id.startswith("__"):
                raise ActionRejected("dunder names are forbidden")
            self.env[target.id] = value
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            vals = list(value)
            if len(vals) != len(target.elts):
                raise ActionRejected("unpack length mismatch")
            for t, v in zip(target.elts, vals):
                self._assign(t, v)
            return
        if isinstance(target, ast.Subscript):
            obj = self._expr(target.value)
            key = self._slice(target.slice)
            if not isinstance(obj, (list, dict)):
                raise ActionRejected("subscript assignment allowed only on list/dict")
            obj[key] = value
            self._bounded(obj)
            return
        raise ActionRejected(f"assignment target not allowed: {type(target).__name__}")

    def _stmt(self, n):
        self._tick()
        if isinstance(n, ast.Assign):
            v = self._expr(n.value)
            for t in n.targets:
                self._assign(t, v)
        elif isinstance(n, ast.AnnAssign):
            if n.value is None:
                raise ActionRejected("annotation-only assignment is not supported")
            self._assign(n.target, self._expr(n.value))
        elif isinstance(n, ast.AugAssign):
            if not isinstance(n.target, ast.Name) or type(n.op) not in self.BIN:
                raise ActionRejected("unsupported augmented assignment")
            if n.target.id not in self.env:
                raise ActionRejected("augmented assignment to unknown variable")
            a, b = self.env[n.target.id], self._expr(n.value)
            self._preflight_binop(n.op, a, b)
            self._assign(n.target, self.BIN[type(n.op)](a, b))
        elif isinstance(n, ast.Expr):
            self._expr(n.value)
        elif isinstance(n, ast.If):
            body = n.body if self._expr(n.test) else n.orelse
            for s in body:
                self._stmt(s)
        elif isinstance(n, ast.For):
            seq = self._expr(n.iter)
            if not isinstance(seq, (list, tuple, range)):
                raise ActionRejected("for-loop iterable must be bounded list/tuple/range")
            if len(seq) > self.max_loop:
                raise ActionRejected("loop exceeds iteration budget")
            for v in seq:
                self._assign(n.target, v)
                try:
                    for s in n.body:
                        self._stmt(s)
                except _Continue:
                    continue
                except _Break:
                    break
            else:
                for s in n.orelse:
                    self._stmt(s)
        elif isinstance(n, ast.Break):
            raise _Break()
        elif isinstance(n, ast.Continue):
            raise _Continue()
        elif isinstance(n, ast.Pass):
            return
        else:
            raise ActionRejected(f"statement not allowed: {type(n).__name__}")

    def _slice(self, n):
        if isinstance(n, ast.Slice):
            return slice(self._expr(n.lower) if n.lower else None,
                         self._expr(n.upper) if n.upper else None,
                         self._expr(n.step) if n.step else None)
        return self._expr(n)

    def _call_named(self, name, args, kw):
        if name in self.SAFE_FUNCS:
            return self._bounded(self.SAFE_FUNCS[name](*args, **kw))
        if name == "range":
            r = range(*map(int, args))
            if kw:
                raise ActionRejected("range does not accept keyword arguments here")
            if len(r) > self.max_loop:
                raise ActionRejected("range exceeds loop budget")
            return r
        if name == "print":
            if kw:
                raise ActionRejected("print keyword arguments are disabled")
            s = " ".join(map(str, args))
            if self._printed_chars + len(s) > self.max_print_chars:
                raise ActionRejected("print output exceeds budget")
            self._printed_chars += len(s)
            self._prints.append(s)
            return None
        if name in self.tools:
            if len(self._calls) >= self.max_tool_calls:
                raise ActionRejected("tool-call budget exceeded")
            # Arguments are copied plain data before they cross into trusted
            # tool code, and outputs are copied/validated on the way back.
            aa = self._plain(args)
            kk = self._plain(kw)
            self._calls.append({"tool": name, "args": aa, "kwargs": kk})
            return self._plain(self.tools[name](*aa, **kk))
        raise ActionRejected(f"call target is not an allowed capability: {name}")

    def _expr(self, n):
        self._tick()
        if n is None:
            return None
        if isinstance(n, ast.Constant):
            return self._bounded(n.value)
        if isinstance(n, ast.Name):
            if n.id in self.env:
                return self.env[n.id]
            raise ActionRejected(f"name not available as a value: {n.id}")
        if isinstance(n, ast.BinOp) and type(n.op) in self.BIN:
            a, b = self._expr(n.left), self._expr(n.right)
            self._preflight_binop(n.op, a, b)
            return self._bounded(self.BIN[type(n.op)](a, b))
        if isinstance(n, ast.UnaryOp) and type(n.op) in self.UN:
            return self._bounded(self.UN[type(n.op)](self._expr(n.operand)))
        if isinstance(n, ast.BoolOp):
            # Preserve Python short-circuit semantics so a skipped branch cannot
            # invoke a tool as an accidental side effect.
            if isinstance(n.op, ast.And):
                result = True
                for x in n.values:
                    result = self._expr(x)
                    if not result:
                        return result
                return result
            result = False
            for x in n.values:
                result = self._expr(x)
                if result:
                    return result
            return result
        if isinstance(n, ast.Compare):
            left = self._expr(n.left)
            for op, comp in zip(n.ops, n.comparators):
                right = self._expr(comp)
                if type(op) not in self.CMP or not self.CMP[type(op)](left, right):
                    return False
                left = right
            return True
        if isinstance(n, ast.IfExp):
            return self._expr(n.body if self._expr(n.test) else n.orelse)
        if isinstance(n, ast.List):
            return self._bounded([self._expr(x) for x in n.elts])
        if isinstance(n, ast.Tuple):
            return self._bounded(tuple(self._expr(x) for x in n.elts))
        if isinstance(n, ast.Dict):
            return self._bounded({self._expr(k): self._expr(v)
                                  for k, v in zip(n.keys, n.values)})
        if isinstance(n, ast.Subscript):
            return self._bounded(self._expr(n.value)[self._slice(n.slice)])
        if isinstance(n, ast.Call):
            if not isinstance(n.func, ast.Name):
                raise ActionRejected("only direct named calls are allowed")
            if n.keywords and any(k.arg is None for k in n.keywords):
                raise ActionRejected("**kwargs expansion is forbidden")
            args = [self._expr(x) for x in n.args]
            kw = {k.arg: self._expr(k.value) for k in n.keywords}
            return self._call_named(n.func.id, args, kw)
        # Attribute access, lambdas, comprehensions, generators, function/class
        # definitions, imports, await/yield and exception machinery stay absent.
        raise ActionRejected(f"expression not allowed: {type(n).__name__}")
