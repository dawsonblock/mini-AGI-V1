#!/usr/bin/env python3
"""Campaign 3B.3 Priority 1 — map the control plane, find duplicate authority.

The plan asks for "one authoritative control plane" and the elimination
of "duplicate authority and unresolved trust boundaries". Neither is
possible until it is known WHICH authority is authoritative: the tree
carries several generations of authority code, and only one generation
can be live.

This tool answers that with evidence rather than by reading. It:

  1. finds every module that defines authority-like symbols;
  2. builds the import graph reachable from the LIVE entry points
     (the CLI, the release/qualification scripts, the runtime);
  3. builds a second graph reachable from the TEST tree;
  4. classifies each authority module LIVE / TEST-ONLY / ORPHAN.

ORPHAN does not mean "delete": a donor tree kept for provenance is
legitimate. It means "no live path reaches this, so it is not part of
the control plane" — which is the fact needed to stop treating it as one.

Usage:
  python3 scripts/analysis/control_plane_map.py --output CONTROL_PLANE_MAP.json
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src-python"

#: Symbols that mark a module as part of an authority surface.
AUTHORITY_MARKERS = re.compile(
    r"\b(Authority|AuthorityRegistry|AuthorityLedger|assert_authorized|"
    r"AuthorizedLaunchRecord|PromotionAuthority|qualify|admit|revocation|"
    r"trust_root|TrustRecord)\b")

#: Where the live control plane is entered from. Anything not reachable
#: from here is not part of it.
LIVE_ENTRY_GLOBS = ("scripts/*.py", "scripts/**/*.py",
                    "src-python/minagi/cli.py")


#: Synthetic namespace for files outside src-python (scripts, tests).
#: They are the real entry points but are not importable modules, so they
#: need names of their own for the graph to seed from them. Without this
#: the graph seeds only from minagi.cli and reports one live module,
#: which is exactly the wrong answer.
OUTSIDE = "_outside"


def _module_name(path: Path) -> str | None:
    if path.suffix != ".py":
        return None
    rp = path.resolve()
    try:
        rel = rp.relative_to(SRC.resolve())
        parts = list(rel.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        return ".".join(parts) if parts else None
    except ValueError:
        pass
    try:
        rel = rp.relative_to(ROOT.resolve())
    except ValueError:
        return None
    parts = list(rel.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join([OUTSIDE, *parts]) if parts else None


def _imports(path: Path) -> set[str]:
    """Module names imported by `path`, absolute-form, best effort."""
    try:
        tree = ast.parse(path.read_text())
    except (SyntaxError, UnicodeDecodeError, OSError):
        return set()
    found: set[str] = set()
    pkg = _module_name(path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                found.add(a.name)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level and pkg:
                head = pkg.split(".")
                base = ".".join(head[:len(head) - node.level] + ([base] if base else []))
            for a in node.names:
                found.add(f"{base}.{a.name}" if base else a.name)
            if base:
                found.add(base)
    return found


def _resolve(name: str, index: dict) -> str | None:
    """Longest module-prefix of `name` present in the index."""
    parts = name.split(".")
    for i in range(len(parts), 0, -1):
        cand = ".".join(parts[:i])
        if cand in index:
            return cand
    return None


def _reachable(seeds: list[str], index: dict) -> set[str]:
    seen: set[str] = set()
    stack = [s for s in seeds if s]
    while stack:
        cur = stack.pop()
        if cur in seen or cur not in index:
            continue
        seen.add(cur)
        for dep in index[cur]:
            r = _resolve(dep, index)
            if r and r not in seen:
                stack.append(r)
    return seen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    index: dict[str, set[str]] = {}
    for p in SRC.rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        name = _module_name(p)
        if name:
            index[name] = _imports(p)
    # scripts and tests are entry points too; index them under the
    # synthetic namespace so the graph can seed from them
    for base in ("scripts", "tests-python"):
        for p in (ROOT / base).rglob("*.py"):
            if "__pycache__" in p.parts:
                continue
            name = _module_name(p)
            if name:
                index[name] = _imports(p)

    live_seeds, test_seeds = [], []
    for pattern in LIVE_ENTRY_GLOBS:
        for p in ROOT.glob(pattern):
            n = _module_name(p)
            if n:
                live_seeds.append(n)
    for p in (ROOT / "tests-python").rglob("*.py"):
        n = _module_name(p)
        if n:
            test_seeds.append(n)

    live = _reachable(live_seeds, index)
    tested = _reachable(test_seeds, index)

    # authority surfaces: modules whose source mentions authority symbols
    surfaces = []
    for name, _deps in index.items():
        p = SRC / (name.replace(".", "/") + ".py")
        if not p.is_file():
            p = SRC / name.replace(".", "/") / "__init__.py"
        if not p.is_file():
            continue
        text = p.read_text(errors="replace")
        hits = sorted(set(AUTHORITY_MARKERS.findall(text)))
        if not hits:
            continue
        # A module with a __main__ guard or a main() is OPERATOR-LAUNCHED:
        # reachable by invocation, not by import, so a static import graph
        # cannot see it. Without this class the production service reads
        # as "test-only", which would be a false and alarming claim.
        entry = ('if __name__ == "__main__"' in text
                 or re.search(r"^def main\(", text, re.M) is not None)
        state = ("LIVE" if name in live else
                 "SERVICE-ENTRYPOINT" if entry else
                 "TEST-ONLY" if name in tested else "ORPHAN")
        surfaces.append({"module": name, "state": state,
                         "operator_launchable": bool(entry),
                         "markers": hits,
                         "lines": len(text.splitlines())})

    order = {"LIVE": 0, "SERVICE-ENTRYPOINT": 1, "TEST-ONLY": 2, "ORPHAN": 3}
    surfaces.sort(key=lambda s: (order[s["state"]], s["module"]))
    counts = {k: sum(1 for s in surfaces if s["state"] == k)
              for k in ("LIVE", "SERVICE-ENTRYPOINT", "TEST-ONLY", "ORPHAN")}

    report = {
        "schema": "mini-agi-campaign3b3-control-plane-map-v1",
        "live_entry_points": sorted(set(live_seeds)),
        "n_modules_indexed": len(index),
        "n_reachable_from_live": len(live),
        "n_reachable_from_tests": len(tested),
        "authority_surface_counts": counts,
        "authority_surfaces": surfaces,
        "note": ("LIVE = reachable by import from a script or CLI entry "
                 "point. SERVICE-ENTRYPOINT = has a main()/__main__ guard, "
                 "so it is launched directly by an operator and is "
                 "invisible to a static import graph; it may well be the "
                 "production path. TEST-ONLY = reachable only from tests. "
                 "ORPHAN = neither. A donor tree kept for provenance is "
                 "legitimate; the point is to stop treating it as an "
                 "authority."),
        "static_analysis_limit": ("This map proves reachability, not "
                                  "deployment. A SERVICE-ENTRYPOINT module "
                                  "is outside what an import graph can "
                                  "decide, so whether it is the production "
                                  "path cannot be settled from the tree "
                                  "alone."),
    }
    Path(args.output).write_text(json.dumps(report, indent=2,
                                            sort_keys=True) + "\n")

    print(f"modules indexed: {len(index)} | reachable from live: {len(live)}")
    print(f"authority surfaces: {counts['LIVE']} LIVE, "
          f"{counts['SERVICE-ENTRYPOINT']} SERVICE-ENTRYPOINT, "
          f"{counts['TEST-ONLY']} TEST-ONLY, {counts['ORPHAN']} ORPHAN")
    print("\nLIVE authority modules (the control plane):")
    for s in surfaces:
        if s["state"] == "LIVE":
            print(f"  {s['module']:52} {s['lines']:>5} lines")
    print("\nSERVICE-ENTRYPOINT authority modules (operator-launched, "
          "invisible to an import graph):")
    for s_ in surfaces:
        if s_["state"] == "SERVICE-ENTRYPOINT":
            print(f"  {s_['module']:52} {s_['lines']:>5} lines")
    print("\nORPHAN authority modules (not reachable from any live path):")
    for s in surfaces:
        if s["state"] == "ORPHAN":
            print(f"  {s['module']:52} {s['lines']:>5} lines")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
