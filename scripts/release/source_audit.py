#!/usr/bin/env python3
"""Mini-AGI release source audit.

Gate that runs before a release build: the release-identity files must
exist and agree, and the source tree must not carry private-key
material, model-weight artifacts, or developer-specific absolute paths.
A gate that can never pass is worse than no gate — every check here is
about THIS tree.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]

# Absolute user-home paths baked into release source (e.g. a developer's
# machine layout). Matches /Users/name/... and /home/name/...
DEVELOPER_PATH = re.compile(r"(?:/Users/|/home/)[A-Za-z0-9._-]+/")

PRIVATE_KEY_MATERIAL = re.compile(
    rb"-----BEGIN [A-Z ]*PRIVATE KEY")

RELEASE_REQUIRED = [
    ROOT / "VERSION",
    ROOT / "LICENSE",
    ROOT / "pyproject.toml",
    ROOT / "SBOM.cdx.json",
    ROOT / "SOURCE_MANIFEST.json",
    ROOT / "RELEASE_SIGNATURE.bin",
    ROOT / "RELEASE_PUBLIC_KEY.pem",
    ROOT / "RELEASE_CHANGE_MANIFEST.json",
    ROOT / "RELEASE_VALIDATION.json",
    ROOT / "RELEASE_ATTESTATION.json",
    ROOT / "CMakeLists.txt",
    ROOT / "src-python" / "minagi" / "__init__.py",
    ROOT / "src-python" / "minagi" / "runtime" / "service.py",
    ROOT / "src-python" / "minagi" / "runtime" / "worker_isolation.py",
    ROOT / "src-python" / "minagi" / "runtime" / "worker_backend.py",
    ROOT / "src-python" / "minagi" / "runtime" / "worker_protocol.py",
    ROOT / "docs" / "research" / "V16_4_6_REPAIR_REPORT.md",
    ROOT / "docs" / "research" / "WORKER_ISOLATION_SPEC_V1646.md",
    ROOT / "docs" / "research" / "REMAINING_DEFECTS_V1646.md",
    ROOT / "configs" / "minagi-worker-seatbelt.sb",
]

# Files whose text gets scanned for developer paths. Scoped to the
# governed runtime + release tooling: scripts/kvmem_eval/ and
# references/ are donor research archives shipped for completeness —
# known-legacy content, not release-critical.
RELEASE_CRITICAL = [
    ROOT / "src-python" / "minagi",
    ROOT / "scripts" / "release",
    ROOT / "scripts" / "validation",
    ROOT / "configs",
    ROOT / "tools",
]

SELF = Path(__file__).resolve()

ALLOWED_PEM = {str(ROOT / "RELEASE_PUBLIC_KEY.pem")}
FORBIDDEN_SUFFIXES = {
    ".pem", ".p12", ".key",
    ".gguf", ".safetensors", ".ckpt", ".pt", ".pth",
}
PRUNE_DIRS = {
    ".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "node_modules", "dist", "dist-v1646", "dist-v1645",
}
PRUNE_PREFIXES = ("build", ".venv", "venv")


def iter_text_files(path: Path):
    if path.is_file():
        yield path
        return
    for item in path.rglob("*"):
        if item.is_file() and item.suffix not in {
                ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip"}:
            yield item


def _version_identities() -> list[str]:
    """VERSION, pyproject and the package __version__ must agree — the
    v16.4.5 release shipped one stale site and only a Colab run saw it."""
    problems = []
    try:
        version = (ROOT / "VERSION").read_text().strip()
    except OSError:
        return ["cannot read VERSION"]
    # VERSION carries the full release name (e.g.
    # "16.4.6-worker-isolation-and-process-closure"); the package
    # identities carry the semver prefix — same rule as
    # test_version_identities_agree.
    number = version.split("-", 1)[0]
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.M)
    if not m or m.group(1) != number:
        problems.append(
            f"pyproject.toml version {m.group(1) if m else '?'} "
            f"!= VERSION number {number}")
    init = (ROOT / "src-python" / "minagi" / "__init__.py") \
        .read_text(encoding="utf-8")
    m = re.search(r'__version__\s*=\s*"([^"]+)"', init)
    if not m or m.group(1) != number:
        problems.append(
            f"minagi.__version__ {m.group(1) if m else '?'} "
            f"!= VERSION number {number}")
    sbom = json.loads((ROOT / "SBOM.cdx.json").read_text(encoding="utf-8"))
    if sbom["metadata"]["component"]["version"] != number:
        problems.append(
            f"SBOM version {sbom['metadata']['component']['version']} "
            f"!= VERSION number {number}")
    return problems


def _manifest_coherence() -> list[str]:
    """The shipped manifest must parse and reference this tree."""
    problems = []
    manifest = ROOT / "SOURCE_MANIFEST.json"
    if not manifest.is_file():
        return ["SOURCE_MANIFEST.json missing"]
    try:
        doc = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"SOURCE_MANIFEST.json does not parse: {exc}"]
    files = doc.get("files")
    if not isinstance(files, dict) or len(files) < 100:
        problems.append(
            "SOURCE_MANIFEST.json files map missing or suspiciously "
            f"small ({0 if not isinstance(files, dict) else len(files)} entries)")
        return problems
    if str(ROOT / "VERSION").endswith("VERSION") and \
            "VERSION" not in files:
        problems.append("SOURCE_MANIFEST.json does not list VERSION")
    return problems


def main() -> int:
    problems: list[str] = []
    for required in RELEASE_REQUIRED:
        if not required.is_file():
            problems.append(
                f"missing required release file: "
                f"{required.relative_to(ROOT)}")

    problems.extend(_version_identities())
    problems.extend(_manifest_coherence())

    for base in RELEASE_CRITICAL:
        if not base.exists():
            continue
        for path in iter_text_files(base):
            if path.resolve() == SELF:
                continue
            try:
                raw = path.read_bytes()
            except OSError:
                continue
            if PRIVATE_KEY_MATERIAL.search(raw):
                problems.append(
                    "private key material in release-critical file: "
                    f"{path.relative_to(ROOT)}")
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                continue
            if DEVELOPER_PATH.search(text):
                problems.append(
                    "developer-specific absolute path in "
                    f"release-critical file: {path.relative_to(ROOT)}")

    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [
            d for d in dirnames
            if d not in PRUNE_DIRS and not any(
                d.startswith(p) for p in PRUNE_PREFIXES)]
        base = Path(dirpath)
        for name in filenames:
            path = base / name
            if str(path) in ALLOWED_PEM:
                continue
            if path.suffix.lower() in FORBIDDEN_SUFFIXES:
                problems.append(
                    "forbidden local artifact in source release: "
                    f"{path.relative_to(ROOT)}")

    if problems:
        print("release source audit: FAIL", file=sys.stderr)
        for problem in problems:
            print(f" - {problem}", file=sys.stderr)
        return 1
    print("release source audit: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
