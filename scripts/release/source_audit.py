#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
DEVELOPER_PATH = re.compile(r"/(?:home|data)/chaidi(?:/|\\b)")
RELEASE_CRITICAL = [
    ROOT / "src",
    ROOT / "include",
    ROOT / "tests",
    ROOT / "benchmark" / "mini_swe_deepswe",
]
RELEASE_SCRIPTS = [
    ROOT / "scripts" / "kvmem_eval" / name
    for name in (
        "dataset.py",
        "judge.py",
        "prompt.py",
        "run_eval.py",
        "run_kvmem_eval.py",
        "run_memoryagentbench_baselines.py",
    )
]


def iter_text_files(path: Path):
    if path.is_file():
        yield path
        return
    for item in path.rglob("*"):
        if item.is_file() and item.suffix not in {".png", ".jpg", ".jpeg", ".gif", ".pdf"}:
            yield item


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    problems: list[str] = []
    for required in (
        ROOT / "LICENSE",
        ROOT / "THIRD_PARTY_NOTICES.md",
        ROOT / "release" / "SOURCE_PROVENANCE.json",
        ROOT / "release" / "BUILD_PROFILE.env",
        ROOT / "release" / "KVMEM_PROFILES.json",
        ROOT / "schemas" / "kvmem-memory-receipt.schema.json",
        ROOT / "schemas" / "kvmem-session-receipt.schema.json",
        ROOT / "schemas" / "kvmem-session-snapshot.schema.json",
        ROOT / "schemas" / "kvmem-executor-scheduler.schema.json",
        ROOT / "schemas" / "kvmem-resource-admission.schema.json",
        ROOT / "schemas" / "kvmem-executor-pool.schema.json",
        ROOT / "schemas" / "kvmem-physical-executor-pool.schema.json",
        ROOT / "docs" / "RC9_NATIVE_POOL_WIRING.md",
        ROOT / "release" / "RC9_QUALIFICATION.json",
        ROOT / "release" / "RC9_1_QUALIFICATION.json",
        ROOT / "release" / "RC9_1_VALIDATION_REPORT.md",
        ROOT / "docs" / "RC9_1_CORRECTNESS_HARDENING.md",
        ROOT / "release" / "RC9_2_QUALIFICATION.json",
        ROOT / "release" / "RC9_2_VALIDATION_REPORT.md",
        ROOT / "docs" / "RC9_2_FULL_HARDENING.md",
        ROOT / "docs" / "RC10_COHERENT_CONTINUAL_MERGE.md",
        ROOT / "docs" / "RC10_IMPLEMENTATION_STATUS.md",
        ROOT / "release" / "RC10_VALIDATION_REPORT.md",
        ROOT / "continual" / "pyproject.toml",
        ROOT / "continual" / "src" / "kvcontinual" / "registry.py",
        ROOT / "continual" / "src" / "kvcontinual" / "cache" / "identity.py",
        ROOT / "docs" / "kvmem_runtime_profiles.md",
        ROOT / "docs" / "kvmem_session_manager.md",
        ROOT / "docs" / "kvmem_executor_scheduler.md",
        ROOT / "docs" / "kvmem_resource_admission.md",
        ROOT / "docs" / "kvmem_executor_runtime.md",
        ROOT / "docs" / "kvmem_session_snapshots.md",
        ROOT / "tests" / "kvmem_exactmass_largeblocks.cu",
    ):
        if not required.is_file():
            problems.append(f"missing required release file: {required.relative_to(ROOT)}")

    for base in [*RELEASE_CRITICAL, *RELEASE_SCRIPTS]:
        for path in iter_text_files(base):
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if DEVELOPER_PATH.search(text):
                problems.append(
                    f"developer-specific absolute path remains in release-critical file: {path.relative_to(ROOT)}"
                )

    # Never package local secrets or model weights. Prune generated trees at
    # traversal time so a local build cannot make the release audit unbounded.
    forbidden_suffixes = {".pem", ".p12", ".gguf", ".safetensors", ".ckpt", ".pt", ".pth"}
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [
            d for d in dirnames
            if d not in {".git", "__pycache__", ".pytest_cache"} and not d.startswith("build")
        ]
        base = Path(dirpath)
        for name in filenames:
            path = base / name
            if path.suffix.lower() in forbidden_suffixes:
                problems.append(f"forbidden local artifact in source release: {path.relative_to(ROOT)}")

    provenance = ROOT / "release" / "SOURCE_PROVENANCE.json"
    if provenance.is_file():
        try:
            info = json.loads(provenance.read_text())
            digest = info.get("source_archive_sha256", "")
            parent_digest = info.get("parent_hardening_artifact_sha256", "")
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                problems.append("SOURCE_PROVENANCE.json has an invalid source_archive_sha256")
            if not re.fullmatch(r"[0-9a-f]{64}", parent_digest):
                problems.append("SOURCE_PROVENANCE.json has an invalid parent_hardening_artifact_sha256")
            if info.get("release_name") != "kvmem-qw3-coherent-continual-rc10":
                problems.append("SOURCE_PROVENANCE.json release_name is not coherent-continual rc10")
        except Exception as exc:
            problems.append(f"invalid SOURCE_PROVENANCE.json: {exc}")

    qualification = ROOT / "release" / "QUALIFICATION.json"
    if qualification.is_file():
        try:
            info = json.loads(qualification.read_text())
            if info.get("release") != "kvmem-qw3-coherent-continual-rc10":
                problems.append("QUALIFICATION.json release is not coherent-continual rc10")
            if info.get("session_architecture", {}).get("physical_executor_pool_scope") != "persistent-session-runtime":
                problems.append("QUALIFICATION.json physical executor pool scope is missing or stale")
        except Exception as exc:
            problems.append(f"invalid QUALIFICATION.json: {exc}")

    release_readme = ROOT / "release" / "README.md"
    if release_readme.is_file() and not release_readme.read_text(encoding="utf-8").startswith(
            "# Coherent Hybrid Continual Memory RC10 release notes"):
        problems.append("release/README.md is not the RC10 canonical release note")

    build_profile = ROOT / "release" / "BUILD_PROFILE.env"
    if build_profile.is_file() and "RC10" not in build_profile.read_text(encoding="utf-8").splitlines()[0]:
        problems.append("release/BUILD_PROFILE.env is not labeled RC10")

    paper = ROOT / "docs" / "KV_Memory_Paper.md"
    if paper.is_file():
        paper_text = paper.read_text(encoding="utf-8")
        if "mean_attention | content_mean" in paper_text:
            problems.append("KV_Memory_Paper.md still advertises removed retrieval method names")


    for schema_name in ("kvmem-executor-pool.schema.json",
                        "kvmem-physical-executor-pool.schema.json"):
        schema_path = ROOT / "schemas" / schema_name
        if schema_path.is_file():
            try:
                schema = json.loads(schema_path.read_text())
                if schema.get("properties", {}).get("scope", {}).get("const") != "persistent-session-runtime":
                    problems.append(f"{schema_name} does not pin the persistent-session-runtime scope")
                if "scope" not in schema.get("required", []):
                    problems.append(f"{schema_name} does not require scope")
            except Exception as exc:
                problems.append(f"invalid {schema_name}: {exc}")

    runtime_doc = ROOT / "docs" / "kvmem_executor_runtime.md"
    if runtime_doc.is_file() and not runtime_doc.read_text(encoding="utf-8").startswith(
            "# RC9.2 physical executor runtime boundary"):
        problems.append("kvmem_executor_runtime.md is not labeled RC9.2")

    if problems:
        print("release source audit: FAIL", file=sys.stderr)
        for problem in problems:
            print(f" - {problem}", file=sys.stderr)
        return 1
    print("release source audit: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
