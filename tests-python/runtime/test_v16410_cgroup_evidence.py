"""v16.4.10 SEC-901 — cgroup occupancy evidence is strictly fail closed.

The kernel's recursive occupancy flag lives in ``cgroup.events``'s
``populated`` field with EXACTLY two valid values. Every other shape —
malformed lines, duplicate keys, unexpected values, missing files,
permission failures, and bare ``cgroup.procs`` emptiness — is Unknown,
and Unknown is never proof of emptiness.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))

from minagi.runtime.worker_launch_broker import (  # noqa: E402
    ContainmentState, cgroup_populated, evaluate_containment)


def _cg(tmp_path, events: str | None = "populated 1\n",
        procs: str | None = None):
    d = tmp_path / "cg"
    d.mkdir(parents=True)
    if events is not None:
        (d / "cgroup.events").write_text(events)
    if procs is not None:
        (d / "cgroup.procs").write_text(procs)
    return d


# ---- the six-row evidence table --------------------------------------

def test_populated_one_is_occupied(tmp_path):
    assert cgroup_populated(_cg(tmp_path, "populated 1\n")) is True


def test_populated_zero_is_empty(tmp_path):
    assert cgroup_populated(_cg(tmp_path, "populated 0\n")) is False


def test_unexpected_populated_value_is_unknown(tmp_path):
    for i, v in enumerate(("2", "-1", "yes", "", "01", "1x")):
        assert cgroup_populated(
            _cg(tmp_path / str(i), f"populated {v}\n")) is None, v


def test_malformed_lines_are_unknown(tmp_path):
    # anything but exactly 'key value' per line is not kernel output
    for i, bad in enumerate(("populated 1 extra\n", "populated\n",
                            "populated 1\nx\n",
                            "populated  1\nfoo\n\n\n")):
        assert cgroup_populated(_cg(tmp_path / str(i), bad)) is None, \
            bad


def test_duplicate_fields_are_unknown(tmp_path):
    assert cgroup_populated(_cg(
        tmp_path, "populated 0\npopulated 1\n")) is None


def test_missing_events_file_is_unknown(tmp_path):
    assert cgroup_populated(_cg(tmp_path, None)) is None


def test_empty_procs_is_not_evidence(tmp_path):
    # an empty cgroup.procs WITHOUT authoritative events is Unknown —
    # the removed-fallback that certified emptiness is gone
    assert cgroup_populated(
        _cg(tmp_path / "a", None, procs="")) is None
    assert cgroup_populated(
        _cg(tmp_path / "b", None, procs="1234\n")) is None


def test_unreadable_events_is_unknown(tmp_path):
    # a directory where the file belongs makes read_text raise an
    # OSError under EVERY uid — including root, which bypasses mode
    # bits — so this reproduces an unreadable cgroup reliably
    d = _cg(tmp_path, "populated 1\n")
    (d / "cgroup.events").unlink()
    (d / "cgroup.events").mkdir()
    assert cgroup_populated(d) is None


def test_missing_cgroup_dir_is_unknown(tmp_path):
    assert cgroup_populated(tmp_path / "nonexistent") is None


# ---- the shared decision rule (SEC-902 pairing) -----------------------

def test_evaluator_populated_one_wins_over_inactive_unit():
    # the audit's exact case: unit inactive + client exited while the
    # cgroup still reports populated 1 -> RUNNING, never empty
    state = evaluate_containment(
        populated=True, cgroup_gone=False, unit_active=False,
        unit_same=True)
    assert state == ContainmentState.RUNNING


def test_evaluator_populated_zero_is_empty():
    assert evaluate_containment(
        populated=False, cgroup_gone=False, unit_active=False,
        unit_same=True) == ContainmentState.EMPTY


def test_evaluator_gone_requires_same_invocation():
    # cgroup removed + unit inactive + the SAME invocation -> removed
    assert evaluate_containment(
        populated=None, cgroup_gone=True, unit_active=False,
        unit_same=True) == ContainmentState.REMOVED_VERIFIED
    # ... but a REUSED unit name (different invocation) is not the
    # same teardown — unknown, never verified
    assert evaluate_containment(
        populated=None, cgroup_gone=True, unit_active=False,
        unit_same=False) == ContainmentState.UNKNOWN
    # an ACTIVE unit (a new invocation under the old name) is also
    # not this worker's teardown
    assert evaluate_containment(
        populated=None, cgroup_gone=True, unit_active=True,
        unit_same=True) == ContainmentState.UNKNOWN


def test_evaluator_unknown_for_everything_else():
    for populated in (None,):
        for gone in (False, True):
            for active in (False, True):
                for same in (False, True):
                    st = evaluate_containment(
                        populated=populated, cgroup_gone=gone,
                        unit_active=active, unit_same=same)
                    if gone and not active and same:
                        continue  # the one verified-removal case
                    assert st == ContainmentState.UNKNOWN, (
                        populated, gone, active, same)
