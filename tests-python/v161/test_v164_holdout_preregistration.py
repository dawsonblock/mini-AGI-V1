"""v16.4.0 batch 2 — Campaign 3A holdout sealing + preregistration.

The final holdout is the confirmation partition the worker never sees:
the plan binds only its manifest digest, the file stays with the
evaluation authority, and the qualifier re-derives the digest from the
supplied file. These tests cover the generator (in-repo refusal,
determinism, vocabulary/ID/family disjointness), the sealer round-trip,
and the campaign-3 preregistration invariants.

The digest re-derivation test needs the authority-held file (outside
this repository); it skips on hosts that do not hold it, while the
binding-format and corpus-cleanliness checks always run.
"""
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))

from egai.common.canonical import sha256_bytes  # noqa: E402
from minagi.v161.dataset_manifest import (DatasetMember,  # noqa: E402
                                          DatasetMembershipManifest)

GEN = ROOT / "scripts" / "generate_holdout.py"
SEAL = ROOT / "scripts" / "seal_final_holdout.py"
CAMPAIGN3_CORPUS = ROOT / "configs" / "campaign3_tasks.jsonl"
HOLDOUT_DEFAULT = (Path.home() / ".local" / "share" / "minagi" /
                   "holdouts" / "campaign3a-holdout.jsonl")

_spec = importlib.util.spec_from_file_location("seal_final_holdout", SEAL)
_sealer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sealer)

DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _write_corpus(path: Path, words, family="tr-reverse"):
    rows = [{"id": f"{family}-{i:02d}", "family": family, "split": "train",
             "prompt": f"Task: Reverse the characters of the input word."
                       f"\nInput: {w}\nAnswer:",
             "expected": w[::-1]} for i, w in enumerate(words)]
    path.write_text("\n".join(json.dumps(r, sort_keys=True)
                              for r in rows) + "\n")
    return rows


def _run_gen(corpus, out, *, seed=7, families=2, per_family=3, extra=()):
    return subprocess.run(
        [sys.executable, str(GEN), "--corpus", str(corpus),
         "--out", str(out), "--seed", str(seed),
         "--families", str(families), "--per-family", str(per_family),
         *extra], capture_output=True, text=True)


def _input_words(rows):
    out = set()
    for r in rows:
        m = re.search(r"Input: (\S+)", str(r.get("prompt", "")))
        if m:
            out.add(m.group(1))
    return out


# ---------- generator ----------------------------------------------------

def test_generator_refuses_in_repo_output(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    _write_corpus(corpus, ["cobalt", "amber"])
    sneaky = ROOT / "configs" / "sneaky-holdout.jsonl"
    proc = _run_gen(corpus, sneaky)
    assert proc.returncode == 2
    assert "refusing" in proc.stderr
    assert not sneaky.exists()


def test_generator_deterministic_shape_and_meta(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus_rows = _write_corpus(corpus, ["cobalt", "amber"])
    out1, out2 = tmp_path / "h1.jsonl", tmp_path / "h2.jsonl"
    for out in (out1, out2):
        proc = _run_gen(corpus, out)
        assert proc.returncode == 0, proc.stderr
    assert out1.read_bytes() == out2.read_bytes()  # deterministic
    rows = [json.loads(l) for l in out1.read_text().splitlines()]
    assert len(rows) == 6
    assert all(r["split"] == "final_holdout" for r in rows)
    assert all(r["family"].startswith("holdout-") for r in rows)
    assert not (_input_words(rows) & _input_words(corpus_rows))
    assert not ({r["id"] for r in rows} &
                {r["id"] for r in corpus_rows})
    assert not ({r["family"] for r in rows} &
                {r["family"] for r in corpus_rows})
    meta = json.loads((tmp_path / "h1.jsonl.meta.json").read_text())
    assert meta["seed"] == 7 and meta["n_rows"] == 6
    # different seed -> different draw (fresh words), still valid shape
    proc = _run_gen(corpus, tmp_path / "h3.jsonl", seed=8)
    assert proc.returncode == 0
    assert (tmp_path / "h3.jsonl").read_bytes() != out1.read_bytes()


def test_generator_filters_corpus_vocabulary(tmp_path):
    """Corpus inputs drawn from the holdout pool must not reappear."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from generate_holdout import HOLDOUT_WORDS  # noqa: E402
    corpus = tmp_path / "corpus.jsonl"
    _write_corpus(corpus, list(HOLDOUT_WORDS[:5]))
    out = tmp_path / "h.jsonl"
    proc = _run_gen(corpus, out, families=2, per_family=3)
    assert proc.returncode == 0, proc.stderr
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert not (_input_words(rows) & set(HOLDOUT_WORDS[:5]))


def test_generator_rejects_unknown_rule(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    _write_corpus(corpus, ["cobalt"])
    proc = _run_gen(corpus, tmp_path / "h.jsonl",
                    extra=("--rules", "hid-nonsense"))
    assert proc.returncode == 1
    assert "unknown rules" in proc.stderr


# ---------- sealer round-trip -------------------------------------------

def _seal(holdout, corpus):
    return subprocess.run(
        [sys.executable, str(SEAL), str(holdout), str(corpus)],
        capture_output=True, text=True)


def test_sealer_accepts_generated_holdout(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    _write_corpus(corpus, ["cobalt", "amber"])
    out = tmp_path / "h.jsonl"
    assert _run_gen(corpus, out).returncode == 0
    proc = _seal(out, corpus)
    assert proc.returncode == 0, proc.stderr
    sealed = json.loads(proc.stdout)["final_holdout_digest"]
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    recomputed = DatasetMembershipManifest(
        "final_holdout",
        tuple(_sealer._member(r) for r in rows)).digest
    assert sealed == recomputed


def test_sealer_rejects_leaky_holdout(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus_rows = _write_corpus(corpus, ["cobalt", "amber"])
    leaky = tmp_path / "leaky.jsonl"
    leaky.write_text(json.dumps(dict(corpus_rows[0], split="final_holdout"),
                                sort_keys=True) + "\n")
    proc = _seal(leaky, corpus)
    assert proc.returncode == 1
    assert "leakage" in proc.stderr


# ---------- campaign-3 preregistration invariants ------------------------

def _config(name):
    return yaml.safe_load((ROOT / "configs" / name).read_text())


def _corpus_rows():
    return [json.loads(l) for l in CAMPAIGN3_CORPUS.read_text().splitlines()
            if l.strip()]


def test_campaign3a_binds_a_sealed_holdout_digest():
    digest = str(_config("campaign3a.yaml").get("final_holdout_digest") or "")
    assert DIGEST_RE.fullmatch(digest), \
        "campaign3a must bind a sealed final_holdout_digest before signing"


def test_corpus_contains_no_holdout_rows():
    rows = _corpus_rows()
    assert rows, "campaign3 corpus missing"
    assert not any(r["split"] == "final_holdout" for r in rows), \
        "the sealed holdout must never live inside the worker corpus"


def test_campaign3bc_holdout_invariants():
    b = str(_config("campaign3b.yaml").get("final_holdout_digest") or "")
    c = str(_config("campaign3c.yaml").get("final_holdout_digest") or "")
    for name, d in (("3b", b), ("3c", c)):
        assert d == "" or DIGEST_RE.fullmatch(d), f"{name} digest malformed"
    if b and c:
        assert b == c, "3C must use the same sealed holdout as 3B"


def test_campaign3a_holdout_rederives_from_authority_file():
    holdout = Path(os.environ.get("MINIAGI_HOLDOUT_PATH",
                                  HOLDOUT_DEFAULT)).expanduser()
    if not holdout.is_file():
        pytest.skip("authority-held holdout file not present on this host "
                    "(sealed digest binding is still enforced above)")
    digest = str(_config("campaign3a.yaml")["final_holdout_digest"])
    rows = [json.loads(l) for l in holdout.read_text().splitlines()
            if l.strip()]
    man = DatasetMembershipManifest(
        "final_holdout", tuple(_sealer._member(r) for r in rows))
    assert man.digest == digest, \
        "bound digest does not re-derive from the authority-held file"
    corpus = _corpus_rows()
    assert not ({r["id"] for r in rows} & {r["id"] for r in corpus})
    assert not ({r["family"] for r in rows} & {r["family"] for r in corpus})
