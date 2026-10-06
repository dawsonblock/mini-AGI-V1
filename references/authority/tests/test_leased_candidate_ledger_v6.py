from minagi.candidate_pipeline import LeasedCandidateLedger


def test_leased_candidate_ledger_roundtrip(tmp_path):
    led = LeasedCandidateLedger(tmp_path / "authority")
    cid = led.propose(base_generation="g1", source="v6")
    led.append(cid, "trained")
    led.append(cid, "evaluated")
    assert LeasedCandidateLedger(tmp_path / "authority").state(cid) == "evaluated"
