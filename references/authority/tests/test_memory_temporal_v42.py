import time
import pytest
from minagi.memory import EpisodicMemory


def test_future_valid_memory_is_not_retrieved_early(tmp_path):
    now=time.time()
    mem=EpisodicMemory(str(tmp_path/'m.db'))
    eid=mem.append('user','Project Helios launches after winter','test',valid_from=now+1000)
    assert all(h['id'] != eid for h in mem.search('Helios winter',limit=5,as_of=now))
    assert any(h['id'] == eid for h in mem.search('Helios winter',limit=5,as_of=now+2000))


def test_temporal_reference_must_exist(tmp_path):
    mem=EpisodicMemory(str(tmp_path/'m.db'))
    with pytest.raises(KeyError):
        mem.append('user','replacement','test',supersedes=999)
