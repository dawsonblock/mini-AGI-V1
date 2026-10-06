import pytest
from egai.common.canonical import digest
from minagi.v161.dataset_manifest import DatasetMember, DatasetMembershipManifest, DatasetPartitionSet

def m(i,f): return DatasetMember(i,f,digest({'i':i,'f':f}))
def test_real_disjointness_passes():
    p=DatasetPartitionSet(DatasetMembershipManifest('train',(m('t','ft'),)),DatasetMembershipManifest('val',(m('v','fv'),)),DatasetMembershipManifest('hidden',(m('h','fh'),)))
    assert p.proof()['verified'] is True

def test_family_leakage_fails():
    with pytest.raises(PermissionError):
        DatasetPartitionSet(DatasetMembershipManifest('train',(m('t','same'),)),DatasetMembershipManifest('val',(m('v','fv'),)),DatasetMembershipManifest('hidden',(m('h','same'),)))
