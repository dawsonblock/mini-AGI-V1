def test_rc14_donor_imports():
    from minagi.rc14.executed_runs import DatasetSplitManifestRC14
    from minagi.rc14.independent_reproduction import IndependentReproductionCertificateRC14
    assert DatasetSplitManifestRC14 and IndependentReproductionCertificateRC14

def test_v16_selected_imports():
    from minagi.v16.plasticity_execution import PlasticityExecutionHarnessV160
    from minagi.v16.isolated_solver import IsolatedJSONSolverV160
    assert PlasticityExecutionHarnessV160 and IsolatedJSONSolverV160

def test_dream_rsi_imports():
    from dream_rsi_governed.policy import ExplorationPolicy
    assert ExplorationPolicy

def test_grounded_replay_imports():
    from egai.authority.replay_policy import GroundedReplayPolicyAuthority
    from egai.authority.replay_canary import ReplayLiveCanaryAuthority
    assert GroundedReplayPolicyAuthority and ReplayLiveCanaryAuthority
