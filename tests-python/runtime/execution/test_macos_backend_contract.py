import numpy as np
from kvcontinual.execution.recurrent.affine import AffineSummary
from kvcontinual.execution.recurrent.interfaces import BackendCapabilities, backend_capabilities
from kvcontinual.execution.recurrent.macos_affine import MacAffineExecutor


def test_macos_affine_numpy_fallback_matches_reference():
    ex = MacAffineExecutor(prefer_mlx=False)
    s = AffineSummary(np.eye(2) * .9, np.eye(2) * .1)
    state = np.arange(4, dtype=np.float64).reshape(2,2)
    assert np.allclose(ex.apply(s, state), s.apply(state))
    assert ex.info.name == "numpy"


def test_legacy_backend_capabilities_fail_closed_for_acceleration():
    class Legacy: pass
    caps = backend_capabilities(Legacy())
    assert not caps.hypic_seam8
    assert not caps.exact_prefix_checkpoint
    assert not caps.causal_conv_seam_repair
    assert not caps.full_attention_relocation
    assert caps.exact_selected_replay


def test_explicit_exact_only_capabilities():
    class ExactOnly:
        def capabilities(self):
            return BackendCapabilities(hypic_seam8=False, single_state_init=False, exact_prefix_checkpoint=False, device="mps")
    caps = backend_capabilities(ExactOnly())
    assert not caps.hypic_seam8
    assert not caps.exact_prefix_checkpoint
    assert caps.device == "mps"
