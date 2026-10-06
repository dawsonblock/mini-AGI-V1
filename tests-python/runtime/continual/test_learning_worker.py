import pytest
from egai.common.crypto import Ed25519Signer, Ed25519Verifier
from kvcontinual.continual.experience import Episode
from kvcontinual.continual.experience_store import ExperienceStore
from kvcontinual.continual.dataset_export import export_mlx_dataset
from kvcontinual.continual.registry import AdapterRegistry
from kvcontinual.continual.qualification import QualificationPolicy
from minagi.v14.verification import EpisodeVerificationAuthority, EpisodeVerificationValidator


def test_learning_worker_can_build_but_cannot_self_promote(tmp_path):
    store = ExperienceStore(str(tmp_path / "exp.sqlite3"))
    e = Episode(prompt="p", response="r"); store.append(e)
    signer=Ed25519Signer.generate("verifier"); verifier=Ed25519Verifier(); verifier.register(signer.key_id,signer.public_bytes())
    receipt=EpisodeVerificationAuthority(verifier_id="v",signer=signer).issue(
        episode_id=e.id,prompt=e.prompt,attempted_output=e.response,repaired_output=e.response,
        evidence_root_digest="sha256:"+"b"*64,score=1.0)
    store.install_verification_receipt(receipt,EpisodeVerificationValidator(verifier=verifier,trusted_key_ids={signer.key_id}))
    train, dataset_digest, ids = export_mlx_dataset(store, str(tmp_path / "data"))
    run = store.create_learning_run(dataset_digest, len(ids), {"train": str(train)})
    adapter = tmp_path / "adapter.safetensors"; adapter.write_bytes(b"adapter-v1")
    reg = AdapterRegistry(str(tmp_path / "registry"))
    m = reg.register_candidate(str(adapter), "sha256:base", dataset_digest, {"lr": 1e-5})
    store.bind_candidate(run, m.candidate_id)
    q = QualificationPolicy().evaluate(m.candidate_id, 0.05, 0.0, True, True)
    with pytest.raises(RuntimeError, match="qualification bundle"):
        reg.write_qualification(q)
    with pytest.raises(RuntimeError, match="no qualification record"):
        reg.promote(m.candidate_id)
