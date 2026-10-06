from egai.common.crypto import Ed25519Signer, Ed25519Verifier
from kvcontinual.continual.experience import Episode
from kvcontinual.continual.experience_store import ExperienceStore
from kvcontinual.continual.dataset_export import export_mlx_dataset
from minagi.v14.verification import EpisodeVerificationAuthority, EpisodeVerificationValidator


def _verify(store, e, importance=0.9):
    signer = Ed25519Signer.generate("episode-verifier")
    verifier = Ed25519Verifier(); verifier.register(signer.key_id, signer.public_bytes())
    receipt = EpisodeVerificationAuthority(verifier_id="independent-test-verifier", signer=signer).issue(
        episode_id=e.id, prompt=e.prompt, attempted_output=e.response, repaired_output=e.response,
        evidence_root_digest="sha256:" + "a" * 64, score=1.0, importance=importance,
    )
    validator = EpisodeVerificationValidator(verifier=verifier, trusted_key_ids={signer.key_id})
    store.install_verification_receipt(receipt, validator)


def test_experience_persistence_and_export_requires_signed_receipt(tmp_path):
    db = tmp_path / "exp.sqlite3"
    store = ExperienceStore(str(db))
    e = Episode(prompt="new fact?", response="answer")
    store.append(e, importance=0.9)
    assert store.list_training_ready() == []
    try:
        store.mark_verified(e.id, importance=0.9)
    except PermissionError:
        pass
    else:
        raise AssertionError("boolean verification remained an authority path")
    _verify(store, e)
    ready = store.list_training_ready()
    assert len(ready) == 1 and ready[0].id == e.id
    train, digest, ids = export_mlx_dataset(store, str(tmp_path / "data"))
    assert train.is_file() and digest.startswith("sha256:") and ids == [e.id]
    assert store.list_training_ready() == []
    reopened = ExperienceStore(str(db))
    assert reopened.conn.execute("select count(*) from episodes").fetchone()[0] == 1
    assert reopened.verification_receipt_digest(e.id).startswith("sha256:")
