from dataclasses import dataclass
from egai.common.canonical import digest,validate_digest
@dataclass(frozen=True)
class ExperimentClosure:
    experiment_manifest_digest:str;model_digest:str;environment_digest:str;skill_snapshot_digest:str;journal_head_digest:str;source_manifest_digest:str;backend_digest:str
    replay_audit_digest:str='';evidence_head_digest:str='';belief_snapshot_digest:str='';learning_package_digests:tuple[str,...]=()
    def __post_init__(self):
        for d in (self.experiment_manifest_digest,self.model_digest,self.environment_digest,self.skill_snapshot_digest,self.journal_head_digest,self.source_manifest_digest,self.backend_digest):validate_digest(d)
        for d in (self.replay_audit_digest,self.evidence_head_digest,self.belief_snapshot_digest):
            if d:validate_digest(d)
        for d in self.learning_package_digests:validate_digest(d)
    @property
    def digest(self):return digest(self)
