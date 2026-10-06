from dataclasses import asdict,replace
from pathlib import Path
import json
from egai.common.canonical import digest
from egai.bench.sequential_governed import GovernedSequentialExperiment
from egai.bench.repair import ReferenceRepairProvider
from egai.cognition.agent import GovernedAdaptiveAgent
from egai.common.environment import EnvironmentFingerprint
from egai.evidence.ledger import EvidenceLedger
from egai.replay.audit import ReplayAuditor
from egai.skills.learner import SynthesizedSkillMiner
from egai.skills.synthesis import PatternSkillSynthesizer
from egai.integration.learning_artifacts import LearningArtifactBuilder
from .manifest import verify_manifest
from .result import ExperimentResult
from .closure import ExperimentClosure

class ExperimentOrchestrator:
    """RC8.6 governed experiment: evidence-first learning with no dataset-authorized persistence."""
    def __init__(self,runner_id,signer,verifier,trusted_registrar_keys,journal,artifact_store,feedback_signer=None,trusted_feedback_keys=(),source_root=None,repair_provider=None,skill_synthesizer=None):
        self.runner_id=runner_id;self.signer=signer;self.verifier=verifier;self.trusted_registrar_keys=tuple(trusted_registrar_keys);self.journal=journal;self.artifact_store=artifact_store
        self.feedback_signer=feedback_signer or signer;self.trusted_feedback_keys=tuple(trusted_feedback_keys) or (self.feedback_signer.key_id,);self.source_root=Path(source_root) if source_root else None
        self.repair_provider=repair_provider;self.skill_synthesizer=skill_synthesizer or PatternSkillSynthesizer()
    def _source_digest(self):
        if not self.source_root or not self.source_root.exists():return digest({'source':'unbound-test-harness'})
        rows=[]
        for p in sorted(self.source_root.rglob('*.py')):
            if p.is_file():rows.append((str(p.relative_to(self.source_root)),digest({'text':p.read_text(encoding='utf-8')})))
        return digest(rows)
    def run(self,result_id,manifest,model,experience_cases,evaluation_cases,scorer):
        verify_manifest(manifest,self.verifier,self.trusted_registrar_keys,model.model_digest,experience_cases,evaluation_cases)
        env=EnvironmentFingerprint.capture(runtime_notes={'runner_id':self.runner_id,'backend_digest':model.backend_digest})
        source_digest=self._source_digest();self.journal.append(manifest.digest,'start',{'model_digest':model.model_digest,'backend_digest':model.backend_digest,'environment_digest':env.digest,'source_digest':source_digest,'experience_digest':manifest.experience_digest,'evaluation_digest':manifest.evaluation_digest})
        repair=self.repair_provider
        if repair is None:
            if not manifest.learner_config.get('synthetic_reference_repairs',False):raise RuntimeError('RC8.6 requires an explicit RepairProvider; reference answer repair is test-only')
            repair=ReferenceRepairProvider()
        evidence_path=self.journal.path.parent/('evidence-'+manifest.digest.split(':')[1][:16]+'.sqlite')
        ledger=EvidenceLedger(evidence_path,self.signer,self.verifier)
        agent=GovernedAdaptiveAgent(model,skill_miner=SynthesizedSkillMiner(self.skill_synthesizer,min_distinct_inputs=2))
        exp=GovernedSequentialExperiment(experience_cases,evaluation_cases,scorer,repair,ledger,self.feedback_signer,self.verifier,self.trusted_feedback_keys,checkpoints=manifest.checkpoints,bootstrap_samples=int(manifest.learner_config.get('bootstrap_samples',1000)),near_leakage_threshold=manifest.near_leakage_threshold)
        report=exp.run(model,agent,env.digest,checkpoint_callback=lambda p:self.journal.append(manifest.digest,'checkpoint',asdict(p)),verification_callback=lambda r:self.journal.append(manifest.digest,'verification_receipt',{'receipt_digest':r.digest,'case_id':r.case_id,'verifier_id':r.verifier_id}))
        ledger.checkpoint()
        if not ledger.verify_chain():raise RuntimeError('evidence ledger verification failed')
        self.journal.append(manifest.digest,'evidence_closure',{'evidence_head':report.evidence_head,'belief_snapshot_digest':report.belief_snapshot_digest,'records':len(ledger.all())})
        replay_audit=ReplayAuditor(self.signer,self.verifier).audit_traces('RA-'+result_id,report.traces)
        replay_artifact=self.artifact_store.put_bytes(json.dumps(asdict(replay_audit),sort_keys=True,separators=(',',':')).encode())
        self.journal.append(manifest.digest,'replay_audit',{'audit_digest':replay_audit.digest,'artifact_digest':replay_artifact,'grounded_coverage':replay_audit.grounded_coverage,'prefix_violations':replay_audit.prefix_violations})
        skill_blob=agent.export_skill_bundle();skill_artifact=self.artifact_store.put_bytes(skill_blob);builder=LearningArtifactBuilder(self.artifact_store);packages=[]
        for proc in agent.skills.all():
            pkg=builder.build(proc,report);pd=self.artifact_store.put_bytes(json.dumps(asdict(pkg),sort_keys=True,separators=(',',':')).encode());packages.append(pd)
            self.journal.append(manifest.digest,'learning_proposal',{'package_digest':pd,'proposal_digest':pkg.proposal_digest,'skill_manifest_digest':pkg.skill_manifest.manifest_digest,'evidence_root':pkg.evidence_root})
        health=agent.skills.health();self.journal.append(manifest.digest,'complete',{'skill_artifact':skill_artifact,'learned_procedures':report.learned_procedures,'verified_repairs':report.verified_repairs,'skill_failures':report.skill_failures,'skill_health':health,'negative_transfer_rate':report.max_negative_transfer_rate,'learning_packages':packages})
        if not self.journal.verify(manifest.digest):raise RuntimeError('experiment journal verification failed')
        head=self.journal.entries()[-1].entry_hash
        closure=ExperimentClosure(manifest.digest,model.model_digest,env.digest,skill_artifact,head,source_digest,model.backend_digest,replay_audit.digest,report.evidence_head,report.belief_snapshot_digest,tuple(packages))
        closure_artifact=self.artifact_store.put_bytes(json.dumps(asdict(closure),sort_keys=True,separators=(',',':')).encode());first,last=report.points[0],report.points[-1]
        u=ExperimentResult(result_id,manifest.digest,model.model_digest,skill_artifact,head,report.leakage_clean,report.model_digest_constant,last.gain_vs_frozen,last.ci_low,last.ci_high,max(0.,last.baseline_retention_success-last.retention_success),max(0.,last.baseline_security_success-last.security_success),report.learned_procedures,self.runner_id,closure.digest,closure_artifact,env.digest,model.backend_digest,report.verified_repairs,report.skill_failures,report.max_negative_transfer_rate,replay_audit.digest,replay_artifact,replay_audit.grounded_coverage,replay_audit.prefix_violations,last.future_n,report.degraded_skills,report.retired_skills,report.evidence_head,report.belief_snapshot_digest,tuple(packages))
        signed=self.signer.sign(asdict(u));return replace(u,runner_key_id=signed.key_id,signature_b64=signed.signature_b64),report
