from dataclasses import replace
import math
import pytest
from chain_helpers import prepare_chain
from minagi.egai import PromotionAuthority, ImprovementLedger, PromotionVerdict
from kvcontinual.execution.authority import Ed25519ReceiptSigner
from egai.common.trust import AuthorityTrust
from egai.authority.model import RawResult
from egai.common.canonical import digest
from minagi.egai.models import LearningLevel,LearningProposal,LearningAction
from minagi.egai.qualification import QualificationBundle,QualificationMetrics,IndependentQualificationGate


def inputs():
    from minagi.egai.proposals import LeastPermanentPlasticityProposer,PlasticityDecisionContext
    from minagi.egai.models import ProposalOption
    p=LeastPermanentPlasticityProposer(proposer_id='test').propose(
        options=[ProposalOption(LearningAction.CREATE_PROCEDURE,LearningLevel.L3_PROCEDURAL_SKILL,.2,.01,reason='verified')],
        target='repair',evidence_digests=[digest('evidence')],production_identity_digest=digest('identity'),
        context=PlasticityDecisionContext(.5,.01,1.,.2,.01,{}))
    m=QualificationMetrics(.1,0.,.01,0.,0,0,1.,.08)
    b=QualificationBundle(p.digest,digest('candidate'),p.production_identity_digest,'independent-q',8,
        ('episode','procedure'),True,True,m,digest('suite'))
    return p,b


def test_unsigned_bundle_never_authorizes(tmp_path):
    p,b=inputs(); authority=PromotionAuthority(gate=IndependentQualificationGate(),
        improvement_ledger=ImprovementLedger(tmp_path/'ledger'),signer=Ed25519ReceiptSigner.generate())
    d,receipt,_=authority.decide(proposal=p,bundle=b,origin_evidence=p.evidence_digests,
        affected_components=('repair',),production_identity_before=p.production_identity_digest)
    assert d.verdict is PromotionVerdict.REJECT and receipt is None


def test_real_chain_authorizes_once_and_rejects_reuse(tmp_path):
    p,b=inputs();g,b,c,_,_=prepare_chain(tmp_path,p,b)
    authority=PromotionAuthority(gate=g,improvement_ledger=ImprovementLedger(tmp_path/'ledger'),signer=Ed25519ReceiptSigner.generate())
    args=dict(proposal=p,bundle=b,research_chain=c,origin_evidence=p.evidence_digests,
        affected_components=('repair',),production_identity_before=p.production_identity_digest)
    assert authority.decide(**args)[1] is not None
    with pytest.raises(PermissionError,match='already consumed'):authority.decide(**args)


@pytest.mark.parametrize('field',['evaluation','results','benchmark','qualifier_binding','evaluator_binding'])
def test_chain_signature_tampering_rejected(tmp_path,field):
    p,b=inputs();g,b,c,_,_=prepare_chain(tmp_path,p,b)
    obj=getattr(c,field)
    bad=replace(obj,signature_b64='invalid')
    assert g.evaluate(p,b,replace(c,**{field:bad})).verdict is PromotionVerdict.REJECT


def test_role_revocation_rejects_previously_valid_chain(tmp_path):
    p,b=inputs();g,b,c,cv,_=prepare_chain(tmp_path,p,b)
    assert g.evaluate(p,b,c).verdict is PromotionVerdict.APPROVE
    cv.trust=AuthorityTrust(cv.trust.roles,revoked_keys=frozenset([c.evaluation.evaluator_key_id]))
    assert g.evaluate(p,b,c).verdict is PromotionVerdict.REJECT


def test_binding_to_another_candidate_rejected(tmp_path):
    p,b=inputs();g,b,c,_,_=prepare_chain(tmp_path,p,b)
    assert g.evaluate(p,replace(b,candidate_digest=digest('another')),c).verdict is PromotionVerdict.REJECT


@pytest.mark.parametrize('value',[math.nan,math.inf,-1.])
def test_nonfinite_or_negative_runner_metrics_rejected(value):
    with pytest.raises(ValueError):RawResult('case','future',0.,value)
