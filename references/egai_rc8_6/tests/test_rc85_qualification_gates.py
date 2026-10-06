import unittest
from dataclasses import asdict,replace
from egai.common.crypto import Ed25519Signer,Ed25519Verifier
from egai.experiment.result import ExperimentResult
from egai.experiment.qualify import ContinualLearningQualifier
class T(unittest.TestCase):
 def signed(self,runner,**kw):
  base=dict(result_id='r',experiment_digest='sha256:'+'1'*64,model_digest='sha256:'+'2'*64,
   final_skill_snapshot_digest='sha256:'+'3'*64,journal_head_digest='sha256:'+'4'*64,
   leakage_clean=True,model_digest_constant=True,final_future_gain=.2,final_future_ci_low=.05,final_future_ci_high=.3,
   retention_regression=0.,security_regression=0.,learned_procedures=2,runner_id='runner',
   verified_repairs=3,negative_transfer_rate=0.,replay_audit_digest='sha256:'+'5'*64,
   replay_grounded_coverage=1.,replay_prefix_violations=0,future_eval_n=10,retired_skills=0)
  base.update(kw);u=ExperimentResult(**base);e=runner.sign(asdict(u));return replace(u,runner_key_id=e.key_id,signature_b64=e.signature_b64)
 def test_strict_gates(self):
  r=Ed25519Signer.generate('runner');q=Ed25519Signer.generate('q');v=Ed25519Verifier();v.register(r.key_id,r.public_bytes());v.register(q.key_id,q.public_bytes())
  qual=ContinualLearningQualifier('q',q,v,(r.key_id,),max_negative_transfer_rate=.05,min_replay_grounded_coverage=1.,min_future_eval_n=5,max_retired_skills=0)
  self.assertTrue(qual.evaluate(self.signed(r)).passed)
  self.assertFalse(qual.evaluate(self.signed(r,negative_transfer_rate=.2)).passed)
  self.assertFalse(qual.evaluate(self.signed(r,replay_grounded_coverage=.5)).passed)
  self.assertFalse(qual.evaluate(self.signed(r,future_eval_n=2)).passed)
  self.assertFalse(qual.evaluate(self.signed(r,retired_skills=1)).passed)
