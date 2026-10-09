import unittest
from egai.bench.engine import TaskCase
from egai.bench.sequential import SequentialExperiment
from egai.cognition.model import FrozenModel,FrozenModelIdentity
from egai.cognition.agent import SandboxAdaptiveAgent

class RuleModel(FrozenModel):
 def __init__(self):self.ident=FrozenModelIdentity('rule',(('w','sha256:'+'1'*64),),'','','test')
 @property
 def model_digest(self):return self.ident.digest
 def generate(self,prompt):
  task=prompt.split('TASK:\n',1)[1].split('\nReturn only',1)[0]
  if 'Reverse the input characters.' in prompt:return task[::-1]
  return 'UNKNOWN'

class T(unittest.TestCase):
 def test_future_improves_without_weight_change(self):
  model=RuleModel();agent=SandboxAdaptiveAgent(model)
  meta={'feedback_verified':True,'verifier_id':'oracle'}
  exp=[TaskCase('e1','experience','abc','cba','reverse','Reverse the input characters.',('reverse',),meta),TaskCase('e2','experience','cat','tac','reverse','Reverse the input characters.',('reverse',),meta)]
  ev=[TaskCase('f1','future','dog','god','reverse'),TaskCase('r1','retention','plain','UNKNOWN','other')]
  def score(a, b):
   return 1. if a==b else 0.
  report=SequentialExperiment(exp,ev,score,checkpoints=(0,1,2),bootstrap_samples=100,near_leakage_threshold=None).run(model,agent)
  self.assertTrue(report.model_digest_constant);self.assertEqual(report.points[0].future_success,0.)
  self.assertEqual(report.points[-1].future_success,1.);self.assertEqual(report.learned_procedures,1)
