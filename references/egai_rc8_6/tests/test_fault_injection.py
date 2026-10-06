import unittest
from egai.experiment.faults import FaultInjector, FaultPlan
class T(unittest.TestCase):
 def test_backend_failure_injection(self):
  f=FaultInjector(FaultPlan(fail_backend_after_calls=2))
  f.before_model_call(); f.before_model_call()
  with self.assertRaises(RuntimeError): f.before_model_call()
