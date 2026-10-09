"""Governed benchmark harness regression tests.

Covers the paths that were previously unrunnable: the verified-repair
branch (a VerifiedEpisode arity defect), the skill-success path
(mark_success did not exist) and the end-of-run health summary
(health() did not exist). The harness had no test coverage at all.
"""
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))

from egai.bench.engine import TaskCase  # noqa: E402
from egai.bench.sequential_governed import (  # noqa: E402
    GovernedSequentialExperiment)
from egai.cognition.agent import SandboxAdaptiveAgent  # noqa: E402
from egai.cognition.model import FrozenModel, FrozenModelIdentity  # noqa: E402
from egai.common.crypto import Ed25519Signer, Ed25519Verifier  # noqa: E402
from egai.evidence.ledger import EvidenceLedger  # noqa: E402
from egai.skills.memory import (LearnedProcedure,  # noqa: E402
                                SandboxSkillMemory)


class RuleModel(FrozenModel):
    def __init__(self):
        self.ident = FrozenModelIdentity(
            'rule', (('w', 'sha256:' + '1' * 64),), '', '', 'test')

    @property
    def model_digest(self):
        return self.ident.digest

    def generate(self, prompt):
        task = prompt.split('TASK:\n', 1)[1].split('\nReturn only', 1)[0]
        if 'Reverse the input characters.' in prompt:
            return task[::-1]
        return 'UNKNOWN'


@dataclass
class Repair:
    output_text: str
    provider_id: str = 'repair-oracle'
    provenance_digest: str = 'sha256:' + '2' * 64


class RepairProvider:
    def repair(self, case, attempt):
        return Repair(str(case.expected))


def scorer(a, b):
    return 1.0 if a == b else 0.0


META = {'feedback_verified': True, 'verifier_id': 'test'}


class GovernedHarnessTest(unittest.TestCase):
    def _run(self, tmp_path, experience):
        signer = Ed25519Signer.generate('test')
        verifier = Ed25519Verifier()
        verifier.register(signer.key_id, signer.public_bytes())
        ledger = EvidenceLedger(tmp_path / 'evidence.db', signer, verifier)
        model = RuleModel()
        agent = SandboxAdaptiveAgent(model)
        # seed a skill so the successful row exercises mark_success
        agent.skills.upsert(LearnedProcedure(
            'p-reverse', 'reverse', 'Reverse the input characters.',
            'Reverse the input characters.', ('e0',), successes=2))
        evaluation = [TaskCase('f1', 'future', 'dog', 'god', 'reverse',
                               'Reverse the input characters.')]
        report = GovernedSequentialExperiment(
            experience, evaluation, scorer, RepairProvider(), ledger,
            signer, verifier, [signer.key_id], checkpoints=(0, 1),
            bootstrap_samples=50, near_leakage_threshold=None).run(model, agent)
        return report, ledger, agent

    def test_success_and_repair_paths_run(self):
        tmp = Path(tempfile.mkdtemp(prefix='governed-test-'))
        experience = [
            TaskCase('e1', 'experience', 'abc', 'cba', 'reverse',
                     'Reverse the input characters.', ('reverse',), META),
            TaskCase('e2', 'experience', 'abc', 'cba', 'mystery',
                     'Do the mystery transform.', ('mystery',), META),
        ]
        report, ledger, agent = self._run(tmp, experience)
        # the failing row went through the verified-repair branch
        self.assertEqual(report.verified_repairs, 1)
        self.assertEqual(len(report.traces), 1)
        self.assertEqual(report.traces[0].repaired_output, 'cba')
        self.assertEqual(report.traces[0].attempted_score, 0.0)
        self.assertEqual(report.traces[0].repaired_score, 1.0)
        # the successful row credited its retrieved skill
        skills = agent.skills.all()
        self.assertEqual(len(skills), 1)
        self.assertEqual(skills[0].successes, 3)   # seeded 2 + mark_success
        self.assertEqual(skills[0].failures, 0)
        # health summary and report fields
        self.assertEqual(report.degraded_skills, 0)
        self.assertEqual(report.retired_skills, 0)
        self.assertEqual(report.learned_procedures, 1)
        # evidence chain is intact and the report binds its head
        self.assertTrue(ledger.verify_chain())
        self.assertEqual(report.evidence_head, ledger.head()[1])

    def test_health_reports_degraded_skills(self):
        mem = SandboxSkillMemory()
        mem.upsert(LearnedProcedure('p1', 'reverse', 't', 'p', ('e',),
                                    successes=1))
        self.assertEqual(mem.health(), {'degraded': 0, 'retired': 0})
        mem.mark_failure('p1')
        mem.mark_failure('p1')                     # failures(2) > successes(1)
        self.assertEqual(mem.health(), {'degraded': 1, 'retired': 0})
        mem.mark_success('p1')
        mem.mark_success('p1')                     # successes(3) > failures(2)
        self.assertEqual(mem.health(), {'degraded': 0, 'retired': 0})
        # unknown ids are no-ops, exactly like mark_failure
        mem.mark_success('missing')
        mem.mark_failure('missing')
        self.assertEqual(len(mem.all()), 1)


if __name__ == '__main__':
    unittest.main()
