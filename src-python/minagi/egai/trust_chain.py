"""Role-scoped research evidence verification, with no runtime promotion capability."""
from dataclasses import asdict, dataclass
import math
import json
from statistics import mean
from egai.common.crypto import SignedEnvelope
from egai.common.canonical import digest


@dataclass(frozen=True)
class ResearchChain:
    candidate: object
    build: object
    benchmark: object
    results: object
    evaluation: object
    qualification: object
    evaluator_binding: SignedEnvelope
    qualifier_binding: SignedEnvelope

    def binding(self, bundle):
        return {
            'schema': 'mini-agi-research-binding-v1',
            'bundle_digest': bundle.digest,
            'candidate': self.candidate.digest,
            'build': self.build.digest,
            'benchmark': self.benchmark.digest,
            'results': self.results.digest,
            'evaluation': self.evaluation.digest,
            'qualification': self.qualification.digest,
        }

    @property
    def digest(self):
        return digest(self)


class ResearchChainVerifier:
    def __init__(self, verifier, trust, artifact_store):
        self.verifier = verifier
        self.trust = trust
        self.artifacts = artifact_store

    def _sig(self, role, obj, key, signature):
        self.trust.require(role, key)
        if not self.verifier.verify(asdict(obj.unsigned()), SignedEnvelope(key, signature)):
            raise PermissionError(f'invalid {role} signature')

    def verify(self, proposal, bundle, chain):
        if not isinstance(chain, ResearchChain):
            raise PermissionError('signed independent research chain required')
        c, b, bm, r, e, q = (chain.candidate, chain.build, chain.benchmark,
                            chain.results, chain.evaluation, chain.qualification)
        keys = (b.builder_key_id, bm.benchmark_key_id, r.runner_key_id,
                e.evaluator_key_id, q.qualifier_key_id)
        if len(set(keys)) != len(keys):
            raise PermissionError('research roles must use distinct keys')
        for role, obj, key, signature in (
            ('builder', b, b.builder_key_id, b.signature_b64),
            ('benchmark', bm, bm.benchmark_key_id, bm.signature_b64),
            ('runner', r, r.runner_key_id, r.signature_b64),
            ('evaluator', e, e.evaluator_key_id, e.signature_b64),
            ('qualifier', q, q.qualifier_key_id, q.signature_b64),
        ):
            self._sig(role, obj, key, signature)
        if not bm.preregistered or not q.passed:
            raise PermissionError('research qualification failed or benchmark not preregistered')
        pairs = (
            (c.proposal_digest, proposal.digest), (c.artifact_digest, bundle.candidate_digest),
            (b.candidate_digest, c.digest), (b.output_digest, bundle.candidate_digest),
            (r.build_digest, b.digest), (r.benchmark_digest, bm.digest),
            (e.build_digest, b.digest), (e.benchmark_digest, bm.digest),
            (e.result_bundle_digest, r.digest), (q.evaluation_digest, e.digest),
            (bundle.suite_digest, bm.digest), (bundle.evaluator_id, e.evaluator_id),
        )
        if any(a != z for a, z in pairs):
            raise PermissionError('research chain binding mismatch')
        for role, envelope, expected_key in (
            ('evaluator', chain.evaluator_binding, e.evaluator_key_id),
            ('qualifier', chain.qualifier_binding, q.qualifier_key_id),
        ):
            self.trust.require(role, envelope.key_id)
            if envelope.key_id != expected_key or not self.verifier.verify(chain.binding(bundle), envelope):
                raise PermissionError('unsigned or substituted governance measurements')
        # Every preregistered artifact is checked against its content address.
        for d in (c.artifact_digest, b.environment_digest, b.output_digest,
                  bm.task_set_digest, bm.retention_set_digest, bm.security_set_digest):
            self.artifacts.get_bytes(d)
        expected_ids={}
        for split, dataset in (('future',bm.task_set_digest),('retention',bm.retention_set_digest),('security',bm.security_set_digest)):
            rows=json.loads(self.artifacts.get_bytes(dataset))['value']
            for row in rows:
                if row['split']!=split or row['case_id'] in expected_ids:
                    raise PermissionError('invalid preregistered split or repeated case id')
                expected_ids[row['case_id']]=split
        if {x.case_id:x.split for x in r.results}!=expected_ids:
            raise PermissionError('runner results do not cover the preregistered benchmark')
        future = [x for x in r.results if x.split == 'future']
        retention = [x for x in r.results if x.split == 'retention']
        security = [x for x in r.results if x.split == 'security']
        if not future or not retention or not security:
            raise PermissionError('all three benchmark splits required')
        ids = [x.case_id for x in r.results]
        if len(ids) != len(set(ids)) or bundle.qualification_worlds > len(future):
            raise PermissionError('duplicate or insufficient independent worlds')
        metrics = bundle.metrics
        actual = {
            'forward_transfer_delta': mean(x.candidate_score-x.baseline_score for x in future),
            'forgetting': max(0., mean(x.baseline_score-x.candidate_score for x in retention)),
            'calibration_regression': max(0., mean(x.candidate_calibration-x.baseline_calibration for x in r.results)),
            'security_regressions': sum(bool(x.security_regression) for x in r.results),
            'unauthorized_writes': sum(not x.authorized_change for x in r.results),
            'provenance_closure': e.provenance_closure,
        }
        for name, value in actual.items():
            if not math.isfinite(value) or not math.isclose(getattr(metrics, name), value, rel_tol=1e-9, abs_tol=1e-12):
                raise PermissionError(f'governance metric differs from signed results: {name}')
        return chain.digest

    def consume(self, proposal, bundle, chain):
        """One-shot benchmark consumption, subordinate to the existing promotion plane."""
        import sqlite3
        self.verify(proposal, bundle, chain)
        db=sqlite3.connect(self.artifacts.root/'research_consumption.sqlite3')
        try:
            db.execute('PRAGMA journal_mode=WAL'); db.execute('PRAGMA synchronous=FULL')
            db.execute('CREATE TABLE IF NOT EXISTS consumed (benchmark TEXT PRIMARY KEY, bundle TEXT UNIQUE NOT NULL)')
            with db:
                db.execute('INSERT INTO consumed VALUES(?,?)',(chain.benchmark.digest,bundle.digest))
        except sqlite3.IntegrityError as exc:
            raise PermissionError('one-shot research benchmark already consumed') from exc
        finally:
            db.close()
