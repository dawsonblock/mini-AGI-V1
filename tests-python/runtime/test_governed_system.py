from minagi.system import GovernedSystem
from kvcontinual.execution.types import ExecutionIdentity,ModelIdentity,ReconstructionMode,ReconstructionAction
from kvcontinual.execution.source import SourceSegment
from kvcontinual.execution.authority import Ed25519ReceiptSigner
from kvcontinual.execution.recurrent.interfaces import BackendCapabilities


class Backend:
    def capabilities(self):return BackendCapabilities(hypic_seam8=True,causal_conv_seam_repair=True,full_attention_relocation=True)
    def exact_selected_replay(self,ids):return ('exact',ids)
    def materialize(self,*args):raise AssertionError('unqualified acceleration reached')


def test_canonical_system_replays_exactly_without_target_qualification(tmp_path):
    system=GovernedSystem(tmp_path,Backend(),research_verifier=None,promotion_signer=Ed25519ReceiptSigner.generate())
    seg=SourceSegment([1,2,3],'test');system.sources.put(seg)
    identity=ExecutionIdentity(ModelIdentity('w','a','t','architecture','rope'),'abi','gdn','layout','f32','f16')
    result=system.reconstruct([seg.id],identity,ReconstructionMode.FAST)
    assert result.action is ReconstructionAction.EXACT_SELECTED_REPLAY
    assert result.state==('exact',[seg.id])
    system.close()


def test_signed_target_binding_and_revocation_use_real_rc11_contracts(tmp_path,monkeypatch):
    from dataclasses import asdict,replace
    import pytest
    import minagi.system as assembly
    from egai.common.canonical import digest
    from kvcontinual.execution.authority import Ed25519ReceiptVerifier
    from kvcontinual.execution.qualification_bundle import build_qualification_bundle
    from kvcontinual.execution.qualification_harness import QualificationObservation,MacQualificationThresholds
    from kvcontinual.execution.hardware_qualification import (
        HardwareRuntimeFingerprint,AccelerationQualificationObservation,
        AccelerationQualificationThresholds,build_acceleration_qualification_certificate,
        QualificationSuiteCase,build_qualification_suite_manifest)
    from kvcontinual.execution.deployment_binding import build_deployment_binding
    from kvcontinual.execution.macos_qualification_preflight import MacQualificationPreflight,PreflightCheck
    system=GovernedSystem(tmp_path,Backend(),research_verifier=None,promotion_signer=Ed25519ReceiptSigner.generate())
    segment=SourceSegment([1,2,3],'test');system.sources.put(segment)
    identity=ExecutionIdentity(ModelIdentity('w','a','t','architecture','rope'),'abi','gdn','layout','f32','f16')
    hw=HardwareRuntimeFingerprint('Darwin','arm64','test','3.12','test','test','test','metal',True,True,'backend')
    pf=MacQualificationPreflight(1,'test','backend',hw.digest,(PreflightCheck('test',True,'synthetic test host'),))
    pf=replace(pf,report_digest=pf.expected_digest())
    monkeypatch.setattr(assembly,'run_macos_qualification_preflight',lambda **k:pf)
    monkeypatch.setattr(assembly,'detect_hardware_runtime_fingerprint',lambda **k:hw)
    probe={'hardware_toolchain_fingerprint':digest('probe'),'apple_silicon':True,'metal_available':True,'mps_available':True}
    release=digest('release');head=digest('ledger')
    observations=[QualificationObservation('case',.001,1.,True)]
    bundle=asdict(build_qualification_bundle(hardware_probe=probe,capture_manifest={'test':True},observations=observations,
        model_weights_digest=digest('model'),tokenizer_digest=digest('tokenizer'),kernel_build_digest=digest('kernel'),
        execution_identity_digest=identity.digest,runtime_build_digest=release,thresholds=MacQualificationThresholds(min_cases=1)))
    suite=build_qualification_suite_manifest('test',[QualificationSuiteCase('case',digest('case'),'ARBITRARY',1)])
    cert=build_acceleration_qualification_certificate(identity=identity,backend_fingerprint='backend',hardware_runtime=hw,
        release_manifest_digest=release,observations=[AccelerationQualificationObservation('case',.001,1.,True,2.,1.,'ARBITRARY',1)],
        qualification_suite=suite,qualification_ledger_head=head,thresholds=AccelerationQualificationThresholds(min_case_count=1))
    binding=build_deployment_binding(cert,backend_fingerprint='backend',hardware_runtime_digest=hw.digest,
        release_manifest_digest=release,qualification_ledger_head=head,qualification_suite_digest=suite.suite_digest)
    signer=Ed25519ReceiptSigner.generate();revoked=set()
    verifier=Ed25519ReceiptVerifier(signer.private_key.public_key(),key_id=signer.key_id,revocation_check=lambda k:k in revoked)
    signed={'schema':'mini-agi-target-deployment-v1','bundle':bundle,'certificate_digest':cert.certificate_digest,
            'deployment_binding_digest':binding.binding_digest,'hardware_runtime_digest':hw.digest}
    kwargs=dict(identity=identity,bundle=bundle,receipt=signer.issue(signed),verifier=verifier,certificate=cert,
        deployment_binding=binding,backend_fingerprint='backend',release_manifest_digest=release,
        qualification_ledger_head=head,qualification_suite_digest=suite.suite_digest)
    bad={**kwargs,'receipt':signer.issue(bundle)}
    with pytest.raises(PermissionError,match='signature required'):system.bind_target_qualification(**bad)
    system.bind_target_qualification(**kwargs)
    with pytest.raises(AssertionError,match='unqualified acceleration reached'):
        system.reconstruct([segment.id],identity,ReconstructionMode.FAST)
    revoked.add(signer.key_id)
    result=system.reconstruct([segment.id],identity,ReconstructionMode.FAST)
    assert result.action is ReconstructionAction.EXACT_SELECTED_REPLAY
    system.close()
