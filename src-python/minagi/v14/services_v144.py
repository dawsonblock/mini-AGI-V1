from __future__ import annotations

from dataclasses import asdict
from .verification import EpisodeVerificationReceipt
from .evidence_v144 import EvidenceStageReceiptV144, EvidenceStrengthProofV144
from .services_v143 import UnixJsonAuthorityClientV143


class VerificationAuthorityClientV144:
    is_local = True
    __slots__ = ("__issue",)
    def __init__(self, authority): self.__issue = authority.issue
    def issue(self, **kwargs): return self.__issue(**kwargs)


class EvidenceStageAuthorityClientV144:
    is_local = True
    __slots__ = ("__issue",)
    def __init__(self, authority): self.__issue = authority.issue
    def issue(self, **kwargs): return self.__issue(**kwargs)


class EvidenceStrengthAuthorityClientV144:
    is_local = True
    __slots__ = ("__issue",)
    def __init__(self, authority): self.__issue = authority.issue
    def issue(self, **kwargs): return self.__issue(**kwargs)


class UnixVerificationAuthorityClientV144(UnixJsonAuthorityClientV143):
    def __init__(self, socket_path): super().__init__(socket_path, "verification.issue")
    def issue(self, **kwargs): return EpisodeVerificationReceipt(**self.call(kwargs))


class UnixEvidenceStageAuthorityClientV144(UnixJsonAuthorityClientV143):
    def __init__(self, socket_path): super().__init__(socket_path, "evidence_stage.issue")
    def issue(self, **kwargs): return EvidenceStageReceiptV144(**self.call(kwargs))


class UnixEvidenceStrengthAuthorityClientV144(UnixJsonAuthorityClientV143):
    def __init__(self, socket_path): super().__init__(socket_path, "evidence_strength.issue")
    def issue(self, **kwargs): return EvidenceStrengthProofV144(**self.call(kwargs))


def authority_handlers_v144(*, verification_authority=None, evidence_stage_authority=None, evidence_strength_authority=None,
                            base_handlers=None):
    handlers = dict(base_handlers or {})
    if verification_authority is not None:
        handlers["verification.issue"] = lambda payload: asdict(verification_authority.issue(**payload))
    if evidence_stage_authority is not None:
        handlers["evidence_stage.issue"] = lambda payload: asdict(evidence_stage_authority.issue(**payload))
    if evidence_strength_authority is not None:
        handlers["evidence_strength.issue"] = lambda payload: asdict(evidence_strength_authority.issue(**payload))
    return handlers
