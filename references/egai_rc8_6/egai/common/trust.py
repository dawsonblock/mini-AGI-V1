from dataclasses import dataclass

ROLES=("benchmark","builder","runner","evaluator","qualifier","promotion","rollback")

@dataclass(frozen=True)
class AuthorityTrust:
    roles: dict[str, tuple[str,...]]

    def __post_init__(self):
        unknown=set(self.roles)-set(ROLES)
        if unknown: raise ValueError(f"unknown trust roles: {sorted(unknown)}")

    def require(self, role:str, key_id:str):
        allowed=set(self.roles.get(role,()))
        if not key_id or key_id not in allowed:
            raise PermissionError(f"untrusted {role} key: {key_id!r}")

    @classmethod
    def from_signers(cls, **role_to_signer):
        return cls({r:(s.key_id,) for r,s in role_to_signer.items()})
