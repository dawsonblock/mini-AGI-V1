from dataclasses import dataclass

ROLES=("benchmark","builder","runner","evaluator","qualifier","promotion","rollback")

@dataclass(frozen=True)
class AuthorityTrust:
    roles: dict[str, tuple[str,...]]
    revoked_keys: frozenset[str] = frozenset()

    def __post_init__(self):
        object.__setattr__(self, "roles", {r: tuple(keys) for r, keys in self.roles.items()})
        object.__setattr__(self, "revoked_keys", frozenset(self.revoked_keys))
        unknown=set(self.roles)-set(ROLES)
        if unknown: raise ValueError(f"unknown trust roles: {sorted(unknown)}")

    def require(self, role:str, key_id:str):
        allowed=set(self.roles.get(role,()))
        if key_id in self.revoked_keys:
            raise PermissionError(f"revoked {role} key: {key_id}")
        if not key_id or key_id not in allowed:
            raise PermissionError(f"untrusted {role} key: {key_id!r}")

    @classmethod
    def from_signers(cls, **role_to_signer):
        return cls({r:(s.key_id,) for r,s in role_to_signer.items()})
