from __future__ import annotations

import hmac
import os
from fastapi import Depends, FastAPI, Header, HTTPException

app = FastAPI(title="KVContinual V14 Admin Control API", version="14.0.0")


def require_admin(x_admin_token: str | None = Header(default=None)) -> None:
    expected = os.getenv("KVCONTINUAL_ADMIN_TOKEN")
    if not expected:
        raise HTTPException(503, "admin mutation API disabled: KVCONTINUAL_ADMIN_TOKEN is not configured")
    if x_admin_token is None or not hmac.compare_digest(x_admin_token, expected):
        raise HTTPException(401, "invalid admin token")


@app.get("/health")
def health():
    return {
        "ok": True,
        "version": "14.0.0",
        "authority": "operator-observer",
        "persistent_mutation": False,
        "message": "qualification/promotion moved to the signed v14 authority pipeline",
    }


@app.post("/qualification", dependencies=[Depends(require_admin)])
def legacy_qualification_disabled():
    raise HTTPException(
        410,
        "legacy local qualification authority removed; use independent EvaluationBundle -> QualificationRecord",
    )


@app.post("/promotion/{candidate_id}", dependencies=[Depends(require_admin)])
def legacy_promotion_disabled(candidate_id: str):
    raise HTTPException(
        410,
        "legacy admin promotion removed; activation requires signed PromotionAuthorization and hardened execution registry",
    )


@app.post("/rollback", dependencies=[Depends(require_admin)])
def legacy_rollback_disabled():
    raise HTTPException(
        410,
        "legacy unsigned rollback removed; use the signed production-transition authority",
    )
