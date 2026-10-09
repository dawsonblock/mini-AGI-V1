"""v16.4.3 supervised runtime — the protected model-loading path.

The serving supervisor owns every production model load. Artifacts move
through an explicit state machine (`activation_state`); grant
reservations, lifecycle events, and the serving pointer share one
transactional authority (`authority_store`, SQLite); the durable event
log is hash-chained and verified on read (`journal_v2`); operation
authorization derives from authenticated OS principals
(`access_policy`); backend identity is bound by a signed manifest
(`backend_manifest`); inference traffic routes only through the
committed+healthy handle (`serving_router`); and the AF_UNIX service
wrapper (`service`) carries the boundary across processes — an
`isinstance` check inside one interpreter is not a security boundary.
"""
