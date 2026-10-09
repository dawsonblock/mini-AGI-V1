"""v16.4.2 supervised runtime — the protected model-loading path.

The serving supervisor owns every production model load. Artifacts
move through an explicit state machine (`activation_state`), every
transition is journaled before it takes effect (`durable_journal`),
and the previous healthy model is retained until a new candidate's
activation is durably committed (`supervisor`). The AF_UNIX service
wrapper (`service`) carries the boundary across processes — an
`isinstance` check inside one interpreter is not a security boundary
(UPGRADE_PLAN §3.1).
"""
