"""v16.4.2 security primitives — authenticated authorization evidence.

This package holds the artifacts that carry authority across the
production boundary:

    signed_revocations    RevocationSnapshotV2 — signed, monotonic,
                          freshness-and-future bounded revocation
                          evidence with an atomic durable store
    admission_grants      AdmissionGrantV1 — the short-lived signed
                          authorization a trusted admission service
                          issues to the serving supervisor
    trusted_authority_client
                          the client side of the supervised launch
                          service (runtime/service.py)

Nothing here is a policy decision by itself: these modules encode and
verify evidence. Who may sign which record lives in
`minagi.v161.authority` (the trust root); what a document must contain
lives in `minagi.v161.strict_schema`.
"""
