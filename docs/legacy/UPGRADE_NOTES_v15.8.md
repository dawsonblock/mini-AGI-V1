# Upgrade Notes — v15.8.0

v15.8 is an experiment-control release, not a new mutation-authority release.

The key change is preregistration. A real-weight claim is not accepted from a loose A0/A1 comparison anymore. The exact arms, ordered hidden-task commitments, gate parameters, and scoring/retention/security policy identities must be content-addressed before the first task is consumed. The completed campaign then emits a single result bundle pointing to all evidence.

Use `QW3LaunchSpec` to construct each native arm and `tools/render_qw3_launch.py` to render the exact command. QW3 still independently measures deployed bytes at startup; paths in the launch spec are deployment inputs, not trust anchors.

A PASS result remains proposal-only. Promotion must still proceed through the existing independent qualification and promotion authority chain.
