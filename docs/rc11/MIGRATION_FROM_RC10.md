# Migration from RC10.x

Use RC11 as the only active branch.

- Keep existing RC10.5 native and gateway configuration; they are retained at repository root.
- Install `execution_memory` in the same virtual environment.
- Do not import old RC10.16 source databases into mutable mode. RC11 migrates missing digest columns but treats existing segment IDs as immutable thereafter.
- Re-qualify promoted adapters if you want signed promotion authority. Existing unsigned records are migration evidence, not cryptographic production authorization.
- Rebuild execution artifacts under RC11 so they carry the source content digest.
- Treat v4.3 modules under `oracle_reference/` as independent research/oracle code, not the serving runtime.
- Treat `authority/` as the retained v6 reference/control implementation; the production adapter registry now incorporates its core digest/signature principles directly.
