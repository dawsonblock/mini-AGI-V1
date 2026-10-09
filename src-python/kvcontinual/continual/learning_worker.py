from __future__ import annotations

import argparse
import json
from pathlib import Path

from kvcontinual.continual.dataset_export import export_mlx_dataset
from kvcontinual.continual.experience_store import ExperienceStore
from kvcontinual.continual.registry import AdapterRegistry


def cmd_export(args: argparse.Namespace) -> None:
    store = ExperienceStore(args.db)
    train, digest, ids = export_mlx_dataset(store, args.out, limit=args.limit, mark_exported=not args.no_mark)
    run_id = store.create_learning_run(digest, len(ids), {"train": str(train), "episode_ids": ids})
    print(json.dumps({"run_id": run_id, "train": str(train), "dataset_digest": digest, "examples": len(ids)}, sort_keys=True))


def cmd_register(args: argparse.Namespace) -> None:
    reg = AdapterRegistry(args.registry)
    manifest = reg.register_candidate(
        args.adapter,
        base_model_digest=args.base_model_digest,
        dataset_digest=args.dataset_digest,
        training_config=json.loads(Path(args.training_config).read_text()) if args.training_config else {},
        parent_adapter_digest=args.parent_adapter_digest,
    )
    if args.db and args.run_id is not None:
        ExperienceStore(args.db).bind_candidate(args.run_id, manifest.candidate_id)
    print(json.dumps(manifest.__dict__, sort_keys=True))


def cmd_qualify(args: argparse.Namespace) -> None:
    raise PermissionError(
        "v14 learning workers are proposal/build authorities only; independent qualification must run in the authority pipeline"
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Offline continual-learning worker. Never auto-promotes production adapters.")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("export")
    e.add_argument("--db", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--limit", type=int, default=1000)
    e.add_argument("--no-mark", action="store_true")
    e.set_defaults(func=cmd_export)

    r = sub.add_parser("register")
    r.add_argument("--registry", required=True)
    r.add_argument("--adapter", required=True)
    r.add_argument("--base-model-digest", required=True)
    r.add_argument("--dataset-digest", required=True)
    r.add_argument("--training-config")
    r.add_argument("--parent-adapter-digest")
    r.add_argument("--db")
    r.add_argument("--run-id", type=int)
    r.set_defaults(func=cmd_register)

    q = sub.add_parser("qualify")
    q.add_argument("--registry", required=True)
    q.add_argument("--candidate-id", required=True)
    q.add_argument("--metrics", required=True)
    q.add_argument("--evaluator-digest", required=True)
    q.add_argument("--min-target-gain", type=float, default=0.03)
    q.add_argument("--max-general-regression", type=float, default=0.01)
    q.set_defaults(func=cmd_qualify)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
