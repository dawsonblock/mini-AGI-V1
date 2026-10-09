"""Render an exact governed QW3 command from a v15.8 launch-spec JSON file."""
from __future__ import annotations
import argparse
import json
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src-python"))
from minagi.integration.qw3_state import ServedArtifactManifest
from minagi.v15.neural_campaign import QW3LaunchSpec


def load(path: str | Path) -> QW3LaunchSpec:
    row=json.loads(Path(path).read_text())
    manifest=ServedArtifactManifest(**row.pop("manifest"))
    if "extra_args" in row:
        row["extra_args"]=tuple(row["extra_args"])
    row.pop("schema", None)
    return QW3LaunchSpec(manifest=manifest, **row)


def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument("spec", help="JSON QW3LaunchSpec")
    p.add_argument("--json", action="store_true", help="emit argv JSON instead of shell text")
    a=p.parse_args(); spec=load(a.spec); argv=spec.argv()
    print(json.dumps(list(argv), separators=(",", ":")) if a.json else shlex.join(argv))
    return 0

if __name__=="__main__": raise SystemExit(main())
