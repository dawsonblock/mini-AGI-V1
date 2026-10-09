#!/usr/bin/env python3
from pathlib import Path
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src-python'))
from kvcontinual import __version__
from kvcontinual.continual.memory.store import MemoryStore, MemoryRecord
from kvcontinual.execution.source import SourceSegmentStore, SourceSegment
from kvcontinual.execution.types import ModelIdentity, ExecutionIdentity
from kvcontinual.execution.cache.persistent import PersistentExecutionArtifactStore
from kvcontinual.execution.cache.block import ExecutionArtifact

def main():
    assert __version__ == '14.0.0'
    with tempfile.TemporaryDirectory() as td:
        mem=MemoryStore(str(Path(td)/'memory.sqlite3'))
        mid=mem.put(MemoryRecord('verified integration event','episode','smoke','rc11.7',verified=True))
        assert mem.get(mid) is not None
        src=SourceSegmentStore(str(Path(td)/'source.sqlite3'))
        seg=SourceSegment(tokens=[1,2,3,4],source_locator='integration://smoke',tokenizer_digest='sha256:tok',normalization_digest='sha256:norm')
        src.put(seg)
        model=ModelIdentity('sha256:weights','sha256:adapter','sha256:tok','qwen-hybrid','rope','bf16')
        ident=ExecutionIdentity(model,'rc11.7-host','gdn-v1','column-major','bf16','fp16')
        art=ExecutionArtifact(seg.id, ident, 0, {}, source_content_digest=seg.content_digest)
        store=PersistentExecutionArtifactStore(Path(td)/'execution-artifacts'); store.put(art); store.close()
        store=PersistentExecutionArtifactStore(Path(td)/'execution-artifacts')
        restored=store.get(seg.id, ident, seg.content_digest); assert restored is not None and restored.artifact_id == art.artifact_id
        assert store.get(seg.id, ident, 'sha256:wrong') is None
        assert store.verify_all() == {'ok':1,'bad':0}; store.close()
    print('mini-AGI 14 cross-plane persistent integration smoke passed')
if __name__ == '__main__': main()
