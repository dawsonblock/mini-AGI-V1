from __future__ import annotations
from pathlib import Path
import sqlite3
import time

class SnapshotRollbackGuard:
    """Durable anti-rollback floor for accepted snapshot lineages."""
    def __init__(self,path:str|Path):
        self.path=Path(path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self._init()

    def _connect(self):
        c=sqlite3.connect(self.path)
        c.row_factory=sqlite3.Row
        c.execute('PRAGMA journal_mode=WAL')
        c.execute('PRAGMA synchronous=FULL')
        return c

    def _init(self):
        with self._connect() as c:
            c.execute('''CREATE TABLE IF NOT EXISTS snapshot_floors(
                root_snapshot_id TEXT PRIMARY KEY,
                max_lineage_depth INTEGER NOT NULL,
                snapshot_id TEXT NOT NULL,
                chain_hash TEXT NOT NULL,
                accepted_at REAL NOT NULL)''')
            c.commit()

    def state(self,root_snapshot_id:str)->dict|None:
        with self._connect() as c:
            r=c.execute('SELECT * FROM snapshot_floors WHERE root_snapshot_id=?',(root_snapshot_id,)).fetchone()
        return None if r is None else dict(r)

    def assert_allowed(self,manifest:dict)->None:
        root=str(manifest.get('root_snapshot_id') or manifest.get('snapshot_id') or '')
        sid=str(manifest.get('snapshot_id') or '')
        chain=str(manifest.get('chain_hash') or '')
        depth=int(manifest.get('lineage_depth',0))
        if not root or not sid:
            raise RuntimeError('snapshot anti-rollback requires snapshot id/root id')
        cur=self.state(root)
        if cur is None:
            return
        old_depth=int(cur['max_lineage_depth'])
        if depth < old_depth:
            raise RuntimeError(f'snapshot rollback rejected: depth {depth} < floor {old_depth}')
        if depth == old_depth and sid != str(cur['snapshot_id']):
            raise RuntimeError('snapshot fork rejected at anti-rollback floor')
        if depth > old_depth and not chain:
            raise RuntimeError('snapshot chain hash required above anti-rollback floor')

    def record(self,manifest:dict)->None:
        self.assert_allowed(manifest)
        root=str(manifest.get('root_snapshot_id') or manifest.get('snapshot_id'))
        sid=str(manifest.get('snapshot_id'))
        chain=str(manifest.get('chain_hash') or '')
        depth=int(manifest.get('lineage_depth',0))
        now=time.time()
        with self._connect() as c:
            c.execute('''INSERT INTO snapshot_floors(root_snapshot_id,max_lineage_depth,snapshot_id,chain_hash,accepted_at)
                VALUES(?,?,?,?,?)
                ON CONFLICT(root_snapshot_id) DO UPDATE SET
                  max_lineage_depth=excluded.max_lineage_depth,
                  snapshot_id=excluded.snapshot_id,
                  chain_hash=excluded.chain_hash,
                  accepted_at=excluded.accepted_at
                WHERE excluded.max_lineage_depth>=snapshot_floors.max_lineage_depth''',
                (root,depth,sid,chain,now))
            c.commit()

    def integrity_check(self)->tuple[bool,str]:
        try:
            with self._connect() as c:
                result=str(c.execute('PRAGMA integrity_check').fetchone()[0])
                bad=int(c.execute("SELECT COUNT(*) FROM snapshot_floors WHERE max_lineage_depth<0 OR length(snapshot_id)<>64 OR length(root_snapshot_id)<>64").fetchone()[0])
            return result=='ok' and bad==0,f'integrity={result};invalid={bad}'
        except sqlite3.Error as e:
            return False,f'snapshot_floor_error:{type(e).__name__}'
