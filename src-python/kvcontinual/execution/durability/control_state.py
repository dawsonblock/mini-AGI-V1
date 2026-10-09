from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import sqlite3
import time
import hashlib
import json
from .file_lock import ProcessFileLock

@dataclass(frozen=True, slots=True)
class MutationRecord:
    sequence: int
    transaction_id: str
    kind: str
    phase: str
    expected_generation: int
    created_at: float
    updated_at: float

class ControlState:
    """Shared monotonic mutation sequencer and durable recovery checkpoint."""
    def __init__(self, path: str | Path, *, lock_path: str | Path | None = None):
        self.path=Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path=Path(lock_path) if lock_path is not None else self.path.with_suffix('.lock')
        self._init()

    def _connect(self):
        c=sqlite3.connect(self.path)
        c.execute('PRAGMA journal_mode=WAL'); c.execute('PRAGMA synchronous=FULL')
        c.row_factory=sqlite3.Row
        return c

    def _init(self):
        with self._connect() as c:
            c.execute('''CREATE TABLE IF NOT EXISTS mutations(
                sequence INTEGER PRIMARY KEY,
                transaction_id TEXT NOT NULL UNIQUE,
                kind TEXT NOT NULL CHECK(kind IN ('promotion','rollback')),
                phase TEXT NOT NULL,
                expected_generation INTEGER NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL)''')
            c.execute('''CREATE TABLE IF NOT EXISTS recovery_checkpoint(
                singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                last_sequence INTEGER NOT NULL DEFAULT 0,
                last_transaction_id TEXT,
                last_kind TEXT,
                checkpoint_digest TEXT NOT NULL DEFAULT '',
                updated_at REAL NOT NULL)''')
            cols={str(r[1]) for r in c.execute('PRAGMA table_info(recovery_checkpoint)')}
            if 'checkpoint_digest' not in cols: c.execute("ALTER TABLE recovery_checkpoint ADD COLUMN checkpoint_digest TEXT NOT NULL DEFAULT ''")
            c.execute('INSERT OR IGNORE INTO recovery_checkpoint(singleton,last_sequence,checkpoint_digest,updated_at) VALUES(1,0,?,?)',('',time.time()))
            c.commit()

    def reserve(self, *, transaction_id: str, kind: str, expected_generation: int, phase: str='prepared') -> int:
        if kind not in {'promotion','rollback'}: raise ValueError('invalid mutation kind')
        with ProcessFileLock(self.lock_path, timeout=10.0):
            with self._connect() as c:
                c.execute('BEGIN IMMEDIATE')
                existing=c.execute('SELECT sequence FROM mutations WHERE transaction_id=?',(transaction_id,)).fetchone()
                if existing is not None:
                    c.commit(); return int(existing['sequence'])
                seq=int(c.execute('SELECT COALESCE(MAX(sequence),0)+1 FROM mutations').fetchone()[0])
                now=time.time()
                c.execute('INSERT INTO mutations(sequence,transaction_id,kind,phase,expected_generation,created_at,updated_at) VALUES(?,?,?,?,?,?,?)',
                          (seq,transaction_id,kind,phase,int(expected_generation),now,now))
                c.commit(); return seq

    def update_phase(self, transaction_id: str, phase: str) -> None:
        with self._connect() as c:
            cur=c.execute('UPDATE mutations SET phase=?,updated_at=? WHERE transaction_id=?',(phase,time.time(),transaction_id))
            if cur.rowcount != 1: raise KeyError(transaction_id)
            c.commit()

    def record(self, transaction_id: str) -> MutationRecord | None:
        with self._connect() as c:
            r=c.execute('SELECT * FROM mutations WHERE transaction_id=?',(transaction_id,)).fetchone()
        if r is None: return None
        return MutationRecord(int(r['sequence']),str(r['transaction_id']),str(r['kind']),str(r['phase']),
                              int(r['expected_generation']),float(r['created_at']),float(r['updated_at']))

    def pending(self) -> tuple[MutationRecord,...]:
        with self._connect() as c:
            rows=c.execute("SELECT * FROM mutations WHERE phase NOT IN ('committed','aborted') ORDER BY sequence").fetchall()
        return tuple(MutationRecord(int(r['sequence']),str(r['transaction_id']),str(r['kind']),str(r['phase']),
                                    int(r['expected_generation']),float(r['created_at']),float(r['updated_at'])) for r in rows)

    @staticmethod
    def _checkpoint_digest(previous_digest:str, record:MutationRecord)->str:
        payload={'previous_digest':previous_digest,'sequence':record.sequence,'transaction_id':record.transaction_id,
                 'kind':record.kind,'phase':record.phase,'expected_generation':record.expected_generation}
        return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()

    def recovery_plan(self) -> tuple[MutationRecord,...]:
        """Return deterministic pending work and reject checkpoint-skipping state."""
        cp=self.checkpoint_state(); last=int(cp['last_sequence'])
        with self._connect() as c:
            bad=c.execute("SELECT transaction_id,sequence,phase FROM mutations WHERE sequence<=? AND phase NOT IN ('committed','aborted') ORDER BY sequence",(last,)).fetchall()
            if bad:
                first=bad[0]; raise RuntimeError(f"nonterminal mutation behind checkpoint: {first['transaction_id']}@{first['sequence']}:{first['phase']}")
            rows=c.execute("SELECT * FROM mutations WHERE sequence>? AND phase NOT IN ('committed','aborted') ORDER BY sequence,transaction_id",(last,)).fetchall()
        return tuple(MutationRecord(int(r['sequence']),str(r['transaction_id']),str(r['kind']),str(r['phase']),
                                    int(r['expected_generation']),float(r['created_at']),float(r['updated_at'])) for r in rows)

    def checkpoint(self, record: MutationRecord) -> None:
        with ProcessFileLock(self.lock_path, timeout=10.0):
            with self._connect() as c:
                c.execute('BEGIN IMMEDIATE')
                if record.phase not in {'committed','aborted'}: raise RuntimeError('checkpoint requires terminal mutation')
                row=c.execute('SELECT last_sequence,last_transaction_id,last_kind,checkpoint_digest FROM recovery_checkpoint WHERE singleton=1').fetchone(); last=int(row['last_sequence'])
                if record.sequence < last: raise RuntimeError(f'recovery checkpoint regression: {record.sequence} < {last}')
                if record.sequence == last:
                    if row['last_transaction_id'] not in (None,record.transaction_id) or row['last_kind'] not in (None,record.kind):
                        raise RuntimeError('checkpoint sequence collision')
                    c.commit(); return
                bad=int(c.execute("SELECT COUNT(*) FROM mutations WHERE sequence<? AND phase NOT IN ('committed','aborted')",(record.sequence,)).fetchone()[0])
                if bad: raise RuntimeError('cannot advance checkpoint past nonterminal mutation')
                digest=self._checkpoint_digest(str(row['checkpoint_digest'] or ''),record)
                c.execute('UPDATE recovery_checkpoint SET last_sequence=?,last_transaction_id=?,last_kind=?,checkpoint_digest=?,updated_at=? WHERE singleton=1',
                          (record.sequence,record.transaction_id,record.kind,digest,time.time()))
                c.commit()

    def checkpoint_state(self) -> dict:
        with self._connect() as c:
            r=c.execute('SELECT * FROM recovery_checkpoint WHERE singleton=1').fetchone()
        return {'last_sequence':int(r['last_sequence']),'last_transaction_id':r['last_transaction_id'],
                'last_kind':r['last_kind'],'checkpoint_digest':str(r['checkpoint_digest'] or ''),'updated_at':float(r['updated_at'])}

    def integrity_check(self) -> tuple[bool,str]:
        try:
            with self._connect() as c:
                result=str(c.execute('PRAGMA integrity_check').fetchone()[0])
                dup=int(c.execute('SELECT COUNT(*) FROM (SELECT transaction_id,COUNT(*) n FROM mutations GROUP BY transaction_id HAVING n>1)').fetchone()[0])
                bad=int(c.execute("SELECT COUNT(*) FROM mutations WHERE sequence<=0 OR expected_generation<0 OR kind NOT IN ('promotion','rollback')").fetchone()[0])
                cp_row=c.execute('SELECT last_sequence,last_transaction_id,last_kind,checkpoint_digest FROM recovery_checkpoint WHERE singleton=1').fetchone(); cp=int(cp_row['last_sequence'])
                maxseq=int(c.execute('SELECT COALESCE(MAX(sequence),0) FROM mutations').fetchone()[0])
                behind=int(c.execute("SELECT COUNT(*) FROM mutations WHERE sequence<=? AND phase NOT IN ('committed','aborted')",(cp,)).fetchone()[0])
                cp_bad=0
                if cp>0:
                    rr=c.execute('SELECT * FROM mutations WHERE sequence=?',(cp,)).fetchone()
                    if rr is None or str(rr['transaction_id'])!=str(cp_row['last_transaction_id']) or str(rr['kind'])!=str(cp_row['last_kind']): cp_bad=1
                    elif not str(cp_row['checkpoint_digest'] or ''): cp_bad=1
            ok=result=='ok' and dup==0 and bad==0 and cp<=maxseq and behind==0 and cp_bad==0
            return ok,f'integrity={result};duplicates={dup};invalid={bad};checkpoint={cp};max_sequence={maxseq};nonterminal_behind={behind};checkpoint_binding_errors={cp_bad}'
        except sqlite3.Error as e:
            return False,f'control_state_error:{type(e).__name__}'
