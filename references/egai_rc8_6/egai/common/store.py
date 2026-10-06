import sqlite3
from pathlib import Path
from contextlib import contextmanager
class SQLiteStore:
    def __init__(self,path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(self.path,timeout=30,isolation_level=None);self.db.row_factory=sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL');self.db.execute('PRAGMA synchronous=FULL');self.db.execute('PRAGMA foreign_keys=ON')
    def execute(self,sql,args=()):
        cur=self.db.execute(sql,args);return cur
    @contextmanager
    def immediate(self):
        self.db.execute('BEGIN IMMEDIATE')
        try: yield self.db; self.db.execute('COMMIT')
        except Exception:
            self.db.execute('ROLLBACK');raise
    def close(self):
        db=getattr(self,'db',None)
        if db is not None: db.close();self.db=None
    def __enter__(self):return self
    def __exit__(self,*exc):self.close()
    def __del__(self):
        try:self.close()
        except Exception:pass
