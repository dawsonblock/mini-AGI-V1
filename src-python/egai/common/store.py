import sqlite3
from pathlib import Path

class SQLiteStore:
    def __init__(self,path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(self.path);self.db.row_factory=sqlite3.Row
    def execute(self,sql,args=()):
        cur=self.db.execute(sql,args);self.db.commit();return cur
    def close(self):
        db=getattr(self,"db",None)
        if db is not None:
            db.close();self.db=None
    def __enter__(self):return self
    def __exit__(self,*exc):self.close()
    def __del__(self):
        try:self.close()
        except Exception:pass
