import json
from dataclasses import asdict
from egai.common.store import SQLiteStore
from .model import SkillManifest

class SkillRegistry(SQLiteStore):
    """Candidate registry. No public promotion surface exists here."""
    def __init__(self,path):
        super().__init__(path)
        self.execute("""CREATE TABLE IF NOT EXISTS skills(skill_id TEXT,version TEXT,manifest_digest TEXT UNIQUE,
          body TEXT,status TEXT,runtime_digest TEXT,PRIMARY KEY(skill_id,version))""")
    def register_candidate(self,s:SkillManifest):
        self.execute("INSERT INTO skills VALUES(?,?,?,?,?,?)",(s.skill_id,s.version,s.manifest_digest,json.dumps(asdict(s)),'candidate',''))
    def promoted(self): return list(self.db.execute("SELECT * FROM skills WHERE status='promoted'"))
    def candidate(self,skill_id,version): return self.db.execute("SELECT * FROM skills WHERE skill_id=? AND version=?",(skill_id,version)).fetchone()
