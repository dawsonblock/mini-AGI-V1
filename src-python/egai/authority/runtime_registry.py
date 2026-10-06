import json
from dataclasses import asdict
from egai.common.store import SQLiteStore
from egai.common.crypto import SignedEnvelope

class RuntimeRegistry(SQLiteStore):
    def __init__(self,path):
        super().__init__(path)
        self.execute("""CREATE TABLE IF NOT EXISTS runtimes(seq INTEGER PRIMARY KEY AUTOINCREMENT,runtime_digest TEXT UNIQUE,body TEXT,active INTEGER,decision_nonce TEXT UNIQUE)""")
    def nonce_used(self,n):return self.db.execute('SELECT 1 FROM runtimes WHERE decision_nonce=?',(n,)).fetchone() is not None
    def current_digest(self):
        r=self.db.execute('SELECT runtime_digest FROM runtimes WHERE active=1 ORDER BY seq DESC LIMIT 1').fetchone();return r['runtime_digest'] if r else None
    def activate(self,runtime,nonce):
        with self.db:
            self.db.execute('UPDATE runtimes SET active=0 WHERE active=1')
            self.db.execute('INSERT INTO runtimes(runtime_digest,body,active,decision_nonce) VALUES(?,?,1,?)',(runtime.digest,json.dumps(asdict(runtime)),nonce))
    def rollback(self,authorization,verifier,trust):
        trust.require('rollback',authorization.authority_key_id)
        if not verifier.verify(asdict(authorization.unsigned()),SignedEnvelope(authorization.authority_key_id,authorization.signature_b64)):raise PermissionError('invalid rollback signature')
        if self.current_digest()!=authorization.from_runtime_digest:raise ValueError('rollback source is not current runtime')
        target=self.db.execute('SELECT 1 FROM runtimes WHERE runtime_digest=?',(authorization.to_runtime_digest,)).fetchone()
        if not target:raise ValueError('rollback target unknown')
        with self.db:
            self.db.execute('UPDATE runtimes SET active=0 WHERE active=1')
            self.db.execute('UPDATE runtimes SET active=1 WHERE runtime_digest=?',(authorization.to_runtime_digest,))
