from __future__ import annotations

from dataclasses import asdict
from io import BytesIO
from pathlib import Path
import hashlib, json, os, sqlite3, tempfile, zipfile, time, math
from kvcontinual.execution.durability.resource_budget import ResourceBudget
import numpy as np

from kvcontinual.execution.attention.relocation import FullAttentionRelocationMetadata
from kvcontinual.execution.cache.block import ArtifactQualification, BoundaryAnchors, ExecutionArtifact, RecurrentTailArtifact
from kvcontinual.execution.durability.cas import ContentAddressedStore
from kvcontinual.execution.recurrent.affine import AffineSummary
from kvcontinual.execution.recurrent.conv_boundary import ConvBoundaryState
from kvcontinual.execution.types import CacheTier, ExecutionIdentity, ModelIdentity, TransitionOrientation

_SCHEMA=1


def _json_safe(v):
    if v is None or isinstance(v,(str,int,float,bool)): return v
    if isinstance(v,dict): return {str(k):_json_safe(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)): return [_json_safe(x) for x in v]
    raise TypeError(f"non-JSON metadata type: {type(v).__name__}")


def encode_artifact(a: ExecutionArtifact) -> bytes:
    a.validate(); arrays={}
    def add(name, arr):
        arr=np.asarray(arr)
        bio=BytesIO(); np.save(bio,arr,allow_pickle=False); data=bio.getvalue()
        arrays[name]=data
        return {'entry':name,'sha256':hashlib.sha256(data).hexdigest(),'dtype':str(arr.dtype),'shape':list(arr.shape)}
    rec=[]
    for (layer,head), r in sorted(a.recurrent.items()):
        item={'layer':layer,'head':head,'seam_width':r.seam_width,'orientation':r.tail_summary.orientation.value,
              'T':add(f'arrays/recurrent-{layer}-{head}-T.npy',r.tail_summary.T),
              'Z':add(f'arrays/recurrent-{layer}-{head}-Z.npy',r.tail_summary.Z),
              'stats':_json_safe(r.stats)}
        if r.trailing_conv_payload is not None: item['trailing_conv_payload']=add(f'arrays/recurrent-{layer}-{head}-conv.npy',r.trailing_conv_payload)
        rec.append(item)
    conv=[]
    for layer,c in sorted(a.conv_boundaries.items()): conv.append({'layer':layer,'kernel_size':c.kernel_size,'history':add(f'arrays/conv-{layer}.npy',c.history)})
    bound={'leading_tokens':list(a.boundary.leading_tokens),'original_hidden':{},'original_attention_output':{},'original_route_ids':{}}
    for field in ('original_hidden','original_attention_output','original_route_ids'):
        for k,v in sorted(getattr(a.boundary,field).items()): bound[field][str(k)]=add(f'arrays/boundary-{field}-{k}.npy',v)
    ident={'model':asdict(a.identity.model),**{k:v for k,v in asdict(a.identity).items() if k!='model'}}
    man={'schema':_SCHEMA,'artifact_id':a.artifact_id,'source_segment_id':a.source_segment_id,'source_content_digest':a.source_content_digest,
         'identity':ident,'identity_digest':a.identity.digest,'fixed_seam_width':a.fixed_seam_width,'recurrent':rec,'conv_boundaries':conv,
         'attention_kv_ref':a.attention_kv_ref,'position_metadata':_json_safe(a.position_metadata),'boundary':bound,
         'attention_relocation':[asdict(v) for _,v in sorted(a.attention_relocation.items())],
         'capture_manifest_digest':a.capture_manifest_digest,'qualification':None if a.qualification is None else asdict(a.qualification),
         'tier':a.tier.value,'byte_size':a.byte_size,'created_at':a.created_at}
    mbytes=json.dumps(man,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    out=BytesIO()
    with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_STORED) as z:
        for name,data in [('manifest.json',mbytes),*sorted(arrays.items())]:
            info=zipfile.ZipInfo(name,(1980,1,1,0,0,0)); info.compress_type=zipfile.ZIP_STORED; info.external_attr=0o100644<<16
            z.writestr(info,data)
    return out.getvalue()


def decode_artifact(data: bytes) -> ExecutionArtifact:
    with zipfile.ZipFile(BytesIO(data),'r') as z:
        names=set(z.namelist())
        if 'manifest.json' not in names: raise ValueError('artifact manifest missing')
        m=json.loads(z.read('manifest.json'))
        if m.get('schema')!=_SCHEMA: raise ValueError('unsupported artifact schema')
        def arr(spec):
            name=spec['entry']
            if name not in names: raise ValueError(f'missing array {name}')
            raw=z.read(name)
            if hashlib.sha256(raw).hexdigest()!=spec['sha256']: raise ValueError(f'array digest mismatch: {name}')
            a=np.load(BytesIO(raw),allow_pickle=False)
            if list(a.shape)!=spec['shape'] or str(a.dtype)!=spec['dtype']: raise ValueError(f'array metadata mismatch: {name}')
            return a
        md=m['identity']['model']; model=ModelIdentity(**md)
        idkw={k:v for k,v in m['identity'].items() if k!='model'}; ident=ExecutionIdentity(model=model,**idkw)
        if ident.digest!=m['identity_digest']: raise ValueError('execution identity digest mismatch')
        recurrent={}
        for r in m['recurrent']:
            key=(int(r['layer']),int(r['head']))
            recurrent[key]=RecurrentTailArtifact(layer=key[0],head=key[1],seam_width=int(r['seam_width']),
                tail_summary=AffineSummary(arr(r['T']),arr(r['Z']),TransitionOrientation(r['orientation'])),
                trailing_conv_payload=arr(r['trailing_conv_payload']) if 'trailing_conv_payload' in r else None,stats=dict(r.get('stats',{})))
        conv={int(c['layer']):ConvBoundaryState(arr(c['history']),int(c['kernel_size'])) for c in m['conv_boundaries']}
        b=m['boundary']; boundary=BoundaryAnchors(leading_tokens=list(b['leading_tokens']))
        for field in ('original_hidden','original_attention_output','original_route_ids'):
            setattr(boundary,field,{int(k):arr(v) for k,v in b[field].items()})
        reloc={int(x['layer']):FullAttentionRelocationMetadata(**x) for x in m['attention_relocation']}
        q=ArtifactQualification(**m['qualification']) if m['qualification'] is not None else None
        a=ExecutionArtifact(source_segment_id=m['source_segment_id'],identity=ident,fixed_seam_width=int(m['fixed_seam_width']),recurrent=recurrent,
            attention_kv_ref=m.get('attention_kv_ref'),position_metadata=m.get('position_metadata',{}),boundary=boundary,conv_boundaries=conv,
            attention_relocation=reloc,capture_manifest_digest=m.get('capture_manifest_digest',''),qualification=q,tier=CacheTier(m['tier']),
            byte_size=int(m.get('byte_size',0)),source_content_digest=m.get('source_content_digest',''),artifact_id=m['artifact_id'],created_at=m['created_at'])
        a.validate(); return a


class PersistentExecutionArtifactStore:
    """Crash-safe persistent execution-artifact store backed by SHA-256 CAS + SQLite index.

    SQLite stores only lookup metadata. Artifact bytes are immutable CAS objects and are
    verified before deserialization. Corrupt objects fail closed and are quarantined.
    """
    def __init__(self, root: str|Path, *, budget: ResourceBudget | None = None):
        self.budget = budget or ResourceBudget()
        self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True)
        self.cas=ContentAddressedStore(self.root/'objects')
        self.db=sqlite3.connect(self.root/'index.sqlite3')
        self.db.execute('PRAGMA journal_mode=WAL'); self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS artifacts (source_segment_id TEXT NOT NULL, identity_digest TEXT NOT NULL, source_content_digest TEXT NOT NULL, object_digest TEXT NOT NULL, tier TEXT NOT NULL, byte_size INTEGER NOT NULL, PRIMARY KEY(source_segment_id,identity_digest))')
        if 'token_count' not in {x[1] for x in self.db.execute('PRAGMA table_info(artifacts)')}:
            self.db.execute('ALTER TABLE artifacts ADD COLUMN token_count INTEGER NOT NULL DEFAULT -1')
        self.db.execute('CREATE TABLE IF NOT EXISTS lease_owners (owner TEXT PRIMARY KEY, generation INTEGER NOT NULL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS leases (digest TEXT, owner TEXT, generation INTEGER, expires REAL, PRIMARY KEY(digest,owner))')
        self.db.execute('CREATE TABLE IF NOT EXISTS mutations (sequence INTEGER PRIMARY KEY, operation TEXT NOT NULL, payload TEXT NOT NULL, previous_digest TEXT NOT NULL, digest TEXT NOT NULL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS checkpoint (singleton INTEGER PRIMARY KEY CHECK(singleton=1), sequence INTEGER NOT NULL, digest TEXT NOT NULL)')
        self.db.execute("INSERT OR IGNORE INTO checkpoint VALUES(1,0,'')")
        self.db.commit()
        self.verify_recovery_checkpoint()
    def close(self): self.db.close()
    def put(self,a:ExecutionArtifact, *, source_token_count: int | None = None)->None:
        token_count = -1 if source_token_count is None else source_token_count
        if not isinstance(token_count, int) or token_count < -1: raise ValueError('invalid source token count')
        payload=encode_artifact(a)
        # Serialize admission with every index, lease and GC mutation across processes.
        self.db.execute('BEGIN IMMEDIATE')
        try:
            blocks=int(self.db.execute('SELECT COUNT(*) FROM artifacts WHERE NOT (source_segment_id=? AND identity_digest=?)', (a.source_segment_id,a.identity.digest)).fetchone()[0])+1
            digest=self.cas.digest_bytes(payload)
            disk_bytes=sum(self.cas.path_for(d).stat().st_size for d in self.cas.iter_digests())
            added=0 if self.cas.path_for(digest).exists() else len(payload)
            tokens=int(self.db.execute('SELECT COALESCE(SUM(MAX(token_count,0)),0) FROM artifacts WHERE NOT (source_segment_id=? AND identity_digest=?)',(a.source_segment_id,a.identity.digest)).fetchone()[0])+max(0,token_count)
            self.budget.validate(blocks=blocks,tokens=tokens,payload_bytes=disk_bytes+added)
            obj=self.cas.put(payload)
            self.db.execute('INSERT INTO artifacts VALUES(?,?,?,?,?,?,?) ON CONFLICT(source_segment_id,identity_digest) DO UPDATE SET source_content_digest=excluded.source_content_digest,object_digest=excluded.object_digest,tier=excluded.tier,byte_size=excluded.byte_size,token_count=excluded.token_count',
                (a.source_segment_id,a.identity.digest,a.source_content_digest,obj.digest,a.tier.value,len(payload),token_count))
            self._record_mutation('put', {'source':a.source_segment_id,'identity':a.identity.digest,'object':obj.digest})
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise
    def get(self,source_segment_id:str,identity:ExecutionIdentity,source_content_digest:str|None=None):
        row=self.db.execute('SELECT source_content_digest,object_digest FROM artifacts WHERE source_segment_id=? AND identity_digest=?',(source_segment_id,identity.digest)).fetchone()
        if row is None: return None
        if source_content_digest is not None and row[0] and row[0]!=source_content_digest: return None
        try: payload=self.cas.get(row[1]); a=decode_artifact(payload)
        except Exception:
            self.cas.verify_or_quarantine(row[1])
            with self.db:
                self.db.execute('DELETE FROM artifacts WHERE source_segment_id=? AND identity_digest=? AND object_digest=?',(source_segment_id,identity.digest,row[1]))
                self._record_mutation('quarantine', {'object':row[1]})
            return None
        if a.identity.digest!=identity.digest or a.source_segment_id!=source_segment_id or a.source_content_digest!=row[0]: return None
        return a
    def resolve(self,source_segment_ids:list[str],identity:ExecutionIdentity,source_content_digests:dict[str,str]|None=None):
        hits=[]; misses=[]
        for sid in source_segment_ids:
            a=self.get(sid,identity,source_content_digests.get(sid) if source_content_digests else None)
            (hits if a is not None else misses).append(a if a is not None else sid)
        return hits,misses
    def move_tier(self,source_segment_id:str,identity:ExecutionIdentity,tier:CacheTier)->None:
        a=self.get(source_segment_id,identity)
        if a is None: raise KeyError(source_segment_id)
        count=self.db.execute('SELECT token_count FROM artifacts WHERE source_segment_id=? AND identity_digest=?',(source_segment_id,identity.digest)).fetchone()[0]
        a.tier=tier; self.put(a,source_token_count=None if count<0 else count)
    def evict_namespace(self,identity:ExecutionIdentity)->int:
        with self.db:
            cur=self.db.execute('DELETE FROM artifacts WHERE identity_digest=?',(identity.digest,))
            self._record_mutation('evict', {'identity':identity.digest})
        return int(cur.rowcount)
    def bytes_by_tier(self):
        return {row[0]:int(row[1]) for row in self.db.execute('SELECT tier,COALESCE(SUM(byte_size),0) FROM artifacts GROUP BY tier')}
    def verify_all(self)->dict[str,int]:
        ok=bad=0
        rows=list(self.db.execute('SELECT source_segment_id,identity_digest,object_digest,source_content_digest FROM artifacts'))
        for sid,ident,d,source_digest in rows:
            try:
                a=decode_artifact(self.cas.get(d))
                valid=a.source_segment_id==sid and a.identity.digest==ident and a.source_content_digest==source_digest
            except Exception:
                valid=False
            if valid: ok+=1
            else:
                bad+=1; self.cas.verify_or_quarantine(d)
                with self.db:
                    self.db.execute('DELETE FROM artifacts WHERE source_segment_id=? AND identity_digest=? AND object_digest=?',(sid,ident,d))
                    self._record_mutation('quarantine', {'object':d})
        return {'ok':ok,'bad':bad}

    def _record_mutation(self, operation, payload):
        cp=self.db.execute('SELECT sequence,digest FROM checkpoint WHERE singleton=1').fetchone()
        seq=int(cp[0])+1
        body=json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False)
        d=hashlib.sha256(json.dumps([seq,operation,body,cp[1]],separators=(',',':')).encode()).hexdigest()
        self.db.execute('INSERT INTO mutations VALUES(?,?,?,?,?)',(seq,operation,body,cp[1],d))
        self.db.execute('UPDATE checkpoint SET sequence=?,digest=? WHERE singleton=1',(seq,d))

    def verify_recovery_checkpoint(self):
        previous=''; last=0
        # One read transaction keeps the history and checkpoint in a consistent snapshot.
        self.db.execute('BEGIN')
        try:
            if self.db.execute('PRAGMA integrity_check').fetchone()[0]!='ok':
                raise RuntimeError('artifact index integrity failed')
            for seq,op,body,prev,d in self.db.execute('SELECT * FROM mutations ORDER BY sequence'):
                want=hashlib.sha256(json.dumps([seq,op,body,prev],separators=(',',':')).encode()).hexdigest()
                if seq!=last+1 or prev!=previous or d!=want:
                    raise RuntimeError('artifact recovery order or hash chain invalid')
                previous=d; last=seq
            cp=self.db.execute('SELECT sequence,digest FROM checkpoint WHERE singleton=1').fetchone()
            if cp!=(last,previous): raise RuntimeError('artifact recovery checkpoint mismatch')
            return {'sequence':last,'digest':previous}
        finally:
            self.db.rollback()

    def fence_lease_owner(self, owner):
        if not isinstance(owner,str) or not owner: raise ValueError('lease owner required')
        self.db.execute('BEGIN IMMEDIATE')
        try:
            row=self.db.execute('SELECT generation FROM lease_owners WHERE owner=?',(owner,)).fetchone()
            generation=1 if row is None else int(row[0])+1
            self.db.execute('INSERT INTO lease_owners VALUES(?,?) ON CONFLICT(owner) DO UPDATE SET generation=excluded.generation',(owner,generation))
            self.db.execute('DELETE FROM leases WHERE owner=?',(owner,))
            self._record_mutation('fence',{'owner':owner,'generation':generation})
            self.db.commit(); return generation
        except BaseException:
            self.db.rollback(); raise

    def _require_generation(self,owner,generation):
        row=self.db.execute('SELECT generation FROM lease_owners WHERE owner=?',(owner,)).fetchone()
        if row is None or generation!=row[0]: raise RuntimeError('stale lease owner generation')

    def acquire_lease(self,digest,owner,generation,*,ttl_seconds=30.,now=None):
        if not math.isfinite(ttl_seconds) or ttl_seconds<=0: raise ValueError('positive finite lease TTL required')
        now=time.time() if now is None else float(now)
        if not math.isfinite(now): raise ValueError('finite lease time required')
        self.db.execute('BEGIN IMMEDIATE')
        try:
            self._require_generation(owner,generation)
            self.cas.get(digest)
            self.db.execute('INSERT OR REPLACE INTO leases VALUES(?,?,?,?)',(digest,owner,generation,now+ttl_seconds))
            self._record_mutation('lease',{'object':digest,'owner':owner,'generation':generation,'expires':now+ttl_seconds})
            self.db.commit(); return now+ttl_seconds
        except BaseException:
            self.db.rollback(); raise

    def renew_lease(self,digest,owner,generation,*,ttl_seconds=30.,now=None):
        now=time.time() if now is None else float(now)
        if not math.isfinite(now) or not math.isfinite(ttl_seconds) or ttl_seconds<=0:
            raise ValueError('finite time and positive TTL required')
        self.db.execute('BEGIN IMMEDIATE')
        try:
            self._require_generation(owner,generation)
            row=self.db.execute('SELECT expires FROM leases WHERE digest=? AND owner=? AND generation=?',(digest,owner,generation)).fetchone()
            if row is None or row[0]<=now: raise RuntimeError('lease expired or missing')
            self.cas.get(digest)
            self.db.execute('UPDATE leases SET expires=? WHERE digest=? AND owner=? AND generation=?',(now+ttl_seconds,digest,owner,generation))
            self._record_mutation('renew',{'object':digest,'owner':owner,'generation':generation,'expires':now+ttl_seconds})
            self.db.commit(); return now+ttl_seconds
        except BaseException:
            self.db.rollback(); raise

    def release_lease(self,digest,owner,generation):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            self._require_generation(owner,generation)
            cur=self.db.execute('DELETE FROM leases WHERE digest=? AND owner=? AND generation=?',(digest,owner,generation))
            self._record_mutation('release',{'object':digest,'owner':owner,'generation':generation})
            self.db.commit(); return bool(cur.rowcount)
        except BaseException:
            self.db.rollback(); raise

    def collect_orphans(self,*,now=None):
        now=time.time() if now is None else float(now)
        if not math.isfinite(now): raise ValueError('finite GC time required')
        self.db.execute('BEGIN IMMEDIATE')
        try:
            self.db.execute('DELETE FROM leases WHERE expires<=?',(now,))
            retained={x[0] for x in self.db.execute('SELECT object_digest FROM artifacts')}
            retained.update(x[0] for x in self.db.execute('SELECT digest FROM leases JOIN lease_owners USING(owner) WHERE leases.generation=lease_owners.generation AND expires>?',(now,)))
            removed=0
            for d in self.cas.iter_digests():
                if d not in retained: removed+=int(self.cas.delete(d))
            self._record_mutation('gc',{'removed':removed})
            self.db.commit(); return removed
        except BaseException:
            self.db.rollback(); raise

    def admission(self):
        self.verify_recovery_checkpoint()
        report=self.verify_all()
        if self.db.execute('SELECT COUNT(*) FROM artifacts WHERE token_count<0').fetchone()[0]:
            raise RuntimeError('source token accounting required before acceleration admission')
        count=self.db.execute('SELECT COUNT(*) FROM artifacts').fetchone()[0]
        total=sum(self.cas.path_for(d).stat().st_size for d in self.cas.iter_digests())
        tokens=self.db.execute('SELECT COALESCE(SUM(MAX(token_count,0)),0) FROM artifacts').fetchone()[0]
        self.budget.validate(blocks=count,tokens=tokens,payload_bytes=total)
        if report['bad']: raise RuntimeError('corrupt payloads fail resource admission')
        return report
