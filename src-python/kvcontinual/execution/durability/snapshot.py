from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import time
import uuid

GENESIS='0'*64

def _sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def _canonical(obj:dict)->bytes:
    return json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf-8')

def _sqlite_integrity(path:Path)->str:
    c=sqlite3.connect(str(path))
    try: return str(c.execute('PRAGMA integrity_check').fetchone()[0])
    finally: c.close()

@dataclass(frozen=True,slots=True)
class SnapshotFile:
    name:str; sha256:str; size_bytes:int

@dataclass(frozen=True,slots=True)
class SnapshotManifest:
    format:str
    created_at:float
    files:tuple[SnapshotFile,...]
    snapshot_id:str|None=None
    parent_snapshot_id:str|None=None
    parent_snapshot_name:str|None=None
    lineage_depth:int=0
    root_snapshot_id:str|None=None
    parent_chain_hash:str|None=None
    chain_hash:str|None=None
    authority_key_id:str|None=None
    snapshot_signature:str|None=None

class SQLiteSnapshotManager:
    """SQLite-consistent snapshots with v4 lineage commitments and optional signatures."""
    def __init__(self,destination_root:str|Path,*,authority=None):
        self.destination_root=Path(destination_root); self.destination_root.mkdir(parents=True,exist_ok=True); self.authority=authority

    @staticmethod
    def _backup_one(src:Path,dst:Path)->None:
        s=sqlite3.connect(str(src)); d=sqlite3.connect(str(dst))
        try:
            s.backup(d); d.commit(); result=d.execute('PRAGMA integrity_check').fetchone()[0]
            if result!='ok': raise RuntimeError(f'snapshot integrity check failed: {result}')
        finally: d.close(); s.close()

    @staticmethod
    def _manifest_core(manifest:dict)->dict:
        return {k:manifest.get(k) for k in ('format','created_at','files','parent_snapshot_id','parent_snapshot_name','lineage_depth')}

    @classmethod
    def _snapshot_id(cls,manifest:dict)->str:
        return hashlib.sha256(_canonical(cls._manifest_core(manifest))).hexdigest()

    @staticmethod
    def _chain_hash(*,snapshot_id:str,root_snapshot_id:str,parent_chain_hash:str,lineage_depth:int)->str:
        body={'snapshot_id':snapshot_id,'root_snapshot_id':root_snapshot_id,'parent_chain_hash':parent_chain_hash,'lineage_depth':int(lineage_depth)}
        return hashlib.sha256(_canonical(body)).hexdigest()

    @staticmethod
    def _signature_payload(manifest:dict)->dict:
        p=dict(manifest); p.pop('snapshot_signature',None); return p

    def create(self,sources:dict[str,str|Path],*,name:str|None=None,parent_snapshot:str|Path|None=None)->Path:
        stamp=name or time.strftime('%Y%m%dT%H%M%SZ',time.gmtime()); final=self.destination_root/stamp
        if final.exists(): raise FileExistsError(final)
        parent_id=parent_name=None; parent_chain=GENESIS; root_id=None; depth=0
        if parent_snapshot is not None:
            parent=Path(parent_snapshot)
            ok,problems=self.verify(parent,authority=self.authority,require_signature=self.authority is not None)
            if not ok: raise RuntimeError('parent snapshot verification failed: '+','.join(problems))
            pm=json.loads((parent/'SNAPSHOT.json').read_text()); parent_id=pm.get('snapshot_id') or self._snapshot_id(pm)
            parent_name=parent.name; depth=int(pm.get('lineage_depth',0))+1
            root_id=pm.get('root_snapshot_id') or parent_id
            parent_chain=pm.get('chain_hash') or hashlib.sha256(('legacy:'+parent_id).encode()).hexdigest()
        tmp=Path(tempfile.mkdtemp(prefix=f'.{stamp}.',dir=self.destination_root))
        try:
            files=[]
            for logical,source in sorted(sources.items()):
                if not logical or '/' in logical or '\\' in logical: raise ValueError('invalid logical snapshot name')
                src=Path(source)
                if not src.is_file(): raise FileNotFoundError(src)
                dst=tmp/f'{logical}.sqlite3'; self._backup_one(src,dst); files.append(SnapshotFile(dst.name,_sha256(dst),dst.stat().st_size))
            manifest={'format':'rc10-snapshot-v4','created_at':time.time(),'files':[asdict(x) for x in files],
                      'parent_snapshot_id':parent_id,'parent_snapshot_name':parent_name,'lineage_depth':depth}
            sid=self._snapshot_id(manifest); root_id=sid if depth==0 else root_id
            manifest['snapshot_id']=sid; manifest['root_snapshot_id']=root_id; manifest['parent_chain_hash']=parent_chain
            manifest['chain_hash']=self._chain_hash(snapshot_id=sid,root_snapshot_id=root_id,parent_chain_hash=parent_chain,lineage_depth=depth)
            if self.authority is not None:
                manifest['authority_key_id']=self.authority.active_key_id
                key_id,sig=self.authority.sign_payload(self._signature_payload(manifest),purpose='snapshot_manifest')
                if key_id!=manifest['authority_key_id']: raise RuntimeError('control authority rotated during snapshot signing')
                manifest['snapshot_signature']=sig
            mp=tmp/'SNAPSHOT.json'; mp.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
            with mp.open('rb') as f: os.fsync(f.fileno())
            os.replace(tmp,final)
            try:
                dfd=os.open(self.destination_root,os.O_RDONLY); os.fsync(dfd); os.close(dfd)
            except OSError: pass
            return final
        finally:
            if tmp.exists(): shutil.rmtree(tmp,ignore_errors=True)

    @classmethod
    def verify(cls,snapshot_dir:str|Path,*,verify_lineage:bool=True,_seen:set[str]|None=None,authority=None,
               require_signature:bool=False)->tuple[bool,list[str]]:
        root=Path(snapshot_dir); problems=[]
        try: manifest=json.loads((root/'SNAPSHOT.json').read_text())
        except Exception: return False,['invalid_or_missing_manifest']
        fmt=manifest.get('format')
        if fmt not in {'rc10-snapshot-v1','rc10-snapshot-v2','rc10-snapshot-v3','rc10-snapshot-v4'}: problems.append('unsupported_snapshot_format')
        rows=manifest.get('files',[])
        if not isinstance(rows,list) or not rows: problems.append('missing_files')
        for row in rows:
            p=root/row.get('name','')
            if p.parent.resolve()!=root.resolve(): problems.append(f"{row.get('name')}:unsafe_path"); continue
            if not p.is_file(): problems.append(f"{row.get('name')}:missing"); continue
            if _sha256(p)!=row.get('sha256'): problems.append(f'{p.name}:digest_mismatch')
            try:
                result=_sqlite_integrity(p)
                if result!='ok': problems.append(f'{p.name}:sqlite_{result}')
            except sqlite3.Error: problems.append(f'{p.name}:sqlite_error')
        if fmt in {'rc10-snapshot-v3','rc10-snapshot-v4'}:
            got=manifest.get('snapshot_id'); want=cls._snapshot_id(manifest)
            if got!=want: problems.append('snapshot_id_mismatch')
            depth=int(manifest.get('lineage_depth',0)); parent_id=manifest.get('parent_snapshot_id'); parent_name=manifest.get('parent_snapshot_name')
            if depth==0 and (parent_id or parent_name): problems.append('lineage_root_has_parent')
            if depth>0 and (not parent_id or not parent_name): problems.append('lineage_parent_missing')
            if fmt=='rc10-snapshot-v4':
                root_id=manifest.get('root_snapshot_id'); parent_chain=manifest.get('parent_chain_hash')
                if depth==0 and root_id!=got: problems.append('lineage_root_id_mismatch')
                if not root_id: problems.append('lineage_root_id_missing')
                if not isinstance(parent_chain,str) or len(parent_chain)!=64: problems.append('parent_chain_hash_invalid')
                elif got and root_id:
                    want_chain=cls._chain_hash(snapshot_id=got,root_snapshot_id=root_id,parent_chain_hash=parent_chain,lineage_depth=depth)
                    if manifest.get('chain_hash')!=want_chain: problems.append('lineage_chain_hash_mismatch')
                sig=manifest.get('snapshot_signature'); key_id=manifest.get('authority_key_id')
                if require_signature and (not sig or not key_id): problems.append('snapshot_signature_missing')
                if authority is not None and sig and key_id:
                    if not authority.verify_payload(cls._signature_payload(manifest),purpose='snapshot_manifest',key_id=key_id,signature=sig):
                        problems.append('snapshot_signature_invalid')
                elif require_signature and authority is None: problems.append('snapshot_authority_unavailable')
            if verify_lineage and parent_id and parent_name:
                if Path(parent_name).name!=parent_name or parent_name in {'.','..'}:
                    problems.append('lineage_parent_unsafe_name'); return not problems,problems
                seen=set() if _seen is None else set(_seen)
                if got in seen: problems.append('lineage_cycle')
                else:
                    seen.add(got); parent=root.parent/parent_name
                    if not parent.is_dir(): problems.append('lineage_parent_not_found')
                    else:
                        pok,pproblems=cls.verify(parent,verify_lineage=True,_seen=seen,authority=authority,require_signature=require_signature)
                        if not pok: problems.extend(f'parent:{x}' for x in pproblems)
                        else:
                            pm=json.loads((parent/'SNAPSHOT.json').read_text()); pid=pm.get('snapshot_id') or cls._snapshot_id(pm)
                            if pid!=parent_id: problems.append('lineage_parent_id_mismatch')
                            if int(pm.get('lineage_depth',0))+1!=depth: problems.append('lineage_depth_mismatch')
                            if fmt=='rc10-snapshot-v4':
                                if manifest.get('root_snapshot_id')!=(pm.get('root_snapshot_id') or pid): problems.append('lineage_root_continuity_mismatch')
                                pchain=pm.get('chain_hash') or hashlib.sha256(('legacy:'+pid).encode()).hexdigest()
                                if manifest.get('parent_chain_hash')!=pchain: problems.append('lineage_parent_chain_mismatch')
        return not problems,problems

    @classmethod
    def restore(cls,snapshot_dir:str|Path,destinations:dict[str,str|Path],*,overwrite:bool=False,authority=None,
                require_signature:bool=False)->dict[str,str]:
        root=Path(snapshot_dir); ok,problems=cls.verify(root,authority=authority,require_signature=require_signature)
        if not ok: raise RuntimeError('snapshot verification failed: '+','.join(problems))
        manifest=json.loads((root/'SNAPSHOT.json').read_text()); by_name={str(r['name']):r for r in manifest['files']}
        staged:dict[str,Path]={}; backups:dict[str,Path]={}; published=[]
        try:
            for logical,dest_raw in sorted(destinations.items()):
                src=root/f'{logical}.sqlite3'; row=by_name.get(src.name)
                if row is None: raise KeyError(f'snapshot does not contain {logical!r}')
                dest=Path(dest_raw); dest.parent.mkdir(parents=True,exist_ok=True)
                if dest.exists() and not overwrite: raise FileExistsError(dest)
                fd,tmp_name=tempfile.mkstemp(prefix=f'.{dest.name}.restore.',dir=dest.parent); os.close(fd)
                tmp=Path(tmp_name); shutil.copy2(src,tmp)
                with tmp.open('rb') as f: os.fsync(f.fileno())
                if _sha256(tmp)!=row['sha256'] or _sqlite_integrity(tmp)!='ok': raise RuntimeError(f'restore staging verification failed: {logical}')
                staged[logical]=tmp
            for logical,dest_raw in sorted(destinations.items()):
                dest=Path(dest_raw)
                if dest.exists():
                    backup=dest.with_name(dest.name+f'.pre-restore-{uuid.uuid4().hex}'); os.replace(dest,backup); backups[logical]=backup
                os.replace(staged[logical],dest); published.append(logical)
                try:
                    dfd=os.open(dest.parent,os.O_RDONLY); os.fsync(dfd); os.close(dfd)
                except OSError: pass
            for logical,dest_raw in sorted(destinations.items()):
                dest=Path(dest_raw); row=by_name[f'{logical}.sqlite3']
                if _sha256(dest)!=row['sha256'] or _sqlite_integrity(dest)!='ok': raise RuntimeError(f'published restore verification failed: {logical}')
            for b in backups.values():
                try: b.unlink()
                except FileNotFoundError: pass
            return {k:str(Path(v)) for k,v in destinations.items()}
        except Exception:
            for logical in reversed(published):
                dest=Path(destinations[logical])
                try: dest.unlink()
                except FileNotFoundError: pass
                if logical in backups and backups[logical].exists(): os.replace(backups[logical],dest)
            raise
        finally:
            for p in staged.values():
                try: p.unlink()
                except FileNotFoundError: pass
            for logical,b in backups.items():
                dest=Path(destinations[logical])
                if b.exists() and not dest.exists(): os.replace(b,dest)
                elif b.exists(): b.unlink()
