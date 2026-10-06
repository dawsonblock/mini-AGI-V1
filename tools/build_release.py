"""Regenerate source integrity metadata and deterministic ZIP using an external key."""
import argparse,hashlib,json,zipfile,tomllib,sys
sys.dont_write_bytecode = True
from pathlib import Path
from cryptography.hazmat.primitives import serialization
from verify_release import ENVELOPE,verify


def write_json(path,obj):path.write_text(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+'\n')

def build(root,key_path,zip_path=None):
    root=Path(root);metadata=tomllib.loads((root/'pyproject.toml').read_text())['project'];version=metadata['version']
    banned={'__pycache__','.pytest_cache','build','dist','.git'}
    files=[p for p in root.rglob('*') if p.is_file()]
    for p in files:
        if p.is_symlink() or any(x in banned or x.endswith('.egg-info') for x in p.relative_to(root).parts) or p.suffix in {'.pyc','.pyo','.whl','.o','.a','.so','.dylib'}:raise ValueError('release contains generated material: '+str(p))
    components=[{'type':'library','name':name,'version':version,'bom-ref':name+'@'+version} for name in ['mini-agi-converged','kvcontinual-execution','minagi-egai-governance','egai-research-kernel']]
    # Requirements are declared ranges, not a fully resolved production lockfile.
    sbom={'bomFormat':'CycloneDX','specVersion':'1.5','version':1,
          'metadata':{'component':components[0]},'components':components[1:],
          'properties':[{'name':'mini-agi:dependency-resolution','value':'declared ranges; see pyproject.toml, not a production lockfile'}],
          'dependencies':[{'ref':components[0]['bom-ref'],'dependsOn':[c['bom-ref'] for c in components[1:]]}]}
    import re
    requirement_groups=[('required',metadata['dependencies'])]+list(metadata.get('optional-dependencies',{}).items())
    for group,requirements in requirement_groups:
        for requirement in requirements:
            name=re.split(r'[<>=!~;\[]',requirement,1)[0].strip()
            ref='python-requirement:'+group+':'+name
            sbom['components'].append({'type':'library','name':name,'bom-ref':ref,
                'properties':[{'name':'declared-requirement','value':requirement},
                              {'name':'dependency-group','value':group}]})
            sbom['dependencies'][0]['dependsOn'].append(ref)
    # Inventory vendored source separately; licenses supplied with native source remain authoritative.
    for vendor in sorted((root/'third_party').iterdir()):
        sbom['components'].append({'type':'library','name':vendor.name,'bom-ref':'vendored:'+vendor.name,
            'properties':[{'name':'source-location','value':'third_party/'+vendor.name}]})
    write_json(root/'SBOM.cdx.json',sbom)
    files=[p for p in root.rglob('*') if p.is_file() and p.relative_to(root).as_posix() not in ENVELOPE]
    manifest={'schema':'mini-agi-source-manifest-v1','name':metadata['name'],'version':version,
              'envelope_exclusions':sorted(ENVELOPE),'files':{p.relative_to(root).as_posix():{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(files)}}
    write_json(root/'SOURCE_MANIFEST.json',manifest)
    key=serialization.load_pem_private_key(Path(key_path).read_bytes(),password=None)
    pub=key.public_key().public_bytes(serialization.Encoding.Raw,serialization.PublicFormat.Raw)
    att={'schema':'mini-agi-detached-release-attestation-v1','name':metadata['name'],'version':version,
         'manifest_sha256':hashlib.sha256((root/'SOURCE_MANIFEST.json').read_bytes()).hexdigest(),
         'sbom_sha256':hashlib.sha256((root/'SBOM.cdx.json').read_bytes()).hexdigest(),
         'validation_sha256':hashlib.sha256((root/'RELEASE_VALIDATION.json').read_bytes()).hexdigest(),
         'release_key_sha256':hashlib.sha256(pub).hexdigest(),
         'signing_scope':'local-build-integrity; newly generated key, no inherited maintainer trust',
         'production_mac_qualified':False,'execution_baseline':'exact-selected-replay'}
    write_json(root/'RELEASE_ATTESTATION.json',att)
    (root/'RELEASE_SIGNATURE.bin').write_bytes(key.sign((root/'RELEASE_ATTESTATION.json').read_bytes()))
    (root/'RELEASE_PUBLIC_KEY.pem').write_bytes(key.public_key().public_bytes(serialization.Encoding.PEM,serialization.PublicFormat.SubjectPublicKeyInfo))
    result=verify(root)
    if zip_path:
        with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
            for p in sorted(root.rglob('*')):
                if not p.is_file():continue
                info=zipfile.ZipInfo(root.name+'/'+p.relative_to(root).as_posix(),(1980,1,1,0,0,0))
                info.compress_type=zipfile.ZIP_DEFLATED;info.create_system=3
                mode=0o755 if p.suffix=='.sh' else 0o644
                info.external_attr=(0o100000|mode)<<16
                z.writestr(info,p.read_bytes())
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(Path(__file__).resolve().parents[1]));p.add_argument('--private-key',required=True);p.add_argument('--zip')
    a=p.parse_args();print(json.dumps(build(a.root,a.private_key,a.zip),sort_keys=True))
