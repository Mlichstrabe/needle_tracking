"""Package only the T12 deliverables and their reproducible source inputs."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/cad_laser100m_T12_simple_bracket'
ZIP=ROOT/'output/T12_SIMPLE_BRACKET_STEP_STL.zip'

def main():
    report=json.loads((OUT/'independent_validation.json').read_text(encoding='utf-8'))
    assert report['independent_validation']=='PASS'
    sources=['laser100m_T12_simple_bracket.py','render_T12_simple_bracket.py',
             'verify_T12_simple_bracket.py','package_T12_simple_bracket.py','T12_simple_bracket_design.md']
    entries={f.relative_to(OUT).as_posix():f for f in OUT.rglob('*')
             if f.is_file() and f.name!='sha256.json'}
    entries.update({'source/cad/'+n:ROOT/'cad'/n for n in sources})
    reference=Path('output/cad_laser100m_T10_onepiece/T10_REFERENCE_INSTALLED_NEEDLE.step')
    entries['source/'+reference.as_posix()]=ROOT/reference
    manifest={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in sorted(entries.items())}
    (OUT/'sha256.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    with zipfile.ZipFile(ZIP,'w',zipfile.ZIP_DEFLATED) as z:
        for n,p in sorted(entries.items()): z.write(p,n)
        z.write(OUT/'sha256.json','sha256.json')
    with zipfile.ZipFile(ZIP) as z:
        assert z.testzip() is None
        for n,digest in manifest.items():
            assert hashlib.sha256(z.read(n)).hexdigest()==digest,n
    print(json.dumps(dict(archive=str(ZIP),entries=len(entries)+1,crc='PASS',sha256='PASS'),ensure_ascii=False))

if __name__=='__main__':
    main()
