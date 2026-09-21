"""Package the new IMU latch revision without changing T12 files."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/cad_laser100m_T13_imu_slide'
ZIP=ROOT/'output/T13_IMU_SLIDE_LATCH_STEP_STL.zip'

def main():
    assert json.loads((OUT/'independent_validation.json').read_text(encoding='utf-8'))['independent_validation']=='PASS'
    entries={f.relative_to(OUT).as_posix():f for f in OUT.rglob('*') if f.is_file() and f.name!='sha256.json'}
    for n in ['laser100m_T13_imu_slide.py','verify_T13_imu_slide.py','render_T13_imu_slide.py',
              'package_T13_imu_slide.py','laser100m_T12_simple_bracket.py','verify_T12_simple_bracket.py',
              'render_T12_simple_bracket.py']:
        entries['source/cad/'+n]=ROOT/'cad'/n
    needle='output/cad_laser100m_T10_onepiece/T10_REFERENCE_INSTALLED_NEEDLE.step'
    entries['source/'+needle]=ROOT/needle
    manifest={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in sorted(entries.items())}
    (OUT/'sha256.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    with zipfile.ZipFile(ZIP,'w',zipfile.ZIP_DEFLATED) as z:
        for n,p in entries.items():z.write(p,n)
        z.write(OUT/'sha256.json','sha256.json')
    with zipfile.ZipFile(ZIP) as z:
        assert z.testzip() is None
        for n,d in manifest.items():assert hashlib.sha256(z.read(n)).hexdigest()==d
    print(json.dumps(dict(archive=str(ZIP),archive_integrity='PASS',sha256='PASS'),ensure_ascii=False))

if __name__=='__main__':main()
