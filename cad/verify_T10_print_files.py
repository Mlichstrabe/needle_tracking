"""Independent final STL verification without importing CAD/OCP/VTK libraries."""
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
import trimesh
from shapely.geometry import Polygon

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/cad_laser100m_T10_onepiece'
checks={}
for name in ['T10_PRINT_ONE_PIECE_6mm_WITH_BRIM','T10_CAD_ONE_PIECE_FINISHED']:
    mesh=trimesh.load_mesh(OUT/(name+'.stl'))
    assert mesh.is_watertight and mesh.is_winding_consistent and len(mesh.split())==1
    assert abs(mesh.bounds[0,2])<1e-5
    previous=None
    rows=[]
    for height in [.001]+list(np.arange(.1,mesh.bounds[1,2],.2)):
        path=mesh.section(plane_origin=[0,0,height],plane_normal=[0,0,1])
        assert path is not None and len(path.discrete)==1
        poly=Polygon(path.discrete[0][:,:2])
        assert poly.is_valid and not poly.interiors
        extra=0 if previous is None else poly.difference(previous.buffer(.2)).area
        assert extra<.01,(name,height,extra)
        rows.append(dict(z=float(height),islands=1,area_mm2=poly.area,unsupported_beyond_02mm_mm2=extra))
        previous=poly
    checks[name]=dict(watertight=True,solid_components=1,bounds=mesh.bounds.tolist(),layers=rows)
    if 'PRINT' in name:
        assert mesh.bounds[0,:2].min()>5 and mesh.bounds[1,:2].max()<175
    else:
        assert abs(mesh.extents[0]-212)<1e-4

report=dict(independent_stl_validation='passed',checks=checks,
    generator_runtime_note='CAD-generated artifacts and STEP roundtrips completed, but Windows CAD native library teardown returned an access violation. This independent process does not import CAD/OCP/VTK.',
    gcode_sliced=False,physical_print_verified=False)
(OUT/'independent_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
note='\n\n生成环境记录：本机CAD原生库在文件生成及STEP回读完成后的退出阶段发生访问异常；单独导入CAD库也可复现。最终STL随后在不加载CAD库的独立进程重新校验，水密、单实体、落床、逐层轮廓检查通过。详见 independent_validation.json。实际切片与实物打印未验证。\n'
readme=OUT/'START_HERE_打印说明.md'
txt=readme.read_text(encoding='utf-8')
if '生成环境记录' not in txt: readme.write_text(txt+note,encoding='utf-8')
manifest={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in OUT.iterdir() if f.is_file() and f.name!='sha256.json'}
(OUT/'sha256.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
bundle=ROOT/'output/T10_ONE_PIECE_6mm_PRINT_PACKAGE.zip'
sources=['laser100m_T10_onepiece.py','verify_T10_print_files.py','T10_design.md','laser100m_T5.py','laser100m_T2.py','laser100m_T2_parameters.json']
with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
    for f in OUT.iterdir():
        if f.is_file(): z.write(f,f.name)
    for n in sources: z.write(ROOT/'cad'/n,'source/'+n)
with zipfile.ZipFile(bundle) as z:
    assert z.testzip() is None
    for name,digest in manifest.items():
        assert hashlib.sha256(z.read(name)).hexdigest()==digest
print(json.dumps(dict(independent_validation='PASS',stls=2,section_samples_per_stl=27,
                      archive_integrity='PASS',bundle=str(bundle)),ensure_ascii=False))
