"""Independent T13 mesh verification without importing CAD/OCP."""
import hashlib
import json
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from verify_T12_simple_bracket import region

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/cad_laser100m_T13_imu_slide'


def main():
    checks={};meshes={}
    for n in ('T13_BRACKET','T13_PRINT_SIDE_DOWN'):
        m=trimesh.load_mesh(OUT/(n+'.stl'),process=True)
        assert m.is_watertight and m.is_winding_consistent and len(m.split())==1
        assert m.volume>0
        meshes[n]=m
        checks[n]=dict(watertight=True,single_component=True,dimensions=m.extents.tolist())
    v=json.loads((OUT/'validation.json').read_text(encoding='utf-8'))
    raw=meshes['T13_BRACKET'];m=meshes['T13_PRINT_SIDE_DOWN']
    verts=raw.vertices.copy();verts[:,1],verts[:,2]=-raw.vertices[:,2],raw.vertices[:,1]
    verts+=v['print_translation_mm']
    assert cKDTree(m.vertices).query(verts)[0].max()<1e-4
    assert abs(m.bounds[0,2])<1e-5
    first=region(m,.001);assert first.geom_type=='Polygon'
    previous=first;overhangs=[]
    heights=np.arange(.1,m.bounds[1,2],.2)
    for z in heights:
        current=region(m,float(z));area=current.difference(previous.buffer(.2)).area
        if area>.05:overhangs.append(dict(z_mm=float(z),overhang_area_mm2=area))
        previous=current
    assert overhangs
    assert hashlib.sha256(Path(v['needle_source']).read_bytes()).hexdigest()==v['needle_sha256']
    result=dict(independent_validation='PASS',checks=checks,first_layer_connected=True,
                first_layer_area_mm2=first.area,layer_samples=len(heights),overhangs=overhangs,
                support_required=True,physical_snap_cycle_test=False,
                runtime_note='CAD export/checks completed; CAD process exit 1. Independent mesh checks do not load CAD libraries.')
    (OUT/'independent_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(independent_validation='PASS',first_layer_area=first.area,overhang_samples=len(overhangs))))


if __name__=='__main__':main()
