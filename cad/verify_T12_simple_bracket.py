"""Independent mesh/bed-contact checks. Does not load CAD/OCP libraries."""
from pathlib import Path
import hashlib
import json
import numpy as np
import trimesh
from shapely.geometry import Polygon

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/cad_laser100m_T12_simple_bracket'


def region(mesh,z):
    sec=mesh.section(plane_origin=[0,0,z],plane_normal=[0,0,1])
    assert sec is not None, ('missing section',z)
    # Symmetric difference handles internal hole loops without assuming order.
    p=Polygon()
    for loop in sec.discrete:
        poly=Polygon(loop[:,:2])
        assert poly.is_valid
        p=p.symmetric_difference(poly)
    assert p.is_valid and p.area>0
    return p


def main():
    meshes={}
    report={}
    for name in ('T12_BRACKET','T12_PRINT_SIDE_DOWN'):
        mesh=trimesh.load_mesh(OUT/(name+'.stl'),process=True)
        assert mesh.is_watertight and mesh.is_winding_consistent
        assert len(mesh.split())==1 and mesh.volume>0
        meshes[name]=mesh
        report[name]=dict(watertight=True,winding_consistent=True,components=1,
                          dimensions_mm=mesh.extents.tolist(),volume_mm3=mesh.volume)
    assembly=meshes['T12_BRACKET']
    printing=meshes['T12_PRINT_SIDE_DOWN']
    assert abs(assembly.volume-printing.volume)<.05
    assert abs(printing.bounds[0,2])<1e-5
    assert printing.bounds[0,:2].min()>9.9
    # Verify every vertex is the same part after the stated print transform.
    validation=json.loads((OUT/'validation.json').read_text(encoding='utf-8'))
    from scipy.spatial import cKDTree
    verts=assembly.vertices.copy()
    verts[:,1],verts[:,2]=-assembly.vertices[:,2],assembly.vertices[:,1]
    verts+=np.array(validation['print_translation_mm'])
    max_vertex_delta=float(cKDTree(printing.vertices).query(verts)[0].max())
    assert max_vertex_delta<1e-4
    first=region(printing,.001)
    assert first.geom_type=='Polygon', 'first layer must be connected'
    rows=[]
    previous=first
    for z in np.arange(.1,printing.bounds[1,2],.2):
        p=region(printing,float(z))
        extra=p.difference(previous.buffer(.2)).area
        rows.append(dict(z_mm=round(float(z),3),area_mm2=round(p.area,4),
                         area_beyond_previous_02mm_mm2=round(extra,4)))
        previous=p
    unsupported=[r for r in rows if r['area_beyond_previous_02mm_mm2']>.05]
    # We expect overhangs: report them, never claim a support-free print.
    assert unsupported
    expected=validation['needle_sha256']
    assert hashlib.sha256(Path(validation['needle_source']).read_bytes()).hexdigest()==expected
    result=dict(independent_validation='PASS',meshes=report,
                print_orientation_vertex_delta_mm=max_vertex_delta,
                first_layer_area_mm2=first.area,first_layer_connected=True,
                first_layer_holes=len(first.interiors),layer_samples=rows,
                overhang_samples=unsupported,support_required=True,
                unchanged_existing_needle_sha256=expected,
                generator_exit_code=1,
                generator_runtime_note='CAD wrote all files and passed STEP roundtrips; exit code 1 follows native-library teardown. A bare cadquery import also exited 1. This independent process loads no CAD library.',
                slicing_or_physical_fit_verified=False)
    (OUT/'independent_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(independent_validation='PASS',single_printed_part=True,
                         first_layer_connected=True,first_layer_area_mm2=round(first.area,2),
                         layer_samples=len(rows),overhang_samples=len(unsupported),support_required=True)))


if __name__=='__main__':
    main()
