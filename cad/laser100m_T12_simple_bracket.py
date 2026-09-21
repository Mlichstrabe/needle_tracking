"""One printed holder, existing T10 needle, two purchased electronics modules.

uv run --python 3.12 --offline --with cadquery==2.6.1 --with trimesh --with matplotlib --with shapely --with networkx --with scipy python -X utf8 cad/laser100m_T12_simple_bracket.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cadquery as cq
import trimesh

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/cad_laser100m_T12_simple_bracket'
NEEDLE = ROOT / 'output/cad_laser100m_T10_onepiece/T10_REFERENCE_INSTALLED_NEEDLE.step'
P = dict(revision='T12-simple-open-bracket', units='mm', printed_part_count=1,
         plate_length=55.0, plate_width=43.8, plate_thickness=2.4,
         needle_interface_x=-3.0, needle_axis_z=4.2, socket_arc_diameter=4.5,
         socket_flat_z=2.95, socket_depth=12.4, needle_insertion_length=12.0,
         laser_length=46.47, laser_width=37.0, laser_height=24.6,
         laser_front_x=1.0, laser_top_z=-1.0, laser_bottom_z=-25.6,
         laser_side_clearance_each=1.0, support_thickness=2.4,
         imu_base_length=42.0, imu_base_width=30.0, imu_front_x=12.0,
         imu_base_bottom_z=5.4, imu_pcb_thickness_assumed=1.6,
         imu_underside_clearance_assumed=3.0, imu_side_clearance_each=1.0,
         imu_upper_module_length_reference=22.0, imu_upper_module_width_reference=20.0,
         imu_upper_module_height_reference=7.0, imu_header_spacing_assumed=3.0,
         actual_connector_geometry_verified=False, physical_fit_verified=False)


def box(x0, x1, y0, y1, z0, z1):
    return cq.Workplane('XY').box(x1-x0, y1-y0, z1-z0).translate(
        ((x0+x1)/2, (y0+y1)/2, (z0+z1)/2)).val()


def bounds(shape):
    from OCP.BRepBndLib import BRepBndLib
    from OCP.Bnd import Bnd_Box
    b = Bnd_Box()
    BRepBndLib.AddOptimal_s(shape.wrapped, b, False, False)
    return list(b.Get())


def dimensions(shape):
    b = bounds(shape)
    return [round(b[i+3]-b[i], 6) for i in range(3)]


def one(shape):
    assert shape.isValid(), 'invalid solid'
    ss = shape.Solids()
    assert len(ss) == 1, ('expected one body', len(ss))
    return ss[0]


def overlap(a, b):
    return sum(s.Volume() for s in a.intersect(b).Solids())


def build_holder():
    plate = cq.Workplane(obj=box(0,55,-21.9,21.9,0,2.4)).edges('|Z').fillet(1.2).val()
    socket = cq.Workplane(obj=box(-3,11.8,-4.6,4.6,0,8.9)).edges('|X').fillet(.6).val()
    s = plate.fuse(socket)
    # Four robust support feet, open fore/aft and between the feet.
    for x0 in (11.0,34.0):
        for sign in (-1,1):
            ya,yb = (19.5,21.9) if sign > 0 else (-21.9,-19.5)
            fa,fb = (15.5,21.9) if sign > 0 else (-21.9,-15.5)
            s = s.fuse(box(x0,x0+7,ya,yb,-28,2.4))
            s = s.fuse(box(x0,x0+7,fa,fb,-28,-25.6))
    # IMU glue lands under board corners. PCB bottom is 3 mm above plate.
    for x0 in (13.0,48.0):
        for sign in (-1,1):
            a,b = (11.0,15.0) if sign > 0 else (-15.0,-11.0)
            s = s.fuse(box(x0,x0+4,a,b,2.4,5.4))
            a,b = (16.0,18.4) if sign > 0 else (-18.4,-16.0)
            s = s.fuse(box(x0,x0+4,a,b,2.4,6.6))
    for a,b in ((11,15),(-15,-11)):
        s = s.fuse(box(8.6,11,a,b,2.4,6.6))
    # D profile copied from T10 interface, with radial/flat glue allowance.
    bore = cq.Solid.makeCylinder(2.25,12.9,cq.Vector(-3.5,0,4.2),cq.Vector(1,0,0))
    bore = bore.intersect(box(-4,10,-4,4,2.95,8))
    return one(s.cut(bore).clean())


def reference_parts():
    needle = one(cq.importers.importStep(str(NEEDLE)).val())
    laser = box(1,47.47,-18.5,18.5,-25.6,-1)
    pcb = box(12,54,-15,15,5.4,7)
    # Illustrative height only: no clamping surface depends on this height.
    imu_header = box(24,42,-8,8,7,10)
    imu_module = box(22,44,-10,10,10,17)
    imu = one(pcb.fuse(imu_header).fuse(imu_module).clean())
    return needle, laser, imu, pcb, imu_header, imu_module


def export_mesh(path, shape):
    verts, faces = shape.tessellate(.025,.08)
    mesh = trimesh.Trimesh(vertices=[v.toTuple() for v in verts], faces=faces,process=True)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    assert mesh.is_watertight and mesh.is_winding_consistent
    assert len(mesh.split()) == 1
    mesh.export(path)
    return mesh


def export_step(path, shape, count=1):
    cq.exporters.export(shape,str(path))
    back = cq.importers.importStep(str(path)).val()
    dv = abs(back.Volume()-shape.Volume())
    db = max(abs(a-b) for a,b in zip(bounds(back),bounds(shape)))
    assert back.isValid() and len(back.Solids()) == count
    assert dv < max(1e-4,shape.Volume()*1e-8) and db < 1e-5
    return dict(valid=True, solids=count, volume_delta_mm3=dv,bounds_delta_mm=db)


def assert_clear(name, holder, ref):
    value = overlap(holder,ref)
    assert value < 1e-6, (name,value)
    return round(value,9)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    needle_hash = hashlib.sha256(NEEDLE.read_bytes()).hexdigest()
    holder = build_holder()
    needle,laser,imu,pcb,header,module = reference_parts()
    checks = {n:assert_clear(n,holder,p) for n,p in
              [('existing_needle',needle),('laser_full_envelope',laser),('imu_assembly',imu)]}
    # Explicitly verify insertion approaches, not merely the final pose.
    paths={}
    for name,ref,shifts in [
        ('needle_from_front',needle,[(-v,0,0) for v in (30,15,8,4,1,0)]),
        ('laser_from_rear',laser,[(v,0,0) for v in (65,50,35,20,10,4,1,0)]),
        ('imu_from_above',imu,[(0,0,v) for v in (30,15,8,2,0)])]:
        paths[name]=[assert_clear(name,holder,ref.translate(v)) for v in shifts]
    optical=box(-210,1,-18.5,18.5,-25.6,-1)
    checks['whole_front_optical_corridor']=assert_clear('optical',holder,optical)
    checks['needle_optical_corridor']=assert_clear('needle optical',needle,optical)
    # Preserve original T10 coordinates. Print just the holder on a broad side.
    printing=holder.rotate((0,0,0),(1,0,0),90)
    b=bounds(printing)
    translation=(10-b[0],10-b[1],-b[2])
    printing=printing.translate(translation)
    steps={}
    for name,shape in [('T12_BRACKET',holder),('T12_PRINT_SIDE_DOWN',printing)]:
        steps[name]=export_step(OUT/(name+'.step'),shape)
        export_mesh(OUT/(name+'.stl'),shape)
    refdir=OUT/'reference_DO_NOT_PRINT'
    refdir.mkdir(exist_ok=True)
    refs=[('needle',needle,(.24,.27,.29)),('laser_envelope',laser,(.22,.24,.26)),
          ('imu_pcb',pcb,(.75,.14,.12)),('imu_header_assumed',header,(.16,.17,.19)),
          ('imu_module_envelope',module,(.18,.21,.24))]
    asm=cq.Assembly(name='T12_ONE_PRINTED_HOLDER_WITH_PURCHASED_MODULES')
    asm.add(holder,name='ONLY_PRINTED_PART',color=cq.Color(.79,.83,.84))
    for name,shape,color in refs:
        asm.add(shape,name='REFERENCE_'+name,color=cq.Color(*color))
        export_mesh(refdir/(name+'.stl'),shape)
    asm.export(str(OUT/'T12_ASSEMBLY_REFERENCE.step'))
    back=cq.importers.importStep(str(OUT/'T12_ASSEMBLY_REFERENCE.step')).val()
    assert back.isValid() and len(back.Solids())==6
    assert abs(back.Volume()-(holder.Volume()+sum(s.Volume() for _,s,_ in refs)))<1e-3
    steps['assembly']=dict(valid=True,solids=6)
    # Reference lens positions visualize direction only. Full front corridor
    # above is the actual conservative clearance check.
    for name,y,z,radius in [('large_lens',-3,-10.5,6.3),('small_aperture',11,-10.5,2.1)]:
        visual=cq.Solid.makeCylinder(radius,.35,cq.Vector(.65,y,z),cq.Vector(1,0,0))
        export_mesh(refdir/(name+'_illustrative.stl'),visual)
    report=dict(parameters=P,dimensions_mm=dimensions(holder),volume_mm3=holder.Volume(),
                PLA_mass_estimate_g=holder.Volume()*1.24/1000,
                step_roundtrip=steps,overlap_mm3=checks,insertion_paths_sampled_mm3=paths,
                print_rotation_degrees_about_X=90,print_translation_mm=translation,
                print_bounds_mm=bounds(printing),needle_source=str(NEEDLE),needle_sha256=needle_hash,
                needle_unchanged=hashlib.sha256(NEEDLE.read_bytes()).hexdigest()==needle_hash,
                support_required=True,physical_fit_verified=False,slicer_validated=False,
                notes=['模块参考体按最大外廓，不含实测接插件。',
                       '上装IMU位置、焊点高度和胶粘垫台接触面需实物确认。',
                       '镜头圆形仅指示朝向；通过全前端外廓通道校核，不作为光学精度数据。',
                       '针体仅引用既有STEP，未重新设计或导出打印针体。'])
    (OUT/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'parameters.json').write_text(json.dumps(P,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(output=str(OUT),dimensions_mm=dimensions(holder),
                         single_printed_body=True,checks=checks),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
