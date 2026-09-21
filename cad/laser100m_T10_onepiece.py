"""T10: solid one-piece 200 mm exposed probe, continuous flat bed and coplanar brim.

Run: uv run --python 3.12 --offline --with cadquery==2.6.1 --with trimesh --with matplotlib --with shapely --with networkx --with scipy python -X utf8 cad/laser100m_T10_onepiece.py
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import zipfile
import cadquery as cq
import numpy as np
import trimesh
from shapely.geometry import Polygon

CAD = Path(__file__).resolve().parent
ROOT = CAD.parent
OUT = ROOT / 'output/cad_laser100m_T10_onepiece'
spec = importlib.util.spec_from_file_location('t10_housing_base', CAD / 'laser100m_T5.py')
t5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t5)

P = dict(revision='T10', exposed_length=200.0, insertion_length=12.0,
         shaft_diameter=6.0, root_diameter=8.5, tenon_diameter=4.0,
         exit_x=-3.0, axis_z=4.2, flat_offset=-1.0, nose_plan_radius=3.0,
         brim_thickness=0.2, brim_width=16.5, brim_end_extension=4.0,
         print_rotation_degrees=45.0, assumed_bed_size=180.0,
         layer_check_height=0.2)

def build():
    tip = P['exit_x'] - P['exposed_length']
    end = P['exit_x'] + P['insertion_length']
    z = P['axis_z']
    flat = z + P['flat_offset']
    r = P['shaft_diameter'] / 2
    rr = P['root_diameter'] / 2
    tr = P['tenon_diameter'] / 2
    def cone(x, length, r0, r1):
        return cq.Workplane(obj=cq.Solid.makeCone(r0, r1, length, cq.Vector(x,0,z), cq.Vector(1,0,0)))
    rod = t5.base.cyl_x(tip, -24-tip+.05, r, z)
    rod = rod.union(cone(-24,14,r,rr))
    rod = rod.union(t5.base.cyl_x(-10,4.05,rr,z))
    rod = rod.union(cone(-6,3,rr,tr))
    rod = rod.union(t5.base.cyl_x(-3,end+3,tr,z))
    rod = rod.intersect(t5.base.box(tip-1,end+1,-10,10,flat,z+10))
    # Rounded in plan, extending all the way down to the bed. No spherical undercut.
    nr = P['nose_plan_radius']
    nose = cq.Workplane('XY',origin=(tip+nr,0,flat-1)).circle(nr).extrude(20)
    mask = nose.union(t5.base.box(tip+nr,end+1,-10,10,flat-1,z+10))
    clean = t5.solid(rod.intersect(mask))
    clean = clean.translate((0,0,-flat))
    ext = P['brim_end_extension']
    brim = t5.base.box(tip-ext,end+ext,-P['brim_width']/2,P['brim_width']/2,
                       0,P['brim_thickness']).edges('|Z').fillet(4)
    printable = t5.solid(cq.Workplane(obj=clean).union(brim))
    return clean, printable

def mesh_of(shape):
    vertices, faces = shape.tessellate(.008, .05)
    mesh = trimesh.Trimesh(vertices=[v.toTuple() for v in vertices],faces=faces,process=True)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    assert mesh.is_watertight and mesh.is_winding_consistent and len(mesh.split()) == 1
    return mesh

def horizontal_polygon(mesh, z):
    section = mesh.section(plane_origin=[0,0,z], plane_normal=[0,0,1])
    assert section is not None, ('empty layer',z)
    loops = section.discrete
    assert len(loops) == 1, ('islands or holes',z,len(loops))
    polygon = Polygon(loops[0][:,:2])
    assert polygon.is_valid and polygon.area>0
    return polygon

def layer_checks(mesh):
    # This is a geometric cross-section check, not a G-code/slicer verification.
    first = horizontal_polygon(mesh,.001)
    rows=[]
    previous=first
    heights=np.arange(.1,mesh.bounds[1,2],P['layer_check_height'])
    for h in heights:
        current=horizontal_polygon(mesh,float(h))
        outside=current.difference(previous.buffer(.2)).area
        assert outside<.01, ('expansion exceeds 0.2 mm lateral support allowance',h,outside)
        rows.append(dict(z_mm=round(float(h),3),islands=1,holes=0,
                         area_mm2=round(current.area,4),outside_previous_buffer_mm2=round(outside,7)))
        previous=current
    return dict(first_layer_area_mm2=first.area, samples=rows,
                method='Horizontal STL contours; 0.2 mm vertical sampling, 0.2 mm lateral allowance. Not slicer or physical validation.')

def on_diagonal(shape):
    rotated=shape.rotate((0,0,0),(0,0,1),P['print_rotation_degrees'])
    b=t5.exact_bounds(rotated)
    center=P['assumed_bed_size']/2
    return rotated.translate((center-(b[0]+b[3])/2,center-(b[1]+b[4])/2,-b[2]))

def export_part(name,shape,reports):
    step=OUT/(name+'.step')
    cq.exporters.export(shape,str(step))
    _,rt=t5.roundtrip(step,shape,1)
    mesh=mesh_of(shape)
    stl=OUT/(name+'.stl')
    mesh.export(stl)
    reread=trimesh.load_mesh(stl)
    assert reread.is_watertight and reread.is_winding_consistent and len(reread.split())==1
    reports[name]=dict(step=rt,stl_watertight=True,stl_components=1,bounds_mm=reread.bounds.tolist())
    return reread

def preview(clean,printable,clean_mesh,print_mesh,diagonal_mesh):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif']=['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus']=False
    fig=plt.figure(figsize=(16,10),facecolor='#f4f7fa',layout='constrained')
    gs=fig.add_gridspec(2,2,height_ratios=[1,1.1])
    ax=fig.add_subplot(gs[0,:])
    for mesh,z,color in [(print_mesh,.001,'#e4a84a'),(clean_mesh,1.0,'#496978')]:
        polygon=horizontal_polygon(mesh,z)
        ax.fill(*polygon.exterior.xy,color=color)
    ax.set(xlim=(-212,16),ylim=(-13,13));ax.set_aspect('equal');ax.set_axis_off()
    ax.set_title('T10 一体实心针体俯视｜外露 200 mm + 插入端 12 mm\n主体宽 6 mm；橙色为 0.20 mm 厚、与底面齐平的可剪薄边',fontsize=16)
    bed=fig.add_subplot(gs[1,0]); bed.set_aspect('equal')
    poly=horizontal_polygon(diagonal_mesh,.001)
    bed.fill(*poly.exterior.xy,color='#e4a84a',label='首层整片接触范围')
    bed.set(xlim=(0,180),ylim=(0,180),xlabel='X / mm',ylabel='Y / mm')
    bed.grid(alpha=.2);bed.legend(loc='upper left');bed.set_title('45° 斜放｜按 180 × 180 mm 平台核对包络')
    sub=gs[1,1].subgridspec(2,1)
    side=fig.add_subplot(sub[0]); side.set_aspect('equal')
    section=clean_mesh.section(plane_origin=[-100,0,0],plane_normal=[1,0,0])
    yz=section.discrete[0][:,1:]
    side.fill(yz[:,0],yz[:,1],color='#496978')
    side.fill([-8.25,8.25,8.25,-8.25],[0,0,.2,.2],color='#e4a84a')
    side.axhline(0,color='#444');side.set(xlim=(-9,9),ylim=(-.5,5),xlabel='横向 / mm',ylabel='高度 / mm')
    side.set_title('主体截面：宽 6，高 4，平底宽 5.66 mm')
    tip=fig.add_subplot(sub[1]);tip.set_aspect('equal')
    first=horizontal_polygon(print_mesh,.001); cp=horizontal_polygon(clean_mesh,.001)
    tip.fill(*first.exterior.xy,color='#e4a84a');tip.fill(*cp.exterior.xy,color='#496978')
    tip.set(xlim=(-208,-187),ylim=(-9,9),xlabel='轴向 / mm',ylabel='横向 / mm')
    tip.set_title('针尖首层俯视：圆头从底面开始，无架空防滑脚')
    fig.savefig(OUT/'T10_design_preview.png',dpi=150)
    plt.close(fig)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    clean,printable=build()
    reports={}
    primary=on_diagonal(printable)
    diagonal_mesh=export_part('T10_PRINT_ONE_PIECE_6mm_WITH_BRIM',primary,reports)
    clean_mesh=export_part('T10_CAD_ONE_PIECE_FINISHED',clean,reports)
    print_mesh=mesh_of(printable)
    b=t5.exact_bounds(clean)
    assert abs(b[3]-b[0]-212)<1e-6 and abs(P['exit_x']-b[0]-200)<1e-6
    bottom_faces=[f for f in clean.Faces() if f.geomType()=='PLANE' and abs(f.Center().z)<1e-6]
    assert len(bottom_faces)==1
    bottom=bottom_faces[0]
    bottom_bounds=t5.exact_bounds(bottom)
    assert abs(bottom_bounds[0]-b[0])<1e-6 and abs(bottom_bounds[3]-b[3])<1e-6
    bedbounds=t5.exact_bounds(primary)
    assert min(bedbounds[:2])>5 and max(bedbounds[3:5])<175
    checks_clean=layer_checks(clean_mesh)
    checks_print=layer_checks(print_mesh)
    # Confirm root insert still fits the existing carrier and clears other housing solids.
    bp=json.loads((CAD/'laser100m_T2_parameters.json').read_text(encoding='utf-8'))
    bp['housing']['width']=46
    shells,env=t5.build_parts(bp)
    installed=clean.translate((0,0,P['axis_z']+P['flat_offset']))
    fit={k:t5.overlap(installed,s) for k,s in shells.items()}
    assert max(fit.values())<1e-5,fit
    # Exactly one needle solid, no pin sockets, no assembly seams, no sacrificial material in CAD reference.
    cq.exporters.export(installed,str(OUT/'T10_REFERENCE_INSTALLED_NEEDLE.step'))
    _,ref_rt=t5.roundtrip(OUT/'T10_REFERENCE_INSTALLED_NEEDLE.step',installed,1)
    preview(clean,printable,clean_mesh,print_mesh,diagonal_mesh)
    report=dict(parameters=P,files=reports,installed_reference_roundtrip=ref_rt,
                finished_dimensions_mm=t5.dims(clean),flat_bottom_face_area_mm2=bottom.Area(),
                flat_bottom_face_x_extent_mm=[bottom_bounds[0],bottom_bounds[3]],
                body_bottom_width_mm=2*math.sqrt(3**2-1**2),
                previous_5mm_body_bottom_width_mm=2*math.sqrt(2.5**2-1**2),
                print_bed_bounds_mm=bedbounds,finished_layer_checks=checks_clean,print_layer_checks=checks_print,
                housing_overlap_mm3=fit,physical_print_verified=False,gcode_sliced=False)
    (OUT/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'parameters.json').write_text(json.dumps(P,ensure_ascii=False,indent=2),encoding='utf-8')
    readme=f'''# T10 一体针体打印说明

首选打印文件：T10_PRINT_ONE_PIECE_6mm_WITH_BRIM.stl（也提供同名 STEP）。仅一个实体，已平底朝下、Z=0、45°斜放。不要自动旋转成立式。按180×180 mm平台核对几何包络；切片软件仍需核对实际可打印区与预留区域。

尺寸：主体宽6 mm（原5 mm），高度4 mm，主体平底宽5.66 mm（原4.58 mm）；根部最大宽8.5 mm；外露200 mm，加12 mm插入端，总长212 mm。插入端圆弧外径4 mm、底部同样切平，沿用原孔胶粘。实心一体，无分段、销孔或单独连接件。

针尖改为俯视圆头，圆头一直延伸至底面，不再使用底部悬空的球头。整根针体底部是一张从针尖到插入端连续的平面。

带薄边打印版：最底0.20 mm为连续加宽层，总宽16.5 mm、轴向两端各延长4 mm。薄边向侧面扩展，与针体底面齐平，不把针体抬高。打印后沿针体边缘剪除薄边、修整底部和插入端。薄边和针体熔接，拆除便利性仍需实测。

切片起点：首层高度设0.20 mm以匹配薄边；普通层高0.20 mm。无需另加底筏或抬高模型的支撑；已有模型薄边，先关闭额外brim以保持包络。选实际打印机、实际PLA预设，先仅打印这一根。温度、流量等沿用已验证的耗材设置。检查切片首层应为连续整片，第二层应是一根连续针体。首层异常则停止，不继续消耗整盘材料。

T10_CAD_ONE_PIECE_FINISHED.step / .stl 是剪除薄边后的成品几何；T10_REFERENCE_INSTALLED_NEEDLE.step 是原装配坐标参考。首选上方 PRINT 文件直接切片。

数字检查：STEP回读、单实体STL水密性、整根底面连续性、180 mm平台斜放包络、逐0.20 mm水平截面、上一层侧向0.20 mm范围内支撑、原外壳配合干涉。首层截面积约{checks_print['first_layer_area_mm2']:.0f} mm²；成品底面面积约{bottom.Area():.0f} mm²。

这是几何截面检查，不是实际切片或成功试印证据。尚未验证G-code、床面附着、热翘曲、薄边拆除和弯曲强度。此前T9脚垫把主体抬高0.8 mm导致间隙；本版移除该结构。

附着原理参考：https://help.prusa3d.com/article/skirt-and-brim_133969 （扩大首层面积）。床面清洁与首层问题：https://help.prusa3d.com/article/first-layer-issues_1804 。
'''
    (OUT/'START_HERE_打印说明.md').write_text(readme,encoding='utf-8')
    manifest={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in OUT.iterdir() if f.is_file() and f.name!='sha256.json'}
    (OUT/'sha256.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    bundle=ROOT/'output/T10_ONE_PIECE_6mm_PRINT_PACKAGE.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
        for f in OUT.iterdir():
            if f.is_file(): z.write(f,f.name)
        for src in [Path(__file__),CAD/'T10_design.md',CAD/'laser100m_T5.py',CAD/'laser100m_T2.py',CAD/'laser100m_T2_parameters.json']:
            z.write(src,'source/'+src.name)
    print(json.dumps(dict(output=str(OUT),bundle=str(bundle),dimensions=t5.dims(clean),
                         bottom_area=bottom.Area(),first_layer_area=checks_print['first_layer_area_mm2'],
                         plate_bounds=bedbounds,layer_samples=len(checks_print['samples'])),ensure_ascii=False))

if __name__=='__main__':
    main()
