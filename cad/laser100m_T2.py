"""Build the approved compact conformal thin-shell STEP assembly; sensor parts are explicitly proxies.

Run with CadQuery 2.8, trimesh and matplotlib. Does not touch the legacy STP CAD.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import zipfile

import cadquery as cq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "cad_laser100m_T2_tapered_needle"
PARAM = Path(__file__).with_name("laser100m_T2_parameters.json")


def box(x0, x1, y0, y1, z0, z1):
    return cq.Workplane("XY").box(x1-x0, y1-y0, z1-z0, centered=False).translate((x0, y0, z0))


def cyl_x(x, length, radius, z=4):
    return cq.Workplane("YZ", origin=(x, 0, z)).circle(radius).extrude(length)


def solid(wp):
    shape = wp.val() if isinstance(wp, cq.Workplane) else wp
    ss = shape.Solids()
    if len(ss) != 1 or not shape.isValid():
        raise ValueError(f"Expected one valid solid; solids={len(ss)}, valid={shape.isValid()}")
    return ss[0]


def overlap(a, b):
    return sum(s.Volume() for s in a.intersect(b).Solids())


def exact_bounds(shape):
    # Loft tessellation can inflate the default bounding box. Measure CAD surfaces.
    from OCP.BRepBndLib import BRepBndLib
    from OCP.Bnd import Bnd_Box
    box3d=Bnd_Box()
    BRepBndLib.AddOptimal_s(shape.wrapped,box3d,False,False)
    return box3d.Get()


def dims(shape):
    b=exact_bounds(shape)
    return [round(b[i+3]-b[i],6) for i in range(3)]


def mirrored_box(x0, x1, y0, y1, z0, z1, sign):
    return box(x0, x1, y0 if sign > 0 else -y1, y1 if sign > 0 else -y0, z0, z1)


def rounded_wire(x0, x1, y0, y1, z, r):
    return (cq.Workplane("XY", origin=((x0+x1)/2,(y0+y1)/2,z))
            .rect(x1-x0-2*r,y1-y0-2*r).offset2D(r).val())


def loft_rect(bottom, top, radius):
    return cq.Workplane(obj=cq.Solid.makeLoft([
        rounded_wire(*bottom,radius), rounded_wire(*top,radius)],True))


def optical_tunnel(p, x0, x1):
    a=p['optical_opening']; r=a['radius']
    return (cq.Workplane('YZ',origin=(x0,a['center_y'],a['center_z']))
            .rect(a['width']-2*r,a['height']-2*r).offset2D(r).extrude(x1-x0))


def build(p):
    h,n,l,im=(p[k] for k in ('housing','needle','laser_reference','imu_reference'))
    L,W,t=h['length'],h['width'],h['wall']
    bottom,shoulder,seam=h['bottom_z'],h['shoulder_z'],h['seam_z']
    floor=bottom+t
    outer=box(0,L,-W/2,W/2,bottom,shoulder).edges('|Z').fillet(h['corner_radius'])
    crown=p['crown']
    outer=outer.union(loft_rect(
        (crown['base_x'][0],crown['base_x'][1],-crown['base_width']/2,crown['base_width']/2,crown['base_z']),
        (crown['top_x'][0],crown['top_x'][1],-crown['top_width']/2,crown['top_width']/2,crown['top_z']),crown['radius']))
    inner=box(t,L-t,-W/2+t,W/2-t,floor,shoulder-t).edges('|Z').fillet(h['corner_radius']-t)
    # Crown location follows the known small IMU body but remains a fit assumption.
    inner=inner.union(loft_rect((18.6,48.4,-14.4,14.4,6.7),(21.6,45.4,-11.4,11.4,18.4),.9))
    shell=outer.cut(inner)
    lower=shell.intersect(box(-1,L+1,-W,W,bottom-1,seam))
    upper=shell.intersect(box(-1,L+1,-W,W,seam+h['seam_clearance'],h['top_z']+1))
    lower=lower.cut(optical_tunnel(p,-1,l['front_x']+1))
    # A top-open root notch allows the complete carrier and rod to be lifted out.
    lower=lower.cut(box(-1,2,-3.9,3.9,.3,seam+1))
    upper=upper.cut(cyl_x(-1,3,n['socket_radius']+.25,n['axis_z']))
    lower=lower.cut(box(L-3,L+1,-7,7,-22,-11))
    usb_clear=box(L-3,L+1,-8,8,2.3,7.8)
    lower=lower.cut(usb_clear); upper=upper.cut(usb_clear)

    # Laser loads vertically before the carrier is installed.
    for sign in (-1,1):
        lower=lower.union(mirrored_box(4,51,18.8,20.6,floor,-6.5,sign))
        for x in (8,40):
            lower=lower.union(mirrored_box(x,x+9,13,17,floor,l['bottom_z'],sign))
        lower=lower.union(mirrored_box(2.2,3.7,13,20.6,floor,-22.8,sign))
        lower=lower.union(mirrored_box(50.77,52.37,13,20.6,floor,-4,sign))

    carrier=box(8.5,55,-19,19,.2,1.8)
    carrier=carrier.cut(box(17,50,-12,12,-1,3))
    carrier=carrier.union(box(6,14,-4,4,.2,1.8))
    carrier=carrier.union(cyl_x(n['exit_x'],n['insertion'],n['socket_radius'],n['axis_z']))
    carrier=carrier.cut(cyl_x(n['exit_x']-1,n['insertion']+1.2,n['socket_diameter']/2,n['axis_z']))
    laser_top=l['bottom_z']+l['height']
    for sign in (-1,1):
        for x in (19,38):
            carrier=carrier.union(mirrored_box(x,x+8,18.7,20.1,.2,1.8,sign))
            carrier=carrier.union(mirrored_box(x,x+8,19.1,20.1,-6,1,sign))
            carrier=carrier.union(mirrored_box(x+4,x+8,20,20.9,-4,-2,sign))
            lower=lower.cut(mirrored_box(x-.25,x+8.25,19.8,23,-4.25,-1.75,sign))
        # Trial pads contact a possible upper-edge datum, never a modeled lens.
        for x in (14,43):
            carrier=carrier.union(mirrored_box(x,x+6,14,17,laser_top,.3,sign))
        carrier=carrier.union(mirrored_box(10,11.7,0,16.7,1.8,3.8,sign))
        carrier=carrier.union(mirrored_box(54.3,55.5,11,17,1.8,3.8,sign))
        carrier=carrier.union(mirrored_box(15,18,15.3,16.7,1.8,3.6,sign))
        carrier=carrier.union(mirrored_box(17,31,15.3,16.7,2.05,3.6,sign))
        carrier=carrier.union(mirrored_box(28,31,14.5,16.7,3.45,4.5,sign))

    # Lid clips lie outside the carrier's PCB support, with separate release slots.
    for sign in (-1,1):
        for x in (5,48):
            upper=upper.union(mirrored_box(x,x+8,19.15,20.55,1.5,7,sign))
            upper=upper.union(mirrored_box(x+5,x+8,20.45,21.2,2.4,3.6,sign))
            lower=lower.cut(mirrored_box(x-.25,x+8.25,19,20.8,1.25,5.51,sign))
            lower=lower.cut(mirrored_box(x+4.75,x+8.25,20,23,2.15,3.85,sign))

    ix,iy,iz=im['front_x'],im['width']/2,im['bottom_z']
    pcb_top=iz+im['pcb_thickness_assumed']
    imu=box(ix,ix+im['length'],-iy,iy,iz,pcb_top)
    imu=imu.union(box(ix+10,ix+32,-10,10,pcb_top,iz+im['height']))
    imu_full=solid(box(ix,ix+im['length'],-iy,iy,iz,iz+im['height']))
    laser_full=solid(box(l['front_x'],l['front_x']+l['length'],-l['width']/2,l['width']/2,l['bottom_z'],laser_top))
    laser=cq.Workplane(obj=laser_full)
    # Cosmetic optical recesses in the reference only; not a precision lens model.
    for y,r in ((0,7),(-11.35,2.2)):
        recess=cq.Workplane('YZ',origin=(l['front_x']-.1,y,-9.35)).circle(r).extrude(2.1)
        laser=laser.cut(recess)
    r=n['diameter']/2; tip=n['exit_x']-n['exposed_length']
    # Reference-inspired profile: rear half cylindrical, front half tapered.
    # Keep the existing 4 mm root and 12 mm insertion; exposed length stays 200.
    nose_radius=0.4
    transition=n['exit_x']-n['exposed_length']/2
    rod=(cq.Workplane('XY').moveTo(tip,0)
         .threePointArc((tip+nose_radius*(1-2**-.5),nose_radius*2**-.5),
                        (tip+nose_radius,nose_radius))
         .lineTo(transition,r).lineTo(n['exit_x']+n['insertion'],r)
         .lineTo(n['exit_x']+n['insertion'],0).close()
         .revolve(360,(0,0),(1,0)).translate((0,0,n['axis_z'])))
    parts={'01_lower_chassis':solid(lower),'02_upper_cover':solid(upper),'03_sensor_carrier':solid(carrier),
           'REF_04_rod_d4_L212':solid(rod),'REF_05_HWT906P_approximate':solid(imu),'REF_06_laser100m_envelope':solid(laser)}
    checks={'optical_corridor':solid(optical_tunnel(p,-6,l['front_x']-.01)),
            'laser_top_insertion_sweep':solid(box(4,50.47,-18.5,18.5,-25.1,34)),
            'laser_full_envelope':laser_full,'imu_full_envelope':imu_full}
    return parts,checks




def coupon(p):
    # Three 12 mm-deep blind socket trials; order is 4.1, 4.2, 4.3 from left to right.
    w = box(0, 48, 0, 14, 0, 15)
    for x, d in zip((8, 24, 40), p["fit"]["coupon_holes"]):
        hole = cq.Workplane("XY", origin=(x, 7, 3)).circle(d/2).extrude(13)
        w = w.cut(hole)
    return solid(w)


def render(parts):
    import vtk
    from PIL import Image, ImageDraw, ImageFont
    colors = [(.17,.19,.21),(.22,.24,.26),(.25,.47,.54),(.60,.67,.70),(.19,.59,.43),(.38,.40,.42)]
    scratch = ROOT / "tmp" / "cad_laser100m_T2_tapered_needle"
    scratch.mkdir(parents=True, exist_ok=True)

    def frame(names, path, size, target, camera, scale, cut=None, render_parts=None):
        ren = vtk.vtkRenderer()
        ren.SetBackground(.953,.965,.973)
        for index, (name, original) in enumerate((parts if render_parts is None else render_parts).items()):
            if name not in names:
                continue
            shape = original if cut is None or name.startswith("REF") else original.intersect(cut)
            if not shape.Solids():
                continue
            vertices, triangles = shape.tessellate(.08)
            points, cells = vtk.vtkPoints(), vtk.vtkCellArray()
            for vertex in vertices:
                points.InsertNextPoint(*vertex.toTuple())
            for indices in triangles:
                cell = vtk.vtkTriangle()
                for j, idx in enumerate(indices):
                    cell.GetPointIds().SetId(j, idx)
                cells.InsertNextCell(cell)
            poly = vtk.vtkPolyData()
            poly.SetPoints(points); poly.SetPolys(cells)
            normals = vtk.vtkPolyDataNormals()
            normals.SetInputData(poly); normals.SetFeatureAngle(40); normals.Update()
            mapper = vtk.vtkPolyDataMapper(); mapper.SetInputConnection(normals.GetOutputPort())
            actor = vtk.vtkActor(); actor.SetMapper(mapper)
            actor.GetProperty().SetColor(*colors[index])
            actor.GetProperty().SetAmbient(.3); actor.GetProperty().SetDiffuse(.7)
            actor.GetProperty().SetSpecular(.12); actor.GetProperty().SetSpecularPower(25)
            ren.AddActor(actor)
        window = vtk.vtkRenderWindow()
        window.SetOffScreenRendering(1); window.SetSize(*size); window.AddRenderer(ren)
        cam = ren.GetActiveCamera()
        cam.SetPosition(*camera); cam.SetFocalPoint(*target); cam.SetViewUp(0,0,1)
        cam.ParallelProjectionOn(); cam.SetParallelScale(scale)
        ren.ResetCameraClippingRange(); window.Render()
        capture = vtk.vtkWindowToImageFilter(); capture.SetInput(window)
        capture.SetInputBufferTypeToRGB(); capture.ReadFrontBufferOff(); capture.Update()
        writer = vtk.vtkPNGWriter(); writer.SetFileName(str(path)); writer.SetInputConnection(capture.GetOutputPort()); writer.Write()
        window.Finalize()
        with Image.open(path) as im:
            return im.convert("RGB").copy()

    def font(size):
        return ImageFont.truetype("C:/Windows/Fonts/arial.ttf",size)

    names = list(parts)
    raw = frame(names,scratch/"full.png",(1800,720),(-50,0,0),(-300,-360,220),75)
    canvas = Image.new("RGB",(1800,850),(243,246,248)); canvas.paste(raw,(0,65))
    draw = ImageDraw.Draw(canvas)
    draw.text((900,24),"T2 | 100 m laser + HWT906P | Conformal thin-shell STEP / fit draft",font=font(30),fill="#163e50",anchor="mt")
    draw.text((900,812),"Body 58 x 44 x 47 mm (+3 mm rod socket) | Root diameter 4 mm | Tapered front 100 mm | Exposed 200 mm",font=font(23),fill="#29424f",anchor="mt")
    canvas.save(OUT/"assembly_preview.png")
    internal = [n for n in names if "rod" not in n and "upper_cover" not in n]
    cut = solid(box(-1,101,0,30,-31,31))
    left = frame(internal,scratch/"cut.png",(1000,760),(29,0,0),(-110,-160,95),48,cut)
    right = frame(internal,scratch/"open.png",(1000,760),(29,0,0),(-90,-140,230),48)
    canvas = Image.new("RGB",(2000,900),(243,246,248)); canvas.paste(left,(0,85)); canvas.paste(right,(1000,85))
    draw = ImageDraw.Draw(canvas)
    draw.text((1000,18),"T2 internal layout | reference sensors are approximate",font=font(32),fill="#163e50",anchor="mt")
    draw.text((500,65),"Section view | green IMU / gray laser proxy",font=font(22),fill="#29424f",anchor="mt")
    draw.text((1500,65),"Cover removed | removable IMU carrier",font=font(22),fill="#29424f",anchor="mt")
    draw.text((1000,860),"Verify physical fit, optical clearance and connector orientation before final printing.",font=font(22),fill="#29424f",anchor="mt")
    canvas.save(OUT/"internal_layout.png")


    expanded = {}
    for name, shape in parts.items():
        dz = 62 if name == '02_upper_cover' else (32 if name in ('03_sensor_carrier','REF_05_HWT906P_approximate','REF_04_rod_d4_L212') else 0)
        expanded[name] = shape.translate((0,0,dz))
    exploded = frame([n for n in parts if 'rod' not in n], scratch/'exploded.png',(1200,1000),(28,0,27),(-125,-200,130),70,render_parts=expanded)
    canvas=Image.new('RGB',(1200,1110),(243,246,248));canvas.paste(exploded,(0,65))
    draw=ImageDraw.Draw(canvas)
    draw.text((600,18),'T2 | Assembly sequence / exploded view',font=font(29),fill='#163e50',anchor='mt')
    draw.text((600,1050),'Laser in lower shell > IMU carrier > removable top cover',font=font(20),fill='#29424f',anchor='mt')
    canvas.save(OUT/'exploded_view.png')


def main():
    import math
    import trimesh
    p=json.loads(PARAM.read_text(encoding='utf-8'))
    OUT.mkdir(parents=True,exist_ok=True)
    parts,envelopes=build(p)
    structural={k:s for k,s in parts.items() if not k.startswith('REF')}
    print('Thin-shell geometry built',flush=True)
    checks={}

    def require_clear(label,a,b):
        volume=overlap(a,b)
        if not math.isfinite(volume) or volume>1e-5:
            raise RuntimeError(f'{label}: unexpected overlap {volume} mm3')
        checks[label]=round(volume,8)

    for (an,a),(bn,b) in itertools.combinations(parts.items(),2):
        require_clear(f'{an} / {bn}',a,b)
    for name,s in structural.items():
        require_clear(f'optical_corridor / {name}',envelopes['optical_corridor'],s)
        require_clear(f'full_laser_box / {name}',envelopes['laser_full_envelope'],s)
    require_clear('laser_vertical_insertion_without_carrier',envelopes['laser_top_insertion_sweep'],parts['01_lower_chassis'])
    for dz in (10,15,25):
        require_clear(f'carrier_extraction_after_release_z{dz}',parts['03_sensor_carrier'].translate((0,0,dz)),parts['01_lower_chassis'])
    assumptions={f'full_IMU_box_vs_{name}':round(overlap(envelopes['imu_full_envelope'],s),6) for name,s in structural.items()}

    rows={}; rt={}; meshes={}
    def roundtrip(name,shape,expected_solids):
        imported=cq.importers.importStep(str(OUT/name)).val()
        delta=abs(imported.Volume()-shape.Volume())
        bounds_delta=max(abs(a-b) for a,b in zip(exact_bounds(imported),exact_bounds(shape)))
        if not imported.isValid() or len(imported.Solids())!=expected_solids or delta>max(1e-4,shape.Volume()*1e-8) or bounds_delta>1e-5:
            raise RuntimeError(f'STEP reimport failed: {name}; valid={imported.isValid()}; solids={len(imported.Solids())}; volume_delta={delta}; bbox_delta={bounds_delta}; imported_bbox={dims(imported)}; source_bbox={dims(shape)}; imported_volume={imported.Volume()}; source_volume={shape.Volume()}')
        rt[name]={'valid':True,'solid_count':expected_solids,'volume_delta_mm3':delta,'bbox_size_delta_mm':bounds_delta}
        return imported

    def check_mesh(name,shape):
        path=OUT/f'{name}.stl'
        cq.exporters.export(shape,str(path),tolerance=.003 if 'rod' in name else .03,angularTolerance=.03 if 'rod' in name else .08)
        mesh=trimesh.load_mesh(str(path))
        # STL float rounding collapses one triangle at the spherical pole.
        if 'rod' in name:
            mesh.update_faces(mesh.nondegenerate_faces())
            mesh.remove_unreferenced_vertices()
            mesh.export(str(path))
            mesh=trimesh.load_mesh(str(path))
        info={'watertight':bool(mesh.is_watertight),'winding_consistent':bool(mesh.is_winding_consistent),'components':len(mesh.split())}
        if not info['watertight'] or not info['winding_consistent'] or info['components']!=1:
            raise RuntimeError(f'Invalid STL: {name}: {info}')
        meshes[name]=info

    for name,s in parts.items():
        rows[name]={'valid':s.isValid(),'solids':len(s.Solids()),'dimensions_mm':dims(s),'volume_mm3':round(s.Volume(),6)}
        cq.exporters.export(s,str(OUT/f'{name}.step'))
        roundtrip(f'{name}.step',s,1)
        if name in structural or name=='REF_04_rod_d4_L212':
            check_mesh(name,s)
    trial=coupon(p)
    cq.exporters.export(trial,str(OUT/'needle_socket_fit_coupon.step'))
    roundtrip('needle_socket_fit_coupon.step',trial,1)
    check_mesh('needle_socket_fit_coupon',trial)

    palette=[(.17,.19,.21),(.22,.24,.26),(.25,.47,.54),(.60,.67,.70),(.19,.59,.43),(.38,.40,.42)]
    assembly=cq.Assembly(name='T2_CONFORMAL_THIN_SHELL_FIT_DRAFT')
    shell_assembly=cq.Assembly(name='T2_THREE_PRINTED_COMPONENTS')
    exploded=cq.Assembly(name='T2_EXPLODED_REFERENCE_ASSEMBLY')
    exploded_shapes=[]
    for (name,s),color in zip(parts.items(),palette):
        assembly.add(s,name=name,color=cq.Color(*color))
        if name in structural:
            shell_assembly.add(s,name=name,color=cq.Color(*color))
        dz=62 if name=='02_upper_cover' else (32 if name in ('03_sensor_carrier','REF_05_HWT906P_approximate','REF_04_rod_d4_L212') else 0)
        shifted=s.translate((0,0,dz))
        exploded_shapes.append(shifted)
        exploded.add(shifted,name=name,color=cq.Color(*color))
    assembly.export(str(OUT/'laser100m_HWT906P_T2_assembly.step'))
    shell_assembly.export(str(OUT/'T2_shell_only.step'))
    exploded.export(str(OUT/'T2_exploded_assembly.step'))
    compound=cq.Compound.makeCompound(list(parts.values()))
    printed=cq.Compound.makeCompound(list(structural.values()))
    cq.exporters.export(compound,str(OUT/'laser100m_HWT906P_T2_multibody.step'))
    imported=roundtrip('laser100m_HWT906P_T2_assembly.step',compound,6)
    roundtrip('laser100m_HWT906P_T2_multibody.step',compound,6)
    roundtrip('T2_shell_only.step',printed,3)
    roundtrip('T2_exploded_assembly.step',cq.Compound.makeCompound(exploded_shapes),6)

    n,l=p['needle'],p['laser_reference']
    setback=l['front_x']-n['exit_x']
    distances={'setback_from_rod_exit_mm':setback,'initial_geometric_distance_mm':n['exposed_length']+setback,
               'distance_after_150mm_stroke_mm':n['exposed_length']+setback-150,'near_range_verified':False}
    report={'revision':p['revision'],'cadquery_version':cq.__version__,'body_envelope_mm':[58,44,47],
            'structure_including_socket_envelope_mm':dims(printed),'whole_assembly_envelope_mm':dims(imported),
            'parts':rows,'zero_overlap_checks_mm3':checks,'assumption_intersections_mm3':assumptions,
            'assumption_note':'Full uniform IMU box is NOT claimed to fit. Crown cavity assumes raised components are confined to the approximate central body.',
            'step_roundtrip':rt,'stl_checks':meshes,'geometric_distances':distances,'limitations':p['limitations']}
    (OUT/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUT/'parameters.json').write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('STEP reimport, mesh and geometric clearance checks passed',flush=True)
    render(parts)
    write_readme(report)
    manifest={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(OUT.iterdir()) if f.is_file() and f.name!='sha256.json'}
    (OUT/'sha256.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    bundle=ROOT/'output'/'laser100m_HWT906P_T2_STEP_bundle.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(OUT.iterdir()):
            if f.is_file():
                z.write(f,f'{OUT.name}/{f.name}')
        z.write(Path(__file__),'source/laser100m_T2.py')
        z.write(PARAM,'source/laser100m_T2_parameters.json')
    print(json.dumps({'output':str(OUT),'assembly_bbox_mm':dims(imported),'structure_bbox_mm':dims(printed),'geometric_distances':distances,'zip':str(bundle)},ensure_ascii=False),flush=True)


def write_readme(r):
    lines=[
        '# T2 渐细针体 STEP 交付说明','',
        '针体仿照 acupuncter_needle.stl 的圆柱加长渐细轮廓重新建模：外露 200 mm，后段 100 mm 为 Ø4 圆柱，前段 100 mm 渐细至 Ø0.8 并以 R0.4 圆钝端收尾；插入段 Ø4 × 12 mm。保留现有根部连接尺寸，不是把原 STL 等比例放大，也不是精确复刻其端面。针体强度与实物试配未验证。','',
        '依据用户确认的薄壳概念图建立的第一版可编辑几何。当前为实物试配前版本；概念图不是计量图，未直接按图中像素比例加工。','',
        '## 文件入口','',
        '- `laser100m_HWT906P_T2_assembly.step`：带名称和颜色的 6 实体装配。',
        '- `T2_shell_only.step`：下壳、内部载架、上盖，共 3 个结构件；不包含传感器和针杆参考件。',
        '- `T2_exploded_assembly.step`：便于检查装配顺序的分解装配。',
        '- `laser100m_HWT906P_T2_multibody.step`：不保留装配层级时的备用格式。',
        '- `01_`、`02_`、`03_` 单件 STEP/STL；针孔试配件及验证报告。','',
        '## 尺寸与结构','',
        '- 主体 58 × 44 × 47 mm；前突针杆座 3 mm，含针杆座的结构包络 61 × 44 × 47 mm。',
        '- 平直壁名义厚度 1.6 mm；上盖边缘位于 Z=5.65，下壳接缝顶面 Z=5.5，接缝间隙 0.15 mm。局部卡扣、圆角和斜面厚度不同。',
        '- 下壳承接测距模块；独立载架固定 IMU、连接针杆并限制测距模块上移；上盖仅包覆和辅助保护。',
        '- 针杆 Ø4 mm，外露 200 mm，胶接深度 12 mm，总长 212 mm；出壳平面 X=-3，针尖 X=-203。',
        '- 测距模块包络 46.47 × 37 × 24.6 mm，光学前端 X=4；IMU 载板 42 × 30 mm，总高 16 mm。',
        '- 顶部隆起贴近近似 IMU 本体，尾部两处开口直接出线，没有尾部线缆腔。','',
        '## 与效果图的必要区别','',
        '效果图中的镜头、线路和壳体孔位不是精确厂家几何。本版前端采用一个 31 × 20.4 mm 的圆角共用避让口，露出收发光学区域，未把未经实物核对的两个圆孔定成精密安装孔。参考模块上的两个浅凹圆面只供识别光学朝向。',
        '顶部隆起按 IMU 中部凸起示意建模。它不是能够容纳任意 42 × 30 × 16 mm 实心长方体的通用盖。','',
        '## 数字检查及证据边界','',
        '- 6 个装配零件均为单一有效实体；装配、单件、仅结构件 STEP 经过重导入，体积与包络尺寸保持一致。',
        '- 6 件两两无体积干涉（阈值 0.00001 mm³）；完整激光参考包络与 3 个结构件无体积干涉。',
        '- 光窗通道与结构件无体积相交；先拆载架时，测距模块的竖直装入包络不碰下壳。',
        '- 载架卡钩解除后，上移 10、15、25 mm 的三个位置不碰下壳；没有验证整个弹性解锁过程。',
        '- 结构件及针孔试配 STL 均水密、法向一致、单一连通分量。',
        f"- 另外检查完整等高 IMU 长方体：它与顶盖相交约 {r['assumption_intersections_mm3']['full_IMU_box_vs_02_upper_cover']:.2f} mm³，与载架 PCB 卡钩相交约 {r['assumption_intersections_mm3']['full_IMU_box_vs_03_sensor_carrier']:.2f} mm³。这两项是明确保留的几何假设，不能用近似组件的零干涉替代实物校核。",'',
        '## 装配与试配','',
        '1. 先用 4.1、4.2、4.3 mm 的 12 mm 深试配孔确定针杆胶接孔径；三孔按 X 从小到大排列。',
        '2. 测距模块从上方放进下壳，对齐前后限位和底部支点。确认支点及上压垫不碰镜头、排线座、焊点或脆弱器件。',
        '3. 先将 IMU 安装到载架，再将载架压入下壳并使两侧卡钩就位；针杆试配后胶接到载架前座。',
        '4. 检查 USB-C 实际方向和插头外形，连接线束，最后合上上盖；拆卸先释放上盖，再释放载架。',
        '5. 先试配局部固定与卡扣，再决定完整壳体打印。STL 沿装配坐标输出，切片时自行选朝向及支撑，未声称可直接无支撑打印。',
        '6. 卡扣根部强度、反复开合、保持力、实际插头、线缆应力释放和壳体局部厚度没有经过机械试验。','',
        '## 测距几何已改变','',
        '测距模块前端距针杆出壳面仅 7 mm。假设靶面垂直针轴且光轴平行针轴，针尖刚接触靶面时几何距离 207 mm，推进 150 mm 后为 57 mm。',
        '57 mm 是否可用尚未验证，不能默认仍能完成原计划的 150 mm 测距行程。必须用当前 100 m 模块、采样模式及靶片验证；若近端无效，应限行程或调整模块位置。',
        '上述数值是几何量，不是模块读数；软件/硬件距离基准与补偿另行核对。软件参数及旧 CAD 未修改。','',
        '## 复现','',
        '`uv run --with cadquery --with matplotlib --with trimesh python cad/laser100m_T2.py`',
        '参数见 `cad/laser100m_T2_parameters.json`。厂家原始二维图为 `D:/From Zero/模块资料/尺寸图.jpg`；IMU 尺寸沿用已确认的项目记录。',''
    ]
    (OUT/'README_交付说明.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    main()
