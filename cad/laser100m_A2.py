"""Build the approved A-layout STEP assembly; sensor parts are explicitly proxies.

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
OUT = ROOT / "output" / "cad_laser100m_A2_compact"
PARAM = Path(__file__).with_name("laser100m_A2_parameters.json")


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


def dims(shape):
    b = shape.BoundingBox()
    return [round(b.xlen, 6), round(b.ylen, 6), round(b.zlen, 6)]


def mirrored_box(x0, x1, y0, y1, z0, z1, sign):
    return box(x0, x1, y0 if sign > 0 else -y1, y1 if sign > 0 else -y0, z0, z1)


def build(p):
    h, n, l, im = (p[k] for k in ("housing", "needle", "laser_reference", "imu_reference"))
    L, W, H, t, seam = h["length"], h["width"], h["height"], h["wall"], h["seam_z"]
    bottom, top = h["bottom_z"], h["bottom_z"]+H
    outer = box(0,L,-W/2,W/2,bottom,top).edges("|X").fillet(h["corner_radius"])
    inner = box(t,L-t,-W/2+t,W/2-t,bottom+t,top-t).edges("|X").fillet(h["corner_radius"]-t)
    shell = outer.cut(inner)
    lower = shell.intersect(box(-1,L+1,-W,W,bottom-1,seam))
    upper = shell.intersect(box(-1,L+1,-W,W,seam,top+1))
    laser_top = l["bottom_z"]+l["height"]
    optical = box(-3,l["front_x"]-.01,-19,19,l["bottom_z"]-.5,laser_top+.5)
    lower = lower.cut(optical)
    lower = lower.cut(box(L-3,L+2,-9,9,-24.5,-6.5))
    upper = upper.cut(box(L-3,L+2,-10,10,3,18))

    lower = lower.union(cyl_x(0,n["insertion"]+2,4,n["axis_z"]))
    # Taper the bridge down behind the shaft socket instead of lifting the IMU.
    bridge = cq.Workplane("XZ",origin=(0,5,0)).polyline([(1,.1),(32,.1),(32,2),(14,8),(1,8)]).close().extrude(10)
    lower = lower.union(bridge)
    lower = lower.cut(cyl_x(-1,n["insertion"]+1.2,n["socket_diameter"]/2,n["axis_z"]))
    upper = upper.cut(box(-1,2.3,-5.3,5.3,1.7,8.3))

    lower = lower.union(box(28,80,-22.8,22.8,2,4))
    for sign in (-1,1):
        lower = lower.union(mirrored_box(30,78,22.6,23.2,-26,2,sign))
        lower = lower.union(mirrored_box(30,78,18.8,20.8,-26,-1.5,sign))
        for x in (34,65):
            lower = lower.union(mirrored_box(x,x+10,13,17,-26,l["bottom_z"],sign))
    lower = lower.union(box(76.77,78.77,-20.8,20.8,-26,-3.5))

    strap = box(54,62,-22.35,22.35,laser_top,.6)
    for sign in (-1,1):
        strap = strap.union(mirrored_box(54,62,20.95,22.35,-7.5,.6,sign))
        strap = strap.union(mirrored_box(54,62,20.25,21.1,-6.5,-5.5,sign))
        lower = lower.cut(mirrored_box(53.75,62.25,20,22.6,-6.75,-5.25,sign))

    ix, iy, iz = im["front_x"], im["width"]/2, im["bottom_z"]
    pcb_top = iz+im["pcb_thickness_assumed"]
    imu = box(ix,ix+im["length"],-iy,iy,iz,pcb_top)
    imu = imu.union(box(ix+10,ix+32,-10,10,pcb_top,iz+im["height"]))
    imu_keepout = solid(box(ix,ix+im["length"],-iy,iy,iz,iz+im["height"]))
    for sign in (-1,1):
        lower = lower.union(mirrored_box(30,31.7,0,16.7,4,6,sign))
        lower = lower.union(mirrored_box(74.3,76.3,10,16.7,4,6,sign))
        lower = lower.union(mirrored_box(35,38,15.3,16.7,3.8,5.8,sign))
        lower = lower.union(mirrored_box(37,51,15.3,16.7,4.25,5.8,sign))
        lower = lower.union(mirrored_box(48,51,14.5,16.7,pcb_top+.05,6.7,sign))
    lower = lower.cut(box(53,63,-22.95,22.95,-.7,31))

    for sign in (-1,1):
        for x in (15,72):
            upper = upper.union(mirrored_box(x,x+12,21.8,23.1,-5,4,sign))
            upper = upper.union(mirrored_box(x+7,x+12,23.05,24,-4,-2,sign))
            # Rear latches share space with the datum plate: relieve both together.
            lower = lower.cut(mirrored_box(x-.25,x+12.25,21.5,23.35,-5.25,4.25,sign))
            lower = lower.cut(mirrored_box(x+6.75,x+12.25,22.8,26,-4.25,-1.75,sign))

    r = n["diameter"]/2
    needle = cyl_x(-n["exposed_length"]+r,n["exposed_length"]-r+n["insertion"],r,n["axis_z"])
    needle = needle.union(cq.Workplane("XY").sphere(r).translate((-n["exposed_length"]+r,0,n["axis_z"])))
    laser = box(l["front_x"],l["front_x"]+l["length"],-l["width"]/2,l["width"]/2,l["bottom_z"],laser_top)
    parts = {"01_lower_chassis":solid(lower),"02_upper_cover":solid(upper),
             "03_laser_trial_retainer":solid(strap),"REF_04_rod_d4_L212":solid(needle),
             "REF_05_HWT906P_approximate":solid(imu),"REF_06_laser100m_envelope":solid(laser)}
    return parts,solid(optical),imu_keepout


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
    colors = [(0.30,.51,.58),(.66,.77,.82),(.84,.64,.27),(.60,.67,.70),(.19,.59,.43),(.87,.53,.30)]
    scratch = ROOT / "tmp" / "cad_laser100m_A2_compact"
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
    draw.text((900,24),"A2 | 100 m laser + HWT906P | STEP layout / fit draft",font=font(30),fill="#163e50",anchor="mt")
    draw.text((900,812),"Housing 86 x 50 x 52 mm | Rod diameter 4 mm | Exposed length 200 mm",font=font(23),fill="#29424f",anchor="mt")
    canvas.save(OUT/"assembly_preview.png")
    internal = [n for n in names if "rod" not in n and "upper_cover" not in n]
    cut = solid(box(-1,101,0,30,-31,31))
    left = frame(internal,scratch/"cut.png",(1000,760),(50,0,0),(-110,-160,95),66,cut)
    right = frame(internal,scratch/"open.png",(1000,760),(50,0,0),(-90,-140,230),66)
    canvas = Image.new("RGB",(2000,900),(243,246,248)); canvas.paste(left,(0,85)); canvas.paste(right,(1000,85))
    draw = ImageDraw.Draw(canvas)
    draw.text((1000,18),"A2 internal layout | reference sensors are approximate",font=font(32),fill="#163e50",anchor="mt")
    draw.text((500,65),"Section view | green IMU / orange laser proxy",font=font(22),fill="#29424f",anchor="mt")
    draw.text((1500,65),"Cover removed | retaining strap access below IMU",font=font(22),fill="#29424f",anchor="mt")
    draw.text((1000,860),"Verify physical fit, optical clearance and connector orientation before final printing.",font=font(22),fill="#29424f",anchor="mt")
    canvas.save(OUT/"internal_layout.png")

    # Same camera and scale: differences are actual geometry, not image zoom.
    a1dir = ROOT / "output" / "cad_laser100m_A"
    if a1dir.exists():
        original = {name: cq.importers.importStep(str(a1dir/f"{name}.step")).val() for name in parts}
        handle = [name for name in parts if "rod" not in name]
        before = frame(handle,scratch/"before.png",(1000,700),(45,0,0),(-130,-180,120),66,render_parts=original)
        after = frame(handle,scratch/"after.png",(1000,700),(45,0,0),(-130,-180,120),66)
        canvas = Image.new("RGB",(2000,850),(243,246,248))
        canvas.paste(before,(0,85)); canvas.paste(after,(1000,85))
        draw = ImageDraw.Draw(canvas)
        draw.text((1000,15),"Housing comparison | identical view and scale",font=font(32),fill="#163e50",anchor="mt")
        draw.text((500,62),"A1 | 100 x 54 x 60 mm",font=font(27),fill="#29424f",anchor="mt")
        draw.text((1500,62),"A2 COMPACT | 86 x 50 x 52 mm",font=font(27),fill="#29424f",anchor="mt")
        draw.text((1000,808),"31% smaller bounding volume | Sensor sizes and exposed rod length unchanged",font=font(25),fill="#29424f",anchor="mt")
        canvas.save(OUT/"A1_A2_comparison.png")


def main():
    import trimesh
    p = json.loads(PARAM.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    parts, optical, imu_keepout = build(p)
    print("Geometry built", flush=True)
    checks = {}
    for (an, a), (bn, b) in itertools.combinations(parts.items(), 2):
        value = overlap(a, b)
        checks[f"{an} / {bn}"] = round(value, 8)
        if value > 1e-5:
            raise RuntimeError(f"Interference {an} / {bn}: {value:.6f} mm3")
    for name, shape in parts.items():
        if name.startswith("REF"):
            continue
        value = overlap(shape, optical)
        checks[f"optical_corridor / {name}"] = round(value, 8)
        if value > 1e-5:
            raise RuntimeError(f"Optical corridor obstructed by {name}: {value}")
    # Full IMU envelope overlaps only the intentional PCB-edge retention regions.
    full_imu_overlap = overlap(imu_keepout, parts["01_lower_chassis"])
    checks["full_IMU_box_vs_edge_hooks_expected_not_electronics_proxy"] = round(full_imu_overlap, 8)
    laser_slide = solid(box(-47,76.47,-18.5,18.5,-25.5,-.9))
    strap_access = solid(box(53.9,62.1,-22.5,22.5,-.6,32))
    for name, envelope in (("laser_front_insertion_sweep",laser_slide), ("strap_top_access_after_latch_release",strap_access)):
        volume = overlap(envelope,parts["01_lower_chassis"])
        checks[name] = round(volume,8)
        if volume > 1e-5:
            raise RuntimeError(f"Assembly access blocked: {name}: {volume}")
    rows, roundtrips, meshes = {}, {}, {}
    for name, shape in parts.items():
        cq.exporters.export(shape, str(OUT/f"{name}.step"))
        rows[name] = {"valid": shape.isValid(), "solids": len(shape.Solids()), "dimensions_mm": dims(shape), "volume_mm3": round(shape.Volume(), 6)}
        imported = cq.importers.importStep(str(OUT/f"{name}.step")).val()
        if not imported.isValid() or len(imported.Solids()) != 1:
            raise RuntimeError(f"STEP roundtrip failed: {name}")
        delta = abs(imported.Volume()-shape.Volume())
        if delta > max(1e-4, shape.Volume()*1e-8):
            raise RuntimeError(f"STEP volume changed: {name}: {delta}")
        roundtrips[name] = {"valid": True, "solids": 1, "volume_delta_mm3": delta}
        if not name.startswith("REF"):
            cq.exporters.export(shape, str(OUT/f"{name}.stl"), tolerance=.04, angularTolerance=.1)
            mesh = trimesh.load_mesh(str(OUT/f"{name}.stl"))
            meshes[name] = {"watertight": bool(mesh.is_watertight), "components": len(mesh.split()), "winding_consistent": bool(mesh.is_winding_consistent)}
            if not mesh.is_watertight or not mesh.is_winding_consistent or len(mesh.split()) != 1:
                raise RuntimeError(f"Mesh invalid: {name}: {meshes[name]}")
    trial = coupon(p)
    cq.exporters.export(trial, str(OUT/"fit_coupon_4p1_4p2_4p3.step"))
    cq.exporters.export(trial, str(OUT/"fit_coupon_4p1_4p2_4p3.stl"), tolerance=.04, angularTolerance=.1)
    mesh = trimesh.load_mesh(str(OUT/"fit_coupon_4p1_4p2_4p3.stl"))
    if not mesh.is_watertight or len(mesh.split()) != 1:
        raise RuntimeError("Fit coupon mesh invalid")
    assembly = cq.Assembly(name="Laser100m_HWT906P_A2_LAYOUT_FIT_DRAFT")
    colors = [(0.30,.51,.58),(.66,.77,.82),(.84,.64,.27),(.60,.67,.70),(.19,.59,.43),(.87,.53,.30)]
    for (name, shape), rgb in zip(parts.items(), colors):
        assembly.add(shape, name=name, color=cq.Color(*rgb))
    assembly.export(str(OUT/"laser100m_HWT906P_A2_assembly.step"))
    compound = cq.Compound.makeCompound(list(parts.values()))
    cq.exporters.export(compound, str(OUT/"laser100m_HWT906P_A2_multibody.step"))
    reimport = cq.importers.importStep(str(OUT/"laser100m_HWT906P_A2_assembly.step")).val()
    if not reimport.isValid() or len(reimport.Solids()) != len(parts):
        raise RuntimeError("Assembly STEP roundtrip failed")
    data = {"revision": p["revision"], "comparison": p["comparison"], "cadquery_version": cq.__version__, "parts": rows,
            "intersection_volumes_mm3": checks, "step_roundtrip": roundtrips,
            "assembly_roundtrip_solids": len(reimport.Solids()), "assembly_dimensions_mm": dims(reimport),
            "stl_checks": meshes, "limitations": p["limitations"]}
    (OUT/"validation.json").write_text(json.dumps(data, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    (OUT/"parameters.json").write_text(json.dumps(p, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print("Geometry, STEP reimport and mesh checks passed", flush=True)
    render(parts)
    report = ["# A 方案 STEP 交付说明", "", "版本：A2 紧凑布局及试配版；不是实物装配认证或直接量产定稿。", "",
              "## 打开文件", "", "优先打开 `laser100m_HWT906P_A2_assembly.step`：带名称和颜色的六实体装配。",
              "若软件未保留装配层级，可打开 `laser100m_HWT906P_A2_multibody.step` 或单独零件 STEP。", "",
              "## 主要尺寸", "", "相对 A1 的 100 × 54 × 60 mm，长度减 14 mm、宽度减 4 mm、高度减 8 mm，外包络体积减少约 30.99%。这不是打印材料体积或质量的减少比例。", "", "外壳 86 × 50 × 52 mm；实体杆 Ø4 mm，外露 200 mm，插入 12 mm，总长 212 mm。",
              "针杆出壳平面 X=0，针尖 X=-200，尾部 X=86；针轴位于 Y=0、Z=4。", "",
              "## 已验证", "", f"- {len(parts)} 个零件均为单一有效实体，装配 STEP 重导入得到 {len(reimport.Solids())} 个有效实体。",
              "- 六个装配零件两两相交体积低于 0.00001 mm³。", "- 前端预留光路通道与三个外壳/固定零件无实体相交。",
              "- 测距参考包络的前端滑入通道，以及解除卡钩后压扣上提的检修通道，与下壳无实体相交；弹性解锁过程未仿真。",
              "- 三个壳体/固定零件 STL 与针孔试配件通过水密及单连通分量检查。",
              "- 检查结果见 `validation.json`；无碰撞只针对当前几何与近似传感器，不能证明真实硬件装配合格。", "",
              "## 必须保留的边界", "",
              "1. `REF_` 零件是参考件，不应作为外壳一起打印；测距模块是二维外廓重建的长方体，IMU 内部凸起及 PCB 厚度为近似。",
              "2. IMU 本版暂按 42 mm 边沿针轴布置；USB-C 和坐标轴实际方向尚待核对，不设置未经核实的安装孔。",
              "3. IMU 卡钩接触 PCB 边缘的假定区域。完整 42 × 30 × 16 包络与卡钩存在预期交叠，已单独记录；装入实物前必须确认该边缘无器件。",
              "4. 测距压扣的压紧面、轴向保持和 PCB 支撑点需实物试配，不压镜头或排线座。当前压扣采用平接触面，弹性及保持力未仿真。",
              "5. 光窗通道按宽余量开放；不代表实际收发视场已经测得。尾部缩短后，仅生成接线避让，线束、USB 插头及独立应力夹未建模。允许线束在壳外转弯，不能宣称内置弯曲半径已满足。",
              "6. 光学前端暂后移 30 mm；理论 230→80 mm 仅为几何估算，100 m 模块近端可用性需按当前靶面和模式验证。",
              "7. 先打印针孔和局部固定座试配，再决定完整壳体打印；胶接孔三档按 X 从小到大依次为 4.1、4.2、4.3 mm。", "",
              "## 建议装配顺序", "", "测距模块从前端光窗口滑入 → 从 IMU 平台检修口放入压扣并手动让位卡钩 → 装入 IMU → 连接线缆 → 盖上上盖。检修压扣时先取下上盖和 IMU，再释放压扣两侧卡钩。针杆完成试配后胶接。", "",
              "## 文件来源", "", "源文件：`cad/laser100m_A2.py`；参数：`cad/laser100m_A2_parameters.json`。", "旧 A1 和 STP-23L CAD、IMU 运行配置和测距脚本未修改。", ""]
    (OUT/"README_交付说明.md").write_text("\n".join(report), encoding="utf-8")
    hashes = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(OUT.iterdir()) if f.is_file() and f.name != "sha256.json"}
    (OUT/"sha256.json").write_text(json.dumps(hashes, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    bundle = ROOT/"output"/"laser100m_HWT906P_A2_STEP_bundle.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(OUT.iterdir()):
            if f.is_file():
                z.write(f, f"cad_laser100m_A2_compact/{f.name}")
        z.write(Path(__file__), "source/laser100m_A2.py")
        z.write(PARAM, "source/laser100m_A2_parameters.json")
    print(json.dumps({"output": str(OUT), "assembly_dimensions_mm": dims(reimport), "parts": len(parts), "bundle": str(bundle)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
