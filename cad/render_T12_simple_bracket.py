"""Depth-buffered VTK renders of the actual STL exports."""
from pathlib import Path
import vtk

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/cad_laser100m_T12_simple_bracket'
REF=OUT/'reference_DO_NOT_PRINT'
FONT=r'C:\Windows\Fonts\msyh.ttc'


def mesh_actor(ren,path,col,clip=False):
    reader=vtk.vtkSTLReader(); reader.SetFileName(str(path))
    source=reader
    if clip:
        plane=vtk.vtkPlane(); plane.SetOrigin(-38,0,0); plane.SetNormal(1,0,0)
        cutter=vtk.vtkClipPolyData(); cutter.SetInputConnection(reader.GetOutputPort())
        cutter.SetClipFunction(plane); source=cutter
    normals=vtk.vtkPolyDataNormals(); normals.SetInputConnection(source.GetOutputPort())
    normals.SetFeatureAngle(35); normals.SplittingOn(); normals.ConsistencyOn()
    mapper=vtk.vtkPolyDataMapper(); mapper.SetInputConnection(normals.GetOutputPort())
    actor=vtk.vtkActor(); actor.SetMapper(mapper)
    prop=actor.GetProperty(); prop.SetColor(*col); prop.SetAmbient(.28)
    prop.SetDiffuse(.72); prop.SetSpecular(.15); prop.SetSpecularPower(18)
    ren.AddActor(actor)


def text(ren,string,x,y,size=27):
    actor=vtk.vtkTextActor(); actor.SetInput(string); actor.SetPosition(x,y)
    p=actor.GetTextProperty(); p.SetFontFamily(vtk.VTK_FONT_FILE); p.SetFontFile(FONT)
    p.SetFontSize(size); p.SetColor(.12,.19,.24)
    ren.AddActor2D(actor)


def label(ren,xyz,string,x,y,line_x,line_y):
    ren.SetWorldPoint(*xyz,1); ren.WorldToDisplay(); tx,ty,_=ren.GetDisplayPoint()
    pts=vtk.vtkPoints(); pts.InsertNextPoint(line_x,line_y,0); pts.InsertNextPoint(tx,ty,0)
    lines=vtk.vtkCellArray(); lines.InsertNextCell(2); lines.InsertCellPoint(0); lines.InsertCellPoint(1)
    data=vtk.vtkPolyData(); data.SetPoints(pts); data.SetLines(lines)
    mapper=vtk.vtkPolyDataMapper2D(); mapper.SetInputData(data)
    actor=vtk.vtkActor2D(); actor.SetMapper(mapper)
    actor.GetProperty().SetColor(.23,.36,.42); actor.GetProperty().SetLineWidth(2)
    ren.AddActor2D(actor); text(ren,string,x,y)


def create(assembled):
    ren=vtk.vtkRenderer(); ren.SetBackground(.96,.97,.98)
    win=vtk.vtkRenderWindow(); win.SetOffScreenRendering(1); win.SetSize(1800,1200)
    win.SetMultiSamples(8); win.AddRenderer(ren)
    mesh_actor(ren,OUT/'T12_BRACKET.stl',(.68,.77,.81))
    if assembled:
        for name,col in [('needle',(.24,.29,.32)),('laser_envelope',(.22,.25,.28)),
                         ('imu_pcb',(.69,.18,.14)),('imu_header_assumed',(.16,.18,.2)),
                         ('imu_module_envelope',(.20,.25,.29)),
                         ('large_lens_illustrative',(.19,.47,.53)),
                         ('small_aperture_illustrative',(.08,.15,.17))]:
            mesh_actor(ren,REF/(name+'.stl'),col,name=='needle')
    cam=ren.GetActiveCamera(); target=(12,0,-3) if assembled else (26,0,-8)
    cam.SetFocalPoint(*target); cam.SetPosition(target[0]-110,target[1]-155,target[2]+112)
    cam.SetViewUp(0,0,1); cam.ParallelProjectionOn(); cam.SetParallelScale(51 if assembled else 45)
    ren.ResetCameraClippingRange(); win.Render()
    if assembled:
        text(ren,'T12 一体开放支架 · 实际模型装配',70,1110,46)
        label(ren,(-25,0,4.2),'现有针体\n仅显示尾部',95,530,300,563)
        label(ren,(26,-3,17),'IMU 底板＋上装模块',1070,950,1130,935)
        label(ren,(1,-8,-12),'测距模块\n前方镜头敞开',300,210,450,290)
        label(ren,(36,-21.9,-13),'唯一打印件\n四个承托脚',1370,280,1390,365)
        label(ren,(49,-15,4),'侧后方开放\n接线空间',1400,665,1390,695)
        text(ren,'模块为外廓参考；镜头圆形仅标示朝向。针体沿用原文件。',70,45,25)
        path=OUT/'T12_assembly_preview.png'
    else:
        text(ren,'T12 一体开放支架 · 单个打印件',70,1110,46)
        label(ren,(-3,0,4.2),'D 形针尾插孔\n适配 T10 平底针尾',70,800,320,785)
        label(ren,(26,2,2.4),'共用托板\n厚 2.4 mm',1170,930,1200,910)
        label(ren,(15,-13,5.4),'IMU 胶粘垫台',1360,665,1330,695)
        label(ren,(34,-19.5,-26.5),'测距模块承托脚',1250,220,1280,260)
        text(ren,'58 × 43.8 × 36.9 mm  ·  一个打印实体  ·  侧放切片，需支撑',70,45,26)
        path=OUT/'T12_holder_preview.png'
    win.Render()
    capture=vtk.vtkWindowToImageFilter(); capture.SetInput(win); capture.ReadFrontBufferOff(); capture.Update()
    writer=vtk.vtkPNGWriter(); writer.SetFileName(str(path)); writer.SetInputConnection(capture.GetOutputPort()); writer.Write()
    win.Finalize(); print(str(path),flush=True)


if __name__=='__main__':
    create(True)
    create(False)
