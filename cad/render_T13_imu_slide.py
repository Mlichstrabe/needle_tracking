"""Actual mesh rendering with rear latch and slide rail highlighted."""
from pathlib import Path
import vtk
import render_T12_simple_bracket as base

OUT=Path(__file__).resolve().parents[1]/'output/cad_laser100m_T13_imu_slide'
REF=OUT/'reference_DO_NOT_PRINT'


def create(assembled):
    ren=vtk.vtkRenderer();ren.SetBackground(.96,.97,.98)
    win=vtk.vtkRenderWindow();win.SetOffScreenRendering(1);win.SetSize(1800,1200)
    win.SetMultiSamples(8);win.AddRenderer(ren)
    base.mesh_actor(ren,REF/'rigid_without_latches.stl',(.66,.77,.82))
    for i in (0,1):base.mesh_actor(ren,REF/('latch_'+str(i)+'.stl'),(.90,.55,.23))
    if assembled:
        for n,c in [('imu_pcb',(.69,.18,.14)),('imu_header_assumed',(.16,.18,.2)),
                    ('imu_module_envelope',(.20,.25,.29)),('laser_envelope',(.22,.25,.28))]:
            base.mesh_actor(ren,REF/(n+'.stl'),c)
    camera=ren.GetActiveCamera();camera.SetFocalPoint(30,0,-5)
    # Rear-right view exposes the slide entrance and release tabs.
    camera.SetPosition(150,-145,103);camera.SetViewUp(0,0,1)
    camera.ParallelProjectionOn();camera.SetParallelScale(43)
    ren.ResetCameraClippingRange();win.Render()
    base.text(ren,'T13  IMU 滑入式卡扣 · '+('装配参考' if assembled else '一体支架'),65,1100,45)
    base.label(ren,(50,-13.3,7.4),'两侧压边导轨\n底板从尾部滑入',65,850,370,850)
    base.label(ren,(56,-19.5,6),'尾部止退扣\n向外拨开后取出',1210,735,1200,740)
    base.label(ren,(30,-19.7,5.8),'长弹性臂',70,320,330,370)
    base.label(ren,(11.6,12,6),'前端限位',1200,940,1170,950)
    base.label(ren,(34,-21.9,-20),'测距模块托脚\n沿用 T12',1280,215,1250,300)
    base.text(ren,'橙色仅用于突出卡扣；与蓝色部分连成一个打印件。IMU 不用胶粘或螺丝。',65,55,25)
    win.Render();capture=vtk.vtkWindowToImageFilter();capture.SetInput(win);capture.ReadFrontBufferOff();capture.Update()
    writer=vtk.vtkPNGWriter();writer.SetFileName(str(OUT/('T13_assembly_preview.png' if assembled else 'T13_holder_preview.png')))
    writer.SetInputConnection(capture.GetOutputPort());writer.Write();win.Finalize()


if __name__=='__main__':
    create(False);create(True);print('T13 previews exported',flush=True)
