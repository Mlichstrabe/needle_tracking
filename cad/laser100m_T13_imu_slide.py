"""T12 holder with an integral rear-entry IMU rail and releasable latch pair."""
import hashlib
import json
from pathlib import Path
import cadquery as cq
import laser100m_T12_simple_bracket as base

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/cad_laser100m_T13_imu_slide'
box=base.box
P=dict(base.P,revision='T13-IMU-slide-latch',imu_fixing='rear-entry rail + two release latches',
       imu_adhesive=False,imu_screws=False,pcb_slot_height=2.0,pcb_thickness_assumed=1.6,
       rail_vertical_clearance=.4,rail_edge_overlap_nominal=1.7,
       front_stop_clearance=.4,rear_latch_clearance=.4,latch_thickness=1.4,
       latch_height=3.0,latch_effective_length=33.4,release_outward_each=2.2,
       physical_latch_cycle_test=False)


def build():
    rigid=base.build_holder()
    # Remove only the old IMU pads/guides, preserving every lower feature.
    for x in (13.,48.):
        for a,b in ((11,15),(-15,-11),(16,18.4),(-18.4,-16)):
            rigid=rigid.cut(box(x,x+4,a,b,2.4,10))
    for a,b in ((11,15),(-15,-11)):
        rigid=rigid.cut(box(8.6,11,a,b,2.4,10))
    for x0,x1 in ((12.,18.),(48.,53.)):
        for sign in (-1,1):
            def side(a,b,z0,z1):
                return box(x0,x1,a,b,z0,z1) if sign==1 else box(x0,x1,-b,-a,z0,z1)
            rigid=rigid.fuse(side(11,15,2.4,5.4))
            rigid=rigid.fuse(side(16,18.4,2.4,9.2))
            rigid=rigid.fuse(side(13.3,18.4,7.4,9.2))
    for a,b in ((11,15),(-15,-11)):
        rigid=rigid.fuse(box(9.2,11.6,a,b,2.4,7.4))
    arms=[]
    for sign in (-1,1):
        # Long lateral arms bend outward. They sit above the common plate;
        # open side gaps permit release without bending the PCB itself.
        anchor=box(14,18,16,20.4,2.4,7.4)
        pts=[(17,16),(18,16),(21,19),(58.5,19),(58.5,20.4),(17,20.4)]
        arm=cq.Workplane('XY',origin=(0,0,4.4)).polyline(pts).close().extrude(3).val()
        # Vertical retaining face at x=54.4; rear diagonal is insertion lead-in.
        ramp=[(54.4,14),(55.3,14),(58.5,19),(58.5,20.4),(54.4,20.4)]
        hook=cq.Workplane('XY',origin=(0,0,5.2)).polyline(ramp).close().extrude(2.2).val()
        grip=box(55.8,59,19,21.4,4.4,7.4)
        arm=base.one(arm.fuse(hook).fuse(grip).clean())
        if sign<0:
            anchor=anchor.mirror('XZ')
            arm=arm.mirror('XZ')
        rigid=rigid.fuse(anchor)
        arms.append(arm)
    rigid=base.one(rigid.clean())
    holder=base.one(rigid.fuse(*arms).clean())
    return holder,rigid,arms


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    holder,rigid,arms=build()
    needle,laser,imu,pcb,header,module=base.reference_parts()
    checks={n:base.assert_clear(n,holder,p) for n,p in [('needle',needle),('laser',laser),('imu',imu)]}
    # The lower holder and needle interface remain identical to T12.
    lower_region=box(-10,80,-30,30,-40,2.4)
    old_lower=base.build_holder().intersect(lower_region)
    new_lower=holder.intersect(lower_region)
    delta=sum(s.Volume() for s in old_lower.cut(new_lower).Solids())+sum(s.Volume() for s in new_lower.cut(old_lower).Solids())
    assert delta<1e-6
    # Static checks for retention after insertion.
    retention={}
    for name,shift in [('rear_withdrawal',(1,0,0)),('front_push',(-1,0,0)),('lift',(0,0,.8))]:
        v=base.overlap(holder,imu.translate(shift))
        assert v>.01,(name,v)
        retention[name+'_blocked_mm3']=v
    # An open-latch clearance proxy, not a nonlinear material simulation.
    # Arms are translated outward as rigid envelopes. Attachment mechanics
    # and strain/stress concentrations still require physical validation.
    open_arms=[arms[0].translate((0,-2.2,0)),arms[1].translate((0,2.2,0))]
    paths=[]
    for dx in (65,50,40,30,20,10,5,2,1,0):
        pos=imu.translate((dx,0,0))
        row={'x_shift':dx,'rigid':base.assert_clear('rail path',rigid,pos)}
        for i,a in enumerate(open_arms): row['released_arm_'+str(i)]=base.assert_clear('released path',a,pos)
        paths.append(row)
    for dx in (65,35,10,0): base.assert_clear('laser rear path',holder,laser.translate((dx,0,0)))
    for dx in (-30,-10,0): base.assert_clear('needle path',holder,needle.translate((dx,0,0)))
    base.assert_clear('front optical corridor',holder,box(-210,1,-18.5,18.5,-25.6,-1))
    steps={}
    printing=holder.rotate((0,0,0),(1,0,0),90)
    b=base.bounds(printing); shift=(10-b[0],10-b[1],-b[2]); printing=printing.translate(shift)
    for n,s in [('T13_BRACKET',holder),('T13_PRINT_SIDE_DOWN',printing)]:
        steps[n]=base.export_step(OUT/(n+'.step'),s)
        base.export_mesh(OUT/(n+'.stl'),s)
    ref=OUT/'reference_DO_NOT_PRINT';ref.mkdir(exist_ok=True)
    assembly=cq.Assembly(name='T13_IMU_SLIDE_LATCH_SINGLE_PRINTED_HOLDER')
    assembly.add(holder,name='ONLY_PRINTED_PART',color=cq.Color(.65,.76,.81))
    for n,s,c in [('needle',needle,(.24,.29,.32)),('laser_envelope',laser,(.22,.25,.28)),
                  ('imu_pcb',pcb,(.69,.18,.14)),('imu_header_assumed',header,(.16,.18,.20)),
                  ('imu_module_envelope',module,(.20,.25,.29))]:
        assembly.add(s,name='REF_'+n,color=cq.Color(*c));base.export_mesh(ref/(n+'.stl'),s)
    assembly.export(str(OUT/'T13_ASSEMBLY_REFERENCE.step'))
    rt=cq.importers.importStep(str(OUT/'T13_ASSEMBLY_REFERENCE.step')).val()
    assert rt.isValid() and len(rt.Solids())==6
    # Colored arm meshes are rendering references already present in holder.
    for i,a in enumerate(arms):base.export_mesh(ref/('latch_'+str(i)+'.stl'),a)
    base.export_mesh(ref/'rigid_without_latches.stl',rigid)
    # Provide the same viewing-only lens direction references.
    for n,y,z,r in [('large_lens',-3,-10.5,6.3),('small_aperture',11,-10.5,2.1)]:
        base.export_mesh(ref/(n+'_illustrative.stl'),cq.Solid.makeCylinder(r,.35,cq.Vector(.65,y,z),cq.Vector(1,0,0)))
    report=dict(parameters=P,dimensions_mm=base.dimensions(holder),step_roundtrip=steps,
                overlap_mm3=checks,lower_T12_geometry_difference_mm3=delta,retention_checks=retention,
                released_path_proxy=paths,release_model='Rigid arm-envelope displacement; NOT elastic FEA',
                simple_cantilever_strain_estimate=1.5*1.4*2.2/33.4**2,
                strain_estimate_note='Small-deflection beam estimate only; ignores root stress concentration, layer anisotropy and fatigue. Not a PLA pass/fail criterion.',
                print_translation_mm=shift,needle_source=str(base.NEEDLE),
                needle_sha256=hashlib.sha256(base.NEEDLE.read_bytes()).hexdigest(),
                support_required=True,physical_latch_test=False,slicer_validated=False)
    (OUT/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'parameters.json').write_text(json.dumps(P,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(output=str(OUT),dimensions=base.dimensions(holder),retention=retention),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
