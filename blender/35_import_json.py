import argparse, json, sys
from pathlib import Path
import bpy
from mathutils import Vector

BONES=[("hips","left_hip"),("hips","right_hip"),("hips","left_shoulder"),("hips","right_shoulder"),("left_shoulder","right_shoulder"),("left_shoulder","left_elbow"),("left_elbow","left_wrist"),("right_shoulder","right_elbow"),("right_elbow","right_wrist"),("left_hip","left_knee"),("left_knee","left_ankle"),("left_ankle","left_heel"),("left_heel","left_foot_index"),("right_hip","right_knee"),("right_knee","right_ankle"),("right_ankle","right_heel"),("right_heel","right_foot_index"),("hips","nose")]

argv=sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else []
p=argparse.ArgumentParser(); p.add_argument("--json",required=True); p.add_argument("--output",required=True); a=p.parse_args(argv)
data=json.loads(Path(a.json).read_text(encoding="utf-8"))
bpy.ops.object.select_all(action="SELECT"); bpy.ops.object.delete(use_global=False)

objs={}
for j in data["frames"][0]["joints"]:
    bpy.ops.mesh.primitive_uv_sphere_add(segments=12,ring_count=6,radius=.035)
    objs[j]=bpy.context.object; objs[j].name=f"joint_{j}"

for s,e in BONES:
    if s not in objs or e not in objs: continue
    c=bpy.data.curves.new(f"curve_{s}_{e}","CURVE"); c.dimensions="3D"; c.bevel_depth=.018
    sp=c.splines.new("POLY"); sp.points.add(1)
    o=bpy.data.objects.new(f"bone_{s}_{e}",c); bpy.context.collection.objects.link(o)
    for pi,obj in enumerate([objs[s],objs[e]]):
        for ax,t in enumerate(["LOC_X","LOC_Y","LOC_Z"]):
            d=sp.points[pi].driver_add("co",ax).driver
            v=d.variables.new(); v.name="v"; v.type="TRANSFORMS"; v.targets[0].id=obj; v.targets[0].transform_type=t; v.targets[0].transform_space="WORLD_SPACE"; d.expression="v"

for fr in data["frames"]:
    f=int(fr["frame"])+1
    for j,loc in fr["joints"].items():
        objs[j].location=Vector(loc); objs[j].keyframe_insert("location",frame=f)

bpy.ops.mesh.primitive_plane_add(size=8)
bpy.ops.object.camera_add(location=(3.2,-5.0,2.3),rotation=(1.18,0,.55)); bpy.context.scene.camera=bpy.context.object
bpy.ops.object.light_add(type="AREA",location=(2,-2,4)); bpy.context.object.data.energy=1000; bpy.context.object.data.size=5
sc=bpy.context.scene; sc.frame_start=1; sc.frame_end=data["n_frames"]; sc.render.fps=data["fps"]
engines={e.identifier for e in sc.render.bl_rna.properties["engine"].enum_items}
sc.render.engine="BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engines else "BLENDER_EEVEE"
Path(a.output).parent.mkdir(parents=True,exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=str(Path(a.output)))
print("Saved:",a.output)
