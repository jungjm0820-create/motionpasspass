"""Exercises addon/extract.py + overlay compute path with a fake bpy (no Blender needed)."""
import os, sys, types
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

# ---- fake modules -----------------------------------------------------------
def mod(name, **kw):
    m = types.ModuleType(name); m.__dict__.update(kw); sys.modules[name] = m; return m

class _Obj: pass
bpy = mod("bpy", types=types.SimpleNamespace(Object=_Obj, SpaceView3D=object, Operator=object, Panel=object, PropertyGroup=object))
bpy.app = types.SimpleNamespace(handlers=types.SimpleNamespace(depsgraph_update_post=[], load_post=[]))
mod("bpy.app.handlers", persistent=lambda f: f)
mod("bpy.props", **{k: (lambda *a, **k: None) for k in ("BoolProperty", "EnumProperty", "FloatProperty", "IntProperty", "PointerProperty")})
mod("bpy.types", Operator=object, Panel=object, PropertyGroup=object)
mod("bpy_extras"); mod("bpy_extras.view3d_utils")
mod("gpu"); mod("gpu_extras"); mod("gpu_extras.batch", batch_for_shader=lambda *a, **k: (a, k))

# ---- fake Blender data ---------------------------------------------------------
class KP:
    def __init__(s, x, y, hl, hr, ipo="BEZIER"):
        s.co = (x, y); s.handle_left = hl; s.handle_right = hr; s.interpolation = ipo

class KPs(list):
    def foreach_get(self, attr, arr):
        if attr == "interpolation":
            raise RuntimeError("enum foreach_get unsupported (exercise fallback)")
        arr[:] = np.array([getattr(k, attr) for k in self], dtype=np.float32).ravel()

class FC:
    def __init__(s, path, idx, kps, extrap="CONSTANT"):
        s.data_path, s.array_index = path, idx
        s.keyframe_points = KPs(kps); s.extrapolation = extrap; s.mute = False; s.modifiers = []

class Bone:
    def __init__(s, name, parent, head, length=1.0):
        s.name, s.parent, s.children, s.length = name, parent, [], length
        m = np.eye(4); m[:3, 3] = head; s.matrix_local = m.tolist()
        s.use_inherit_rotation, s.inherit_scale, s.use_local_location = True, "FULL", True
        if parent: parent.children.append(s)

class PB:
    def __init__(s):
        s.rotation_mode = "QUATERNION"; s.location = (0, 0, 0); s.rotation_quaternion = (1, 0, 0, 0)
        s.rotation_euler = (0, 0, 0); s.rotation_axis_angle = (0, 0, 1, 0); s.scale = (1, 1, 1); s.constraints = []

class Pose:
    def __init__(s, names): s.bones = {n: PB() for n in names}
    # extract uses pose.bones[name] and iterates pose.bones -> make dict iterate values
class PBMap(dict):
    def __iter__(s): return iter(s.values())
    def __getitem__(s, k): return dict.__getitem__(s, k)

root = Bone("root", None, (0, 0, 0)); mid = Bone("mid", root, (0, 1, 0)); tip = Bone("tip", mid, (0, 2, 0))
arm = types.SimpleNamespace(bones=[root, mid, tip], pose_position="POSE")
pbm = PBMap({n: PB() for n in ("root", "mid", "tip")})
pbm["mid"].rotation_mode = "XYZ"

lin = lambda x, y: KP(x, y, (x - 1, y), (x + 1, y), "LINEAR")
fcs = [
    FC('pose.bones["root"].location', 0, [lin(0, 0), lin(10, 10)]),                       # root x: 0 -> 10
    FC('pose.bones["mid"].rotation_euler', 2, [KP(0, 0, (-1, 0), (3, 0)), KP(10, np.pi / 2, (7, np.pi / 2), (11, np.pi / 2))]),
    FC('pose.bones["tip"].scale', 1, [lin(0, 1), lin(10, 2)], "LINEAR"),
]
channelbag = types.SimpleNamespace(fcurves=fcs, slot="SLOT")
strip = types.SimpleNamespace(type="KEYFRAME", channelbag=lambda slot: channelbag if slot == "SLOT" else None)
action = types.SimpleNamespace(layers=[types.SimpleNamespace(strips=[strip])])
ad = types.SimpleNamespace(action=action, action_slot="SLOT", nla_tracks=[], drivers=[])
obj = types.SimpleNamespace(name="Rig", type="ARMATURE", data=arm, animation_data=ad,
                            pose=types.SimpleNamespace(bones=pbm))

import importlib
extract = importlib.import_module("addon.extract")
overlay = importlib.import_module("addon.overlay")

rig = extract.build_rig_data(obj)
assert rig.names == ["root", "mid", "tip"], rig.names
assert list(rig.parent) == [-1, 0, 1]
assert rig.curve_n[0 * 10 + 0] == 2           # root location X
assert rig.curve_n[1 * 10 + 3 + 2] == 2       # mid euler Z -> slot 5
assert rig.curve_n[2 * 10 + 7 + 1] == 2       # tip scale Y -> slot 8
assert rig.curve_extrap[2 * 10 + 8] == 1
assert list(rig.key_ipo[:2]) == [1, 1]        # interpolation fallback path worked
assert not rig.warnings, rig.warnings

frames = np.array([0.0, 5.0, 10.0])
for backend in ("NUMPY", "AUTO"):
    name, pose = overlay.compute_pose(rig, frames, backend)
    # root head x follows the location curve
    assert np.allclose(pose[:, 0, 0, 3], [0, 5, 10]), pose[:, 0, 0, 3]
    # mid rotates 0 -> 90deg about Z (bezier, symmetric => 45deg at frame 5): its Y axis swings toward -X
    ang = np.arctan2(pose[1, 1, 1, 0], pose[1, 1, 1, 1])
    assert abs(abs(ang) - np.pi / 4) < 1e-5, ang
    assert np.allclose(pose[2, 1, :3, 1], [-1, 0, 0], atol=1e-6)   # keys are float32 in Blender
    # tip scale Y doubles by frame 10 (bone y axis length = 2)
    assert abs(np.linalg.norm(pose[2, 2, :3, 1]) - 2.0) < 1e-9
    print(backend, "-> backend used:", name, "OK")

# rotation-mode change is detected by read_dynamic
pbm["mid"].rotation_mode = "QUATERNION"
assert extract.read_dynamic(obj, rig) is True
print("extract + compute path OK (mock bpy)")
