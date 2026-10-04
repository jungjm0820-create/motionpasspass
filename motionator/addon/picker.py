"""Weight-based bone picking: click the mesh, select the bone that influences that spot the most.

Repeated clicks on the same spot cycle through the next-strongest influences.
"""
import bpy
from bpy.props import BoolProperty
from bpy.types import Operator
from bpy_extras import view3d_utils

_keymaps = []


def _influences(context, arm_obj, hit_obj, face_index, loc_world, mat_world):
    """Return [(bone_name, weight)] sorted by weight, interpolated at the hit point."""
    dg = context.evaluated_depsgraph_get()
    orig = hit_obj.original
    eo = orig.evaluated_get(dg)
    me_eval = eo.data
    me_orig = orig.data
    if len(me_eval.vertices) != len(me_orig.vertices):
        return None, "Mesh topology is changed by other modifiers; weight pick needs matching vertices."
    if face_index < 0 or face_index >= len(me_eval.polygons):
        return None, "Hit face index out of range."

    p = mat_world.inverted() @ loc_world
    verts = list(me_eval.polygons[face_index].vertices)
    inv = []
    for v in verts:
        d = (me_eval.vertices[v].co - p).length
        inv.append(1.0 / max(d, 1e-9))
    tot = sum(inv)
    inv = [i / tot for i in inv]

    vg_names = {vg.index: vg.name for vg in orig.vertex_groups}
    bone_names = {pb.name for pb in arm_obj.pose.bones}
    acc = {}
    for v, wv in zip(verts, inv):
        for g in me_orig.vertices[v].groups:
            nm = vg_names.get(g.group)
            if nm in bone_names and g.weight > 1e-4:
                acc[nm] = acc.get(nm, 0.0) + g.weight * wv
    return sorted(acc.items(), key=lambda kv: -kv[1]), None


class MOTIONATOR_OT_weight_pick(Operator):
    bl_idname = "motionator.weight_pick"
    bl_label = "Weight Pick"
    bl_description = "Select the bone with the highest skin weight under the mouse cursor"
    bl_options = {"REGISTER", "UNDO"}

    extend: BoolProperty(name="Extend", default=False)

    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return (ob is not None and ob.type == "ARMATURE" and ob.mode == "POSE"
                and context.area is not None and context.area.type == "VIEW_3D")

    def invoke(self, context, event):
        region, rv3d = context.region, context.region_data
        coord = (event.mouse_region_x, event.mouse_region_y)
        origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
        direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
        dg = context.evaluated_depsgraph_get()
        hit, loc, _nor, idx, hit_obj, mat = context.scene.ray_cast(dg, origin, direction)
        if not hit or hit_obj is None or hit_obj.original.type != "MESH":
            self.report({"INFO"}, "No mesh under cursor.")
            return {"CANCELLED"}

        arm_obj = context.active_object
        infl, err = _influences(context, arm_obj, hit_obj, idx, loc, mat)
        if err:
            self.report({"WARNING"}, err)
            return {"CANCELLED"}
        if not infl:
            self.report({"INFO"}, f"'{hit_obj.name}' has no weights for bones of '{arm_obj.name}'.")
            return {"CANCELLED"}

        arm = arm_obj.data
        active = arm.bones.active.name if arm.bones.active else None
        choice = infl[0]
        if not self.extend and active == infl[0][0] and len(infl) > 1:
            choice = infl[1]                      # click again -> next strongest influence
        bone = arm.bones.get(choice[0])
        if bone is None or bone.hide or not getattr(bone, "is_visible", True):
            self.report({"INFO"}, f"Bone '{choice[0]}' is hidden.")
            return {"CANCELLED"}

        if not self.extend:
            for b in arm.bones:
                b.select = b.select_head = b.select_tail = False
        bone.select = bone.select_head = bone.select_tail = True
        arm.bones.active = bone
        top = ", ".join(f"{n} {w * 100:.0f}%" for n, w in infl[:3])
        self.report({"INFO"}, f"Picked {bone.name}   (influences here: {top})")
        for area in context.window.screen.areas:
            area.tag_redraw()
        return {"FINISHED"}


def register():
    bpy.utils.register_class(MOTIONATOR_OT_weight_pick)
    kc = bpy.context.window_manager.keyconfigs.addon
    if kc:
        km = kc.keymaps.new(name="Pose", space_type="EMPTY")
        for shift in (False, True):
            kmi = km.keymap_items.new(MOTIONATOR_OT_weight_pick.bl_idname, "LEFTMOUSE", "PRESS",
                                      ctrl=True, alt=True, shift=shift)
            kmi.properties.extend = shift
            _keymaps.append((km, kmi))


def unregister():
    for km, kmi in _keymaps:
        km.keymap_items.remove(kmi)
    _keymaps.clear()
    bpy.utils.unregister_class(MOTIONATOR_OT_weight_pick)
