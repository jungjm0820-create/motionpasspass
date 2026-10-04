"""Settings, panel and diagnostic operators."""
import time

import bpy
import numpy as np
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty
from bpy.types import Operator, Panel, PropertyGroup

from . import extract, fk_numpy, overlay
from .overlay import STATE


def _tag_redraw(self, context):
    STATE.cache_key = None
    for win in context.window_manager.windows:
        for area in win.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


class MOTIONATOR_PG_settings(PropertyGroup):
    enabled: BoolProperty(name="Motion Path", description="Show the real-time 3D motion path", default=False, update=_tag_redraw)
    frames_before: IntProperty(name="Before", default=30, min=0, max=1000, update=_tag_redraw)
    frames_after: IntProperty(name="After", default=30, min=0, max=1000, update=_tag_redraw)
    use_tail: BoolProperty(name="Bone Tail", description="Trace the bone tail instead of the head", default=False, update=_tag_redraw)
    speed_color: BoolProperty(name="Velocity Colors", description="Blue = slow, red = fast", default=True, update=_tag_redraw)
    show_frame_dots: BoolProperty(name="Frame Dots", default=True, update=_tag_redraw)
    show_keyframes: BoolProperty(name="Keyframes", default=True, update=_tag_redraw)
    depth_test: BoolProperty(name="Occlude by Geometry", default=False, update=_tag_redraw)
    line_width: FloatProperty(name="Width", default=3.0, min=1.0, max=12.0, update=_tag_redraw)
    backend: EnumProperty(
        name="Backend",
        items=[("AUTO", "Auto", "Native C++ if available, else numpy"),
               ("NATIVE", "Native C++", "Force the C++ core"),
               ("NUMPY", "numpy", "Force the pure-python reference")],
        default="AUTO", update=_tag_redraw)


def _pose_armature(context):
    obj = context.active_object
    return obj if obj and obj.type == "ARMATURE" else None


class MOTIONATOR_OT_verify(Operator):
    bl_idname = "motionator.verify"
    bl_label = "Verify Against Blender"
    bl_description = ("Compare the computed bone matrices with Blender's own evaluation (frame_set) "
                      "on a few frames. Run this first to confirm the math is correct for your rig")

    @classmethod
    def poll(cls, context):
        return _pose_armature(context) is not None

    def execute(self, context):
        obj = _pose_armature(context)
        scene = context.scene
        s = scene.motionator
        rig = extract.build_rig_data(obj)
        cur = scene.frame_current
        sample = np.unique(np.round(np.linspace(cur - s.frames_before, cur + s.frames_after, 7)).astype(int))
        _, pose = overlay.compute_pose(rig, sample.astype(np.float64), s.backend)
        err_t = err_r = 0.0
        worst = ""
        try:
            for i, f in enumerate(sample):
                scene.frame_set(int(f))
                dg = context.evaluated_depsgraph_get()
                eo = obj.evaluated_get(dg)
                for bi, name in enumerate(rig.names):
                    m = np.array(eo.pose.bones[name].matrix, dtype=np.float64)
                    d = np.abs(m - pose[i, bi])
                    et, er = d[:3, 3].max(), d[:3, :3].max()
                    if et > err_t or er > err_r:
                        worst = f"{name}@{int(f)}"
                    err_t, err_r = max(err_t, et), max(err_r, er)
        finally:
            scene.frame_set(cur)
        ok = err_t < 1e-4 and err_r < 1e-4
        STATE.verify_text = (f"{'OK' if ok else 'MISMATCH'}  max pos err {err_t:.2e}, "
                             f"max rot err {err_r:.2e}" + ("" if ok else f"  (worst: {worst})"))
        self.report({"INFO" if ok else "WARNING"}, "Motionator verify: " + STATE.verify_text)
        return {"FINISHED"}


class MOTIONATOR_OT_benchmark(Operator):
    bl_idname = "motionator.benchmark"
    bl_label = "Benchmark"
    bl_description = "Time the native core against the numpy reference on the current armature and frame window"

    @classmethod
    def poll(cls, context):
        return _pose_armature(context) is not None

    def execute(self, context):
        obj = _pose_armature(context)
        scene = context.scene
        s = scene.motionator
        t0 = time.perf_counter()
        rig = extract.build_rig_data(obj)
        t_extract = (time.perf_counter() - t0) * 1000
        cur = scene.frame_current
        frames = np.arange(cur - s.frames_before, cur + s.frames_after + 1, dtype=np.float64)
        parts = [f"extract {t_extract:.1f} ms"]
        res = {}
        for label, choice in (("numpy", "NUMPY"), ("native", "NATIVE")):
            if label == "native" and not overlay.native_available():
                parts.append("native: NOT AVAILABLE")
                continue
            n = 3 if label == "numpy" else 30
            t0 = time.perf_counter()
            for _ in range(n):
                _, res[label] = overlay.compute_pose(rig, frames, choice)
            parts.append(f"{label} {(time.perf_counter() - t0) / n * 1000:.2f} ms")
        if "native" in res:
            parts.append(f"max diff {np.abs(res['native'] - res['numpy']).max():.1e}")
        text = f"{len(rig.names)} bones x {len(frames)} frames | " + " | ".join(parts)
        STATE.verify_text = text
        self.report({"INFO"}, text)
        return {"FINISHED"}


class MOTIONATOR_OT_rebuild(Operator):
    bl_idname = "motionator.rebuild"
    bl_label = "Rebuild Cache"
    bl_description = "Force re-reading all F-curves"

    def execute(self, context):
        STATE.keys_dirty = True
        STATE.cache_key = None
        return {"FINISHED"}


class MOTIONATOR_PT_main(Panel):
    bl_label = "Motionator"
    bl_idname = "MOTIONATOR_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Motionator"

    def draw(self, context):
        layout = self.layout
        s = context.scene.motionator
        obj = context.active_object

        row = layout.row()
        row.scale_y = 1.4
        row.prop(s, "enabled", toggle=True, icon="CURVE_PATH")
        if not (obj and obj.type == "ARMATURE" and obj.mode == "POSE"):
            layout.label(text="Select an armature and enter Pose Mode.", icon="INFO")

        col = layout.column(align=True)
        col.prop(s, "frames_before")
        col.prop(s, "frames_after")
        col = layout.column(align=True)
        col.prop(s, "use_tail")
        col.prop(s, "speed_color")
        col.prop(s, "show_frame_dots")
        col.prop(s, "show_keyframes")
        col.prop(s, "depth_test")
        col.prop(s, "line_width")

        box = layout.box()
        box.label(text="Engine", icon="PREFERENCES")
        box.prop(s, "backend", text="")
        native_ok = overlay.native_available()
        box.label(text="Native C++ core: " + ("loaded" if native_ok else "NOT loaded (numpy fallback)"),
                  icon="CHECKMARK" if native_ok else "ERROR")
        if STATE.last_backend:
            box.label(text=f"{STATE.last_backend}: {STATE.last_ms:.2f} ms  ({STATE.last_bones} bones, {STATE.last_frames} frames)")
        row = box.row(align=True)
        row.operator("motionator.verify", icon="CHECKMARK")
        row.operator("motionator.benchmark", icon="TIME")
        if STATE.verify_text:
            box.label(text=STATE.verify_text)

        rig = STATE.rig
        if rig is not None and rig.warnings:
            wb = layout.box()
            wb.label(text="Not reproduced (F-curve eval only):", icon="ERROR")
            for t in rig.warnings[:6]:
                wb.label(text=t)
        if STATE.error:
            eb = layout.box()
            eb.label(text="Error (screenshot this):", icon="CANCEL")
            for line in STATE.error.strip().splitlines()[-4:]:
                eb.label(text=line[:90])

        layout.separator()
        layout.label(text="Weight Pick: Ctrl+Alt+Click on mesh", icon="MOUSE_LMB")
        layout.label(text="(Shift added = extend selection)")
        layout.operator("motionator.rebuild", icon="FILE_REFRESH")


classes = (MOTIONATOR_PG_settings, MOTIONATOR_OT_verify, MOTIONATOR_OT_benchmark,
           MOTIONATOR_OT_rebuild, MOTIONATOR_PT_main)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.motionator = PointerProperty(type=MOTIONATOR_PG_settings)


def unregister():
    del bpy.types.Scene.motionator
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
