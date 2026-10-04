"""Real-time 3D motion path overlay (POST_VIEW).

Pipeline:  extract (bpy, main thread, cached)  ->  native/numpy FK  ->  GPU polyline.
Frames are never stepped with frame_set(): positions come from F-curve evaluation.
"""
import time
import traceback

import bpy
import gpu
import numpy as np
from bpy.app.handlers import persistent
from gpu_extras.batch import batch_for_shader

from . import extract, fk_numpy

try:
    import motionator_core as native
except Exception:  # not installed / wrong platform -> numpy fallback
    native = None


class _State:
    def __init__(self):
        self.rig = None
        self.keys_dirty = True
        self.dyn_dirty = True
        self.cache_key = None
        self.batches = []
        self.shaders = {}
        self.error = ""
        self.last_ms = 0.0
        self.last_backend = ""
        self.last_frames = 0
        self.last_bones = 0
        self.verify_text = ""
        self.draw_handle = None


STATE = _State()


def native_available():
    return native is not None


# ------------------------------------------------------------------ compute

def pick_backend(choice):
    if choice == "NUMPY" or native is None:
        return "numpy", fk_numpy.evaluate_pose
    return "native", native.evaluate_pose


def compute_pose(rig, frames, backend_choice="AUTO"):
    name, fn = pick_backend(backend_choice)
    pose = fn(frames, rig.parent, rig.rest, rig.rot_mode, rig.defaults,
              rig.curve_off, rig.curve_n, rig.curve_extrap, rig.keys, rig.key_ipo, 0)
    return name, pose


def get_rig(obj):
    st = STATE
    rig = st.rig
    if rig is not None and (rig.obj_name != obj.name or len(rig.names) != len(obj.data.bones)):
        rig = None
    if rig is None or st.keys_dirty:
        rig = extract.build_rig_data(obj)
        st.rig = rig
        st.keys_dirty = False
        st.dyn_dirty = False
        st.cache_key = None
    elif st.dyn_dirty:
        if extract.read_dynamic(obj, rig):
            st.keys_dirty = True
            return get_rig(obj)
        st.dyn_dirty = False
        st.cache_key = None
    return rig


# ------------------------------------------------------------------ colors / shaders

def _speed_colors(speed, alpha):
    """blue (slow) -> green -> red (fast)."""
    mx = float(speed.max()) if len(speed) else 0.0
    t = speed / mx if mx > 1e-12 else np.zeros_like(speed)
    col = np.empty((len(t), 4), dtype=np.float32)
    lo = t < 0.5
    u = np.where(lo, t * 2.0, (t - 0.5) * 2.0)
    col[:, 0] = np.where(lo, 0.15 + 0.0 * u, 0.15 + 0.85 * u)
    col[:, 1] = np.where(lo, 0.45 + 0.45 * u, 0.90 - 0.60 * u)
    col[:, 2] = np.where(lo, 1.00 - 0.75 * u, 0.15 - 0.10 * u)
    col[:, 3] = alpha
    return col


def _shader(name):
    sh = STATE.shaders.get(name)
    if sh is None:
        sh = gpu.shader.from_builtin(name)
        STATE.shaders[name] = sh
    return sh


def _line_shader():
    for nm in ("POLYLINE_SMOOTH_COLOR", "SMOOTH_COLOR"):
        try:
            return nm, _shader(nm)
        except Exception:
            continue
    raise RuntimeError("No usable line shader found")


# ------------------------------------------------------------------ batches

def _target_indices(context, obj, rig):
    names = [pb.name for pb in (context.selected_pose_bones or [])]
    if not names and context.active_pose_bone:
        names = [context.active_pose_bone.name]
    return [rig.name_to_idx[n] for n in names if n in rig.name_to_idx]


def _build(context, obj, rig, s):
    cur = context.scene.frame_current
    f0, f1 = cur - s.frames_before, cur + s.frames_after
    frames = np.arange(f0, f1 + 1, dtype=np.float64)
    targets = _target_indices(context, obj, rig)
    if not targets:
        STATE.batches = []
        return

    t0 = time.perf_counter()
    backend, pose = compute_pose(rig, frames, s.backend)
    t_compute = (time.perf_counter() - t0) * 1000.0

    mw = np.array(obj.matrix_world, dtype=np.float64)
    m3, tr = mw[:3, :3], mw[:3, 3]
    line_name, line_sh = _line_shader()
    pt_sh = _shader("UNIFORM_COLOR")

    batches = []
    for bi in targets:
        head = pose[:, bi, :3, 3]
        if s.use_tail:
            head = head + pose[:, bi, :3, 1] * rig.lengths[bi]
        pts = head @ m3.T + tr                                   # world space [F,3]
        if len(pts) < 2:
            continue
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        speed = np.empty(len(pts))
        speed[0], speed[-1] = seg[0], seg[-1]
        speed[1:-1] = 0.5 * (seg[1:] + seg[:-1])
        alpha = np.where(frames < cur, 0.55, 1.0).astype(np.float32)
        col = _speed_colors(speed, alpha) if s.speed_color else np.tile(
            np.array([1.0, 0.8, 0.2, 1.0], dtype=np.float32), (len(pts), 1))
        line = batch_for_shader(line_sh, "LINE_STRIP",
                                {"pos": np.ascontiguousarray(pts, dtype=np.float32), "color": col})
        item = {"line": line, "line_name": line_name, "line_shader": line_sh}

        if s.show_frame_dots:
            item["dots"] = batch_for_shader(pt_sh, "POINTS", {"pos": np.ascontiguousarray(pts, dtype=np.float32)})
        if s.show_keyframes and len(rig.key_frames[bi]):
            kf = rig.key_frames[bi]
            kf = kf[(kf >= f0) & (kf <= f1)]
            if len(kf):
                kp = np.stack([np.interp(kf, frames, pts[:, a]) for a in range(3)], axis=1)
                item["keys"] = batch_for_shader(pt_sh, "POINTS", {"pos": kp.astype(np.float32)})
        ci = int(np.clip(cur - f0, 0, len(pts) - 1))
        item["cur"] = batch_for_shader(pt_sh, "POINTS", {"pos": pts[ci:ci + 1].astype(np.float32)})
        batches.append(item)

    STATE.batches = batches
    STATE.last_ms = t_compute
    STATE.last_backend = backend
    STATE.last_frames = len(frames)
    STATE.last_bones = len(rig.names)


def _cache_key(context, obj, s):
    mw = tuple(round(v, 6) for row in obj.matrix_world for v in row)
    sel = tuple(sorted(pb.name for pb in (context.selected_pose_bones or [])))
    act = context.active_pose_bone.name if context.active_pose_bone else ""
    return (obj.name, context.scene.frame_current, s.frames_before, s.frames_after, s.use_tail,
            s.speed_color, s.show_frame_dots, s.show_keyframes, s.backend, mw, sel, act)


# ------------------------------------------------------------------ draw

def draw_callback():
    context = bpy.context
    scene = context.scene
    s = getattr(scene, "motionator", None)
    if s is None or not s.enabled:
        return
    obj = context.active_object
    if obj is None or obj.type != "ARMATURE" or obj.mode != "POSE":
        return
    try:
        rig = get_rig(obj)
        key = _cache_key(context, obj, s)
        if key != STATE.cache_key:
            _build(context, obj, rig, s)
            STATE.cache_key = key
            STATE.error = ""

        if not STATE.batches:
            return
        region = context.region
        gpu.state.blend_set("ALPHA")
        gpu.state.depth_test_set("LESS_EQUAL" if s.depth_test else "NONE")
        pt_sh = _shader("UNIFORM_COLOR")
        for item in STATE.batches:
            sh = item["line_shader"]
            sh.bind()
            if item["line_name"].startswith("POLYLINE"):
                sh.uniform_float("viewportSize", (region.width, region.height))
                sh.uniform_float("lineWidth", float(s.line_width))
            else:
                gpu.state.line_width_set(float(s.line_width))
            item["line"].draw(sh)

            pt_sh.bind()
            if "dots" in item:
                gpu.state.point_size_set(float(s.line_width) + 1.0)
                pt_sh.uniform_float("color", (1.0, 1.0, 1.0, 0.55))
                item["dots"].draw(pt_sh)
            if "keys" in item:
                gpu.state.point_size_set(float(s.line_width) * 2.6)
                pt_sh.uniform_float("color", (1.0, 0.85, 0.1, 1.0))
                item["keys"].draw(pt_sh)
            gpu.state.point_size_set(float(s.line_width) * 3.2)
            pt_sh.uniform_float("color", (1.0, 1.0, 1.0, 1.0))
            item["cur"].draw(pt_sh)
        gpu.state.blend_set("NONE")
        gpu.state.depth_test_set("NONE")
    except Exception:
        STATE.error = traceback.format_exc(limit=4)
        STATE.batches = []
        STATE.cache_key = None


# ------------------------------------------------------------------ handlers

@persistent
def depsgraph_handler(scene, depsgraph):
    st = STATE
    if depsgraph.id_type_updated("ACTION"):
        st.keys_dirty = True
    for u in depsgraph.updates:
        idb = u.id
        if isinstance(idb, bpy.types.Object) and idb.type == "ARMATURE":
            st.dyn_dirty = True
            if u.is_updated_geometry:       # bones added/removed etc.
                st.keys_dirty = True
            break


@persistent
def load_post_handler(*_):
    STATE.rig = None
    STATE.keys_dirty = True
    STATE.cache_key = None
    STATE.batches = []


def register():
    if STATE.draw_handle is None:
        STATE.draw_handle = bpy.types.SpaceView3D.draw_handler_add(draw_callback, (), "WINDOW", "POST_VIEW")
    if depsgraph_handler not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(depsgraph_handler)
    if load_post_handler not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(load_post_handler)


def unregister():
    if STATE.draw_handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(STATE.draw_handle, "WINDOW")
        STATE.draw_handle = None
    for lst, fn in ((bpy.app.handlers.depsgraph_update_post, depsgraph_handler),
                    (bpy.app.handlers.load_post, load_post_handler)):
        if fn in lst:
            lst.remove(fn)
