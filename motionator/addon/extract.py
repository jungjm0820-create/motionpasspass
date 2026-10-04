"""bpy -> numpy extraction. This is the ONLY module that reads Blender animation data.

Reads F-curves through the Slotted Actions structure
(Action -> layers -> strips -> channelbag for the object's slot -> fcurves).
"""
import re

import bpy
import numpy as np

NSLOT = 10
ROT_MODE = {"QUATERNION": 0, "XYZ": 1, "XZY": 2, "YXZ": 3, "YZX": 4, "ZXY": 5, "ZYX": 6, "AXIS_ANGLE": 7}
_SLOT_BASE = {"location": 0, "rotation_quaternion": 3, "rotation_euler": 3,
              "rotation_axis_angle": 3, "scale": 7}
_SLOT_LEN = {"location": 3, "rotation_quaternion": 4, "rotation_euler": 3,
             "rotation_axis_angle": 4, "scale": 3}
_MODE_PROP = {0: "rotation_quaternion", 1: "rotation_euler", 2: "rotation_euler", 3: "rotation_euler",
              4: "rotation_euler", 5: "rotation_euler", 6: "rotation_euler", 7: "rotation_axis_angle"}
_PATH_RE = re.compile(r'^pose\.bones\["(.+?)"\]\.(location|rotation_quaternion|rotation_euler|rotation_axis_angle|scale)$')


class RigData:
    """Everything the native core needs, as plain numpy arrays."""

    def __init__(self):
        self.obj_name = ""
        self.names = []
        self.name_to_idx = {}
        self.parent = None        # int32 [B]
        self.rest = None          # float64 [B,4,4]
        self.rot_mode = None      # int32 [B]
        self.defaults = None      # float64 [B,10]
        self.curve_off = None     # int32 [B*10]
        self.curve_n = None       # int32 [B*10]
        self.curve_extrap = None  # int32 [B*10]
        self.keys = None          # float64 [K,6]
        self.key_ipo = None       # int32 [K]
        self.lengths = None       # float64 [B]
        self.key_frames = []      # per bone: sorted np.array of keyed frames
        self.warnings = []
        self.n_curves = 0


# ------------------------------------------------------------------ fcurves

def get_object_fcurves(obj):
    """Return (list_of_fcurves, warning_or_None) for the object's CURRENT action slot."""
    ad = obj.animation_data
    if ad is None or ad.action is None:
        return [], "No action on the armature."
    action = ad.action
    if not hasattr(action, "layers"):               # legacy (pre-slotted) actions
        return list(action.fcurves), None

    slot = getattr(ad, "action_slot", None)
    if slot is None:
        return [], "The action has no slot assigned to this armature."

    cb = None
    try:
        from bpy_extras import anim_utils
        cb = anim_utils.action_get_channelbag_for_slot(action, slot)
    except Exception:
        cb = None
    if cb is None:
        for layer in action.layers:
            for strip in layer.strips:
                if strip.type != "KEYFRAME":
                    continue
                try:
                    cb = strip.channelbag(slot)
                except Exception:
                    cb = None
                if cb is None:
                    for cand in getattr(strip, "channelbags", []):
                        if getattr(cand, "slot", None) == slot:
                            cb = cand
                            break
                if cb is not None:
                    break
            if cb is not None:
                break
    if cb is None:
        return [], "No channelbag found for the armature's slot."
    warn = None
    if len(action.layers) > 1:
        warn = "Action has several layers; only the first matching layer is read."
    return list(cb.fcurves), warn


def _read_keys(fc):
    kps = fc.keyframe_points
    n = len(kps)
    co = np.empty(n * 2, dtype=np.float32)
    hl = np.empty(n * 2, dtype=np.float32)
    hr = np.empty(n * 2, dtype=np.float32)
    kps.foreach_get("co", co)
    kps.foreach_get("handle_left", hl)
    kps.foreach_get("handle_right", hr)
    co = co.reshape(n, 2).astype(np.float64)
    hl = hl.reshape(n, 2).astype(np.float64)
    hr = hr.reshape(n, 2).astype(np.float64)
    keys = np.concatenate([co, hl, hr], axis=1)          # [n, 6]
    try:
        ipo = np.empty(n, dtype=np.int32)
        kps.foreach_get("interpolation", ipo)
        if ipo.min() < 0 or ipo.max() > 20:
            raise ValueError("unexpected enum values")
    except Exception:
        m = {"CONSTANT": 0, "LINEAR": 1, "BEZIER": 2}
        ipo = np.array([m.get(kp.interpolation, 3) for kp in kps], dtype=np.int32)
    ipo = np.minimum(ipo, 3).astype(np.int32)
    return keys, ipo


# ------------------------------------------------------------------ rig build

def _bone_order(arm):
    order = []
    stack = [b for b in reversed([b for b in arm.bones if b.parent is None])]
    while stack:
        b = stack.pop()
        order.append(b)
        stack.extend(reversed(list(b.children)))
    return order


def read_dynamic(obj, rig):
    """Refresh values that change without touching F-curves (rotation mode, un-keyed channel values).
    Returns True if a rotation mode changed (curves must then be re-extracted)."""
    mode_changed = False
    pbs = obj.pose.bones
    for i, name in enumerate(rig.names):
        pb = pbs[name]
        m = ROT_MODE.get(pb.rotation_mode, 0)
        if m != rig.rot_mode[i]:
            rig.rot_mode[i] = m
            mode_changed = True
        d = rig.defaults[i]
        d[0:3] = pb.location
        if m == 0:
            d[3:7] = pb.rotation_quaternion
        elif m == 7:
            d[3:7] = pb.rotation_axis_angle
        else:
            d[3:6] = pb.rotation_euler
            d[6] = 0.0
        d[7:10] = pb.scale
    return mode_changed


def build_rig_data(obj):
    rig = RigData()
    rig.obj_name = obj.name
    arm = obj.data
    bones = _bone_order(arm)
    B = len(bones)
    rig.names = [b.name for b in bones]
    rig.name_to_idx = {n: i for i, n in enumerate(rig.names)}
    rig.parent = np.array([rig.name_to_idx[b.parent.name] if b.parent else -1 for b in bones], dtype=np.int32)
    rig.rest = np.array([np.array(b.matrix_local, dtype=np.float64) for b in bones], dtype=np.float64).reshape(B, 4, 4)
    rig.lengths = np.array([b.length for b in bones], dtype=np.float64)
    rig.rot_mode = np.zeros(B, dtype=np.int32)
    rig.defaults = np.zeros((B, NSLOT), dtype=np.float64)
    rig.defaults[:, 3] = 1.0
    rig.defaults[:, 7:10] = 1.0
    read_dynamic(obj, rig)

    # Support warnings: things F-curve evaluation can NOT reproduce.
    w = rig.warnings
    if getattr(arm, "pose_position", "POSE") == "REST":
        w.append("Armature is in Rest Position.")
    if obj.animation_data and len(obj.animation_data.nla_tracks):
        w.append("NLA tracks present (ignored).")
    if obj.animation_data and len(obj.animation_data.drivers):
        w.append("Drivers present (ignored).")
    n_con = sum(len(pb.constraints) for pb in obj.pose.bones)
    if n_con:
        w.append(f"{n_con} bone constraint(s) present (ignored).")
    for b in bones:
        if not b.use_inherit_rotation or b.inherit_scale != "FULL" or not b.use_local_location:
            w.append(f"Bone '{b.name}' uses non-default inheritance (result may differ).")
            break

    fcurves, warn = get_object_fcurves(obj)
    if warn:
        w.append(warn)

    C = B * NSLOT
    curve_off = np.full(C, -1, dtype=np.int32)
    curve_n = np.zeros(C, dtype=np.int32)
    curve_extrap = np.zeros(C, dtype=np.int32)
    key_chunks, ipo_chunks = [], []
    bone_key_frames = [[] for _ in range(B)]
    cursor = 0
    n_used = 0
    mod_warned = False
    for fc in fcurves:
        m = _PATH_RE.match(fc.data_path)
        if not m or fc.mute:
            continue
        name, prop = m.group(1), m.group(2)
        bi = rig.name_to_idx.get(name)
        if bi is None or _MODE_PROP[int(rig.rot_mode[bi])] != prop and prop.startswith("rotation"):
            continue
        idx = fc.array_index
        if idx >= _SLOT_LEN[prop]:
            continue
        if len(fc.modifiers) and not mod_warned:
            w.append("F-curve modifiers present (ignored, e.g. Cycles).")
            mod_warned = True
        n = len(fc.keyframe_points)
        if n == 0:
            continue
        keys, ipo = _read_keys(fc)
        c = bi * NSLOT + _SLOT_BASE[prop] + idx
        curve_off[c] = cursor
        curve_n[c] = n
        curve_extrap[c] = 1 if fc.extrapolation == "LINEAR" else 0
        key_chunks.append(keys)
        ipo_chunks.append(ipo)
        bone_key_frames[bi].append(keys[:, 0])
        cursor += n
        n_used += 1
        if (ipo > 2).any():
            w.append("Easing interpolation found (treated as linear).")

    if key_chunks:
        rig.keys = np.ascontiguousarray(np.concatenate(key_chunks, axis=0), dtype=np.float64)
        rig.key_ipo = np.ascontiguousarray(np.concatenate(ipo_chunks), dtype=np.int32)
    else:
        rig.keys = np.zeros((1, 6), dtype=np.float64)
        rig.key_ipo = np.zeros(1, dtype=np.int32)
    rig.curve_off, rig.curve_n, rig.curve_extrap = curve_off, curve_n, curve_extrap
    rig.key_frames = [np.unique(np.concatenate(k)) if k else np.zeros(0) for k in bone_key_frames]
    rig.n_curves = n_used
    rig.warnings = list(dict.fromkeys(rig.warnings))
    return rig
