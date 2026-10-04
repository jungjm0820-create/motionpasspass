"""Pure numpy reference implementation of the FK evaluator.

Same data contract as the native core (see src/core.cpp). Used as
  * the correctness reference in tests,
  * an automatic fallback when the native module is not available.
Deliberately written differently from the C++ (polynomial root solve for beziers).
No bpy imports: must stay importable outside Blender.
"""
import numpy as np

NSLOT = 10
_EULER = {1: "XYZ", 2: "XZY", 3: "YXZ", 4: "YZX", 5: "ZXY", 6: "ZYX"}


def _rot(axis, a):
    c, s = np.cos(a), np.sin(a)
    if axis == "X":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "Y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rotation_matrix(mode, r):
    if mode == 0:
        q = np.asarray(r[:4], dtype=float)
        n = np.linalg.norm(q)
        w, x, y, z = (1.0, 0.0, 0.0, 0.0) if n < 1e-30 else q / n
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ])
    if 1 <= mode <= 6:
        order = _EULER[mode]
        idx = {"X": 0, "Y": 1, "Z": 2}
        m = np.eye(3)
        for ch in order:               # applied first -> last, so left-multiply
            m = _rot(ch, r[idx[ch]]) @ m
        return m
    ang, axis = r[0], np.asarray(r[1:4], dtype=float)
    n = np.linalg.norm(axis)
    if n < 1e-30:
        return np.eye(3)
    x, y, z = axis / n
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(ang) * k + (1 - np.cos(ang)) * (k @ k)


def _bezier_y(p, q, x):
    v1, v2, v3, v4 = p[0:2].copy(), p[4:6].copy(), q[2:4].copy(), q[0:2].copy()
    h1, h2 = v1 - v2, v4 - v3
    ln = v4[0] - v1[0]
    if ln > 0:
        l1, l2 = abs(h1[0]), abs(h2[0])
        if l1 + l2 > ln:
            fac = ln / (l1 + l2)
            v2 = v1 - fac * h1
            v3 = v4 - fac * h2
    X = (v1[0], v2[0], v3[0], v4[0])
    coef = [-X[0] + 3 * X[1] - 3 * X[2] + X[3],
            3 * X[0] - 6 * X[1] + 3 * X[2],
            -3 * X[0] + 3 * X[1],
            X[0] - x]
    roots = np.roots(coef) if any(abs(c) > 1e-14 for c in coef[:3]) else []
    cands = [r.real for r in roots if abs(r.imag) < 1e-9 and -1e-7 <= r.real <= 1 + 1e-7]
    if not cands:
        cands = [min(max((x - X[0]) / (X[3] - X[0]), 0.0), 1.0)]

    def bx(t):
        s = 1 - t
        return s**3 * X[0] + 3 * s * s * t * X[1] + 3 * s * t * t * X[2] + t**3 * X[3]

    t = min(cands, key=lambda t: abs(bx(t) - x))
    t = min(max(t, 0.0), 1.0)
    s = 1 - t
    return s**3 * v1[1] + 3 * s * s * t * v2[1] + 3 * s * t * t * v3[1] + t**3 * v4[1]


def eval_curve(k, ipo, extrap, x):
    n = len(k)
    if x <= k[0, 0]:
        if extrap == 0 or n == 1 or ipo[0] == 0:
            return k[0, 1]
        if ipo[0] == 2:
            dx = k[0, 0] - k[0, 2]
            sl = (k[0, 1] - k[0, 3]) / dx if dx != 0 else 0.0
        else:
            dx = k[1, 0] - k[0, 0]
            sl = (k[1, 1] - k[0, 1]) / dx if dx != 0 else 0.0
        return k[0, 1] + sl * (x - k[0, 0])
    if x >= k[-1, 0]:
        if extrap == 0 or n == 1 or ipo[-1] == 0:
            return k[-1, 1]
        if ipo[-1] == 2:
            dx = k[-1, 4] - k[-1, 0]
            sl = (k[-1, 5] - k[-1, 1]) / dx if dx != 0 else 0.0
        else:
            dx = k[-1, 0] - k[-2, 0]
            sl = (k[-1, 1] - k[-2, 1]) / dx if dx != 0 else 0.0
        return k[-1, 1] + sl * (x - k[-1, 0])
    i = int(np.searchsorted(k[:, 0], x, side="right")) - 1
    p, q = k[i], k[i + 1]
    if x == p[0]:
        return p[1]
    m = ipo[i]
    if m == 0:
        return p[1]
    if m == 2:
        return _bezier_y(p, q, x)
    return p[1] + (x - p[0]) / (q[0] - p[0]) * (q[1] - p[1])


def evaluate_pose(frames, parent, rest, rot_mode, defaults, curve_off, curve_n,
                  curve_extrap, keys, key_ipo, n_threads=0):
    frames = np.asarray(frames, dtype=float)
    B = len(parent)
    F = len(frames)
    rest = np.asarray(rest, dtype=float)
    relrest = np.empty_like(rest)
    for b in range(B):
        relrest[b] = rest[b] if parent[b] < 0 else np.linalg.inv(rest[parent[b]]) @ rest[b]
    out = np.empty((F, B, 4, 4))
    for f, x in enumerate(frames):
        for b in range(B):
            v = np.empty(NSLOT)
            for s in range(NSLOT):
                c = b * NSLOT + s
                if curve_n[c] > 0:
                    o, n = curve_off[c], curve_n[c]
                    v[s] = eval_curve(keys[o:o + n], key_ipo[o:o + n], curve_extrap[c], x)
                else:
                    v[s] = defaults[b, s]
            basis = np.eye(4)
            basis[:3, :3] = rotation_matrix(int(rot_mode[b]), v[3:7]) * v[7:10][None, :]
            basis[:3, 3] = v[0:3]
            local = relrest[b] @ basis
            out[f, b] = local if parent[b] < 0 else out[f, parent[b]] @ local
    return out
