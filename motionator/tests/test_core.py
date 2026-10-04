"""Native core vs numpy reference. Run:  python tests/test_core.py"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "addon"))
import fk_numpy as ref

try:
    import motionator_core as native
except ImportError:
    print("motionator_core not importable; build/install it first")
    raise

NS = 10


def make_rig(rng, B, nkeys, frame_span=60.0):
    parent = np.full(B, -1, np.int32)
    for b in range(1, B):
        parent[b] = rng.integers(0, b) if rng.random() < 0.9 else -1
    rest = np.tile(np.eye(4), (B, 1, 1))
    for b in range(B):
        a = rng.normal(size=3)
        q = np.concatenate([[1.0], rng.normal(size=3) * 0.3])
        rot = ref.rotation_matrix(0, q)
        rest[b, :3, :3] = rot
        rest[b, :3, 3] = rng.normal(size=3)
    rot_mode = rng.integers(0, 8, size=B).astype(np.int32)
    defaults = np.zeros((B, NS))
    defaults[:, 0:3] = rng.normal(size=(B, 3)) * 0.1
    defaults[:, 3:7] = rng.normal(size=(B, 4)) * 0.2
    for b in range(B):
        if rot_mode[b] == 0:
            defaults[b, 3:7] = [1, 0, 0, 0]
        elif rot_mode[b] == 7:
            defaults[b, 3:7] = [0.7, 0.3, 0.5, 0.8]
    defaults[:, 7:10] = 1.0 + rng.normal(size=(B, 3)) * 0.05

    C = B * NS
    curve_off = np.full(C, -1, np.int32)
    curve_n = np.zeros(C, np.int32)
    curve_extrap = rng.integers(0, 2, size=C).astype(np.int32)
    keys, ipos = [], []
    for c in range(C):
        if rng.random() < 0.55:
            continue
        n = int(rng.integers(1, nkeys + 1))
        xs = np.sort(rng.uniform(0, frame_span, size=n))
        xs = xs + np.arange(n) * 0.5          # strictly increasing
        ys = rng.normal(size=n)
        ipo = rng.integers(0, 3, size=n)
        rows = []
        for i in range(n):
            gl = (xs[i] - xs[i - 1]) if i > 0 else 5.0
            gr = (xs[i + 1] - xs[i]) if i < n - 1 else 5.0
            sc = 1.5 if rng.random() < 0.2 else 0.4    # sometimes overlapping handles
            hlx, hrx = xs[i] - rng.random() * gl * sc, xs[i] + rng.random() * gr * sc
            rows.append([xs[i], ys[i], hlx, ys[i] + rng.normal(), hrx, ys[i] + rng.normal()])
        curve_off[c] = len(keys)
        curve_n[c] = n
        keys.extend(rows)
        ipos.extend(ipo.tolist())
    keys = np.array(keys if keys else np.zeros((1, 6)), dtype=np.float64).reshape(-1, 6)
    key_ipo = np.array(ipos if ipos else [0], dtype=np.int32)
    return dict(parent=parent, rest=rest, rot_mode=rot_mode, defaults=defaults,
                curve_off=curve_off, curve_n=curve_n, curve_extrap=curve_extrap,
                keys=keys, key_ipo=key_ipo)


def test_unit():
    # 90 deg about Z maps +X to +Y (quaternion and euler agree)
    q = ref.rotation_matrix(0, [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
    e = ref.rotation_matrix(1, [0, 0, np.pi / 2])
    assert np.allclose(q @ [1, 0, 0], [0, 1, 0]) and np.allclose(e, q)
    # Euler XYZ applies X first, then Y, then Z: R = Rz Ry Rx
    r = ref.rotation_matrix(1, [0.3, 0.4, 0.5])
    assert np.allclose(r, ref._rot("Z", 0.5) @ ref._rot("Y", 0.4) @ ref._rot("X", 0.3))
    # bezier with 1/3-spaced flat handles = smoothstep-like, midpoint exactly in the middle
    k = np.array([[0, 0, -1, 0, 1 / 3, 0], [3, 3, 8 / 3, 3, 4, 3.0]])
    assert abs(ref.eval_curve(k, np.array([2, 2]), 0, 1.5) - 1.5) < 1e-9
    # linear extrapolation / constant extrapolation
    kl = np.array([[0, 0, 0, 0, 0, 0], [1, 2, 1, 2, 1, 2.0]])
    assert abs(ref.eval_curve(kl, np.array([1, 1]), 1, 3.0) - 6.0) < 1e-9
    assert abs(ref.eval_curve(kl, np.array([1, 1]), 0, 3.0) - 2.0) < 1e-9
    print("unit ok")


def test_native_vs_ref():
    rng = np.random.default_rng(1234)
    worst = 0.0
    for trial in range(25):
        B = int(rng.integers(1, 25))
        d = make_rig(rng, B, 6)
        frames = np.concatenate([np.arange(-8, 75, dtype=float), rng.uniform(-10, 80, 12)])
        a = native.evaluate_pose(frames, **d, n_threads=0)
        b = ref.evaluate_pose(frames, **d)
        err = float(np.abs(a - b).max())
        scale = max(1.0, float(np.abs(b).max()))
        worst = max(worst, err / scale)
        assert err / scale < 1e-8, (trial, err, scale)
    print(f"native == numpy reference on 25 random rigs (worst rel err {worst:.2e})")


def test_threads_deterministic():
    rng = np.random.default_rng(7)
    d = make_rig(rng, 20, 8)
    fr = np.arange(0, 200, dtype=float)
    a = native.evaluate_pose(fr, **d, n_threads=1)
    b = native.evaluate_pose(fr, **d, n_threads=8)
    assert np.array_equal(a, b)
    print("thread count does not change results")


def test_errors():
    rng = np.random.default_rng(3)
    d = make_rig(rng, 4, 3)
    d["parent"][1] = 3                    # parent after child -> must be rejected
    try:
        native.evaluate_pose(np.arange(3.0), **d)
    except ValueError:
        print("bad hierarchy rejected")
    else:
        raise AssertionError("expected ValueError")


def bench():
    rng = np.random.default_rng(9)
    d = make_rig(rng, 100, 20)
    fr = np.arange(-30, 31, dtype=float)
    for name, fn in (("native", native.evaluate_pose), ("numpy ", ref.evaluate_pose)):
        n = 20 if name == "native" else 1
        t = time.perf_counter()
        for _ in range(n):
            fn(fr, **d)
        print(f"{name}: 100 bones x 61 frames = {(time.perf_counter() - t) / n * 1000:8.2f} ms")


if __name__ == "__main__":
    test_unit()
    test_native_vs_ref()
    test_threads_deterministic()
    test_errors()
    bench()
    print("ALL OK  (threads:", native.hardware_threads(), ")")
