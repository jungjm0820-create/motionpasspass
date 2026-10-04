// Motionator native core.
//
// Pure-array FK evaluation: F-curve (bezier) sampling -> local bone matrices ->
// armature-space pose matrices, multithreaded over frames, GIL released.
// Never touches bpy. All inputs are plain numpy arrays prepared on the Python side.
//
// Conventions (match mathutils):
//   * 4x4 matrices are row-major, column-vector convention (M @ v), translation in [i][3]
//   * basis = T * R * S
//   * pose[b] = pose[parent] * (inv(rest[parent]) * rest[b]) * basis[b]   (root: rest[b] * basis[b])
//
// Channel slots per bone (NSLOT = 10):
//   0..2 location | 3..6 rotation (quat wxyz | euler xyz | axis-angle angle,x,y,z) | 7..9 scale
// rot_mode: 0 quaternion, 1 XYZ, 2 XZY, 3 YXZ, 4 YZX, 5 ZXY, 6 ZYX, 7 axis-angle
// key layout [K,6]: co_x, co_y, hl_x, hl_y, hr_x, hr_y
// ipo: 0 constant, 1 linear, 2 bezier (anything else treated as linear)
// extrap: 0 constant, 1 linear

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

namespace nb = nanobind;

namespace {

constexpr int NSLOT = 10;

template <typename T, size_t N>
using In = nb::ndarray<const T, nb::ndim<N>, nb::c_contig, nb::device::cpu>;

// ---------------------------------------------------------------- F-curve

struct CurveRef {
    const double* k;     // n*6
    const int32_t* ipo;  // n
    int n;
    int extrap;
};

inline double slope_between(const double* a, const double* b) {
    double dx = b[0] - a[0];
    return dx != 0.0 ? (b[1] - a[1]) / dx : 0.0;
}

double bezier_segment(const double* p, const double* q, double x) {
    double v1x = p[0], v1y = p[1];
    double v2x = p[4], v2y = p[5];  // right handle of p
    double v3x = q[2], v3y = q[3];  // left handle of q
    double v4x = q[0], v4y = q[1];

    // Same handle correction as BKE_fcurve_correct_bezpart
    double h1x = v1x - v2x, h1y = v1y - v2y;
    double h2x = v4x - v3x, h2y = v4y - v3y;
    double len = v4x - v1x;
    if (len > 0.0) {
        double l1 = std::fabs(h1x), l2 = std::fabs(h2x);
        if (l1 + l2 > len) {
            double fac = len / (l1 + l2);
            v2x = v1x - fac * h1x;
            v2y = v1y - fac * h1y;
            v3x = v4x - fac * h2x;
            v3y = v4y - fac * h2y;
        }
    }

    auto bez = [](double t, double a, double b, double c, double d) {
        double s = 1.0 - t;
        return s * s * s * a + 3.0 * s * s * t * b + 3.0 * s * t * t * c + t * t * t * d;
    };

    double lo = 0.0, hi = 1.0;
    for (int i = 0; i < 48; ++i) {
        double mid = 0.5 * (lo + hi);
        if (bez(mid, v1x, v2x, v3x, v4x) < x) lo = mid; else hi = mid;
    }
    double t = 0.5 * (lo + hi);
    return bez(t, v1y, v2y, v3y, v4y);
}

double eval_curve(const CurveRef& c, double x) {
    const double* k = c.k;
    const int n = c.n;
    const double* first = k;
    const double* last = k + (n - 1) * 6;

    if (x <= first[0]) {
        if (c.extrap == 0 || n == 1 || c.ipo[0] == 0) return first[1];
        double slope;
        if (c.ipo[0] == 2) {
            double dx = first[0] - first[2];
            slope = dx != 0.0 ? (first[1] - first[3]) / dx : 0.0;
        } else {
            slope = slope_between(first, k + 6);
        }
        return first[1] + slope * (x - first[0]);
    }
    if (x >= last[0]) {
        if (c.extrap == 0 || n == 1 || c.ipo[n - 1] == 0) return last[1];
        double slope;
        if (c.ipo[n - 1] == 2) {
            double dx = last[4] - last[0];
            slope = dx != 0.0 ? (last[5] - last[1]) / dx : 0.0;
        } else {
            slope = slope_between(last - 6, last);
        }
        return last[1] + slope * (x - last[0]);
    }

    // binary search: largest i with k[i].x <= x
    int lo = 0, hi = n - 1;
    while (hi - lo > 1) {
        int mid = (lo + hi) >> 1;
        if (k[mid * 6] <= x) lo = mid; else hi = mid;
    }
    const double* p = k + lo * 6;
    const double* q = k + (lo + 1) * 6;
    if (x == p[0]) return p[1];

    switch (c.ipo[lo]) {
        case 0: return p[1];
        case 2: return bezier_segment(p, q, x);
        default: {
            double t = (x - p[0]) / (q[0] - p[0]);
            return p[1] + t * (q[1] - p[1]);
        }
    }
}

// ---------------------------------------------------------------- matrices

inline void mat4_mul(const double* a, const double* b, double* o) {
    double t[16];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            t[i * 4 + j] = a[i * 4] * b[j] + a[i * 4 + 1] * b[4 + j] +
                           a[i * 4 + 2] * b[8 + j] + a[i * 4 + 3] * b[12 + j];
    std::memcpy(o, t, sizeof(t));
}

void mat4_affine_inverse(const double* m, double* o) {
    double a = m[0], b = m[1], c = m[2];
    double d = m[4], e = m[5], f = m[6];
    double g = m[8], h = m[9], i = m[10];
    double det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    double id = std::fabs(det) > 1e-300 ? 1.0 / det : 0.0;
    double r[9] = {(e * i - f * h) * id, (c * h - b * i) * id, (b * f - c * e) * id,
                   (f * g - d * i) * id, (a * i - c * g) * id, (c * d - a * f) * id,
                   (d * h - e * g) * id, (b * g - a * h) * id, (a * e - b * d) * id};
    double tx = m[3], ty = m[7], tz = m[11];
    o[0] = r[0]; o[1] = r[1]; o[2] = r[2];  o[3] = -(r[0] * tx + r[1] * ty + r[2] * tz);
    o[4] = r[3]; o[5] = r[4]; o[6] = r[5];  o[7] = -(r[3] * tx + r[4] * ty + r[5] * tz);
    o[8] = r[6]; o[9] = r[7]; o[10] = r[8]; o[11] = -(r[6] * tx + r[7] * ty + r[8] * tz);
    o[12] = 0; o[13] = 0; o[14] = 0; o[15] = 1;
}

inline void mat3_mul(const double* a, const double* b, double* o) {
    double t[9];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            t[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
    std::memcpy(o, t, sizeof(t));
}

void rot_axis(int axis, double ang, double* m) {
    double c = std::cos(ang), s = std::sin(ang);
    std::memset(m, 0, 9 * sizeof(double));
    if (axis == 0) { m[0] = 1; m[4] = c; m[5] = -s; m[7] = s; m[8] = c; }
    else if (axis == 1) { m[0] = c; m[2] = s; m[4] = 1; m[6] = -s; m[8] = c; }
    else { m[0] = c; m[1] = -s; m[3] = s; m[4] = c; m[8] = 1; }
}

const int EULER_ORDER[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};

void rotation_matrix(int mode, const double* r, double* m) {
    if (mode == 0) {
        double w = r[0], x = r[1], y = r[2], z = r[3];
        double n = std::sqrt(w * w + x * x + y * y + z * z);
        if (n < 1e-30) { w = 1; x = y = z = 0; } else { w /= n; x /= n; y /= n; z /= n; }
        m[0] = 1 - 2 * (y * y + z * z); m[1] = 2 * (x * y - w * z);     m[2] = 2 * (x * z + w * y);
        m[3] = 2 * (x * y + w * z);     m[4] = 1 - 2 * (x * x + z * z); m[5] = 2 * (y * z - w * x);
        m[6] = 2 * (x * z - w * y);     m[7] = 2 * (y * z + w * x);     m[8] = 1 - 2 * (x * x + y * y);
    } else if (mode >= 1 && mode <= 6) {
        const int* o = EULER_ORDER[mode - 1];
        double ma[9], mb[9], mc[9], t[9];
        rot_axis(o[0], r[o[0]], ma);
        rot_axis(o[1], r[o[1]], mb);
        rot_axis(o[2], r[o[2]], mc);
        mat3_mul(mc, mb, t);   // R = Rc * Rb * Ra
        mat3_mul(t, ma, m);
    } else {
        double ang = r[0], x = r[1], y = r[2], z = r[3];
        double n = std::sqrt(x * x + y * y + z * z);
        if (n < 1e-30) { m[0] = m[4] = m[8] = 1; m[1] = m[2] = m[3] = m[5] = m[6] = m[7] = 0; return; }
        x /= n; y /= n; z /= n;
        double c = std::cos(ang), s = std::sin(ang), v = 1 - c;
        m[0] = c + x * x * v;     m[1] = x * y * v - z * s; m[2] = x * z * v + y * s;
        m[3] = y * x * v + z * s; m[4] = c + y * y * v;     m[5] = y * z * v - x * s;
        m[6] = z * x * v - y * s; m[7] = z * y * v + x * s; m[8] = c + z * z * v;
    }
}

}  // namespace

// ---------------------------------------------------------------- entry point

using OutArr = nb::ndarray<nb::numpy, double, nb::ndim<4>>;

OutArr evaluate_pose(In<double, 1> frames,
                     In<int32_t, 1> parent,
                     In<double, 3> rest,
                     In<int32_t, 1> rot_mode,
                     In<double, 2> defaults,
                     In<int32_t, 1> curve_off,
                     In<int32_t, 1> curve_n,
                     In<int32_t, 1> curve_extrap,
                     In<double, 2> keys,
                     In<int32_t, 1> key_ipo,
                     int n_threads) {
    const size_t F = frames.shape(0);
    const size_t B = parent.shape(0);
    if (rest.shape(0) != B || rest.shape(1) != 4 || rest.shape(2) != 4) throw nb::value_error("rest must be [B,4,4]");
    if (rot_mode.shape(0) != B) throw nb::value_error("rot_mode must be [B]");
    if (defaults.shape(0) != B || defaults.shape(1) != NSLOT) throw nb::value_error("defaults must be [B,10]");
    const size_t C = B * NSLOT;
    if (curve_off.shape(0) != C || curve_n.shape(0) != C || curve_extrap.shape(0) != C)
        throw nb::value_error("curve tables must be [B*10]");
    if (keys.shape(1) != 6 || key_ipo.shape(0) != keys.shape(0)) throw nb::value_error("keys must be [K,6] and key_ipo [K]");

    const int32_t* par = parent.data();
    for (size_t b = 0; b < B; ++b)
        if (par[b] >= (int32_t)b) throw nb::value_error("bones must be ordered parents-first");
    const size_t K = keys.shape(0);
    for (size_t c = 0; c < C; ++c)
        if (curve_n.data()[c] > 0 && (curve_off.data()[c] < 0 || (size_t)(curve_off.data()[c] + curve_n.data()[c]) > K))
            throw nb::value_error("curve range out of bounds");

    // Relative rest matrices (precomputed once)
    std::vector<double> relrest(B * 16);
    for (size_t b = 0; b < B; ++b) {
        const double* rb = rest.data() + b * 16;
        if (par[b] < 0) {
            std::memcpy(&relrest[b * 16], rb, 16 * sizeof(double));
        } else {
            double inv[16];
            mat4_affine_inverse(rest.data() + (size_t)par[b] * 16, inv);
            mat4_mul(inv, rb, &relrest[b * 16]);
        }
    }

    double* out = new double[F * B * 16];
    nb::capsule owner(out, [](void* p) noexcept { delete[] static_cast<double*>(p); });

    const double* fr = frames.data();
    const double* dflt = defaults.data();
    const int32_t* coff = curve_off.data();
    const int32_t* cn = curve_n.data();
    const int32_t* cex = curve_extrap.data();
    const double* kd = keys.data();
    const int32_t* ki = key_ipo.data();
    const int32_t* rm = rot_mode.data();

    {
        nb::gil_scoped_release release;

        auto work_frame = [&](size_t f) {
            double* pose = out + f * B * 16;
            const double x = fr[f];
            for (size_t b = 0; b < B; ++b) {
                double v[NSLOT];
                for (int s = 0; s < NSLOT; ++s) {
                    size_t c = b * NSLOT + s;
                    if (cn[c] > 0) {
                        CurveRef cr{kd + (size_t)coff[c] * 6, ki + coff[c], cn[c], cex[c]};
                        v[s] = eval_curve(cr, x);
                    } else {
                        v[s] = dflt[c];
                    }
                }
                double R[9];
                rotation_matrix(rm[b], v + 3, R);
                double basis[16] = {R[0] * v[7], R[1] * v[8], R[2] * v[9], v[0],
                                    R[3] * v[7], R[4] * v[8], R[5] * v[9], v[1],
                                    R[6] * v[7], R[7] * v[8], R[8] * v[9], v[2],
                                    0, 0, 0, 1};
                double local[16];
                mat4_mul(&relrest[b * 16], basis, local);
                if (par[b] < 0) std::memcpy(pose + b * 16, local, sizeof(local));
                else mat4_mul(pose + (size_t)par[b] * 16, local, pose + b * 16);
            }
        };

        size_t nt = n_threads > 0 ? (size_t)n_threads : std::max(1u, std::thread::hardware_concurrency());
        nt = std::min(nt, F);
        if (nt <= 1) {
            for (size_t f = 0; f < F; ++f) work_frame(f);
        } else {
            std::atomic<size_t> next{0};
            std::vector<std::thread> pool;
            pool.reserve(nt);
            for (size_t t = 0; t < nt; ++t)
                pool.emplace_back([&]() {
                    for (;;) {
                        size_t f = next.fetch_add(1);
                        if (f >= F) break;
                        work_frame(f);
                    }
                });
            for (auto& th : pool) th.join();
        }
    }

    return OutArr(out, {F, B, 4, 4}, owner);
}

NB_MODULE(motionator_core, m) {
    m.doc() = "Motionator native FK / F-curve evaluation core";
    m.def("evaluate_pose", &evaluate_pose,
          nb::arg("frames"), nb::arg("parent"), nb::arg("rest"), nb::arg("rot_mode"),
          nb::arg("defaults"), nb::arg("curve_off"), nb::arg("curve_n"), nb::arg("curve_extrap"),
          nb::arg("keys"), nb::arg("key_ipo"), nb::arg("n_threads") = 0,
          "Return armature-space pose matrices, shape [frames, bones, 4, 4] (float64).");
    m.def("hardware_threads", []() { return (int)std::thread::hardware_concurrency(); });
    m.attr("NSLOT") = NSLOT;
    m.attr("__version__") = "0.1.0";
}
