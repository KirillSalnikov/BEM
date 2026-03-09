"""
OpenCL-accelerated BEM assembly and GMRES matvec (float32 GPU kernels).

GPU kernels use float32 for ~32x speedup on consumer GPUs (vs fp64).
Assembly results are accumulated into float64 on host for compatibility.
Float32 gives ~7 significant digits — sufficient for BEM accuracy of 1-3%.

Usage:
    from bem_opencl import assemble_pmchwt_ocl, solve_gmres_ocl, ocl_available
"""

import numpy as np
import time as _time

# ============================================================
# OpenCL kernel source — float32
# ============================================================

_KERNEL_SRC = r"""
// Assembly kernel: one work-item per (b, n) output element.
// Loops over Nq×Nq quadrature pairs internally.
__kernel void assemble_LK_block(
    __global const float *tp,       // (B, Nq, 3) test quad points
    __global const float *sq,       // (N, Nq, 3) source quad points
    __global const float *tf,       // (B, Nq, 3) test basis fn values
    __global const float *sf,       // (N, Nq, 3) source basis fn values
    __global const float *sq_x_sf,  // (N, Nq, 3) cross(sq, sf) precomputed
    __global const float *jw_t,     // (B, Nq) test Jacobian*weights
    __global const float *jw_s,     // (N, Nq) source Jacobian*weights
    __global const int *t_tri,      // (B,) test triangle indices
    __global const int *s_tri,      // (N,) source triangle indices
    __global const float *t_div,    // (B,) test divergence
    __global const float *s_div,    // (N,) source divergence
    const float k_re,
    const float k_im,
    const float inv4pi,
    const int Nq, const int N_src,
    __global float *L_re, __global float *L_im,  // (B, N) output
    __global float *K_re, __global float *K_im)   // (B, N) output
{
    int b = get_global_id(0);
    int n = get_global_id(1);
    int B = get_global_size(0);

    if (b >= B || n >= N_src) return;

    int is_sing = (t_tri[b] == s_tri[n]);

    float Lvec_re = 0.0f, Lvec_im = 0.0f;
    float Lscl_re = 0.0f, Lscl_im = 0.0f;
    float Kacc_re = 0.0f, Kacc_im = 0.0f;

    for (int iq = 0; iq < Nq; iq++) {
        int ti = b * Nq + iq;
        float tpx = tp[ti * 3 + 0];
        float tpy = tp[ti * 3 + 1];
        float tpz = tp[ti * 3 + 2];
        float tfx = tf[ti * 3 + 0];
        float tfy = tf[ti * 3 + 1];
        float tfz = tf[ti * 3 + 2];
        float jwt = jw_t[ti];

        for (int jq = 0; jq < Nq; jq++) {
            int sj = n * Nq + jq;

            float sqx = sq[sj * 3 + 0];
            float sqy = sq[sj * 3 + 1];
            float sqz = sq[sj * 3 + 2];

            float dx = tpx - sqx;
            float dy = tpy - sqy;
            float dz = tpz - sqz;
            float R = sqrt(dx*dx + dy*dy + dz*dz);
            float R_safe = fmax(R, 1e-7f);

            // Green's function: G = exp(ikR) / (4πR)
            float eR = exp(-k_im * R_safe);
            float cosR = cos(k_re * R_safe);
            float sinR = sin(k_re * R_safe);
            float G_re = eR * cosR * inv4pi / R_safe;
            float G_im = eR * sinR * inv4pi / R_safe;
            float inv4piR = inv4pi / R_safe;

            float Gu_re = G_re;
            float Gu_im = G_im;
            if (is_sing) {
                Gu_re -= inv4piR;
            }

            float jw = jwt * jw_s[sj];

            float sfx = sf[sj * 3 + 0];
            float sfy = sf[sj * 3 + 1];
            float sfz = sf[sj * 3 + 2];
            float f_dot = tfx * sfx + tfy * sfy + tfz * sfz;

            float Gjw_re = Gu_re * jw;
            float Gjw_im = Gu_im * jw;
            Lvec_re += f_dot * Gjw_re;
            Lvec_im += f_dot * Gjw_im;
            Lscl_re += Gjw_re;
            Lscl_im += Gjw_im;

            // K operator
            if (!is_sing && R > 1e-6f) {
                float a_re = (-k_im - 1.0f / R_safe) / R_safe;
                float a_im = k_re / R_safe;
                float gc_re = G_re * a_re - G_im * a_im;
                float gc_im = G_re * a_im + G_im * a_re;
                float gGjw_re = gc_re * jw;
                float gGjw_im = gc_im * jw;

                float cx = tfy * tpz - tfz * tpy;
                float cy = tfz * tpx - tfx * tpz;
                float cz = tfx * tpy - tfy * tpx;
                float triple = cx * sfx + cy * sfy + cz * sfz
                    - tfx * sq_x_sf[sj * 3 + 0]
                    - tfy * sq_x_sf[sj * 3 + 1]
                    - tfz * sq_x_sf[sj * 3 + 2];

                Kacc_re += gGjw_re * triple;
                Kacc_im += gGjw_im * triple;
            }
        }
    }

    // Combine: L[b,n] = ik * Lvec - (i/k) * div_prod * Lscl
    float ik_re = -k_im;
    float ik_im = k_re;
    float k_sq = k_re * k_re + k_im * k_im;
    float iok_re = k_im / k_sq;
    float iok_im = k_re / k_sq;
    float div_prod = t_div[b] * s_div[n];

    float term1_re = ik_re * Lvec_re - ik_im * Lvec_im;
    float term1_im = ik_re * Lvec_im + ik_im * Lvec_re;
    float term2_re = iok_re * div_prod * Lscl_re - iok_im * div_prod * Lscl_im;
    float term2_im = iok_re * div_prod * Lscl_im + iok_im * div_prod * Lscl_re;

    int idx = b * N_src + n;
    L_re[idx] = term1_re - term2_re;
    L_im[idx] = term1_im - term2_im;
    K_re[idx] = Kacc_re;
    K_im[idx] = Kacc_im;
}


// Complex matrix-vector product y = Z * x  (float32)
__kernel void complex_matvec(
    __global const float *Z_re,
    __global const float *Z_im,
    __global const float *x_re,
    __global const float *x_im,
    __global float *y_re,
    __global float *y_im,
    const int M)
{
    int i = get_global_id(0);
    if (i >= M) return;

    float acc_re = 0.0f;
    float acc_im = 0.0f;

    for (int j = 0; j < M; j++) {
        float zr = Z_re[i * M + j];
        float zi = Z_im[i * M + j];
        float xr = x_re[j];
        float xi = x_im[j];
        acc_re += zr * xr - zi * xi;
        acc_im += zr * xi + zi * xr;
    }

    y_re[i] = acc_re;
    y_im[i] = acc_im;
}
"""


# ============================================================
# Context management
# ============================================================

_ctx = None
_queue = None
_prg = None
_asm_kernel = None
_mv_kernel = None


def ocl_available():
    """Check if OpenCL is available."""
    try:
        import pyopencl as cl
        platforms = cl.get_platforms()
        return len(platforms) > 0 and any(p.get_devices() for p in platforms)
    except Exception:
        return False


def _get_ocl():
    """Get or create OpenCL context, queue, compiled program, and cached kernels."""
    global _ctx, _queue, _prg, _asm_kernel, _mv_kernel
    if _ctx is not None:
        return _ctx, _queue, _prg, _asm_kernel, _mv_kernel

    import pyopencl as cl
    # Prefer GPU, fall back to CPU
    gpu_dev = None
    cpu_dev = None
    for platform in cl.get_platforms():
        for device in platform.get_devices():
            if device.type & cl.device_type.GPU:
                gpu_dev = device
            elif device.type & cl.device_type.CPU:
                cpu_dev = device

    device = gpu_dev or cpu_dev
    if device is None:
        raise RuntimeError("No OpenCL device found")

    _ctx = cl.Context([device])
    _queue = cl.CommandQueue(_ctx)
    _prg = cl.Program(_ctx, _KERNEL_SRC).build()
    _asm_kernel = cl.Kernel(_prg, 'assemble_LK_block')
    _mv_kernel = cl.Kernel(_prg, 'complex_matvec')

    dev_type = "GPU" if (device.type & cl.device_type.GPU) else "CPU"
    fp64 = "fp64" if device.double_fp_config else "fp32-only"
    print(f"  [OpenCL] {dev_type}: {device.name} ({device.max_compute_units} CU, "
          f"{device.global_mem_size // 1024**2} MB, {fp64})")
    return _ctx, _queue, _prg, _asm_kernel, _mv_kernel


def _to_f32(arr):
    """Convert to contiguous float32 array."""
    return np.ascontiguousarray(arr, dtype=np.float32)


# ============================================================
# 1. Assembly on OpenCL (float32 kernels → float64 result)
# ============================================================

def assemble_L_K_ocl(rwg, verts, tris, k, quad_order=7):
    """
    OpenCL-accelerated L and K operator assembly (float32 GPU kernels).
    Returns float64 complex matrices for compatibility with CPU code.
    """
    import pyopencl as cl

    ctx, queue, prg, asm_kernel, _ = _get_ocl()
    mf = cl.mem_flags

    from bem_core import tri_quadrature, potential_integral_triangle, \
        vector_potential_integral_triangle

    N = rwg['N']
    quad_pts, quad_wts = tri_quadrature(quad_order)
    Nq = len(quad_wts)
    lam0 = 1 - quad_pts[:, 0] - quad_pts[:, 1]

    def get_qpts_batch(tri_indices):
        t = tris[tri_indices]
        v0 = verts[t[:, 0]]; v1 = verts[t[:, 1]]; v2 = verts[t[:, 2]]
        return np.einsum('q,ni->nqi', lam0, v0) + \
               np.einsum('q,ni->nqi', quad_pts[:, 0], v1) + \
               np.einsum('q,ni->nqi', quad_pts[:, 1], v2)

    qpts_p = get_qpts_batch(rwg['tri_p'])
    qpts_m = get_qpts_batch(rwg['tri_m'])

    f_p = (rwg['length'][:, None, None] / (2 * rwg['area_p'][:, None, None])) * \
          (qpts_p - rwg['free_p'][:, None, :])
    f_m = -(rwg['length'][:, None, None] / (2 * rwg['area_m'][:, None, None])) * \
          (qpts_m - rwg['free_m'][:, None, :])

    div_p = rwg['length'] / rwg['area_p']
    div_m = -rwg['length'] / rwg['area_m']

    jw_p = rwg['area_p'][:, None] * quad_wts[None, :]
    jw_m = rwg['area_m'][:, None] * quad_wts[None, :]

    L = np.zeros((N, N), dtype=complex)
    K = np.zeros((N, N), dtype=complex)
    L_cross = np.zeros((N, N), dtype=complex)
    K_cross = np.zeros((N, N), dtype=complex)

    inv4pi = np.float32(1.0 / (4 * np.pi))
    k_re = np.float32(np.real(k))
    k_im = np.float32(np.imag(k))

    halves = [
        (qpts_p, f_p, div_p, jw_p, rwg['tri_p']),
        (qpts_m, f_m, div_m, jw_m, rwg['tri_m']),
    ]

    t0 = _time.time()
    print(f"    [OCL/f32] Assembly: {N} RWG, {Nq} quad pts, k={k:.4f}...")

    # Preallocate output host arrays
    out_re = np.empty((N, N), dtype=np.float32)
    out_im = np.empty((N, N), dtype=np.float32)

    # Preallocate output device buffers (reused across passes)
    out_nbytes = N * N * 4  # float32
    L_re_buf = cl.Buffer(ctx, mf.WRITE_ONLY, out_nbytes)
    L_im_buf = cl.Buffer(ctx, mf.WRITE_ONLY, out_nbytes)
    K_re_buf = cl.Buffer(ctx, mf.WRITE_ONLY, out_nbytes)
    K_im_buf = cl.Buffer(ctx, mf.WRITE_ONLY, out_nbytes)

    for th in range(2):
        for sh in range(2):
            if th > sh:
                continue

            is_cross = (th != sh)
            tgt_L = L_cross if is_cross else L
            tgt_K = K_cross if is_cross else K

            t_qpts, t_f, t_div, t_jw, t_tri = halves[th]
            s_qpts, s_f, s_div, s_jw, s_tri = halves[sh]

            # Upload float32 arrays to GPU
            tp_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                               hostbuf=_to_f32(t_qpts.reshape(N * Nq, 3)))
            sq_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                               hostbuf=_to_f32(s_qpts.reshape(N * Nq, 3)))
            tf_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                               hostbuf=_to_f32(t_f.reshape(N * Nq, 3)))
            sf_flat = _to_f32(s_f.reshape(N * Nq, 3))
            sf_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=sf_flat)
            sq_flat = _to_f32(s_qpts.reshape(N * Nq, 3))
            sxsf = _to_f32(np.cross(sq_flat.astype(np.float64),
                                     sf_flat.astype(np.float64)))
            sxsf_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=sxsf)
            jwt_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                hostbuf=_to_f32(t_jw.ravel()))
            jws_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                hostbuf=_to_f32(s_jw.ravel()))
            ttri_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                 hostbuf=np.ascontiguousarray(t_tri.astype(np.int32)))
            stri_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                 hostbuf=np.ascontiguousarray(s_tri.astype(np.int32)))
            tdiv_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                 hostbuf=_to_f32(t_div))
            sdiv_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                 hostbuf=_to_f32(s_div))

            # Launch kernel
            asm_kernel.set_args(
                tp_buf, sq_buf, tf_buf, sf_buf, sxsf_buf,
                jwt_buf, jws_buf, ttri_buf, stri_buf,
                tdiv_buf, sdiv_buf,
                k_re, k_im, inv4pi,
                np.int32(Nq), np.int32(N),
                L_re_buf, L_im_buf, K_re_buf, K_im_buf)
            cl.enqueue_nd_range_kernel(queue, asm_kernel, (N, N), None)

            # Read back float32 → accumulate into float64
            cl.enqueue_copy(queue, out_re, L_re_buf)
            cl.enqueue_copy(queue, out_im, L_im_buf)
            queue.finish()
            tgt_L += out_re.astype(np.float64) + 1j * out_im.astype(np.float64)

            cl.enqueue_copy(queue, out_re, K_re_buf)
            cl.enqueue_copy(queue, out_im, K_im_buf)
            queue.finish()
            tgt_K += out_re.astype(np.float64) + 1j * out_im.astype(np.float64)

    # Cross-term symmetry
    L += L_cross + L_cross.T
    K += K_cross + K_cross.T

    elapsed_main = _time.time() - t0
    print(f"    [OCL/f32] Main loop: {elapsed_main:.1f}s")

    # --- Singular corrections (CPU, float64 — O(N), fast) ---
    t1 = _time.time()
    ik = 1j * k
    ik_inv = 1j / k

    tri_verts_cache = {}
    for ti in range(len(tris)):
        tri_verts_cache[ti] = (verts[tris[ti, 0]].copy(),
                                verts[tris[ti, 1]].copy(),
                                verts[tris[ti, 2]].copy())

    tri_to_rwg_p = {}
    tri_to_rwg_m = {}
    for n in range(N):
        tp_idx = rwg['tri_p'][n]
        if tp_idx not in tri_to_rwg_p:
            tri_to_rwg_p[tp_idx] = []
        tri_to_rwg_p[tp_idx].append((n, div_p[n],
                                      rwg['length'][n] / (2 * rwg['area_p'][n]),
                                      rwg['free_p'][n], +1))
        tm_idx = rwg['tri_m'][n]
        if tm_idx not in tri_to_rwg_m:
            tri_to_rwg_m[tm_idx] = []
        tri_to_rwg_m[tm_idx].append((n, div_m[n],
                                      rwg['length'][n] / (2 * rwg['area_m'][n]),
                                      rwg['free_m'][n], -1))

    all_sing_tris = set(tri_to_rwg_p.keys()) | set(tri_to_rwg_m.keys())
    tri_PV_cache = {}
    for tri_idx in all_sing_tris:
        tv = tri_verts_cache[tri_idx]
        t = tris[tri_idx]
        v0, v1, v2 = verts[t[0]], verts[t[1]], verts[t[2]]
        tri_qpts = lam0[:, None] * v0 + quad_pts[:, 0:1] * v1 + quad_pts[:, 1:2] * v2
        P = np.zeros(Nq)
        V = np.zeros((Nq, 3))
        for iq in range(Nq):
            P[iq] = potential_integral_triangle(tri_qpts[iq], *tv)
            V[iq] = vector_potential_integral_triangle(tri_qpts[iq], *tv)
        tri_PV_cache[tri_idx] = (P, V)

    for m in range(N):
        for t_f_h, t_div_h, t_jw_h, t_tri_idx in [
            (f_p[m], div_p[m], jw_p[m], rwg['tri_p'][m]),
            (f_m[m], div_m[m], jw_m[m], rwg['tri_m'][m]),
        ]:
            P_vals, V_vals = tri_PV_cache[t_tri_idx]
            scalar_base = np.dot(P_vals, t_jw_h) * (1.0 / (4 * np.pi))
            for src_dict in [tri_to_rwg_p, tri_to_rwg_m]:
                if t_tri_idx not in src_dict:
                    continue
                for n, src_div_val, src_coeff, src_free, src_sign in src_dict[t_tri_idx]:
                    L_sing_scalar = -ik_inv * t_div_h * src_div_val * scalar_base
                    fn_over_R = src_sign * src_coeff * (V_vals - src_free[None, :] * P_vals[:, None])
                    vec_integral = np.sum(np.sum(t_f_h * fn_over_R, axis=1) * t_jw_h) * (1.0 / (4 * np.pi))
                    L[m, n] += L_sing_scalar + ik * vec_integral

    print(f"    [OCL/f32] Singular corrections (CPU/f64): {_time.time() - t1:.1f}s")

    L = (L + L.T) / 2
    K = (K + K.T) / 2
    print(f"    [OCL/f32] Total assembly: {_time.time() - t0:.1f}s")
    return L, K


# ============================================================
# 2. GMRES with OpenCL matvec (float32)
# ============================================================

class OCLMatvec:
    """GPU-accelerated complex matrix-vector product (float32)."""

    def __init__(self, Z):
        import pyopencl as cl
        self.ctx, self.queue, self.prg, _, self.kernel = _get_ocl()
        mf = cl.mem_flags

        self.M = Z.shape[0]
        Z_re = _to_f32(Z.real)
        Z_im = _to_f32(Z.imag)

        self.Z_re_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Z_re)
        self.Z_im_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Z_im)
        self.x_re_buf = cl.Buffer(self.ctx, mf.READ_WRITE, self.M * 4)
        self.x_im_buf = cl.Buffer(self.ctx, mf.READ_WRITE, self.M * 4)
        self.y_re_buf = cl.Buffer(self.ctx, mf.READ_WRITE, self.M * 4)
        self.y_im_buf = cl.Buffer(self.ctx, mf.READ_WRITE, self.M * 4)

        print(f"  [OCL/f32] Z ({self.M}x{self.M}) → GPU: "
              f"{2 * Z_re.nbytes / 1024**2:.0f} MB")

    def matvec(self, x):
        import pyopencl as cl
        x_re = _to_f32(x.real)
        x_im = _to_f32(x.imag)

        cl.enqueue_copy(self.queue, self.x_re_buf, x_re)
        cl.enqueue_copy(self.queue, self.x_im_buf, x_im)

        self.kernel.set_args(
            self.Z_re_buf, self.Z_im_buf,
            self.x_re_buf, self.x_im_buf,
            self.y_re_buf, self.y_im_buf,
            np.int32(self.M))
        cl.enqueue_nd_range_kernel(self.queue, self.kernel, (self.M,), None)

        y_re = np.empty(self.M, dtype=np.float32)
        y_im = np.empty(self.M, dtype=np.float32)
        cl.enqueue_copy(self.queue, y_re, self.y_re_buf)
        cl.enqueue_copy(self.queue, y_im, self.y_im_buf)
        self.queue.finish()

        return y_re.astype(np.float64) + 1j * y_im.astype(np.float64)


def solve_gmres_ocl(Z, b, tol=1e-4, maxiter=500, precond='block_diag'):
    """GMRES with GPU-accelerated matvec (float32). Drop-in for solve_gmres."""
    from scipy.sparse.linalg import gmres, LinearOperator
    from scipy import linalg

    M = Z.shape[0]
    N = M // 2

    gpu_mv = OCLMatvec(Z)
    A_op = LinearOperator((M, M), matvec=gpu_mv.matvec, dtype=complex)

    if precond == 'block_diag':
        lu11 = linalg.lu_factor(Z[:N, :N])
        lu22 = linalg.lu_factor(Z[N:, N:])
        def apply_precond(x):
            y = np.empty(M, dtype=complex)
            y[:N] = linalg.lu_solve(lu11, x[:N])
            y[N:] = linalg.lu_solve(lu22, x[N:])
            return y
        M_op = LinearOperator((M, M), matvec=apply_precond, dtype=complex)
    else:
        M_op = None

    iter_count = [0]
    def callback(rk):
        iter_count[0] += 1

    t0 = _time.time()
    x, info = gmres(A_op, b, M=M_op, rtol=tol, maxiter=maxiter,
                    callback=callback, callback_type='pr_norm')
    elapsed = _time.time() - t0

    if info == 0:
        print(f"  [OCL/f32] GMRES converged in {iter_count[0]} iterations, {elapsed:.1f}s")
    else:
        print(f"  [OCL/f32] GMRES failed (info={info}), {iter_count[0]} iters, {elapsed:.1f}s")

    return x


# ============================================================
# 3. High-level wrappers
# ============================================================

def assemble_pmchwt_ocl(rwg, verts, tris, k_ext, k_int, eta_ext, eta_int, quad_order=7):
    """OpenCL-accelerated PMCHWT assembly (float32 kernels)."""
    N = rwg['N']
    print(f"  [OCL/f32] Assembling {2*N}x{2*N} PMCHWT ({N} RWG)...")

    t0 = _time.time()
    print(f"  [OCL/f32] Exterior (k={k_ext:.4f})...")
    L_ext, K_ext = assemble_L_K_ocl(rwg, verts, tris, k_ext, quad_order)
    print(f"  [OCL/f32] Interior (k={k_int:.4f})...")
    L_int, K_int = assemble_L_K_ocl(rwg, verts, tris, k_int, quad_order)

    K_sum = K_ext + K_int
    Z = np.zeros((2*N, 2*N), dtype=complex)
    Z[:N, :N] = eta_ext * L_ext + eta_int * L_int
    Z[:N, N:] = -K_sum
    Z[N:, :N] = K_sum
    Z[N:, N:] = L_ext / eta_ext + L_int / eta_int

    print(f"  [OCL/f32] Total PMCHWT: {_time.time() - t0:.1f}s")
    return Z, L_ext, K_ext
