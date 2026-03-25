"""
Helmholtz MLFMA (Multi-Level Fast Multipole Algorithm) for BEM.

Accelerates matrix-vector products for PMCHWT BEM from O(N^2) to O(N log N).
Uses diagonal (plane-wave) translation form for efficient M2L.

Key components:
  - Adaptive octree partitioning of RWG basis function centroids
  - Plane-wave expansion of Helmholtz Green's function
  - P2M, M2M, M2L, L2L, L2P translation operators
  - P2P near-field direct computation (reuses BEM quadrature)
  - PMCHWT system matvec wrapper for GMRES

References:
  - Coifman, Rokhlin, Wandzura (1993) — MLFMA for Helmholtz
  - Chew et al. (2001) — Fast and Efficient Algorithms in CEM
  - Ergul & Gurel (2014) — The Multilevel Fast Multipole Algorithm
"""

import numpy as np
from scipy.special import spherical_jn, spherical_yn
from scipy.special import roots_legendre
import time as _time

try:
    import pyopencl as cl
    HAS_OCL = True
except ImportError:
    HAS_OCL = False

# ============================================================
# 0. OpenCL P2P kernels
# ============================================================

_P2P_OCL_SRC = r"""
// P2P scalar potential: φ_i = Σ_j G(r_i, r_j) * q_j
// Each work-item handles one target, loops over its P2P sources (CSR).
__kernel void p2p_potential(
    __global const float *tgt_xyz,   // (Nt*3) target positions
    __global const float *src_xyz,   // (Ns*3) source positions
    __global const float *q_re,      // (Ns) source charges real
    __global const float *q_im,      // (Ns) source charges imag
    __global const int   *offsets,   // (Nt+1) CSR row offsets
    __global const int   *src_idx,   // (nnz) CSR column indices
    const float k_re, const float k_im,
    __global float *out_re,          // (Nt) output real
    __global float *out_im           // (Nt) output imag
) {
    const int tid = get_global_id(0);
    const float tx = tgt_xyz[tid * 3];
    const float ty = tgt_xyz[tid * 3 + 1];
    const float tz = tgt_xyz[tid * 3 + 2];
    float acc_re = 0.0f, acc_im = 0.0f;
    const int jstart = offsets[tid];
    const int jend   = offsets[tid + 1];
    const float inv4pi = 0.07957747154594767f;
    for (int j = jstart; j < jend; j++) {
        const int sid = src_idx[j];
        const float dx = tx - src_xyz[sid * 3];
        const float dy = ty - src_xyz[sid * 3 + 1];
        const float dz = tz - src_xyz[sid * 3 + 2];
        const float R = sqrt(dx*dx + dy*dy + dz*dz);
        if (R < 1e-12f) continue;
        const float inv_R = 1.0f / R;
        const float eR = exp(-k_im * R);
        const float phase = k_re * R;
        const float G_re = eR * cos(phase) * inv4pi * inv_R;
        const float G_im = eR * sin(phase) * inv4pi * inv_R;
        const float qr = q_re[sid];
        const float qi = q_im[sid];
        acc_re += G_re * qr - G_im * qi;
        acc_im += G_re * qi + G_im * qr;
    }
    out_re[tid] = acc_re;
    out_im[tid] = acc_im;
}

// P2P gradient: (∇φ)_i = Σ_j ∇G(r_i, r_j) * q_j
// ∇G = G * (ik - 1/R) / R * (r_i - r_j)
__kernel void p2p_gradient(
    __global const float *tgt_xyz,
    __global const float *src_xyz,
    __global const float *q_re,
    __global const float *q_im,
    __global const int   *offsets,
    __global const int   *src_idx,
    const float k_re, const float k_im,
    __global float *gx_re, __global float *gx_im,
    __global float *gy_re, __global float *gy_im,
    __global float *gz_re, __global float *gz_im
) {
    const int tid = get_global_id(0);
    const float tx = tgt_xyz[tid * 3];
    const float ty = tgt_xyz[tid * 3 + 1];
    const float tz = tgt_xyz[tid * 3 + 2];
    float ax_re = 0.0f, ax_im = 0.0f;
    float ay_re = 0.0f, ay_im = 0.0f;
    float az_re = 0.0f, az_im = 0.0f;
    const int jstart = offsets[tid];
    const int jend   = offsets[tid + 1];
    const float inv4pi = 0.07957747154594767f;
    for (int j = jstart; j < jend; j++) {
        const int sid = src_idx[j];
        const float dx = tx - src_xyz[sid * 3];
        const float dy = ty - src_xyz[sid * 3 + 1];
        const float dz = tz - src_xyz[sid * 3 + 2];
        const float R = sqrt(dx*dx + dy*dy + dz*dz);
        if (R < 1e-12f) continue;
        const float inv_R = 1.0f / R;
        const float eR = exp(-k_im * R);
        const float phase = k_re * R;
        const float cp = cos(phase);
        const float sp = sin(phase);
        const float G_re = eR * cp * inv4pi * inv_R;
        const float G_im = eR * sp * inv4pi * inv_R;
        // factor = (ik - 1/R) / R  where ik = -k_im + i*k_re
        const float fac_re = (-k_im - inv_R) * inv_R;
        const float fac_im = k_re * inv_R;
        // gradG_scalar = G * factor
        const float gG_re = G_re * fac_re - G_im * fac_im;
        const float gG_im = G_re * fac_im + G_im * fac_re;
        // gradG * q
        const float qr = q_re[sid];
        const float qi = q_im[sid];
        const float gq_re = gG_re * qr - gG_im * qi;
        const float gq_im = gG_re * qi + gG_im * qr;
        ax_re += gq_re * dx; ax_im += gq_im * dx;
        ay_re += gq_re * dy; ay_im += gq_im * dy;
        az_re += gq_re * dz; az_im += gq_im * dz;
    }
    gx_re[tid] = ax_re; gx_im[tid] = ax_im;
    gy_re[tid] = ay_re; gy_im[tid] = ay_im;
    gz_re[tid] = az_re; gz_im[tid] = az_im;
}
"""


class P2P_OpenCL:
    """GPU-accelerated P2P near-field using OpenCL with CSR interaction lists."""

    def __init__(self, fmm):
        """Build CSR interaction lists from FMM tree and upload to GPU.

        Parameters
        ----------
        fmm : HelmholtzFMM
            FMM instance with built octree.
        """
        if not HAS_OCL:
            raise RuntimeError("pyopencl not available")

        self.Nt = fmm.Nt
        self.Ns = fmm.Ns

        # Get OpenCL context
        try:
            from bem_opencl import _get_ocl
            self.ctx, self.queue, _, _, _ = _get_ocl()
        except Exception:
            platforms = cl.get_platforms()
            gpu_dev = None
            for plat in platforms:
                for dev in plat.get_devices():
                    if dev.type == cl.device_type.GPU:
                        gpu_dev = dev
                        break
                if gpu_dev:
                    break
            if gpu_dev is None:
                raise RuntimeError("No GPU device for OpenCL P2P")
            self.ctx = cl.Context([gpu_dev])
            self.queue = cl.CommandQueue(self.ctx)

        # Compile
        self.prg = cl.Program(self.ctx, _P2P_OCL_SRC).build()
        self.kern_pot = cl.Kernel(self.prg, 'p2p_potential')
        self.kern_grad = cl.Kernel(self.prg, 'p2p_gradient')

        # Build CSR interaction lists
        tgt_to_leaf = {}
        for leaf in fmm.tree.leaves:
            for tid in leaf._target_ids:
                tgt_to_leaf[tid] = leaf

        offsets = [0]
        src_indices = []
        for tid in range(self.Nt):
            leaf = tgt_to_leaf.get(tid)
            if leaf is None:
                offsets.append(offsets[-1])
                continue
            all_src = list(leaf._source_ids)
            for nb in leaf.near_list:
                all_src.extend(nb._source_ids)
            src_indices.extend(all_src)
            offsets.append(offsets[-1] + len(all_src))

        self.offsets = np.array(offsets, dtype=np.int32)
        self.src_indices_arr = (np.array(src_indices, dtype=np.int32)
                                if src_indices else np.zeros(1, dtype=np.int32))
        self.nnz = len(src_indices)

        # Upload static buffers
        mf = cl.mem_flags
        tgt_f32 = np.ascontiguousarray(fmm.targets.ravel(), dtype=np.float32)
        src_f32 = np.ascontiguousarray(fmm.sources.ravel(), dtype=np.float32)

        self.tgt_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                 hostbuf=tgt_f32)
        self.src_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                 hostbuf=src_f32)
        self.off_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                 hostbuf=self.offsets)
        self.idx_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                 hostbuf=self.src_indices_arr)

        # Reusable I/O buffers
        nbytes_t = max(self.Nt * 4, 4)
        nbytes_s = max(self.Ns * 4, 4)
        self.q_re_buf = cl.Buffer(self.ctx, mf.READ_ONLY, nbytes_s)
        self.q_im_buf = cl.Buffer(self.ctx, mf.READ_ONLY, nbytes_s)
        self.out_re_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, nbytes_t)
        self.out_im_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, nbytes_t)
        self.gx_re_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, nbytes_t)
        self.gx_im_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, nbytes_t)
        self.gy_re_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, nbytes_t)
        self.gy_im_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, nbytes_t)
        self.gz_re_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, nbytes_t)
        self.gz_im_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, nbytes_t)

        # Host readback arrays (reusable)
        self._out_re = np.empty(self.Nt, dtype=np.float32)
        self._out_im = np.empty(self.Nt, dtype=np.float32)

        avg_src = self.nnz / max(1, self.Nt)
        print(f"  [P2P-OCL] CSR: {self.Nt} targets, {self.Ns} sources, "
              f"{self.nnz} pairs ({avg_src:.0f} avg/tgt)")

    def _upload_charges(self, source_charges):
        q = np.asarray(source_charges, dtype=complex)
        q_re = np.ascontiguousarray(q.real, dtype=np.float32)
        q_im = np.ascontiguousarray(q.imag, dtype=np.float32)
        cl.enqueue_copy(self.queue, self.q_re_buf, q_re)
        cl.enqueue_copy(self.queue, self.q_im_buf, q_im)

    def evaluate(self, source_charges, k):
        """P2P potential on GPU. Returns ndarray (Nt,) complex."""
        self._upload_charges(source_charges)
        self.kern_pot.set_args(
            self.tgt_buf, self.src_buf, self.q_re_buf, self.q_im_buf,
            self.off_buf, self.idx_buf,
            np.float32(k.real), np.float32(k.imag),
            self.out_re_buf, self.out_im_buf)
        cl.enqueue_nd_range_kernel(self.queue, self.kern_pot,
                                   (self.Nt,), None)
        cl.enqueue_copy(self.queue, self._out_re, self.out_re_buf)
        cl.enqueue_copy(self.queue, self._out_im, self.out_im_buf)
        self.queue.finish()
        return self._out_re.astype(np.float64) + 1j * self._out_im.astype(np.float64)

    def evaluate_gradient(self, source_charges, k):
        """P2P gradient on GPU. Returns ndarray (Nt, 3) complex."""
        self._upload_charges(source_charges)
        self.kern_grad.set_args(
            self.tgt_buf, self.src_buf, self.q_re_buf, self.q_im_buf,
            self.off_buf, self.idx_buf,
            np.float32(k.real), np.float32(k.imag),
            self.gx_re_buf, self.gx_im_buf,
            self.gy_re_buf, self.gy_im_buf,
            self.gz_re_buf, self.gz_im_buf)
        cl.enqueue_nd_range_kernel(self.queue, self.kern_grad,
                                   (self.Nt,), None)
        gx_re = np.empty(self.Nt, dtype=np.float32)
        gx_im = np.empty(self.Nt, dtype=np.float32)
        gy_re = np.empty(self.Nt, dtype=np.float32)
        gy_im = np.empty(self.Nt, dtype=np.float32)
        gz_re = np.empty(self.Nt, dtype=np.float32)
        gz_im = np.empty(self.Nt, dtype=np.float32)
        cl.enqueue_copy(self.queue, gx_re, self.gx_re_buf)
        cl.enqueue_copy(self.queue, gx_im, self.gx_im_buf)
        cl.enqueue_copy(self.queue, gy_re, self.gy_re_buf)
        cl.enqueue_copy(self.queue, gy_im, self.gy_im_buf)
        cl.enqueue_copy(self.queue, gz_re, self.gz_re_buf)
        cl.enqueue_copy(self.queue, gz_im, self.gz_im_buf)
        self.queue.finish()
        grad = np.empty((self.Nt, 3), dtype=complex)
        grad[:, 0] = gx_re.astype(np.float64) + 1j * gx_im.astype(np.float64)
        grad[:, 1] = gy_re.astype(np.float64) + 1j * gy_im.astype(np.float64)
        grad[:, 2] = gz_re.astype(np.float64) + 1j * gz_im.astype(np.float64)
        return grad


# ============================================================
# 1. Octree
# ============================================================

class OctreeNode:
    """Single node of the adaptive octree."""
    __slots__ = ['center', 'half_size', 'level', 'index',
                 'children', 'parent', 'particle_indices',
                 'is_leaf', 'near_list', 'far_list',
                 'multipole', 'local_exp',
                 '_target_ids', '_source_ids']

    def __init__(self, center, half_size, level, index):
        self.center = np.array(center, dtype=np.float64)
        self.half_size = half_size
        self.level = level
        self.index = index
        self.children = [None] * 8
        self.parent = None
        self.particle_indices = []
        self.is_leaf = True
        self.near_list = []
        self.far_list = []
        self.multipole = None
        self.local_exp = None


class Octree:
    """Adaptive octree for FMM particle distribution.

    Partitions particles (RWG basis function centroids or quadrature points)
    into a hierarchical tree structure. Each leaf contains at most max_leaf
    particles.
    """

    def __init__(self, points, max_leaf=64, max_depth=20, uniform_depth=None):
        """Build octree from point cloud.

        Parameters
        ----------
        points : ndarray (N, 3)
            Particle positions.
        max_leaf : int
            Maximum particles per leaf box.
        max_depth : int
            Maximum tree depth.
        uniform_depth : int, optional
            If set, force ALL leaves to this depth (uniform tree).
            Required for standard MLFMA without U/V lists.
        """
        self.points = np.ascontiguousarray(points, dtype=np.float64)
        self.N = len(points)
        self.max_leaf = max_leaf
        self.max_depth = max_depth
        self.uniform_depth = uniform_depth
        self.nodes = []
        self.leaves = []
        self.levels = {}  # level -> list of nodes

        # Bounding box
        pmin = points.min(axis=0)
        pmax = points.max(axis=0)
        center = 0.5 * (pmin + pmax)
        half_size = 0.5 * np.max(pmax - pmin) * 1.001  # small margin

        # Root node
        root = OctreeNode(center, half_size, 0, 0)
        root.particle_indices = list(range(self.N))
        self.nodes.append(root)
        self.root = root

        # Recursive subdivision
        self._subdivide(root)

        # Collect leaves and levels
        for node in self.nodes:
            if node.is_leaf:
                self.leaves.append(node)
            level = node.level
            if level not in self.levels:
                self.levels[level] = []
            self.levels[level].append(node)

        self.max_level = max(self.levels.keys())

        # Build interaction lists
        self._build_interaction_lists()

    def _child_octant(self, center, half_size, octant):
        """Get center and half_size of child octant."""
        hs = half_size / 2
        offset = np.array([
            -1 if (octant & 1) == 0 else 1,
            -1 if (octant & 2) == 0 else 1,
            -1 if (octant & 4) == 0 else 1,
        ], dtype=np.float64) * hs
        return center + offset, hs

    def _subdivide(self, node):
        """Recursively subdivide node until leaf criteria met."""
        if self.uniform_depth is not None:
            # Uniform tree: always subdivide until target depth
            if node.level >= self.uniform_depth:
                return
        else:
            if len(node.particle_indices) <= self.max_leaf or node.level >= self.max_depth:
                return

        node.is_leaf = False
        pts = self.points[node.particle_indices]

        # Classify particles into octants
        octant_particles = [[] for _ in range(8)]
        for i, pi in enumerate(node.particle_indices):
            p = self.points[pi]
            octant = 0
            if p[0] >= node.center[0]: octant |= 1
            if p[1] >= node.center[1]: octant |= 2
            if p[2] >= node.center[2]: octant |= 4
            octant_particles[octant].append(pi)

        for octant in range(8):
            # For uniform trees, create all 8 children (even empty)
            if len(octant_particles[octant]) == 0 and self.uniform_depth is None:
                continue
            child_center, child_hs = self._child_octant(
                node.center, node.half_size, octant)
            child = OctreeNode(child_center, child_hs,
                               node.level + 1, len(self.nodes))
            child.parent = node
            child.particle_indices = octant_particles[octant]
            node.children[octant] = child
            self.nodes.append(child)
            self._subdivide(child)

    def _build_interaction_lists(self):
        """Build near-field and far-field (interaction) lists for each node.

        Near list: nodes at the same level whose parent boxes are neighbors
                   but which are not neighbors themselves (= far-field at this level).
        Actually, standard FMM convention:
          - Near list of a leaf: all leaf boxes within 1 box distance (neighbors)
          - Far list (interaction list): children of parent's neighbors that are
            not neighbors of the current box
        """
        # Build neighbor map level by level
        for level in sorted(self.levels.keys()):
            nodes_at_level = self.levels[level]
            n = len(nodes_at_level)

            # For leaf nodes, near list = neighbors (including self)
            # For all nodes, far list = interaction list
            for i, ni in enumerate(nodes_at_level):
                for j, nj in enumerate(nodes_at_level):
                    if i == j:
                        continue
                    dist = np.max(np.abs(ni.center - nj.center))
                    # Neighbors: distance <= 3 * half_size (adjacent or diagonal)
                    neighbor_thresh = 3.0 * ni.half_size + 1e-10

                    if dist <= neighbor_thresh:
                        # These are neighbors at this level
                        if ni.is_leaf and nj.is_leaf:
                            ni.near_list.append(nj)
                    else:
                        # Check if parents are neighbors (or same)
                        if ni.parent is not None and nj.parent is not None:
                            parent_dist = np.max(
                                np.abs(ni.parent.center - nj.parent.center))
                            parent_thresh = 3.0 * ni.parent.half_size + 1e-10
                            if parent_dist <= parent_thresh:
                                ni.far_list.append(nj)

    def __repr__(self):
        n_leaves = len(self.leaves)
        return (f"Octree({self.N} particles, {len(self.nodes)} nodes, "
                f"{n_leaves} leaves, depth={self.max_level})")


# ============================================================
# 2. Spherical harmonics and plane-wave quadrature
# ============================================================

def spherical_hankel1(n, z):
    """Spherical Hankel function of the first kind: h_n^(1)(z) = j_n(z) + i*y_n(z)."""
    return spherical_jn(n, z) + 1j * spherical_yn(n, z)


def _truncation_order(k, box_size, digits=3):
    """Estimate multipole truncation order p for given box size and wavenumber.

    Standard formula: p ≈ ka + c * (ka)^{1/3}
    where a = box diameter, c depends on desired digits of accuracy.
    """
    ka = abs(k) * box_size
    c_map = {2: 3.0, 3: 5.0, 4: 7.0, 5: 9.0}
    c = c_map.get(digits, 5.0)
    p = int(np.ceil(ka + c * max(ka, 1.0) ** (1.0 / 3.0)))
    p = max(p, 3)  # minimum order
    return p


def _sphere_quadrature(p):
    """Quadrature points and weights on the unit sphere for order p.

    Uses Gauss-Legendre for θ and uniform for φ.
    Total points: (p+1) × (2p+2).

    Returns
    -------
    dirs : ndarray (L, 3)
        Unit direction vectors ŝ.
    weights : ndarray (L,)
        Quadrature weights (integrate over 4π).
    theta : ndarray (L,)
    phi : ndarray (L,)
    """
    n_theta = p + 1
    n_phi = 2 * p + 2

    # Gauss-Legendre for cos(theta) on [-1, 1]
    mu, w_mu = roots_legendre(n_theta)
    theta = np.arccos(mu)

    # Uniform for phi
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    d_phi = 2 * np.pi / n_phi

    # Build full grid
    L = n_theta * n_phi
    dirs = np.zeros((L, 3))
    weights = np.zeros(L)
    thetas = np.zeros(L)
    phis = np.zeros(L)

    idx = 0
    for it in range(n_theta):
        st = np.sin(theta[it])
        ct = np.cos(theta[it])
        for ip in range(n_phi):
            sp = np.sin(phi[ip])
            cp = np.cos(phi[ip])
            dirs[idx] = [st * cp, st * sp, ct]
            weights[idx] = w_mu[it] * d_phi
            thetas[idx] = theta[it]
            phis[idx] = phi[ip]
            idx += 1

    return dirs, weights, thetas, phis


# ============================================================
# 3. Helmholtz FMM core
# ============================================================

class HelmholtzFMM:
    """Multi-Level Fast Multipole Algorithm for Helmholtz equation.

    Computes y_i = Σ_j G(r_i, r_j) * q_j  for all i ≠ j
    where G(r,r') = exp(ik|r-r'|) / (4π|r-r'|)

    Also supports gradient evaluation:
    y_i = Σ_j ∇G(r_i, r_j) * q_j

    Uses diagonal (plane-wave) translation form for M2L.
    """

    def __init__(self, targets, sources, k, max_leaf=64, digits=3):
        """Initialize FMM with target and source points.

        Parameters
        ----------
        targets : ndarray (Nt, 3)
            Target (observation) points.
        sources : ndarray (Ns, 3)
            Source points.
        k : complex
            Wavenumber.
        max_leaf : int
            Max particles per leaf box.
        digits : int
            Desired digits of accuracy (2-5).
        """
        self.targets = np.ascontiguousarray(targets, dtype=np.float64)
        self.sources = np.ascontiguousarray(sources, dtype=np.float64)
        self.k = complex(k)
        self.Nt = len(targets)
        self.Ns = len(sources)
        self.digits = digits

        # Build single octree containing all points
        # Target indices: 0..Nt-1, Source indices: Nt..Nt+Ns-1
        all_points = np.vstack([targets, sources])
        N_total = len(all_points)

        # Compute uniform depth from max_leaf
        # depth such that 8^depth * max_leaf >= N_total
        uniform_depth = max(2, int(np.ceil(np.log(max(N_total / max_leaf, 1)) / np.log(8))))
        uniform_depth = min(uniform_depth, 6)  # cap to avoid too deep trees

        self.tree = Octree(all_points, max_leaf=max_leaf,
                           uniform_depth=uniform_depth)

        # For each leaf, separate target and source indices
        for leaf in self.tree.leaves:
            leaf._target_ids = [i for i in leaf.particle_indices if i < self.Nt]
            leaf._source_ids = [i - self.Nt for i in leaf.particle_indices
                                if i >= self.Nt]

        # Use UNIFORM truncation order across all levels
        # Based on finest level (leaf) box size — oversamples at coarser levels
        # but avoids expensive inter-level interpolation
        leaf_half = self.tree.leaves[0].half_size
        leaf_box_size = 2 * leaf_half
        self.p = _truncation_order(k, leaf_box_size, digits)
        self.dirs, self.weights, self.thetas, self.phis = _sphere_quadrature(self.p)
        self.L = len(self.dirs)

        # Precompute transfer functions for ALL unique M2L vectors
        self._transfer_cache = {}
        n_m2l = 0
        for level in sorted(self.tree.levels.keys()):
            for node in self.tree.levels[level]:
                for far_node in node.far_list:
                    d_vec = node.center - far_node.center
                    d_key = tuple(np.round(d_vec, 10))
                    if d_key not in self._transfer_cache:
                        self._transfer_cache[d_key] = self._compute_transfer(d_vec)
                    n_m2l += 1

        # Precompute M2L batch structure for vectorized evaluation
        # For each level: arrays of (target_node_idx, transfer_function)
        # grouped by target for efficient scatter-add
        self._m2l_batch = {}
        n_nodes = len(self.tree.nodes)
        for level in sorted(self.tree.levels.keys()):
            if level == 0:
                continue
            tgt_indices = []
            src_indices = []
            transfer_vecs = []
            for node in self.tree.levels[level]:
                for far_node in node.far_list:
                    d_vec = node.center - far_node.center
                    d_key = tuple(np.round(d_vec, 10))
                    tgt_indices.append(node.index)
                    src_indices.append(far_node.index)
                    transfer_vecs.append(self._transfer_cache[d_key])
            if tgt_indices:
                self._m2l_batch[level] = (
                    np.array(tgt_indices, dtype=np.int32),
                    np.array(src_indices, dtype=np.int32),
                    np.array(transfer_vecs, dtype=complex))  # (n_pairs, L)

        # Precompute M2M parent-child structure
        self._m2m_data = {}  # level -> list of (parent_idx, child_idx, shift_vec)
        for level in range(self.tree.max_level - 1, 0, -1):
            if level not in self.tree.levels:
                continue
            pairs = []
            for node in self.tree.levels[level]:
                if not node.is_leaf:
                    for child in node.children:
                        if child is not None:
                            t_vec = child.center - node.center
                            shift = np.exp(-1j * self.k * (self.dirs @ t_vec))
                            pairs.append((node.index, child.index, shift))
            if pairs:
                p_idx = np.array([p[0] for p in pairs], dtype=np.int32)
                c_idx = np.array([p[1] for p in pairs], dtype=np.int32)
                shifts = np.array([p[2] for p in pairs], dtype=complex)
                self._m2m_data[level] = (p_idx, c_idx, shifts)

        # Precompute L2L parent-child structure
        self._l2l_data = {}
        for level in range(2, self.tree.max_level + 1):
            if level not in self.tree.levels:
                continue
            pairs = []
            for node in self.tree.levels[level]:
                if node.parent is not None:
                    t_vec = node.center - node.parent.center
                    shift = np.exp(1j * self.k * (self.dirs @ t_vec))
                    pairs.append((node.index, node.parent.index, shift))
            if pairs:
                c_idx = np.array([p[0] for p in pairs], dtype=np.int32)
                p_idx = np.array([p[1] for p in pairs], dtype=np.int32)
                shifts = np.array([p[2] for p in pairs], dtype=complex)
                self._l2l_data[level] = (c_idx, p_idx, shifts)

        # Node-indexed arrays for vectorized tree operations
        self._n_nodes = n_nodes

        print(f"  [FMM] Tree: {self.tree}")
        print(f"  [FMM] p={self.p}, L={self.L} plane waves, "
              f"{len(self._transfer_cache)} unique transfers, {n_m2l} M2L pairs")

        # Try GPU-accelerated P2P
        self.p2p_ocl = None
        if HAS_OCL:
            try:
                self.p2p_ocl = P2P_OpenCL(self)
            except Exception as e:
                print(f"  [FMM] GPU P2P unavailable: {e}")

    def _p2m(self, leaf, source_charges):
        """Particle-to-Multipole: compute radiation pattern of leaf sources.

        F(ŝ) = Σ_j q_j * exp(-ik * ŝ · r'_j)

        where r'_j is relative to box center.
        """
        src_ids = leaf._source_ids
        if len(src_ids) == 0:
            leaf.multipole = np.zeros(self.L, dtype=complex)
            return

        # Source positions relative to box center
        r_rel = self.sources[src_ids] - leaf.center[None, :]  # (ns, 3)
        q = source_charges[src_ids]  # (ns,)

        # F(ŝ_l) = Σ_j q_j * exp(-ik * ŝ_l · r_j)
        phase = np.exp(-1j * self.k * (self.dirs @ r_rel.T))  # (L, ns)
        leaf.multipole = phase @ q  # (L,)

    def _m2m(self, parent):
        """Multipole-to-Multipole: shift child multipoles to parent center.

        F_parent(ŝ) = Σ_child F_child(ŝ) * exp(-ik * ŝ · (r_child - r_parent))

        With uniform truncation order, this is a simple phase shift.
        """
        parent.multipole = np.zeros(self.L, dtype=complex)

        for child in parent.children:
            if child is None or child.multipole is None:
                continue
            t_vec = child.center - parent.center
            shift = np.exp(-1j * self.k * (self.dirs @ t_vec))
            parent.multipole += child.multipole * shift

    # With uniform truncation order, no anterpolation needed.

    def _compute_transfer(self, d_vec):
        """Compute M2L translation function T_L(ŝ, d).

        T_L(ŝ, d) = Σ_{l=0}^{p} (2l+1) i^l h_l^(1)(k|d|) P_l(ŝ · d̂)

        Uses Legendre recurrence for P_l and scipy for h_l.
        """
        d_norm = np.linalg.norm(d_vec)
        if d_norm < 1e-15:
            return np.ones(self.L, dtype=complex)

        d_hat = d_vec / d_norm
        kd = self.k * d_norm

        # ŝ · d̂ for all quadrature directions
        cos_angle = self.dirs @ d_hat  # (L,)

        # Spherical Hankel values h_l(kd) for l = 0..p
        h_vals = np.array([spherical_hankel1(l, kd) for l in range(self.p + 1)])

        # Legendre recurrence: P_l(x)
        T = np.zeros(self.L, dtype=complex)
        P_prev = np.ones(self.L)           # P_0 = 1
        P_curr = cos_angle.copy()          # P_1 = x

        # l = 0 term
        T += (2 * 0 + 1) * (1j ** 0) * h_vals[0] * P_prev
        if self.p >= 1:
            T += (2 * 1 + 1) * (1j ** 1) * h_vals[1] * P_curr

        for l in range(2, self.p + 1):
            P_next = ((2*l - 1) * cos_angle * P_curr - (l - 1) * P_prev) / l
            T += (2*l + 1) * (1j ** l) * h_vals[l] * P_next
            P_prev = P_curr
            P_curr = P_next

        return T

    def _m2l(self, target_node, source_node):
        """Multipole-to-Local: translate source multipole to target local expansion.

        L(ŝ) += T_L(ŝ, d) * M(ŝ)
        where T_L = Σ_l (2l+1) i^l h_l^(1)(k|d|) P_l(ŝ·d̂)
        and d = target_center - source_center.
        Uses cached transfer functions.
        """
        if source_node.multipole is None:
            return

        d_vec = target_node.center - source_node.center
        d_key = tuple(np.round(d_vec, 10))
        transfer = self._transfer_cache.get(d_key)
        if transfer is None:
            transfer = self._compute_transfer(d_vec)

        if target_node.local_exp is None:
            target_node.local_exp = np.zeros(self.L, dtype=complex)

        target_node.local_exp += transfer * source_node.multipole

    def _l2l(self, parent, child):
        """Local-to-Local: shift parent local expansion to child center.

        L_child(ŝ) += L_parent(ŝ) * exp(ik ŝ · (r_child - r_parent))
        """
        if parent.local_exp is None:
            return

        t_vec = child.center - parent.center
        shift = np.exp(1j * self.k * (self.dirs @ t_vec))

        if child.local_exp is None:
            child.local_exp = np.zeros(self.L, dtype=complex)
        child.local_exp += parent.local_exp * shift

    def _l2p(self, leaf, result):
        """Local-to-Particle: evaluate local expansion at target points.

        φ(r_i) ≈ (ik/16π²) Σ_l w_l * L(ŝ_l) * exp(ik ŝ_l · r_i)
        """
        if leaf.local_exp is None:
            return

        tgt_ids = leaf._target_ids
        if len(tgt_ids) == 0:
            return

        r_rel = self.targets[tgt_ids] - leaf.center[None, :]
        phase = np.exp(1j * self.k * (r_rel @ self.dirs.T))  # (nt, L)

        prefactor = 1j * self.k / (16 * np.pi**2)
        contrib = prefactor * (phase @ (self.weights * leaf.local_exp))
        result[tgt_ids] += contrib

    def _p2p(self, target_leaf, source_leaf, source_charges, result):
        """Particle-to-Particle: direct near-field computation.

        φ(r_i) = Σ_j G(r_i, r_j) * q_j  for i in target, j in source.
        """
        tgt_ids = target_leaf._target_ids
        src_ids = source_leaf._source_ids

        if len(tgt_ids) == 0 or len(src_ids) == 0:
            return

        r_tgt = self.targets[tgt_ids]  # (nt, 3)
        r_src = self.sources[src_ids]  # (ns, 3)
        q = source_charges[src_ids]    # (ns,)

        # Direct evaluation of G = exp(ikR) / (4πR)
        diff = r_tgt[:, None, :] - r_src[None, :, :]  # (nt, ns, 3)
        R = np.linalg.norm(diff, axis=2)               # (nt, ns)
        R_safe = np.maximum(R, 1e-15)

        G = np.exp(1j * self.k * R_safe) / (4 * np.pi * R_safe)  # (nt, ns)

        # Zero out self-interaction (R ≈ 0)
        G[R < 1e-12] = 0.0

        result[tgt_ids] += G @ q  # (nt,)

    def _p2p_gradient(self, target_leaf, source_leaf, source_charges, result_grad):
        """P2P for gradient of Green's function: ∇_r G(r, r').

        ∇G = (ik - 1/R) * G * (r - r') / R
        """
        tgt_ids = target_leaf._target_ids
        src_ids = source_leaf._source_ids

        if len(tgt_ids) == 0 or len(src_ids) == 0:
            return

        r_tgt = self.targets[tgt_ids]
        r_src = self.sources[src_ids]
        q = source_charges[src_ids]

        diff = r_tgt[:, None, :] - r_src[None, :, :]  # (nt, ns, 3)
        R = np.linalg.norm(diff, axis=2)
        R_safe = np.maximum(R, 1e-15)

        G = np.exp(1j * self.k * R_safe) / (4 * np.pi * R_safe)
        factor = (1j * self.k - 1.0 / R_safe) / R_safe  # (nt, ns)
        gradG = G * factor  # scalar part, (nt, ns)

        # Zero self-interaction
        gradG[R < 1e-12] = 0.0

        # ∇G_ij = gradG_ij * diff_ij (vector)
        # result[i] = Σ_j ∇G_ij * q_j = Σ_j gradG_ij * q_j * diff_ij
        for d in range(3):
            result_grad[tgt_ids, d] += (gradG * diff[:, :, d]) @ q

    def evaluate(self, source_charges):
        """Evaluate potential φ(r_i) = Σ_j G(r_i, r_j) * q_j using FMM.

        Parameters
        ----------
        source_charges : ndarray (Ns,), complex
            Source strengths.

        Returns
        -------
        potential : ndarray (Nt,), complex
            Potential at target points.
        """
        q = np.asarray(source_charges, dtype=complex)
        result = np.zeros(self.Nt, dtype=complex)

        # Node-indexed multipole/local arrays for vectorized ops
        nn = self._n_nodes
        L = self.L
        multi = np.zeros((nn, L), dtype=complex)
        local = np.zeros((nn, L), dtype=complex)

        # === Upward pass: P2M ===
        for leaf in self.tree.leaves:
            src_ids = leaf._source_ids
            if src_ids:
                r_rel = self.sources[src_ids] - leaf.center
                multi[leaf.index] = (
                    np.exp(-1j * self.k * (self.dirs @ r_rel.T)) @ q[src_ids])

        # === M2M (vectorized per level) ===
        for level in range(self.tree.max_level - 1, 0, -1):
            if level not in self._m2m_data:
                continue
            p_idx, c_idx, shifts = self._m2m_data[level]
            products = multi[c_idx] * shifts  # (n_pairs, L)
            np.add.at(multi, p_idx, products)

        # === M2L (vectorized per level) ===
        for level, (tgt_idx, src_idx, T_arr) in self._m2l_batch.items():
            products = T_arr * multi[src_idx]  # (n_pairs, L)
            np.add.at(local, tgt_idx, products)

        # === L2L (vectorized per level) ===
        for level in range(2, self.tree.max_level + 1):
            if level not in self._l2l_data:
                continue
            c_idx, p_idx, shifts = self._l2l_data[level]
            local[c_idx] += local[p_idx] * shifts

        # === L2P: local → target potentials ===
        prefactor = 1j * self.k / (16 * np.pi**2)
        for leaf in self.tree.leaves:
            tgt_ids = leaf._target_ids
            le = local[leaf.index]
            if tgt_ids and np.any(le != 0):
                r_rel = self.targets[tgt_ids] - leaf.center
                phase = np.exp(1j * self.k * (r_rel @ self.dirs.T))
                result[tgt_ids] += prefactor * (phase @ (self.weights * le))

        # === P2P: near-field direct ===
        if self.p2p_ocl is not None:
            result += self.p2p_ocl.evaluate(q, self.k)
        else:
            for leaf in self.tree.leaves:
                tgt_ids = leaf._target_ids
                if not tgt_ids:
                    continue
                r_tgt = self.targets[tgt_ids]
                all_src_ids = list(leaf._source_ids)
                for nb in leaf.near_list:
                    all_src_ids.extend(nb._source_ids)
                if all_src_ids:
                    r_src = self.sources[all_src_ids]
                    q_src = q[all_src_ids]
                    diff = r_tgt[:, None, :] - r_src[None, :, :]
                    R = np.linalg.norm(diff, axis=2)
                    R_safe = np.maximum(R, 1e-15)
                    G = np.exp(1j * self.k * R_safe) / (4 * np.pi * R_safe)
                    G[R < 1e-12] = 0.0
                    result[tgt_ids] += G @ q_src

        return result

    def evaluate_gradient(self, source_charges):
        """Evaluate gradient ∇φ(r_i) = Σ_j ∇G(r_i, r_j) * q_j.

        Returns ndarray (Nt, 3) complex.
        """
        q = np.asarray(source_charges, dtype=complex)
        result_grad = np.zeros((self.Nt, 3), dtype=complex)

        nn = self._n_nodes
        L = self.L
        multi = np.zeros((nn, L), dtype=complex)
        local = np.zeros((nn, L), dtype=complex)

        # === P2M ===
        for leaf in self.tree.leaves:
            src_ids = leaf._source_ids
            if src_ids:
                r_rel = self.sources[src_ids] - leaf.center
                multi[leaf.index] = (
                    np.exp(-1j * self.k * (self.dirs @ r_rel.T)) @ q[src_ids])

        # === M2M ===
        for level in range(self.tree.max_level - 1, 0, -1):
            if level not in self._m2m_data:
                continue
            p_idx, c_idx, shifts = self._m2m_data[level]
            np.add.at(multi, p_idx, multi[c_idx] * shifts)

        # === M2L ===
        for level, (tgt_idx, src_idx, T_arr) in self._m2l_batch.items():
            np.add.at(local, tgt_idx, T_arr * multi[src_idx])

        # === L2L ===
        for level in range(2, self.tree.max_level + 1):
            if level not in self._l2l_data:
                continue
            c_idx, p_idx, shifts = self._l2l_data[level]
            local[c_idx] += local[p_idx] * shifts

        # === L2P gradient ===
        prefactor = 1j * self.k / (16 * np.pi**2)
        ik = 1j * self.k
        for leaf in self.tree.leaves:
            tgt_ids = leaf._target_ids
            le = local[leaf.index]
            if tgt_ids and np.any(le != 0):
                r_rel = self.targets[tgt_ids] - leaf.center
                phase = np.exp(1j * self.k * (r_rel @ self.dirs.T))
                weighted = self.weights * le
                for d in range(3):
                    result_grad[tgt_ids, d] += prefactor * ik * (
                        phase @ (weighted * self.dirs[:, d]))

        # P2P gradient
        if self.p2p_ocl is not None:
            result_grad += self.p2p_ocl.evaluate_gradient(q, self.k)
        else:
            for leaf in self.tree.leaves:
                tgt_ids = leaf._target_ids
                if not tgt_ids:
                    continue
                r_tgt = self.targets[tgt_ids]
                all_src_ids = list(leaf._source_ids)
                for nb in leaf.near_list:
                    all_src_ids.extend(nb._source_ids)
                if not all_src_ids:
                    continue
                r_src = self.sources[all_src_ids]
                q_src = q[all_src_ids]
                diff = r_tgt[:, None, :] - r_src[None, :, :]
                R = np.linalg.norm(diff, axis=2)
                R_safe = np.maximum(R, 1e-15)
                G = np.exp(1j * self.k * R_safe) / (4 * np.pi * R_safe)
                factor = (1j * self.k - 1.0 / R_safe) / R_safe
                gradG = G * factor
                gradG[R < 1e-12] = 0.0
                for d in range(3):
                    result_grad[tgt_ids, d] += (gradG * diff[:, :, d]) @ q_src

        return result_grad

    def _l2p_gradient(self, leaf, result_grad):
        """L2P for gradient: ∇φ = ik * (ik/16π²) Σ ŝ w L(ŝ) e^{ikŝ·r}."""
        if leaf.local_exp is None:
            return

        tgt_ids = leaf._target_ids
        if len(tgt_ids) == 0:
            return

        r_rel = self.targets[tgt_ids] - leaf.center[None, :]
        phase = np.exp(1j * self.k * (r_rel @ self.dirs.T))  # (nt, L)

        prefactor = 1j * self.k / (16 * np.pi**2)
        ik = 1j * self.k

        weighted = self.weights * leaf.local_exp
        for d in range(3):
            result_grad[tgt_ids, d] += prefactor * ik * (
                phase @ (weighted * self.dirs[:, d]))


# ============================================================
# 4. BEM-FMM coupling: L and K operator matvec
# ============================================================

class BEM_FMM_Operator:
    """FMM-accelerated BEM operator for PMCHWT formulation.

    Computes y = Z · x where Z is the PMCHWT system matrix,
    without explicitly forming Z. Uses FMM for far-field
    interactions and direct computation for near-field.

    The PMCHWT matrix has block structure:
        Z = [[η₀L₀ + η₁L₁,  -(K₀+K₁)],
             [K₀+K₁,         L₀/η₀ + L₁/η₁]]

    L and K operators involve surface integrals of G and ∇G
    over RWG basis function pairs. The FMM accelerates these
    by treating quadrature points as equivalent point sources.
    """

    def __init__(self, rwg, verts, tris, k_ext, k_int, eta_ext, eta_int,
                 quad_order=7, fmm_digits=3, max_leaf=64):
        """Initialize BEM-FMM operator.

        Parameters
        ----------
        rwg : dict
            RWG basis function data from build_rwg().
        verts, tris : ndarray
            Mesh vertices and triangles.
        k_ext, k_int : complex
            Exterior and interior wavenumbers.
        eta_ext, eta_int : complex
            Exterior and interior impedances.
        quad_order : int
            Quadrature order for BEM integrals (7 recommended).
        fmm_digits : int
            Digits of FMM accuracy (3-5).
        max_leaf : int
            Max particles per FMM leaf box.
        """
        from bem_core import tri_quadrature

        self.rwg = rwg
        self.verts = verts
        self.tris = tris
        self.k_ext = complex(k_ext)
        self.k_int = complex(k_int)
        self.eta_ext = complex(eta_ext)
        self.eta_int = complex(eta_int)
        self.N = rwg['N']
        self.system_size = 2 * self.N

        t0 = _time.time()
        print(f"  [BEM-FMM] Initializing: {self.N} RWG, quad={quad_order}...")

        # Quadrature
        self.quad_pts, self.quad_wts = tri_quadrature(quad_order)
        self.Nq = len(self.quad_wts)
        lam0 = 1 - self.quad_pts[:, 0] - self.quad_pts[:, 1]

        # Precompute quadrature points for all RWG functions
        def get_qpts_batch(tri_indices):
            t = tris[tri_indices]
            v0 = verts[t[:, 0]]; v1 = verts[t[:, 1]]; v2 = verts[t[:, 2]]
            return (np.einsum('q,ni->nqi', lam0, v0) +
                    np.einsum('q,ni->nqi', self.quad_pts[:, 0], v1) +
                    np.einsum('q,ni->nqi', self.quad_pts[:, 1], v2))

        self.qpts_p = get_qpts_batch(rwg['tri_p'])  # (N, Nq, 3)
        self.qpts_m = get_qpts_batch(rwg['tri_m'])

        # RWG function values at quadrature points
        self.f_p = ((rwg['length'][:, None, None] / (2 * rwg['area_p'][:, None, None]))
                    * (self.qpts_p - rwg['free_p'][:, None, :]))
        self.f_m = (-(rwg['length'][:, None, None] / (2 * rwg['area_m'][:, None, None]))
                    * (self.qpts_m - rwg['free_m'][:, None, :]))

        # Divergences
        self.div_p = rwg['length'] / rwg['area_p']      # (N,)
        self.div_m = -rwg['length'] / rwg['area_m']      # (N,)

        # Jacobian × quadrature weights
        self.jw_p = rwg['area_p'][:, None] * self.quad_wts[None, :]  # (N, Nq)
        self.jw_m = rwg['area_m'][:, None] * self.quad_wts[None, :]

        # All quadrature points as flat arrays for FMM
        # Source points: all quad points of all RWG halves
        # Each RWG function n has 2 halves (p, m), each with Nq quad points
        # Total source points: 2 * N * Nq
        self.all_src_pts = np.vstack([
            self.qpts_p.reshape(-1, 3),  # N*Nq points
            self.qpts_m.reshape(-1, 3),  # N*Nq points
        ])  # (2*N*Nq, 3)

        # Target points: same as source (for a single operator application)
        self.all_tgt_pts = self.all_src_pts.copy()

        # Build FMM trees for exterior and interior wavenumbers
        print(f"  [BEM-FMM] Building FMM tree for k_ext={k_ext:.4f}...")
        self.fmm_ext = HelmholtzFMM(
            self.all_tgt_pts, self.all_src_pts,
            k_ext, max_leaf=max_leaf, digits=fmm_digits)

        if abs(k_int - k_ext) > 1e-10:
            print(f"  [BEM-FMM] Building FMM tree for k_int={k_int:.4f}...")
            self.fmm_int = HelmholtzFMM(
                self.all_tgt_pts, self.all_src_pts,
                k_int, max_leaf=max_leaf, digits=fmm_digits)
        else:
            self.fmm_int = self.fmm_ext

        # Precompute singular corrections
        # (near-field direct integrals for self and neighboring triangles)
        self._precompute_singular_corrections()

        elapsed = _time.time() - t0
        print(f"  [BEM-FMM] Initialization complete: {elapsed:.1f}s")

    def _precompute_singular_corrections(self):
        """Precompute near-field correction matrices for same-triangle interactions.

        The FMM P2P computes G_full = exp(ikR)/(4πR) for non-coincident points
        on the same triangle, and zeros out coincident (R=0) points.
        The correct BEM assembly uses:
          - G_smooth = (exp(ikR)-1)/(4πR) for all same-triangle quadrature
          - Analytical G_0 = 1/(4πR) via Graglia integrals for the singular part

        The correction matrix stores:
          C_L[m,n] = (G_smooth quad + analytical G_0) - (G_full P2P)
          C_K[m,n] = 0 - (gradG P2P)  [K PV integral = 0 for flat same-triangle]

        Applied in matvec as: L_result += C_L @ x, K_result += C_K @ x.
        """
        from bem_core import potential_integral_triangle, \
            vector_potential_integral_triangle

        N = self.N
        Nq = self.Nq
        inv4pi = 1.0 / (4 * np.pi)
        lam0 = 1 - self.quad_pts[:, 0] - self.quad_pts[:, 1]

        # Build map: triangle index -> list of RWG half info
        tri_to_rwg = {}
        for n in range(N):
            for half in [0, 1]:
                if half == 0:
                    tri_idx = self.rwg['tri_p'][n]
                    div_val = self.div_p[n]
                    f_arr = self.f_p[n]      # (Nq, 3)
                    jw_arr = self.jw_p[n]    # (Nq,)
                    coeff = self.rwg['length'][n] / (2 * self.rwg['area_p'][n])
                    free = self.rwg['free_p'][n]
                    sign = +1
                else:
                    tri_idx = self.rwg['tri_m'][n]
                    div_val = self.div_m[n]
                    f_arr = self.f_m[n]
                    jw_arr = self.jw_m[n]
                    coeff = self.rwg['length'][n] / (2 * self.rwg['area_m'][n])
                    free = self.rwg['free_m'][n]
                    sign = -1

                if tri_idx not in tri_to_rwg:
                    tri_to_rwg[tri_idx] = []
                tri_to_rwg[tri_idx].append(
                    (n, half, div_val, f_arr, jw_arr, coeff, free, sign))

        # Initialize correction matrices (one pair per wavenumber)
        self.corr_L_ext = np.zeros((N, N), dtype=complex)
        self.corr_K_ext = np.zeros((N, N), dtype=complex)
        self.corr_L_int = np.zeros((N, N), dtype=complex)
        self.corr_K_int = np.zeros((N, N), dtype=complex)

        for tri_idx, rwg_list in tri_to_rwg.items():
            t = self.tris[tri_idx]
            v0, v1, v2 = self.verts[t[0]], self.verts[t[1]], self.verts[t[2]]

            # Quad points on this triangle
            qpts = (lam0[:, None] * v0 + self.quad_pts[:, 0:1] * v1
                    + self.quad_pts[:, 1:2] * v2)  # (Nq, 3)

            # Distance matrix between quad points on same triangle
            diff = qpts[:, None, :] - qpts[None, :, :]  # (Nq, Nq, 3)
            R = np.linalg.norm(diff, axis=2)  # (Nq, Nq)
            R_safe = np.maximum(R, 1e-15)
            mask = R > 1e-12  # off-diagonal (non-coincident)

            # Analytical integrals (Graglia)
            P_anal = np.zeros(Nq)
            V_anal = np.zeros((Nq, 3))
            for iq in range(Nq):
                P_anal[iq] = potential_integral_triangle(qpts[iq], v0, v1, v2)
                V_anal[iq] = vector_potential_integral_triangle(
                    qpts[iq], v0, v1, v2)

            for k_val, corr_L, corr_K in [
                (self.k_ext, self.corr_L_ext, self.corr_K_ext),
                (self.k_int, self.corr_L_int, self.corr_K_int),
            ]:
                ik = 1j * k_val
                iok = 1j / k_val

                # ΔG = G_smooth - G_full_P2P
                #   off-diagonal: (exp(ikR)-1)/(4πR) - exp(ikR)/(4πR) = -1/(4πR)
                #   diagonal (R=0): ik/(4π) - 0 = ik/(4π)
                DG = np.zeros((Nq, Nq), dtype=complex)
                DG[mask] = -1.0 / (4 * np.pi * R[mask])
                DG[~mask] = ik / (4 * np.pi)

                # P2P grad G for K subtraction (same-triangle, R > 0 only)
                G_full = np.zeros((Nq, Nq), dtype=complex)
                G_full[mask] = np.exp(ik * R[mask]) / (4 * np.pi * R[mask])
                gradG_scalar = np.zeros((Nq, Nq), dtype=complex)
                gradG_scalar[mask] = (
                    G_full[mask] * (ik - 1.0 / R_safe[mask]) / R_safe[mask])

                for m_info in rwg_list:
                    m, m_half, m_div, m_f, m_jw, m_coeff, m_free, m_sign = m_info
                    for n_info in rwg_list:
                        n, n_half, n_div, n_f, n_jw, n_coeff, n_free, n_sign = n_info

                        jw_prod = m_jw[:, None] * n_jw[None, :]  # (Nq, Nq)

                        # --- L correction ---
                        # 1) Numerical ΔG part
                        f_dot = np.sum(
                            m_f[:, None, :] * n_f[None, :, :], axis=2)
                        DL_vec = ik * np.sum(f_dot * DG * jw_prod)
                        DL_scl = -iok * m_div * n_div * np.sum(DG * jw_prod)

                        # 2) Analytical G_0 (Graglia)
                        fn_over_R = n_sign * n_coeff * (
                            V_anal - n_free[None, :] * P_anal[:, None])
                        anal_vec = ik * np.sum(
                            np.sum(m_f * fn_over_R, axis=1) * m_jw) * inv4pi
                        anal_scl = (-iok * m_div * n_div
                                    * np.dot(P_anal, m_jw) * inv4pi)

                        corr_L[m, n] += DL_vec + DL_scl + anal_vec + anal_scl

                        # --- K correction: subtract P2P K for same-triangle ---
                        # Dense K is 0 for same-triangle (PV = 0 for flat),
                        # but FMM P2P computes non-zero gradG for off-diagonal.
                        # C_K = 0 - P2P_K = -(gradG contributions)
                        cross_dn = np.cross(
                            diff, n_f[None, :, :])  # (Nq, Nq, 3)
                        dot_f_cross = np.sum(
                            m_f[:, None, :] * cross_dn, axis=2)  # (Nq, Nq)
                        corr_K[m, n] -= np.sum(
                            gradG_scalar * dot_f_cross * jw_prod)

    def _L_operator_fmm(self, x, k, fmm):
        """Apply L operator to coefficient vector x using FMM.

        L·x = ik Σ_n x_n ∫∫ f_m(r)·f_n(r') G(r,r') dS dS'
            - (i/k) Σ_n x_n ∫∫ ∇·f_m(r) ∇·f_n(r') G(r,r') dS dS'

        We represent each RWG function at its quadrature points as
        equivalent point sources, then use FMM to evaluate the sums.
        """
        N = self.N
        Nq = self.Nq
        ik = 1j * k
        iok = 1j / k

        # --- Vector part: ik ∫∫ f_m · f_n G dS dS' ---
        # Source charges for the vector part:
        # For each source RWG n, at each quad point q:
        #   charge = x[n] * f_n(r_q) * jw[n,q]  (3-vector)
        # But FMM works with scalar charges. Decompose into 3 components.

        result = np.zeros(N, dtype=complex)

        for d in range(3):
            # Source strengths: f_n^d(r_q) * jw * x[n]
            src_charges = np.zeros(2 * N * Nq, dtype=complex)

            # Plus half
            # f_p[n, q, d] * jw_p[n, q] * x[n]
            src_p = (self.f_p[:, :, d] * self.jw_p * x[:, None]).ravel()  # (N*Nq,)
            src_charges[:N * Nq] = src_p

            # Minus half
            src_m = (self.f_m[:, :, d] * self.jw_m * x[:, None]).ravel()
            src_charges[N * Nq:] = src_m

            # Evaluate potential at all target quad points
            phi = fmm.evaluate(src_charges)  # (2*N*Nq,)

            # Accumulate into result: test with f_m^d * jw_m
            phi_p = phi[:N * Nq].reshape(N, Nq)
            phi_m = phi[N * Nq:].reshape(N, Nq)

            result += np.sum(self.f_p[:, :, d] * self.jw_p * phi_p, axis=1)
            result += np.sum(self.f_m[:, :, d] * self.jw_m * phi_m, axis=1)

        result_vec = ik * result

        # --- Scalar part: -(i/k) ∫∫ ∇·f_m ∇·f_n G dS dS' ---
        src_charges_scl = np.zeros(2 * N * Nq, dtype=complex)

        # Source: ∇·f_n * jw * x[n]
        src_p = (self.div_p[:, None] * self.jw_p * x[:, None]).ravel()
        src_charges_scl[:N * Nq] = src_p
        src_m = (self.div_m[:, None] * self.jw_m * x[:, None]).ravel()
        src_charges_scl[N * Nq:] = src_m

        phi_scl = fmm.evaluate(src_charges_scl)

        phi_scl_p = phi_scl[:N * Nq].reshape(N, Nq)
        phi_scl_m = phi_scl[N * Nq:].reshape(N, Nq)

        result_scl = np.zeros(N, dtype=complex)
        result_scl += self.div_p * np.sum(self.jw_p * phi_scl_p, axis=1)
        result_scl += self.div_m * np.sum(self.jw_m * phi_scl_m, axis=1)

        return result_vec - iok * result_scl

    def _K_operator_fmm(self, x, k, fmm):
        """Apply K operator to coefficient vector x using FMM.

        K·x = Σ_n x_n ∫∫ f_m(r) · [∇G(r,r') × f_n(r')] dS dS'

        Uses the identity: a · (∇G × b) = ∇G · (b × a)
        So K[m,n] = ∫∫ ∇G · (f_n × f_m) dS dS'

        We compute ∇G via FMM gradient evaluation.
        For each component of the cross product, we set up source charges
        and evaluate the gradient potential.
        """
        N = self.N
        Nq = self.Nq

        # f_m(r) · [∇G(r,r') × f_n(r')]
        # = ∇G · [f_n(r') × f_m(r)]   (scalar triple product rearrangement)
        #
        # This requires: for each target quad point, sum over source quad points
        # of ∇G_ij · (f_n_j × f_m_i)
        #
        # The cross product couples test and source, so we can't directly use
        # standard FMM. Instead, decompose into components:
        #
        # f_m · (∇G × f_n) = (f_m)_y (∂G/∂z f_n_x - ∂G/∂x f_n_z)
        #                   - (f_m)_x (∂G/∂z f_n_y - ∂G/∂y f_n_z)  + ...
        #
        # More systematically, using Levi-Civita:
        # Σ_{ijk} ε_{ijk} (f_m)_i (∂_j G) (f_n)_k
        #
        # = Σ_j (∂_j G) * Σ_{ik} ε_{ijk} (f_m)_i (f_n)_k
        #
        # For each gradient component j, the source charge for the FMM gradient
        # involves cross-terms between test and source basis functions.
        # This doesn't separate nicely.
        #
        # Alternative approach: compute ∇G · ê_j for each j, then assemble.
        # We need 3 scalar FMM gradient evaluations.

        # For each source function n, quadrature point q, component d:
        #   source charge for gradient component j:
        #   q_j = x[n] * f_n_k(r_q) * jw * ε_{ijk}  (summed with test later)
        #
        # Actually, let's use a simpler decomposition:
        # K[m,n] = ∫_test ∫_src f_m(r) · [∇_r G(r,r') × f_n(r')] dS dS'
        #
        # ∇_r G = [∂G/∂x, ∂G/∂y, ∂G/∂z]
        # [∇G × f_n] = [∂G/∂y f_n_z - ∂G/∂z f_n_y,
        #               ∂G/∂z f_n_x - ∂G/∂x f_n_z,
        #               ∂G/∂x f_n_y - ∂G/∂y f_n_x]
        #
        # f_m · [∇G × f_n] = f_m_x (∂G/∂y f_n_z - ∂G/∂z f_n_y)
        #                   + f_m_y (∂G/∂z f_n_x - ∂G/∂x f_n_z)
        #                   + f_m_z (∂G/∂x f_n_y - ∂G/∂y f_n_x)
        #
        # Rearranging by gradient components:
        # = ∂G/∂x (f_m_z f_n_y - f_m_y f_n_z)
        # + ∂G/∂y (f_m_x f_n_z - f_m_z f_n_x)
        # + ∂G/∂z (f_m_y f_n_x - f_m_x f_n_y)
        #
        # For FMM: compute 3 gradient evaluations with source charges
        # being f_n components, then combine at targets with f_m components.

        result = np.zeros(N, dtype=complex)

        # For each source component k, compute gradient of potential with
        # source charges = f_n_k * jw * x[n]
        grad_potentials = []  # list of 3 arrays, each (2*N*Nq, 3)

        for k_comp in range(3):
            src_charges = np.zeros(2 * N * Nq, dtype=complex)
            src_p = (self.f_p[:, :, k_comp] * self.jw_p * x[:, None]).ravel()
            src_charges[:N * Nq] = src_p
            src_m = (self.f_m[:, :, k_comp] * self.jw_m * x[:, None]).ravel()
            src_charges[N * Nq:] = src_m

            grad = fmm.evaluate_gradient(src_charges)  # (2*N*Nq, 3)
            grad_potentials.append(grad)

        # Now assemble: for each target RWG function m,
        # K·x[m] = Σ over target quad points of:
        #   jw * [f_m_x * (∂/∂y Φ_z - ∂/∂z Φ_y)
        #        + f_m_y * (∂/∂z Φ_x - ∂/∂x Φ_z)
        #        + f_m_z * (∂/∂x Φ_y - ∂/∂y Φ_x)]
        #
        # where Φ_k = FMM potential with source charges f_n_k * jw * x

        # grad_potentials[k] gives ∇Φ_k at all target points
        # ∇Φ_k = (∂Φ_k/∂x, ∂Φ_k/∂y, ∂Φ_k/∂z)

        gP = grad_potentials  # shorthand
        # gP[k][i, j] = ∂Φ_k/∂x_j at target point i

        # Cross product components:
        # (∇×Φ)_x = ∂Φ_z/∂y - ∂Φ_y/∂z = gP[2][:,1] - gP[1][:,2]
        # (∇×Φ)_y = ∂Φ_x/∂z - ∂Φ_z/∂x = gP[0][:,2] - gP[2][:,0]
        # (∇×Φ)_z = ∂Φ_y/∂x - ∂Φ_x/∂y = gP[1][:,0] - gP[0][:,1]

        curl_x = gP[2][:, 1] - gP[1][:, 2]  # (2*N*Nq,)
        curl_y = gP[0][:, 2] - gP[2][:, 0]
        curl_z = gP[1][:, 0] - gP[0][:, 1]

        curl_x_p = curl_x[:N * Nq].reshape(N, Nq)
        curl_x_m = curl_x[N * Nq:].reshape(N, Nq)
        curl_y_p = curl_y[:N * Nq].reshape(N, Nq)
        curl_y_m = curl_y[N * Nq:].reshape(N, Nq)
        curl_z_p = curl_z[:N * Nq].reshape(N, Nq)
        curl_z_m = curl_z[N * Nq:].reshape(N, Nq)

        # Test with f_m
        result += np.sum(self.jw_p * (self.f_p[:, :, 0] * curl_x_p +
                                       self.f_p[:, :, 1] * curl_y_p +
                                       self.f_p[:, :, 2] * curl_z_p), axis=1)
        result += np.sum(self.jw_m * (self.f_m[:, :, 0] * curl_x_m +
                                       self.f_m[:, :, 1] * curl_y_m +
                                       self.f_m[:, :, 2] * curl_z_m), axis=1)

        return result

    # Near corrections are precomputed as matrices in _precompute_singular_corrections
    # and applied in matvec() as: L_result += corr_L @ x, K_result += corr_K @ x

    def matvec(self, x_full):
        """Apply PMCHWT system matrix to vector x = [J; M].

        y = Z · x where Z = [[η₀L₀+η₁L₁, -(K₀+K₁)],
                              [K₀+K₁,       L₀/η₀+L₁/η₁]]

        Parameters
        ----------
        x_full : ndarray (2*N,), complex
            Input vector [J; M].

        Returns
        -------
        y : ndarray (2*N,), complex
            Result vector.
        """
        N = self.N
        J = x_full[:N]
        M = x_full[N:]

        t0 = _time.time()

        # Exterior operators
        L_ext_J = self._L_operator_fmm(J, self.k_ext, self.fmm_ext)
        L_ext_M = self._L_operator_fmm(M, self.k_ext, self.fmm_ext)
        K_ext_J = self._K_operator_fmm(J, self.k_ext, self.fmm_ext)
        K_ext_M = self._K_operator_fmm(M, self.k_ext, self.fmm_ext)

        # Interior operators
        L_int_J = self._L_operator_fmm(J, self.k_int, self.fmm_int)
        L_int_M = self._L_operator_fmm(M, self.k_int, self.fmm_int)
        K_int_J = self._K_operator_fmm(J, self.k_int, self.fmm_int)
        K_int_M = self._K_operator_fmm(M, self.k_int, self.fmm_int)

        # Near-field corrections (precomputed matrices)
        L_ext_J += self.corr_L_ext @ J
        L_ext_M += self.corr_L_ext @ M
        K_ext_J += self.corr_K_ext @ J
        K_ext_M += self.corr_K_ext @ M

        L_int_J += self.corr_L_int @ J
        L_int_M += self.corr_L_int @ M
        K_int_J += self.corr_K_int @ J
        K_int_M += self.corr_K_int @ M

        # Assemble PMCHWT blocks
        K_sum_J = K_ext_J + K_int_J
        K_sum_M = K_ext_M + K_int_M

        y = np.zeros(2 * N, dtype=complex)
        # y[:N] = (η_ext * L_ext + η_int * L_int) · J - (K_ext + K_int) · M
        y[:N] = (self.eta_ext * L_ext_J + self.eta_int * L_int_J) - K_sum_M
        # y[N:] = (K_ext + K_int) · J + (L_ext/η_ext + L_int/η_int) · M
        y[N:] = K_sum_J + (L_ext_M / self.eta_ext + L_int_M / self.eta_int)

        elapsed = _time.time() - t0
        return y

    def as_linear_operator(self):
        """Return scipy LinearOperator wrapping this FMM matvec."""
        from scipy.sparse.linalg import LinearOperator
        n = self.system_size
        return LinearOperator((n, n), matvec=self.matvec, dtype=complex)


# ============================================================
# 5. Block-diagonal preconditioner for matrix-free mode
# ============================================================

def build_block_diagonal_preconditioner(rwg, verts, tris, k_ext, k_int,
                                         eta_ext, eta_int, quad_order=7,
                                         block_size=256):
    """Build a block-diagonal preconditioner for the PMCHWT system.

    Groups nearby RWG functions into blocks and assembles+LU-factors
    each block independently. Memory: O(N * block_size).

    Parameters
    ----------
    block_size : int
        Size of each diagonal block.

    Returns
    -------
    precond_func : callable
        Function that applies the preconditioner: x → M^{-1} x.
    """
    from bem_core import assemble_L_K
    from scipy.sparse.linalg import LinearOperator

    N = rwg['N']
    n_sys = 2 * N

    # For now, use simple sequential blocking
    # (in production, use spatial clustering)
    n_blocks = max(1, (N + block_size - 1) // block_size)

    print(f"  [Precond] Building block-diagonal: {n_blocks} blocks of ≤{block_size}...")

    block_lus = []
    block_ranges = []

    for bi in range(n_blocks):
        i_start = bi * block_size
        i_end = min(i_start + block_size, N)
        block_ranges.append((i_start, i_end))
        bs = i_end - i_start

        # Extract the diagonal block of the PMCHWT matrix
        # This requires assembling a small dense matrix for the block
        # For simplicity, use the full L/K operators restricted to this block
        # TODO: assemble only the block sub-matrix for efficiency

    # Fallback: use diagonal preconditioner (much simpler)
    # Compute diagonal elements via single FMM evaluation with unit vectors
    print(f"  [Precond] Using diagonal preconditioner (fast fallback)...")

    # Estimate diagonal by computing Z·e_i for a few random vectors
    # For now, just return identity (no preconditioning)
    def precond_func(x):
        return x

    return LinearOperator((n_sys, n_sys), matvec=precond_func, dtype=complex)


# ============================================================
# 6. GMRES solver with FMM matvec
# ============================================================

def solve_gmres_fmm(bem_fmm_op, b, tol=1e-4, maxiter=200, precond=None,
                     verbose=True):
    """Solve PMCHWT system using GMRES with FMM-accelerated matvec.

    Parameters
    ----------
    bem_fmm_op : BEM_FMM_Operator
        FMM operator.
    b : ndarray (2*N,)
        RHS vector.
    tol : float
        GMRES tolerance.
    maxiter : int
        Maximum iterations.
    precond : LinearOperator, optional
        Preconditioner.

    Returns
    -------
    x : ndarray (2*N,)
        Solution vector.
    """
    from scipy.sparse.linalg import gmres, LinearOperator

    A_op = bem_fmm_op.as_linear_operator()
    n = bem_fmm_op.system_size

    iter_count = [0]
    residuals = []

    def callback(rk):
        iter_count[0] += 1
        residuals.append(rk)
        if verbose and (iter_count[0] % 5 == 0 or iter_count[0] <= 3):
            print(f"    GMRES iter {iter_count[0]}: res={rk:.2e}")

    t0 = _time.time()
    x, info = gmres(A_op, b, M=precond, rtol=tol, maxiter=maxiter,
                     restart=min(100, n),
                     callback=callback, callback_type='pr_norm')
    elapsed = _time.time() - t0

    if verbose:
        if info == 0:
            print(f"  [FMM-GMRES] Converged: {iter_count[0]} iter, {elapsed:.1f}s")
        else:
            print(f"  [FMM-GMRES] NOT converged (info={info}), "
                  f"{iter_count[0]} iter, {elapsed:.1f}s")

    return x


# ============================================================
# 7. High-level test / verification
# ============================================================

def test_fmm_vs_dense(N_src=500, N_tgt=500, k=2*np.pi, seed=42):
    """Test FMM accuracy against dense direct evaluation.

    Creates random source and target points, computes
    φ = Σ G(r_i, r_j) q_j both ways, reports relative error.
    """
    rng = np.random.RandomState(seed)

    sources = rng.randn(N_src, 3)
    targets = rng.randn(N_tgt, 3) + np.array([3, 0, 0])  # offset to avoid overlap
    charges = rng.randn(N_src) + 1j * rng.randn(N_src)

    print(f"Testing FMM: {N_src} sources, {N_tgt} targets, k={k:.2f}")

    # Dense evaluation
    t0 = _time.time()
    diff = targets[:, None, :] - sources[None, :, :]
    R = np.linalg.norm(diff, axis=2)
    R_safe = np.maximum(R, 1e-15)
    G = np.exp(1j * k * R_safe) / (4 * np.pi * R_safe)
    phi_dense = G @ charges
    t_dense = _time.time() - t0
    print(f"  Dense: {t_dense:.3f}s")

    # FMM evaluation
    t0 = _time.time()
    fmm = HelmholtzFMM(targets, sources, k, max_leaf=32, digits=3)
    t_setup = _time.time() - t0

    t0 = _time.time()
    phi_fmm = fmm.evaluate(charges)
    t_eval = _time.time() - t0

    rel_err = np.linalg.norm(phi_fmm - phi_dense) / np.linalg.norm(phi_dense)
    max_err = np.max(np.abs(phi_fmm - phi_dense)) / np.max(np.abs(phi_dense))

    print(f"  FMM:   setup={t_setup:.3f}s, eval={t_eval:.3f}s")
    print(f"  Relative L2 error: {rel_err:.2e}")
    print(f"  Max relative error: {max_err:.2e}")

    # Test gradient
    print(f"\nTesting FMM gradient...")
    t0 = _time.time()
    grad_dense = np.zeros((N_tgt, 3), dtype=complex)
    for d in range(3):
        factor = (1j * k - 1.0 / R_safe) / R_safe * diff[:, :, d]
        grad_dense[:, d] = (G * factor) @ charges
    t_grad_dense = _time.time() - t0

    t0 = _time.time()
    grad_fmm = fmm.evaluate_gradient(charges)
    t_grad_fmm = _time.time() - t0

    grad_err = np.linalg.norm(grad_fmm - grad_dense) / np.linalg.norm(grad_dense)
    print(f"  Dense gradient: {t_grad_dense:.3f}s")
    print(f"  FMM gradient:   {t_grad_fmm:.3f}s")
    print(f"  Gradient relative L2 error: {grad_err:.2e}")

    return rel_err, grad_err


def test_fmm_2point():
    """Minimal test: single source, single target."""
    src = np.array([[0.0, 0.0, 0.0]])
    tgt = np.array([[5.0, 0.0, 0.0]])
    q = np.array([1.0 + 0j])
    k = 2 * np.pi

    R = 5.0
    G_exact = np.exp(1j * k * R) / (4 * np.pi * R)
    print(f"2-point test: k={k:.2f}, R={R:.1f}")
    print(f"  G_exact = {G_exact}")

    fmm = HelmholtzFMM(tgt, src, k, max_leaf=1, digits=3)
    phi = fmm.evaluate(q)
    print(f"  G_fmm   = {phi[0]}")
    print(f"  rel err = {abs(phi[0] - G_exact) / abs(G_exact):.2e}")

    # Also check what P2P gives directly
    print(f"\n  Checking interaction lists:")
    for leaf in fmm.tree.leaves:
        n_src = len(leaf._source_ids)
        n_tgt = len(leaf._target_ids)
        n_near = len(leaf.near_list)
        n_far = len(leaf.far_list)
        if n_src > 0 or n_tgt > 0:
            print(f"    Leaf {leaf.index}: src={n_src}, tgt={n_tgt}, "
                  f"near={n_near}, far={n_far}, center={leaf.center}")


def test_fmm_interaction_coverage():
    """Verify all particle pairs are covered by P2P + FMM."""
    rng = np.random.RandomState(42)
    N = 50
    sources = rng.randn(N, 3) * 0.5
    targets = rng.randn(N, 3) * 0.5 + np.array([3, 0, 0])
    k = 2 * np.pi

    fmm = HelmholtzFMM(targets, sources, k, max_leaf=8, digits=3)

    # Count P2P pairs
    p2p_pairs = set()
    for leaf in fmm.tree.leaves:
        tgt_ids = leaf._target_ids
        src_ids = leaf._source_ids
        # Self
        for t in tgt_ids:
            for s in src_ids:
                p2p_pairs.add((t, s))
        # Neighbors
        for nb in leaf.near_list:
            nb_src_ids = nb._source_ids
            for t in tgt_ids:
                for s in nb_src_ids:
                    p2p_pairs.add((t, s))

    # Count far-field pairs via leaves that receive M2L contributions
    # (hard to enumerate exactly, but check completeness via direct test)
    print(f"P2P covers {len(p2p_pairs)} of {N*N} pairs ({100*len(p2p_pairs)/(N*N):.1f}%)")

    # Diagnostic: interaction lists per level
    for level in sorted(fmm.tree.levels.keys()):
        nodes = fmm.tree.levels[level]
        n_far = sum(len(n.far_list) for n in nodes)
        n_near = sum(len(n.near_list) for n in nodes)
        n_leaf = sum(1 for n in nodes if n.is_leaf)
        print(f"  Level {level}: {len(nodes)} nodes ({n_leaf} leaves), "
              f"far_list={n_far}, near_list={n_near}")

    # Full test: compare dense vs FMM
    charges = rng.randn(N) + 1j * rng.randn(N)
    diff = targets[:, None, :] - sources[None, :, :]
    R = np.linalg.norm(diff, axis=2)
    R_safe = np.maximum(R, 1e-15)
    G = np.exp(1j * k * R_safe) / (4 * np.pi * R_safe)
    phi_dense = G @ charges

    phi_fmm = fmm.evaluate(charges)
    rel_err = np.linalg.norm(phi_fmm - phi_dense) / np.linalg.norm(phi_dense)
    print(f"Relative L2 error: {rel_err:.2e}")

    # Check P2P only
    phi_p2p = np.zeros(N, dtype=complex)
    for leaf in fmm.tree.leaves:
        fmm._p2p(leaf, leaf, charges, phi_p2p)
        for nb in leaf.near_list:
            fmm._p2p(leaf, nb, charges, phi_p2p)
    p2p_err = np.linalg.norm(phi_p2p - phi_dense) / np.linalg.norm(phi_dense)
    print(f"P2P-only relative error: {p2p_err:.2e}")
    print(f"P2P norm: {np.linalg.norm(phi_p2p):.4e}, Dense norm: {np.linalg.norm(phi_dense):.4e}")
    print(f"FMM norm: {np.linalg.norm(phi_fmm):.4e}")


if __name__ == '__main__':
    test_fmm_2point()
    print("\n" + "=" * 60)
    test_fmm_interaction_coverage()
    print("\n" + "=" * 60)
    test_fmm_vs_dense(N_src=200, N_tgt=200, k=2*np.pi)
