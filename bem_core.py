"""
Minimal BEM/SIE solver for electromagnetic scattering (PMCHWT formulation).
Uses RWG basis functions on triangular surface mesh.
Singularity handled via extraction + analytical inner integral (Graglia 1993).
"""

import numpy as np
from scipy import linalg
from scipy.sparse.linalg import gmres as sp_gmres
from concurrent.futures import ThreadPoolExecutor
import time as _time


# ============================================================
# 1. Mesh generation (icosphere)
# ============================================================

def icosphere(radius=1.0, refinements=2):
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
    ], dtype=np.float64)
    verts /= np.linalg.norm(verts[0])

    tris = np.array([
        [0,11,5], [0,5,1], [0,1,7], [0,7,10], [0,10,11],
        [1,5,9], [5,11,4], [11,10,2], [10,7,6], [7,1,8],
        [3,9,4], [3,4,2], [3,2,6], [3,6,8], [3,8,9],
        [4,9,5], [2,4,11], [6,2,10], [8,6,7], [9,8,1],
    ], dtype=np.int32)

    for _ in range(refinements):
        verts_list = list(verts)
        edge_midpoints = {}
        def get_midpoint(v0, v1):
            key = (min(v0, v1), max(v0, v1))
            if key in edge_midpoints:
                return edge_midpoints[key]
            mid = (np.array(verts_list[v0]) + np.array(verts_list[v1])) / 2
            mid /= np.linalg.norm(mid)
            idx = len(verts_list)
            verts_list.append(mid)
            edge_midpoints[key] = idx
            return idx
        new_tris = []
        for tri in tris:
            a, b, c = tri
            ab = get_midpoint(a, b); bc = get_midpoint(b, c); ca = get_midpoint(c, a)
            new_tris.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]])
        verts = np.array(verts_list)
        tris = np.array(new_tris, dtype=np.int32)
    verts *= radius
    return verts, tris


def load_mesh(filename):
    """Load triangular surface mesh from file.

    Supported formats:
      - STL (.stl) — binary and ASCII
      - OBJ (.obj) — triangular faces only
      - Gmsh (.msh) — version 2 and 4

    Returns (verts, tris): vertex array (V×3) and triangle index array (T×3).
    The mesh must be a closed surface (no boundary edges).
    """
    ext = filename.rsplit('.', 1)[-1].lower()
    if ext == 'stl':
        return _load_stl(filename)
    elif ext == 'obj':
        return _load_obj(filename)
    elif ext == 'msh':
        return _load_gmsh(filename)
    else:
        raise ValueError(f"Unsupported mesh format: .{ext}  (use .stl, .obj, or .msh)")


def _load_stl(filename):
    """Load STL file (binary or ASCII)."""
    with open(filename, 'rb') as f:
        header = f.read(80)
        # Try binary first
        n_tri_bytes = f.read(4)
        if len(n_tri_bytes) < 4:
            raise ValueError("Invalid STL file")
        n_tri = int.from_bytes(n_tri_bytes, 'little')
        expected_size = 80 + 4 + n_tri * 50
        f.seek(0, 2)
        file_size = f.tell()

    if file_size == expected_size:
        # Binary STL
        import struct
        with open(filename, 'rb') as f:
            f.read(84)  # skip header + count
            all_verts = []
            for _ in range(n_tri):
                data = f.read(50)
                vals = struct.unpack('<12fH', data)
                # vals[0:3] = normal, vals[3:6] = v0, vals[6:9] = v1, vals[9:12] = v2
                all_verts.append(vals[3:6])
                all_verts.append(vals[6:9])
                all_verts.append(vals[9:12])
    else:
        # ASCII STL
        all_verts = []
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('vertex'):
                    parts = line.split()
                    all_verts.append((float(parts[1]), float(parts[2]), float(parts[3])))

    # Merge duplicate vertices
    all_verts = np.array(all_verts, dtype=np.float64)
    verts, inverse = np.unique(all_verts.round(decimals=10), axis=0, return_inverse=True)
    tris = inverse.reshape(-1, 3).astype(np.int32)
    return verts, tris


def _load_obj(filename):
    """Load OBJ file (triangular faces)."""
    verts = []
    tris = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('v '):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith('f '):
                parts = line.split()[1:]
                # OBJ indices are 1-based; may have v/vt/vn format
                indices = [int(p.split('/')[0]) - 1 for p in parts]
                if len(indices) == 3:
                    tris.append(indices)
                elif len(indices) == 4:
                    # Quad -> 2 triangles
                    tris.append([indices[0], indices[1], indices[2]])
                    tris.append([indices[0], indices[2], indices[3]])
                else:
                    raise ValueError(f"Face with {len(indices)} vertices (need 3 or 4)")
    return np.array(verts, dtype=np.float64), np.array(tris, dtype=np.int32)


def _load_gmsh(filename):
    """Load Gmsh .msh file (v2 or v4, triangles only)."""
    with open(filename, 'r') as f:
        content = f.read()

    lines = content.split('\n')
    i = 0
    verts = None
    tris = []

    while i < len(lines):
        line = lines[i].strip()

        if line == '$MeshFormat':
            i += 1
            version = float(lines[i].split()[0])
            i += 1  # $EndMeshFormat
            i += 1

        elif line == '$Nodes':
            i += 1
            if version >= 4.0:
                # v4 format: num_entity_blocks num_nodes ...
                parts = lines[i].split()
                n_blocks = int(parts[0])
                n_nodes = int(parts[1])
                verts = np.zeros((n_nodes, 3))
                i += 1
                for _ in range(n_blocks):
                    parts = lines[i].split()
                    n_in_block = int(parts[3])
                    i += 1
                    node_ids = []
                    for _ in range(n_in_block):
                        node_ids.append(int(lines[i].strip()) - 1)
                        i += 1
                    for j, nid in enumerate(node_ids):
                        parts = lines[i].split()
                        verts[nid] = [float(parts[0]), float(parts[1]), float(parts[2])]
                        i += 1
            else:
                # v2 format
                n_nodes = int(lines[i].strip())
                verts = np.zeros((n_nodes, 3))
                i += 1
                for _ in range(n_nodes):
                    parts = lines[i].split()
                    nid = int(parts[0]) - 1
                    verts[nid] = [float(parts[1]), float(parts[2]), float(parts[3])]
                    i += 1
            i += 1  # $EndNodes

        elif line == '$Elements':
            i += 1
            if version >= 4.0:
                parts = lines[i].split()
                n_blocks = int(parts[0])
                i += 1
                for _ in range(n_blocks):
                    parts = lines[i].split()
                    elem_type = int(parts[2])
                    n_in_block = int(parts[3])
                    i += 1
                    for _ in range(n_in_block):
                        parts = lines[i].split()
                        if elem_type == 2:  # triangle
                            tris.append([int(parts[1])-1, int(parts[2])-1, int(parts[3])-1])
                        i += 1
            else:
                n_elem = int(lines[i].strip())
                i += 1
                for _ in range(n_elem):
                    parts = lines[i].split()
                    elem_type = int(parts[1])
                    n_tags = int(parts[2])
                    if elem_type == 2:  # triangle
                        idx = 3 + n_tags
                        tris.append([int(parts[idx])-1, int(parts[idx+1])-1, int(parts[idx+2])-1])
                    i += 1
            i += 1  # $EndElements

        else:
            i += 1

    if verts is None or len(tris) == 0:
        raise ValueError("No triangular mesh found in .msh file")

    return verts, np.array(tris, dtype=np.int32)


# ============================================================
# 2. RWG basis functions
# ============================================================

def build_rwg(verts, tris):
    from collections import defaultdict
    edge_to_tris = defaultdict(list)
    for ti, tri in enumerate(tris):
        for i in range(3):
            v0, v1 = tri[i], tri[(i + 1) % 3]
            edge_key = (min(v0, v1), max(v0, v1))
            edge_to_tris[edge_key].append((ti, tri[(i + 2) % 3]))

    tri_p, tri_m, free_p, free_m, lengths = [], [], [], [], []
    for edge_key, tri_list in edge_to_tris.items():
        if len(tri_list) == 2:
            (tp, vp), (tm, vm) = tri_list
            tri_p.append(tp); tri_m.append(tm)
            free_p.append(verts[vp]); free_m.append(verts[vm])
            lengths.append(np.linalg.norm(verts[edge_key[0]] - verts[edge_key[1]]))

    tri_p = np.array(tri_p); tri_m = np.array(tri_m)
    def areas(idx):
        t = tris[idx]; v0 = verts[t[:,0]]; v1 = verts[t[:,1]]; v2 = verts[t[:,2]]
        return 0.5 * np.linalg.norm(np.cross(v1-v0, v2-v0), axis=1)

    return {'N': len(tri_p), 'tri_p': tri_p, 'tri_m': tri_m,
            'free_p': np.array(free_p), 'free_m': np.array(free_m),
            'length': np.array(lengths), 'area_p': areas(tri_p), 'area_m': areas(tri_m)}


# ============================================================
# 3. Quadrature
# ============================================================

def tri_quadrature(order=7):
    if order == 4:
        a = 1/3; b1 = 0.6; b2 = 0.2
        pts = np.array([[a, a], [b1, b2], [b2, b1], [b2, b2]])
        wts = np.array([-27/48, 25/48, 25/48, 25/48])
    elif order == 7:
        a1, a2 = 0.059715871789770, 0.470142064105115
        b1, b2 = 0.797426985353087, 0.101286507323456
        pts = np.array([[1/3, 1/3], [a1, a2], [a2, a1], [a2, a2],
                         [b1, b2], [b2, b1], [b2, b2]])
        w1 = 0.225; w2 = 0.13239415278851; w3 = 0.12593918054483
        wts = np.array([w1, w2, w2, w2, w3, w3, w3])
    else:
        raise ValueError(f"Unsupported order {order}")
    return pts, wts


# ============================================================
# 4. Analytical potential integral (Graglia 1993)
# ============================================================

def potential_integral_triangle(r_obs, v0, v1, v2):
    """
    Compute ∫_T 1/|r_obs - r'| dS' analytically for a flat triangle T
    with vertices v0, v1, v2. Works for r_obs on or off the triangle plane.

    Based on Graglia, "On the numerical integration of the linear shape
    functions times the 3-D Green's function or its gradient on a plane
    triangle", IEEE TAP 1993.

    Returns scalar value of the integral.
    """
    # Triangle edges and their properties
    vertices = [v0, v1, v2]
    result = 0.0

    for i in range(3):
        # Edge from vertex i to vertex (i+1)%3
        vi = vertices[i]
        vj = vertices[(i + 1) % 3]

        # Edge vector and length
        edge = vj - vi
        l_edge = np.linalg.norm(edge)
        if l_edge < 1e-15:
            continue
        t_hat = edge / l_edge  # tangent unit vector

        # Normal to the triangle
        n_tri = np.cross(v1 - v0, v2 - v0)
        n_tri_norm = np.linalg.norm(n_tri)
        if n_tri_norm < 1e-15:
            return 0.0
        n_hat = n_tri / n_tri_norm

        # Inward normal to the edge (in the triangle plane, pointing into triangle)
        # Graglia convention: u_hat points toward the interior
        m_hat = np.cross(n_hat, t_hat)

        # Signed distance from r_obs to the edge line
        d = np.dot(r_obs - vi, m_hat)

        # Height above triangle plane
        h = np.dot(r_obs - vi, n_hat)

        # Projections of vertices onto edge direction
        s_plus = np.dot(vj - r_obs, t_hat)  # t-coordinate of vj
        s_minus = np.dot(vi - r_obs, t_hat)  # t-coordinate of vi

        # Distances from r_obs to edge endpoints
        R_plus = np.linalg.norm(vj - r_obs)
        R_minus = np.linalg.norm(vi - r_obs)

        # Contribution from this edge
        R0 = np.sqrt(d**2 + h**2)

        # ln term
        if R_plus + s_plus > 1e-15 and R_minus + s_minus > 1e-15:
            log_arg = (R_plus + s_plus) / (R_minus + s_minus)
            if log_arg > 0:
                result += d * np.log(log_arg)

        # arctan term (only if h != 0)
        if abs(h) > 1e-15:
            R0_sq = d**2 + h**2
            t1 = np.arctan2(d * s_plus, R0_sq + abs(h) * R_plus)
            t2 = np.arctan2(d * s_minus, R0_sq + abs(h) * R_minus)
            result -= abs(h) * (t1 - t2)

    return result


def gradient_potential_integral_triangle(r_obs, v0, v1, v2):
    """
    Compute ∇_r ∫_T 1/|r - r'| dS' analytically.
    This is the gradient of the potential integral w.r.t. the observation point.

    Returns a 3-vector.
    """
    # Use finite difference for simplicity (analytical formula exists but is complex)
    eps = 1e-6
    grad = np.zeros(3)
    for d in range(3):
        r_plus = r_obs.copy(); r_plus[d] += eps
        r_minus = r_obs.copy(); r_minus[d] -= eps
        grad[d] = (potential_integral_triangle(r_plus, v0, v1, v2) -
                   potential_integral_triangle(r_minus, v0, v1, v2)) / (2 * eps)
    return grad


def _integral_R_triangle(r_obs, v0, v1, v2):
    """Compute ∫_T |r_obs - r'| dS' numerically (7-pt quadrature)."""
    pts, wts = tri_quadrature(7)
    lam0 = 1 - pts[:, 0] - pts[:, 1]
    rr = np.outer(lam0, v0) + np.outer(pts[:, 0], v1) + np.outer(pts[:, 1], v2)
    R = np.linalg.norm(rr - r_obs[None, :], axis=1)
    area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
    return area * np.dot(wts, R)


def vector_potential_integral_triangle(r_obs, v0, v1, v2):
    """
    Compute ∫_T r'/|r_obs - r'| dS' using:
      ∫_T r'/R dS' = r_obs * P(r_obs) - ∇_r W(r_obs)
    where P = ∫ 1/R dS', W = ∫ R dS', ∇_r W = ∫ (r-r')/R dS'.

    Returns 3-vector.
    """
    P = potential_integral_triangle(r_obs, v0, v1, v2)
    # ∇_r W by finite differences on W
    eps = 1e-6
    grad_W = np.zeros(3)
    for d in range(3):
        rp = r_obs.copy(); rp[d] += eps
        rm = r_obs.copy(); rm[d] -= eps
        grad_W[d] = (_integral_R_triangle(rp, v0, v1, v2) -
                     _integral_R_triangle(rm, v0, v1, v2)) / (2 * eps)
    return r_obs * P - grad_W


# ============================================================
# 5. L and K operator assembly with singularity extraction
# ============================================================

def assemble_L_K(rwg, verts, tris, k, quad_order=7):
    """
    Assemble L and K operator matrices (vectorized over test dimension).

    L_mn = jk ∫∫ f_m·f_n G dS dS' - (j/k) ∫∫ (∇·f_m)(∇·f_n) G dS dS'
    K_mn = ∫∫ f_m · (∇G × f_n) dS dS'

    Singularity extraction for L: G = G_0 + G_smooth, where
    G_0 = 1/(4πR), G_smooth = (e^{ikR}-1)/(4πR)
    G_0 inner integral computed analytically, G_smooth with standard quadrature.

    K self-term: PV integral is zero for flat coplanar triangles, skip.
    """
    import time
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

    qpts_p = get_qpts_batch(rwg['tri_p'])  # (N, Nq, 3)
    qpts_m = get_qpts_batch(rwg['tri_m'])

    f_p = (rwg['length'][:, None, None] / (2 * rwg['area_p'][:, None, None])) * \
          (qpts_p - rwg['free_p'][:, None, :])
    f_m = -(rwg['length'][:, None, None] / (2 * rwg['area_m'][:, None, None])) * \
          (qpts_m - rwg['free_m'][:, None, :])

    div_p = rwg['length'] / rwg['area_p']       # ∇_s·f = l/A
    div_m = -rwg['length'] / rwg['area_m']

    jw_p = rwg['area_p'][:, None] * quad_wts[None, :]  # (N, Nq)
    jw_m = rwg['area_m'][:, None] * quad_wts[None, :]

    L = np.zeros((N, N), dtype=complex)
    K = np.zeros((N, N), dtype=complex)
    # Cross-term accumulators for symmetry exploitation
    L_cross = np.zeros((N, N), dtype=complex)
    K_cross = np.zeros((N, N), dtype=complex)

    # Adaptive batch size based on 2D flat array approach
    # Peak memory per batch element: ~N*Nq * 100 bytes (several 2D arrays)
    Nsq = N * Nq
    batch_size = max(1, min(N, int(3e9 / max(1, Nsq * Nq * 100))))

    t0 = time.time()
    print(f"    Assembly: {N} RWG, {Nq} quad pts, k={k:.4f}, batch={batch_size}...")

    # 3 passes instead of 4: (p,p), (p,m), (m,m). Skip (m,p) — use transpose of (p,m).
    # L and K are symmetric: L_mp[m,n] = L_pm[n,m], so L_mp = L_pm.T
    halves = [
        (qpts_p, f_p, div_p, jw_p, rwg['tri_p']),   # plus half
        (qpts_m, f_m, div_m, jw_m, rwg['tri_m']),    # minus half
    ]

    inv4pi = 1.0 / (4 * np.pi)
    ik = 1j * k
    ik_inv = 1j / k

    for th, (t_qpts_all, t_f_all, t_div_all, t_jw_all, t_tri_all) in enumerate(halves):
        for sh, (s_qpts, s_f, s_div, s_jw, s_tri) in enumerate(halves):
            if th > sh:
                continue  # skip (m,p) pass — use transpose of (p,m)

            is_cross = (th != sh)  # (p,m) cross-term
            tgt_L = L_cross if is_cross else L
            tgt_K = K_cross if is_cross else K

            # Precompute flat source arrays (constant across batches)
            sq_flat = s_qpts.reshape(Nsq, 3)      # (N*Nq, 3)
            sf_flat = s_f.reshape(Nsq, 3)          # (N*Nq, 3)
            sq_sq = np.sum(sq_flat**2, axis=1)     # (N*Nq,)
            jw_s_flat = s_jw.ravel()               # (N*Nq,)
            st_rep = np.repeat(s_tri, Nq)          # (N*Nq,)
            sq_cross_sf = np.cross(sq_flat, sf_flat)  # (N*Nq, 3) for K BLAS trick

            for b_start in range(0, N, batch_size):
                b_end = min(b_start + batch_size, N)
                B = b_end - b_start
                BNq = B * Nq

                tp = t_qpts_all[b_start:b_end]   # (B, Nq, 3)
                tf_batch = t_f_all[b_start:b_end] # (B, Nq, 3)
                td = t_div_all[b_start:b_end]     # (B,)
                tw = t_jw_all[b_start:b_end]      # (B, Nq)
                tt = t_tri_all[b_start:b_end]     # (B,)

                # Flatten test arrays
                tp_flat = tp.reshape(BNq, 3)         # (B*Nq, 3)
                tf_flat = tf_batch.reshape(BNq, 3)   # (B*Nq, 3)
                tp_sq = np.sum(tp_flat**2, axis=1)   # (B*Nq,)
                jw_t_flat = tw.ravel()               # (B*Nq,)

                # R via squared expansion (BLAS matrix multiply, no 5D tensor)
                tp_dot_sq = tp_flat @ sq_flat.T      # (B*Nq, N*Nq) - BLAS DGEMM
                R_sq = tp_sq[:, None] + sq_sq[None, :] - 2.0 * tp_dot_sq
                np.maximum(R_sq, 0.0, out=R_sq)
                R = np.sqrt(R_sq, out=R_sq)          # reuse buffer
                R_safe = np.maximum(R, 1e-15)

                # Green's function (use R_safe in exp to get correct G_smooth at R=0)
                expikR = np.exp(ik * R_safe)
                inv4piR = inv4pi / R_safe
                G_full = expikR * inv4piR             # correct for R > 0; huge at R=0 (OK: masked)

                # Singular mask (flat): test_tri[i//Nq] == src_tri[j//Nq]
                tt_rep = np.repeat(tt, Nq)            # (B*Nq,)
                sing_flat = (tt_rep[:, None] == st_rep[None, :])  # (BNq, Nsq)

                # G_use = G_smooth for singular, G_full for non-singular
                # G_smooth = (exp(ikR)-1)/(4πR) = G_full - 1/(4πR)
                # So G_use = G_full - sing_flat * 1/(4πR)
                G_use = G_full - sing_flat * inv4piR
                del expikR

                # Quadrature weights
                jw_flat = jw_t_flat[:, None] * jw_s_flat[None, :]  # (BNq, Nsq)

                # --- L operator ---
                f_dot = tf_flat @ sf_flat.T           # (BNq, Nsq) - BLAS DGEMM
                G_jw = G_use * jw_flat
                # Reshape to (B, Nq, N, Nq), sum over quad dims → (B, N)
                L_vec = (f_dot * G_jw).reshape(B, Nq, N, Nq).sum(axis=(1, 3))
                L_scl = G_jw.reshape(B, Nq, N, Nq).sum(axis=(1, 3))
                div_prod = td[:, None] * s_div[None, :]
                tgt_L[b_start:b_end, :] += ik * L_vec - ik_inv * div_prod * L_scl
                del f_dot, G_jw, L_vec, L_scl

                # --- K operator ---
                # gradG = G * (ik - 1/R) / R. Zero for R < 1e-12 and singular pairs.
                valid = (~sing_flat) & (R > 1e-12)
                gradG_coeff = np.where(valid, (ik - 1.0 / R_safe) / R_safe, 0.0)
                gG_jw = G_full * gradG_coeff * jw_flat  # (BNq, Nsq)
                del G_use, G_full, sing_flat, valid

                # Triple product: tf·((tp-sq)×sf) = cross(tf,tp)@sf.T - tf@cross(sq,sf).T
                # Uses BLAS DGEMM instead of element-wise 2D operations
                triple = np.cross(tf_flat, tp_flat) @ sf_flat.T - tf_flat @ sq_cross_sf.T

                tgt_K[b_start:b_end, :] += (gG_jw * triple).reshape(B, Nq, N, Nq).sum(axis=(1, 3))
                del R, R_safe, inv4piR, gG_jw, triple

    # Add cross-term and its transpose: L_pm + L_mp = L_cross + L_cross.T
    L += L_cross + L_cross.T
    K += K_cross + K_cross.T
    del L_cross, K_cross

    elapsed_main = time.time() - t0
    print(f"    Main loop: {elapsed_main:.1f}s")

    # --- Singular corrections (analytical G_0 part for L) ---
    t1 = time.time()

    # Cache triangle vertices
    tri_verts_cache = {}
    for ti in range(len(tris)):
        tri_verts_cache[ti] = (verts[tris[ti, 0]].copy(),
                                verts[tris[ti, 1]].copy(),
                                verts[tris[ti, 2]].copy())

    # Build triangle → RWG half map for efficient lookup
    tri_to_rwg_p = {}  # tri_idx → list of (rwg_idx, div, coeff, free, sign)
    tri_to_rwg_m = {}
    for n in range(N):
        tp = rwg['tri_p'][n]
        if tp not in tri_to_rwg_p:
            tri_to_rwg_p[tp] = []
        tri_to_rwg_p[tp].append((n, div_p[n],
                                  rwg['length'][n] / (2 * rwg['area_p'][n]),
                                  rwg['free_p'][n], +1))
        tm = rwg['tri_m'][n]
        if tm not in tri_to_rwg_m:
            tri_to_rwg_m[tm] = []
        tri_to_rwg_m[tm].append((n, div_m[n],
                                  rwg['length'][n] / (2 * rwg['area_m'][n]),
                                  rwg['free_m'][n], -1))

    # Precompute analytical integrals P and V for each unique triangle
    tri_PV_cache = {}  # tri_idx → (P_vals, V_vals)
    all_sing_tris = set(tri_to_rwg_p.keys()) | set(tri_to_rwg_m.keys())
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

    # Apply singular corrections
    for m in range(N):
        for t_f_h, t_div_h, t_jw_h, t_tri_idx in [
            (f_p[m], div_p[m], jw_p[m], rwg['tri_p'][m]),
            (f_m[m], div_m[m], jw_m[m], rwg['tri_m'][m]),
        ]:
            P_vals, V_vals = tri_PV_cache[t_tri_idx]
            scalar_base = np.dot(P_vals, t_jw_h) * inv4pi

            # Source halves on the same triangle
            for src_dict in [tri_to_rwg_p, tri_to_rwg_m]:
                if t_tri_idx not in src_dict:
                    continue
                for n, src_div_val, src_coeff, src_free, src_sign in src_dict[t_tri_idx]:
                    L_sing_scalar = -ik_inv * t_div_h * src_div_val * scalar_base
                    fn_over_R = src_sign * src_coeff * (V_vals - src_free[None, :] * P_vals[:, None])
                    vec_integral = np.sum(np.sum(t_f_h * fn_over_R, axis=1) * t_jw_h) * inv4pi
                    L[m, n] += L_sing_scalar + ik * vec_integral

    elapsed_sing = time.time() - t1
    print(f"    Singular corrections: {elapsed_sing:.1f}s")

    # Symmetrize (L and K are symmetric by construction; average reduces numerical noise)
    L = (L + L.T) / 2
    K = (K + K.T) / 2

    print(f"    Total assembly: {time.time() - t0:.1f}s")

    return L, K


# ============================================================
# 6. PMCHWT system
# ============================================================

def assemble_pmchwt(rwg, verts, tris, k_ext, k_int, eta_ext, eta_int,
                    quad_order=7, parallel=True):
    """Assemble PMCHWT system matrix and RHS.

    Standard PMCHWT sign convention:
      [ηL, -K; +K, L/η] · [J; M] = +b
    where K = K_ext + K_int, and b = [<f,E_inc>; <f,H_inc>].
    Far-field uses sM = -1 for the magnetic current contribution.

    parallel=True: assemble exterior and interior operators in parallel threads
    (BLAS releases GIL, so threading gives ~2x speedup).
    """
    N = rwg['N']
    print(f"  Assembling {2*N}x{2*N} PMCHWT matrix ({N} RWG functions)...")

    if parallel:
        print(f"  Parallel assembly: ext (k={k_ext:.4f}) + int (k={k_int:.4f})...")
        t0 = _time.time()
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_ext = pool.submit(assemble_L_K, rwg, verts, tris, k_ext, quad_order)
            fut_int = pool.submit(assemble_L_K, rwg, verts, tris, k_int, quad_order)
            L_ext, K_ext = fut_ext.result()
            L_int, K_int = fut_int.result()
        print(f"  Parallel assembly done: {_time.time() - t0:.1f}s")
    else:
        print(f"  Exterior operators (k={k_ext:.4f})...")
        L_ext, K_ext = assemble_L_K(rwg, verts, tris, k_ext, quad_order)
        print(f"  Interior operators (k={k_int:.4f})...")
        L_int, K_int = assemble_L_K(rwg, verts, tris, k_int, quad_order)

    K_sum = K_ext + K_int
    Z = np.zeros((2*N, 2*N), dtype=complex)
    Z[:N, :N] = eta_ext * L_ext + eta_int * L_int
    Z[:N, N:] = -K_sum
    Z[N:, :N] = K_sum
    Z[N:, N:] = L_ext / eta_ext + L_int / eta_int
    return Z, L_ext, K_ext


def solve_gmres(Z, b, tol=1e-4, maxiter=500, precond='block_diag'):
    """Solve Z·x = b using GMRES with block-diagonal preconditioner.

    Much faster than LU for large systems: O(N²) per iteration vs O(N³) for LU.
    For multiple RHS, LU is still better (factorize once, solve cheaply).

    precond: 'block_diag' uses the 2×2 block-diagonal of Z as preconditioner,
             'diag' uses the main diagonal,
             None for no preconditioning.
    """
    from scipy.sparse.linalg import LinearOperator
    n = Z.shape[0]
    N = n // 2

    if precond == 'block_diag':
        # Block-diagonal preconditioner: invert the two diagonal blocks separately
        Z11 = Z[:N, :N]
        Z22 = Z[N:, N:]
        lu11 = linalg.lu_factor(Z11)
        lu22 = linalg.lu_factor(Z22)
        def apply_precond(x):
            y = np.empty(n, dtype=complex)
            y[:N] = linalg.lu_solve(lu11, x[:N])
            y[N:] = linalg.lu_solve(lu22, x[N:])
            return y
        M = LinearOperator((n, n), matvec=apply_precond)
    elif precond == 'diag':
        d = np.diag(Z)
        def apply_precond(x):
            return x / d
        M = LinearOperator((n, n), matvec=apply_precond)
    else:
        M = None

    iter_count = [0]
    def callback(rk):
        iter_count[0] += 1

    t0 = _time.time()
    x, info = sp_gmres(Z, b, M=M, rtol=tol, maxiter=maxiter,
                        callback=callback, callback_type='pr_norm')
    elapsed = _time.time() - t0

    if info == 0:
        print(f"  GMRES converged in {iter_count[0]} iterations, {elapsed:.1f}s")
    else:
        print(f"  GMRES did not converge (info={info}), {iter_count[0]} iterations, {elapsed:.1f}s")
        res = np.linalg.norm(Z @ x - b) / np.linalg.norm(b)
        print(f"  Relative residual: {res:.2e}")

    return x


# ============================================================
# 7. RHS
# ============================================================

def compute_rhs_planewave(rwg, verts, tris, k_ext, eta_ext,
                           E0=np.array([1, 0, 0]), k_hat=np.array([0, 0, 1]),
                           quad_order=7):
    """Compute PMCHWT RHS for plane wave incidence.

    Returns b = [<f,E_inc>; <f,H_inc>], matching the
    sign convention Z·x = +b used by assemble_pmchwt.
    """
    N = rwg['N']
    quad_pts, quad_wts = tri_quadrature(quad_order)
    lam0 = 1 - quad_pts[:, 0] - quad_pts[:, 1]
    def get_qpts(ti):
        t = tris[ti]; v0 = verts[t[:,0]]; v1 = verts[t[:,1]]; v2 = verts[t[:,2]]
        return np.einsum('q,ni->nqi', lam0, v0) + np.einsum('q,ni->nqi', quad_pts[:,0], v1) + \
               np.einsum('q,ni->nqi', quad_pts[:,1], v2)
    qp = get_qpts(rwg['tri_p']); qm = get_qpts(rwg['tri_m'])
    H0 = np.cross(k_hat, E0) / eta_ext
    b = np.zeros(2*N, dtype=complex)
    for qpts, free, area, sign in [(qp, rwg['free_p'], rwg['area_p'], +1),
                                     (qm, rwg['free_m'], rwg['area_m'], -1)]:
        f = sign * (rwg['length'][:,None,None] / (2*area[:,None,None])) * (qpts - free[:,None,:])
        jw = area[:,None] * quad_wts[None,:]
        phase = np.exp(1j * k_ext * np.einsum('i,nqi->nq', k_hat, qpts))
        b[:N] += np.sum(np.einsum('nqi,i->nq', f, E0) * phase * jw, axis=1)
        b[N:] += np.sum(np.einsum('nqi,i->nq', f, H0) * phase * jw, axis=1)
    return b


# ============================================================
# 8. Far field
# ============================================================

def compute_far_field(rwg, verts, tris, coeffs_J, coeffs_M, k_ext, eta_ext,
                       theta_arr, phi=0.0, sM=-1, quad_order=7):
    """Compute far-field scattering amplitude F(θ,φ).

    F = -ik/(4π) [η J̃_⊥ + sM * r̂ × M̃]
    where J̃ = Σ J_n ∫ f_n e^{-ikr̂·r'} dS'.
    sM = -1 for standard PMCHWT convention.

    Returns F_theta, F_phi arrays.
    """
    quad_pts, quad_wts = tri_quadrature(quad_order)
    N = rwg['N']; lam0 = 1 - quad_pts[:,0] - quad_pts[:,1]
    def get_qpts(ti):
        t = tris[ti]; v0 = verts[t[:,0]]; v1 = verts[t[:,1]]; v2 = verts[t[:,2]]
        return np.einsum('q,ni->nqi', lam0, v0) + np.einsum('q,ni->nqi', quad_pts[:,0], v1) + \
               np.einsum('q,ni->nqi', quad_pts[:,1], v2)
    qp = get_qpts(rwg['tri_p']); qm = get_qpts(rwg['tri_m'])

    F_theta = np.zeros(len(theta_arr), dtype=complex)
    F_phi = np.zeros(len(theta_arr), dtype=complex)
    for it, theta in enumerate(theta_arr):
        cp, sp = np.cos(phi), np.sin(phi)
        ct, st = np.cos(theta), np.sin(theta)
        r_hat = np.array([st*cp, st*sp, ct])
        theta_hat = np.array([ct*cp, ct*sp, -st])
        phi_hat = np.array([-sp, cp, 0.0])
        Jt = np.zeros(3, dtype=complex); Mt = np.zeros(3, dtype=complex)
        for qpts, free, area, sign in [(qp, rwg['free_p'], rwg['area_p'], +1),
                                         (qm, rwg['free_m'], rwg['area_m'], -1)]:
            f = sign * (rwg['length'][:,None,None] / (2*area[:,None,None])) * (qpts - free[:,None,:])
            jw = area[:,None] * quad_wts[None,:]
            phase = np.exp(-1j * k_ext * np.einsum('i,nqi->nq', r_hat, qpts))
            integral = (f * (phase * jw)[:,:,None]).sum(axis=1)
            Jt += (integral * coeffs_J[:,None]).sum(0)
            Mt += (integral * coeffs_M[:,None]).sum(0)
        Jp = Jt - r_hat * np.dot(r_hat, Jt)
        Mc = np.cross(r_hat, Mt)
        Fv = -1j * k_ext / (4*np.pi) * (eta_ext * Jp + sM * Mc)
        F_theta[it] = np.dot(Fv, theta_hat)
        F_phi[it] = np.dot(Fv, phi_hat)
    return F_theta, F_phi


def _compute_far_field_vec(rwg, verts, tris, coeffs_J, coeffs_M, k_ext, eta_ext,
                            r_hat, sM=-1, quad_order=7, _cache=None):
    """Compute 3D far-field vector F at a single direction r_hat.

    Returns complex 3-vector F such that E_sca ~ (e^{ikr}/r) F.
    Optionally pass _cache dict with precomputed qp, qm for speed.
    """
    quad_pts, quad_wts = tri_quadrature(quad_order)
    N = rwg['N']; lam0 = 1 - quad_pts[:,0] - quad_pts[:,1]
    if _cache is not None:
        qp, qm = _cache['qp'], _cache['qm']
    else:
        def get_qpts(ti):
            t = tris[ti]; v0 = verts[t[:,0]]; v1 = verts[t[:,1]]; v2 = verts[t[:,2]]
            return np.einsum('q,ni->nqi', lam0, v0) + np.einsum('q,ni->nqi', quad_pts[:,0], v1) + \
                   np.einsum('q,ni->nqi', quad_pts[:,1], v2)
        qp = get_qpts(rwg['tri_p']); qm = get_qpts(rwg['tri_m'])

    Jt = np.zeros(3, dtype=complex); Mt = np.zeros(3, dtype=complex)
    for qpts, free, area, sign in [(qp, rwg['free_p'], rwg['area_p'], +1),
                                     (qm, rwg['free_m'], rwg['area_m'], -1)]:
        f = sign * (rwg['length'][:,None,None] / (2*area[:,None,None])) * (qpts - free[:,None,:])
        jw = area[:,None] * quad_wts[None,:]
        phase = np.exp(-1j * k_ext * np.einsum('i,nqi->nq', r_hat, qpts))
        integral = (f * (phase * jw)[:,:,None]).sum(axis=1)
        Jt += (integral * coeffs_J[:,None]).sum(0)
        Mt += (integral * coeffs_M[:,None]).sum(0)
    Jp = Jt - r_hat * np.dot(r_hat, Jt)
    Mc = np.cross(r_hat, Mt)
    return -1j * k_ext / (4*np.pi) * (eta_ext * Jp + sM * Mc)


def compute_cross_sections(rwg, verts, tris, coeffs_J, coeffs_M, k_ext, eta_ext,
                            radius, sM=-1, ntheta=181, nphi=36, quad_order=7):
    """Compute Q_ext and Q_sca with proper 2D angular integration.

    Q_ext uses the optical theorem (forward scattering amplitude).
    Q_sca integrates |F|² over all (θ,φ) directions.
    sM = -1 for standard PMCHWT, +1 for PEC (M=0, doesn't matter).
    """
    theta_arr = np.linspace(0.01, np.pi - 0.01, ntheta)
    phi_arr = np.linspace(0, 2*np.pi, nphi, endpoint=False)
    dphi = 2 * np.pi / nphi

    C_sca = 0.0
    Q_ext = None

    for ip, phi in enumerate(phi_arr):
        F_th, F_ph = compute_far_field(rwg, verts, tris, coeffs_J, coeffs_M,
                                        k_ext, eta_ext, theta_arr, phi=phi,
                                        sM=sM, quad_order=quad_order)
        dsigma = np.abs(F_th)**2 + np.abs(F_ph)**2
        C_sca += dphi * np.trapezoid(dsigma * np.sin(theta_arr), theta_arr)

        if ip == 0:
            Q_ext = 4 * np.pi / k_ext * np.imag(F_th[0]) / (np.pi * radius**2)

    Q_sca = C_sca / (np.pi * radius**2)
    return Q_ext, Q_sca


# ============================================================
# 9b. Amplitude scattering matrix and Mueller matrix
# ============================================================

def compute_amplitude_matrix(rwg, verts, tris, k_ext, eta_ext, theta_arr,
                              Z_lu=None, Z=None, k_hat=np.array([0,0,1.0]),
                              sM=-1, quad_order=7):
    """Compute the 2×2 amplitude scattering matrix S(θ) at φ=0.

    Convention (Bohren & Huffman):
      [E_s,par ]   exp(ikr)   [S2  S3] [E_i,par ]
      [E_s,perp] = --------   [S4  S1] [E_i,perp]
                    -ikr

    Two incident polarizations are solved:
      par  = E0 in scattering plane (x for k along z, φ=0)
      perp = E0 perpendicular (y)

    Parameters:
      Z_lu   : precomputed LU factorization (from scipy.linalg.lu_factor)
      Z      : system matrix (used if Z_lu is not provided)
      k_hat  : incident wave direction
      theta_arr : scattering angles

    Returns dict with keys 'S1','S2','S3','S4' — complex arrays (len(theta_arr),).
    Also 'theta' for convenience.
    """
    from scipy.linalg import lu_factor, lu_solve

    N = rwg['N']

    # Determine par/perp polarization vectors from k_hat
    # par = in scattering plane (xz for k along z)
    # Choose e_par perpendicular to k_hat and in xz-plane
    if abs(k_hat[2]) > 0.9:
        e_perp = np.array([0.0, 1.0, 0.0])
    else:
        e_perp = np.cross(k_hat, np.array([0, 0, 1.0]))
        e_perp /= np.linalg.norm(e_perp)
    e_par = np.cross(e_perp, k_hat)
    e_par /= np.linalg.norm(e_par)

    # LU factorization (reusable for multiple RHS)
    if Z_lu is None:
        if Z is None:
            raise ValueError("Provide either Z or Z_lu")
        Z_lu = lu_factor(Z)

    # Solve for two polarizations
    results = {}
    for pol_name, E0 in [('par', e_par), ('perp', e_perp)]:
        b = compute_rhs_planewave(rwg, verts, tris, k_ext, eta_ext,
                                   E0=E0, k_hat=k_hat, quad_order=quad_order)
        coeffs = lu_solve(Z_lu, b)
        J = coeffs[:N]; M = coeffs[N:]
        F_th, F_ph = compute_far_field(rwg, verts, tris, J, M, k_ext, eta_ext,
                                        theta_arr, phi=0.0, sM=sM, quad_order=quad_order)
        results[pol_name] = (F_th, F_ph)

    # Convert F to amplitude matrix elements
    # F = E_sca * r * e^{-ikr}, and S = -ik * r * E_sca * e^{-ikr}
    # So S = -ik * F
    ik = -1j * k_ext
    S2 = ik * results['par'][0]    # θ-component from par incidence
    S4 = ik * results['par'][1]    # φ-component from par incidence
    S3 = ik * results['perp'][0]   # θ-component from perp incidence
    S1 = ik * results['perp'][1]   # φ-component from perp incidence

    return {'S1': S1, 'S2': S2, 'S3': S3, 'S4': S4, 'theta': theta_arr}


def amplitude_to_mueller(S1, S2, S3, S4):
    """Convert amplitude matrix elements to 4×4 Mueller matrix.

    Input: S1, S2, S3, S4 — complex arrays of length N_theta.
    Returns: M — array of shape (4, 4, N_theta).

    Convention follows Bohren & Huffman, eq. 3.16.
    """
    n = len(S1)
    M = np.zeros((4, 4, n))

    M[0,0] = 0.5*(np.abs(S1)**2 + np.abs(S2)**2 + np.abs(S3)**2 + np.abs(S4)**2)
    M[0,1] = 0.5*(np.abs(S2)**2 - np.abs(S1)**2 + np.abs(S4)**2 - np.abs(S3)**2)
    M[0,2] = np.real(S2*np.conj(S3) + S1*np.conj(S4))
    M[0,3] = np.imag(S2*np.conj(S3) - S1*np.conj(S4))

    M[1,0] = 0.5*(np.abs(S2)**2 - np.abs(S1)**2 - np.abs(S4)**2 + np.abs(S3)**2)
    M[1,1] = 0.5*(np.abs(S2)**2 + np.abs(S1)**2 - np.abs(S4)**2 - np.abs(S3)**2)
    M[1,2] = np.real(S2*np.conj(S3) - S1*np.conj(S4))
    M[1,3] = np.imag(S2*np.conj(S3) + S1*np.conj(S4))

    M[2,0] = np.real(S2*np.conj(S4) + S1*np.conj(S3))
    M[2,1] = np.real(S2*np.conj(S4) - S1*np.conj(S3))
    M[2,2] = np.real(S1*np.conj(S2) + S3*np.conj(S4))
    M[2,3] = np.imag(S2*np.conj(S1) + S4*np.conj(S3))

    M[3,0] = np.imag(S4*np.conj(S2) + S1*np.conj(S3))
    M[3,1] = np.imag(S4*np.conj(S2) - S1*np.conj(S3))
    M[3,2] = np.imag(S1*np.conj(S2) - S3*np.conj(S4))
    M[3,3] = np.real(S1*np.conj(S2) - S3*np.conj(S4))

    return M


def compute_mueller_matrix(rwg, verts, tris, k_ext, eta_ext, theta_arr,
                            Z_lu=None, Z=None, k_hat=np.array([0,0,1.0]),
                            sM=-1, quad_order=7):
    """Compute Mueller matrix M(θ) for a fixed particle orientation.

    Returns M of shape (4, 4, N_theta), normalized by 1/k² so that
    M[0,0] = differential scattering cross section dσ/dΩ for unpolarized light.
    """
    S = compute_amplitude_matrix(rwg, verts, tris, k_ext, eta_ext, theta_arr,
                                  Z_lu=Z_lu, Z=Z, k_hat=k_hat, sM=sM,
                                  quad_order=quad_order)
    M = amplitude_to_mueller(S['S1'], S['S2'], S['S3'], S['S4'])
    # Normalize: dσ/dΩ = M11 / k² for unpolarized incidence
    M /= k_ext**2
    return M


def _euler_rotation(alpha, beta, gamma):
    """ZYZ Euler rotation matrix R(α,β,γ)."""
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)
    Rz1 = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    Ry  = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    Rz2 = np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]])
    return Rz1 @ Ry @ Rz2


def orientation_average_mueller(rwg, verts, tris, k_ext, eta_ext, theta_arr,
                                 Z_lu=None, Z=None, sM=-1, quad_order=7,
                                 n_alpha=8, n_beta=8, n_gamma=1):
    """Compute orientation-averaged Mueller matrix ⟨M(θ)⟩.

    Averages over Euler angles (α, β, γ) using Gauss-Legendre quadrature
    for β and uniform quadrature for α and γ:
      ⟨M⟩ = (1/8π²) ∫₀²π ∫₀π ∫₀²π M(α,β,γ) sinβ dα dβ dγ

    The particle is kept fixed; the incident direction k_hat and observation
    directions r̂ are rotated. For each orientation, the amplitude matrix is
    computed in the scattering plane defined by k_hat and r̂.

    Parameters:
      n_alpha : int — number of α quadrature points (default 8)
      n_beta  : int — number of β quadrature points (Gauss-Legendre, default 8)
      n_gamma : int — number of γ points (default 1; increase for no-symmetry particles)

    Returns: M_avg of shape (4, 4, N_theta).
    """
    from scipy.linalg import lu_factor, lu_solve
    from scipy.special import roots_legendre

    N = rwg['N']
    ntheta = len(theta_arr)

    if Z_lu is None:
        if Z is None:
            raise ValueError("Provide either Z or Z_lu")
        Z_lu = lu_factor(Z)

    # Precompute quadrature points for far field
    quad_pts, quad_wts = tri_quadrature(quad_order)
    lam0 = 1 - quad_pts[:,0] - quad_pts[:,1]
    def get_qpts(ti):
        t = tris[ti]; v0 = verts[t[:,0]]; v1 = verts[t[:,1]]; v2 = verts[t[:,2]]
        return np.einsum('q,ni->nqi', lam0, v0) + np.einsum('q,ni->nqi', quad_pts[:,0], v1) + \
               np.einsum('q,ni->nqi', quad_pts[:,1], v2)
    ff_cache = {'qp': get_qpts(rwg['tri_p']), 'qm': get_qpts(rwg['tri_m'])}

    # Quadrature nodes
    mu_nodes, mu_weights = roots_legendre(n_beta)
    beta_nodes = np.arccos(mu_nodes)
    alpha_nodes = np.linspace(0, 2*np.pi, n_alpha, endpoint=False)
    d_alpha = 2*np.pi / n_alpha
    gamma_nodes = np.linspace(0, 2*np.pi, n_gamma, endpoint=False)
    d_gamma = 2*np.pi / n_gamma

    M_avg = np.zeros((4, 4, ntheta))
    total = n_alpha * n_beta * n_gamma
    count = 0

    for ia, alpha in enumerate(alpha_nodes):
        for ib in range(n_beta):
            beta = beta_nodes[ib]
            w_beta = mu_weights[ib]
            for ig, gamma in enumerate(gamma_nodes):
                count += 1
                if count % 10 == 1 or count == total:
                    print(f"    Orientation {count}/{total}...", flush=True)

                # Rotation: particle rotated by R ↔ incident direction k_hat = R^T ẑ
                R = _euler_rotation(alpha, beta, gamma)
                k_hat = R.T @ np.array([0., 0., 1.])

                # Scattering plane basis (lab frame):
                # e_par in scattering plane (contains k_hat), e_perp perpendicular
                if abs(k_hat[2]) > 0.999:
                    e_perp_lab = np.array([0., 1., 0.])
                else:
                    e_perp_lab = np.cross(k_hat, np.array([0., 0., 1.]))
                    e_perp_lab /= np.linalg.norm(e_perp_lab)
                e_par_lab = np.cross(e_perp_lab, k_hat)
                e_par_lab /= np.linalg.norm(e_par_lab)

                # Solve for two polarizations
                solutions = {}
                for pol_name, E0 in [('par', e_par_lab), ('perp', e_perp_lab)]:
                    b = compute_rhs_planewave(rwg, verts, tris, k_ext, eta_ext,
                                               E0=E0, k_hat=k_hat, quad_order=quad_order)
                    coeffs = lu_solve(Z_lu, b)
                    solutions[pol_name] = (coeffs[:N], coeffs[N:])

                # For each scattering angle θ_sca, observation direction:
                # r̂ is at angle θ_sca from k_hat, in the scattering plane
                S1 = np.zeros(ntheta, dtype=complex)
                S2 = np.zeros(ntheta, dtype=complex)
                S3 = np.zeros(ntheta, dtype=complex)
                S4 = np.zeros(ntheta, dtype=complex)

                for it, theta_sca in enumerate(theta_arr):
                    # Observation direction: rotate k_hat by θ_sca in scattering plane
                    ct, st = np.cos(theta_sca), np.sin(theta_sca)
                    r_hat = ct * k_hat + st * e_par_lab

                    # Scattered polarization basis at r̂:
                    # e_sca_par in scattering plane, perpendicular to r̂
                    e_sca_par = -st * k_hat + ct * e_par_lab
                    e_sca_perp = e_perp_lab  # perpendicular stays the same

                    # Compute F vector for each incident polarization
                    for pol_name in ['par', 'perp']:
                        J, M_c = solutions[pol_name]
                        Fv = _compute_far_field_vec(rwg, verts, tris, J, M_c,
                                                     k_ext, eta_ext, r_hat,
                                                     sM=sM, quad_order=quad_order,
                                                     _cache=ff_cache)
                        F_par = np.dot(Fv, e_sca_par)   # parallel component
                        F_perp = np.dot(Fv, e_sca_perp)  # perpendicular component

                        ik = -1j * k_ext
                        if pol_name == 'par':
                            S2[it] = ik * F_par
                            S4[it] = ik * F_perp
                        else:
                            S3[it] = ik * F_par
                            S1[it] = ik * F_perp

                M_orient = amplitude_to_mueller(S1, S2, S3, S4) / k_ext**2
                weight = d_alpha * w_beta * d_gamma / (8 * np.pi**2)
                M_avg += weight * M_orient

    print(f"    Averaged over {count} orientations.")
    return M_avg


# ============================================================
# 10. SNC-tested assembly (n̂×RWG test functions)
# ============================================================

def _compute_tri_normals(verts, tris):
    """Compute outward unit normals for each triangle."""
    v0 = verts[tris[:, 0]]; v1 = verts[tris[:, 1]]; v2 = verts[tris[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= norms
    # Orient outward (for closed surface, normal should point away from centroid)
    centroid = verts.mean(axis=0)
    tri_centers = (v0 + v1 + v2) / 3
    outward = np.einsum('ni,ni->n', normals, tri_centers - centroid)
    normals[outward < 0] *= -1
    return normals


def assemble_L_K_snc(rwg, verts, tris, k, quad_order=7):
    """
    Assemble L and K with SNC (n̂×RWG) test functions.

    SNC testing eliminates the div-div term (since ∇·(n̂×f) = 0 on flat triangles),
    giving a well-conditioned L operator:
      L_mn = jk ∫∫ (n̂×f_m)·f_n G dS dS'
      K_mn = ∫∫ (n̂×f_m) · (∇G × f_n) dS dS'

    The singular part of L only has the weakly-singular vector potential integral
    (no 1/R scalar potential). We use G_smooth + analytical G_0 for the vector part.
    """
    N = rwg['N']
    quad_pts, quad_wts = tri_quadrature(quad_order)
    Nq = len(quad_wts)
    lam0 = 1 - quad_pts[:, 0] - quad_pts[:, 1]

    tri_normals = _compute_tri_normals(verts, tris)

    def get_qpts_batch(tri_indices):
        t = tris[tri_indices]
        v0 = verts[t[:, 0]]; v1 = verts[t[:, 1]]; v2 = verts[t[:, 2]]
        return np.einsum('q,ni->nqi', lam0, v0) + \
               np.einsum('q,ni->nqi', quad_pts[:, 0], v1) + \
               np.einsum('q,ni->nqi', quad_pts[:, 1], v2)

    qpts_p = get_qpts_batch(rwg['tri_p'])
    qpts_m = get_qpts_batch(rwg['tri_m'])

    # RWG basis values at quad points
    f_p = (rwg['length'][:, None, None] / (2 * rwg['area_p'][:, None, None])) * \
          (qpts_p - rwg['free_p'][:, None, :])
    f_m = -(rwg['length'][:, None, None] / (2 * rwg['area_m'][:, None, None])) * \
          (qpts_m - rwg['free_m'][:, None, :])

    # SNC test functions: n̂ × f_m for each half
    n_p = tri_normals[rwg['tri_p']]  # (N, 3)
    n_m = tri_normals[rwg['tri_m']]
    nxf_p = np.cross(n_p[:, None, :], f_p)  # (N, Nq, 3)
    nxf_m = np.cross(n_m[:, None, :], f_m)  # (N, Nq, 3)

    jw_p = rwg['area_p'][:, None] * quad_wts[None, :]
    jw_m = rwg['area_m'][:, None] * quad_wts[None, :]

    # Source (trial) functions — standard RWG, combine both halves
    all_qpts = np.concatenate([qpts_p, qpts_m], axis=1)
    all_f = np.concatenate([f_p, f_m], axis=1)
    all_jw = np.concatenate([jw_p, jw_m], axis=1)

    # For singular extraction, we still need analytical integrals
    tri_verts_cache = {}
    for ti in range(len(tris)):
        tri_verts_cache[ti] = (verts[tris[ti, 0]].copy(),
                                verts[tris[ti, 1]].copy(),
                                verts[tris[ti, 2]].copy())

    # Source basis info for singular contributions
    div_p = rwg['length'] / rwg['area_p']
    div_m = -rwg['length'] / rwg['area_m']

    L = np.zeros((N, N), dtype=complex)
    K = np.zeros((N, N), dtype=complex)

    print(f"    SNC Assembly: {N} RWG, {Nq} quad pts, k={k:.4f}...")

    for m_idx in range(N):
        if m_idx % max(1, N // 10) == 0:
            print(f"      Row {m_idx}/{N}...")

        test_halves = [
            (qpts_p[m_idx], nxf_p[m_idx], jw_p[m_idx], rwg['tri_p'][m_idx]),
            (qpts_m[m_idx], nxf_m[m_idx], jw_m[m_idx], rwg['tri_m'][m_idx]),
        ]

        for test_pts, test_nxf_h, test_jw_h, test_tri_idx in test_halves:
            R_vec = test_pts[None, :, None, :] - all_qpts[:, None, :, :]
            R = np.linalg.norm(R_vec, axis=-1)
            R_safe = np.where(R < 1e-15, 1.0, R)

            G_full = np.exp(1j * k * R) / (4 * np.pi * R_safe)
            G_full = np.where(R < 1e-15, 0.0, G_full)

            # Smooth part for singular extraction
            G_smooth = (np.exp(1j * k * R) - 1.0) / (4 * np.pi * R_safe)
            G_smooth = np.where(R < 1e-15, 1j * k / (4 * np.pi), G_smooth)

            singular_mask_p = (rwg['tri_p'] == test_tri_idx)
            singular_mask_m = (rwg['tri_m'] == test_tri_idx)
            sing_mask = np.zeros((N, 2*Nq), dtype=bool)
            sing_mask[:, :Nq] = singular_mask_p[:, None]
            sing_mask[:, Nq:] = singular_mask_m[:, None]

            G_use = np.where(sing_mask[:, None, :], G_smooth, G_full)

            # --- L operator (SNC): only vector potential term ---
            # L_mn += jk * (n̂×f_m) · f_n * G * jw_test * jw_src
            nxf_dot_f = np.einsum('qi,nji->nqj', test_nxf_h, all_f)
            jw_outer = test_jw_h[None, :, None] * all_jw[:, None, :]

            L_contrib = 1j * k * nxf_dot_f * G_use * jw_outer
            L[m_idx, :] += L_contrib.sum(axis=(1, 2))

            # --- L singular: analytical G_0 vector part ---
            if np.any(sing_mask):
                tv = tri_verts_cache[test_tri_idx]
                P_vals = np.zeros(Nq)
                for iq in range(Nq):
                    P_vals[iq] = potential_integral_triangle(test_pts[iq], *tv)

                for n in range(N):
                    for half_s in range(2):
                        if half_s == 0 and not singular_mask_p[n]:
                            continue
                        if half_s == 1 and not singular_mask_m[n]:
                            continue

                        if half_s == 0:
                            src_coeff = rwg['length'][n] / (2 * rwg['area_p'][n])
                            src_free = rwg['free_p'][n]
                            src_sign = +1
                        else:
                            src_coeff = rwg['length'][n] / (2 * rwg['area_m'][n])
                            src_free = rwg['free_m'][n]
                            src_sign = -1

                        # Vector singular: jk * ∫ (n̂×f_m) · [∫ f_n/R dS']/(4π) dS
                        vec_integral = 0.0
                        for iq in range(Nq):
                            V_r = vector_potential_integral_triangle(test_pts[iq], *tv)
                            fn_over_R = src_sign * src_coeff * (V_r - src_free * P_vals[iq])
                            vec_integral += np.dot(test_nxf_h[iq], fn_over_R) * test_jw_h[iq]
                        vec_integral /= (4 * np.pi)
                        L[m_idx, n] += 1j * k * vec_integral

            # --- K operator (SNC) ---
            gradG_coeff = (1j * k - 1.0 / R_safe) / R_safe
            gradG_coeff = np.where(R < 1e-12, 0.0, gradG_coeff)
            gradG_coeff = np.where(sing_mask[:, None, :], 0.0, gradG_coeff)
            gradG = G_full[:, :, :, None] * gradG_coeff[:, :, :, None] * R_vec

            cross = np.cross(gradG, all_f[:, None, :, :])
            K_integrand = np.einsum('qi,nqji->nqj', test_nxf_h, cross)
            K[m_idx, :] += (K_integrand * jw_outer).sum(axis=(1, 2))

    return L, K


def assemble_pmchwt_snc(rwg, verts, tris, k_ext, k_int, eta_ext, eta_int, quad_order=7):
    """Assemble PMCHWT with SNC testing."""
    N = rwg['N']
    print(f"  Assembling SNC-tested {2*N}x{2*N} PMCHWT ({N} RWG)...")
    print(f"  Exterior operators (k={k_ext:.4f})...")
    L_ext, K_ext = assemble_L_K_snc(rwg, verts, tris, k_ext, quad_order)
    print(f"  Interior operators (k={k_int:.4f})...")
    L_int, K_int = assemble_L_K_snc(rwg, verts, tris, k_int, quad_order)

    Z = np.zeros((2*N, 2*N), dtype=complex)
    Z[:N, :N] = eta_ext * L_ext + eta_int * L_int
    Z[:N, N:] = K_ext + K_int
    Z[N:, :N] = -(K_ext + K_int)
    Z[N:, N:] = L_ext / eta_ext + L_int / eta_int
    return Z, L_ext, K_ext


def compute_rhs_planewave_snc(rwg, verts, tris, k_ext, eta_ext,
                               E0=np.array([1, 0, 0]), k_hat=np.array([0, 0, 1]),
                               quad_order=7):
    """RHS with SNC (n̂×RWG) testing."""
    N = rwg['N']
    tri_normals = _compute_tri_normals(verts, tris)
    quad_pts, quad_wts = tri_quadrature(quad_order)
    lam0 = 1 - quad_pts[:, 0] - quad_pts[:, 1]

    def get_qpts(ti):
        t = tris[ti]; v0 = verts[t[:,0]]; v1 = verts[t[:,1]]; v2 = verts[t[:,2]]
        return np.einsum('q,ni->nqi', lam0, v0) + np.einsum('q,ni->nqi', quad_pts[:,0], v1) + \
               np.einsum('q,ni->nqi', quad_pts[:,1], v2)

    qp = get_qpts(rwg['tri_p']); qm = get_qpts(rwg['tri_m'])
    H0 = np.cross(k_hat, E0) / eta_ext
    b = np.zeros(2*N, dtype=complex)

    for qpts, free, area, sign, tri_idx in [
        (qp, rwg['free_p'], rwg['area_p'], +1, rwg['tri_p']),
        (qm, rwg['free_m'], rwg['area_m'], -1, rwg['tri_m']),
    ]:
        f = sign * (rwg['length'][:,None,None] / (2*area[:,None,None])) * (qpts - free[:,None,:])
        normals = tri_normals[tri_idx]  # (N, 3)
        nxf = np.cross(normals[:, None, :], f)  # (N, Nq, 3)
        jw = area[:,None] * quad_wts[None,:]
        phase = np.exp(1j * k_ext * np.einsum('i,nqi->nq', k_hat, qpts))
        b[:N] += np.sum(np.einsum('nqi,i->nq', nxf, E0) * phase * jw, axis=1)
        b[N:] += np.sum(np.einsum('nqi,i->nq', nxf, H0) * phase * jw, axis=1)
    return b
