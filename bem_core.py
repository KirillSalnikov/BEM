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

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


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


def refine_mesh(verts, tris, mask=None, project_to_sphere=False, sphere_radius=None):
    """Refine triangular mesh by splitting marked triangles.

    Parameters
    ----------
    verts : ndarray (V, 3)
        Vertex coordinates.
    tris : ndarray (T, 3)
        Triangle indices.
    mask : ndarray (T,) of bool, optional
        Which triangles to refine. If None, refine all (uniform).
    project_to_sphere : bool
        If True, project new midpoint vertices onto a sphere.
    sphere_radius : float, optional
        Sphere radius for projection (auto-detected if None).

    Returns
    -------
    new_verts, new_tris : ndarray
        Refined mesh. Marked triangles are split into 4; neighbors that share
        a split edge get a bisection (green closure) to maintain conformality.
    """
    T = len(tris)
    if mask is None:
        mask = np.ones(T, dtype=bool)

    # Build edge → triangles map
    edge_to_tris = {}
    for ti in range(T):
        for j in range(3):
            e = (min(tris[ti, j], tris[ti, (j+1)%3]),
                 max(tris[ti, j], tris[ti, (j+1)%3]))
            if e not in edge_to_tris:
                edge_to_tris[e] = []
            edge_to_tris[e].append(ti)

    # Determine which edges to split
    edge_split = {}  # edge → new vertex index
    verts_list = list(verts)

    if sphere_radius is None and project_to_sphere:
        sphere_radius = np.mean(np.linalg.norm(verts, axis=1))

    for ti in range(T):
        if not mask[ti]:
            continue
        for j in range(3):
            e = (min(tris[ti, j], tris[ti, (j+1)%3]),
                 max(tris[ti, j], tris[ti, (j+1)%3]))
            if e in edge_split:
                continue
            mid = (verts[e[0]] + verts[e[1]]) / 2
            if project_to_sphere:
                r = np.linalg.norm(mid)
                if r > 1e-15:
                    mid = mid * (sphere_radius / r)
            edge_split[e] = len(verts_list)
            verts_list.append(mid)

    # Build new triangles
    new_tris = []
    for ti in range(T):
        a, b, c = tris[ti]
        eab = (min(a, b), max(a, b))
        ebc = (min(b, c), max(b, c))
        eca = (min(c, a), max(c, a))
        split_ab = eab in edge_split
        split_bc = ebc in edge_split
        split_ca = eca in edge_split

        n_split = split_ab + split_bc + split_ca

        if n_split == 3:
            # Red refinement: split into 4
            ab = edge_split[eab]; bc = edge_split[ebc]; ca = edge_split[eca]
            new_tris.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]])
        elif n_split == 2:
            # Green closure: split into 3
            if not split_ab:
                # bc and ca split
                bc = edge_split[ebc]; ca = edge_split[eca]
                new_tris.extend([[a, b, ca], [b, bc, ca], [bc, c, ca]])
            elif not split_bc:
                ab = edge_split[eab]; ca = edge_split[eca]
                new_tris.extend([[b, c, ab], [c, ca, ab], [ca, a, ab]])
            else:
                ab = edge_split[eab]; bc = edge_split[ebc]
                new_tris.extend([[c, a, bc], [a, ab, bc], [ab, b, bc]])
        elif n_split == 1:
            # Bisection: split into 2
            if split_ab:
                ab = edge_split[eab]
                new_tris.extend([[a, ab, c], [ab, b, c]])
            elif split_bc:
                bc = edge_split[ebc]
                new_tris.extend([[b, bc, a], [bc, c, a]])
            else:
                ca = edge_split[eca]
                new_tris.extend([[c, ca, b], [ca, a, b]])
        else:
            new_tris.append([a, b, c])

    return np.array(verts_list), np.array(new_tris, dtype=np.int32)


def adaptive_refine(verts, tris, k, max_edge_per_wavelength=0.2,
                    project_to_sphere=False, sphere_radius=None):
    """Refine mesh until all edges are shorter than max_edge_per_wavelength * λ.

    Parameters
    ----------
    k : float or complex
        Wavenumber. Uses |k| for wavelength calculation.
    max_edge_per_wavelength : float
        Maximum edge length as fraction of wavelength (default 0.2 = λ/5).
    project_to_sphere : bool
        Project new vertices onto sphere surface.
    sphere_radius : float, optional
        Sphere radius for projection.

    Returns
    -------
    new_verts, new_tris : ndarray
    """
    wavelength = 2 * np.pi / abs(k)
    max_edge = max_edge_per_wavelength * wavelength

    for iteration in range(20):
        # Compute edge lengths per triangle
        v0 = verts[tris[:, 0]]; v1 = verts[tris[:, 1]]; v2 = verts[tris[:, 2]]
        e01 = np.linalg.norm(v1 - v0, axis=1)
        e12 = np.linalg.norm(v2 - v1, axis=1)
        e20 = np.linalg.norm(v0 - v2, axis=1)
        max_edges = np.maximum(e01, np.maximum(e12, e20))

        mask = max_edges > max_edge
        n_refine = np.sum(mask)
        if n_refine == 0:
            break
        print(f"    Adaptive refine iter {iteration}: {n_refine}/{len(tris)} tris "
              f"(max edge {np.max(max_edges):.4f}, target {max_edge:.4f})")
        verts, tris = refine_mesh(verts, tris, mask,
                                   project_to_sphere=project_to_sphere,
                                   sphere_radius=sphere_radius)

    print(f"    Final mesh: {len(tris)} tris, {len(verts)} verts")
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
    elif order == 13:
        # Dunavant 13-point rule (degree 7)
        pts_list = [
            (1/3, 1/3),
            (0.260345966079038, 0.260345966079038),
            (0.260345966079038, 0.479308067841923),
            (0.479308067841923, 0.260345966079038),
            (0.065130102902216, 0.065130102902216),
            (0.065130102902216, 0.869739794195568),
            (0.869739794195568, 0.065130102902216),
            (0.048690315425316, 0.312865496004875),
            (0.312865496004875, 0.048690315425316),
            (0.638444188569809, 0.312865496004875),
            (0.312865496004875, 0.638444188569809),
            (0.048690315425316, 0.638444188569809),
            (0.638444188569809, 0.048690315425316),
        ]
        wts_list = [
            -0.149570044467670,
            0.175615257433204, 0.175615257433204, 0.175615257433204,
            0.053347235608839, 0.053347235608839, 0.053347235608839,
            0.077113760890257, 0.077113760890257, 0.077113760890257,
            0.077113760890257, 0.077113760890257, 0.077113760890257,
        ]
        pts = np.array(pts_list)
        wts = np.array(wts_list)
    elif order == 25:
        # Dunavant 25-point rule (degree 10)
        pts_list = [
            (1/3, 1/3),
            (0.028844733232685, 0.485577633383657),
            (0.485577633383657, 0.028844733232685),
            (0.485577633383657, 0.485577633383657),
            (0.781036849029926, 0.109481575485037),
            (0.109481575485037, 0.781036849029926),
            (0.109481575485037, 0.109481575485037),
            (0.141707219414880, 0.307939838764121),
            (0.307939838764121, 0.141707219414880),
            (0.550352941820999, 0.307939838764121),
            (0.307939838764121, 0.550352941820999),
            (0.141707219414880, 0.550352941820999),
            (0.550352941820999, 0.141707219414880),
            (0.025003534762686, 0.246672560639903),
            (0.246672560639903, 0.025003534762686),
            (0.728323904597411, 0.246672560639903),
            (0.246672560639903, 0.728323904597411),
            (0.025003534762686, 0.728323904597411),
            (0.728323904597411, 0.025003534762686),
            (0.009540815400299, 0.066803251012200),
            (0.066803251012200, 0.009540815400299),
            (0.923655933587500, 0.066803251012200),
            (0.066803251012200, 0.923655933587500),
            (0.009540815400299, 0.923655933587500),
            (0.923655933587500, 0.009540815400299),
        ]
        wts_list = [
            0.090817990382754,
            0.036725957756467, 0.036725957756467, 0.036725957756467,
            0.045321059435528, 0.045321059435528, 0.045321059435528,
            0.072757916845420, 0.072757916845420, 0.072757916845420,
            0.072757916845420, 0.072757916845420, 0.072757916845420,
            0.028327242531057, 0.028327242531057, 0.028327242531057,
            0.028327242531057, 0.028327242531057, 0.028327242531057,
            0.009421667726068, 0.009421667726068, 0.009421667726068,
            0.009421667726068, 0.009421667726068, 0.009421667726068,
        ]
        pts = np.array(pts_list)
        wts = np.array(wts_list)
    else:
        raise ValueError(f"Unsupported order {order}; use 4, 7, 13, or 25")
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
# 4b. Numba-accelerated singular corrections
# ============================================================

if HAS_NUMBA:
    @njit(cache=True)
    def _potential_integral_nb(r_obs, v0, v1, v2):
        """∫_T 1/|r_obs - r'| dS' (Graglia 1993), Numba version."""
        verts0 = v0; verts1 = v1; verts2 = v2
        e10 = v1 - v0; e20 = v2 - v0
        n_tri = np.array([e10[1]*e20[2] - e10[2]*e20[1],
                          e10[2]*e20[0] - e10[0]*e20[2],
                          e10[0]*e20[1] - e10[1]*e20[0]])
        n_norm = np.sqrt(n_tri[0]**2 + n_tri[1]**2 + n_tri[2]**2)
        if n_norm < 1e-15:
            return 0.0
        n_hat = n_tri / n_norm
        result = 0.0
        for i in range(3):
            if i == 0: vi = v0; vj = v1
            elif i == 1: vi = v1; vj = v2
            else: vi = v2; vj = v0
            edge = vj - vi
            l_edge = np.sqrt(edge[0]**2 + edge[1]**2 + edge[2]**2)
            if l_edge < 1e-15:
                continue
            t_hat = edge / l_edge
            m_hat = np.array([n_hat[1]*t_hat[2] - n_hat[2]*t_hat[1],
                              n_hat[2]*t_hat[0] - n_hat[0]*t_hat[2],
                              n_hat[0]*t_hat[1] - n_hat[1]*t_hat[0]])
            diff_i = r_obs - vi
            d = diff_i[0]*m_hat[0] + diff_i[1]*m_hat[1] + diff_i[2]*m_hat[2]
            h = diff_i[0]*n_hat[0] + diff_i[1]*n_hat[1] + diff_i[2]*n_hat[2]
            diff_j = vj - r_obs
            diff_ii = vi - r_obs
            s_plus = diff_j[0]*t_hat[0] + diff_j[1]*t_hat[1] + diff_j[2]*t_hat[2]
            s_minus = diff_ii[0]*t_hat[0] + diff_ii[1]*t_hat[1] + diff_ii[2]*t_hat[2]
            R_plus = np.sqrt(diff_j[0]**2 + diff_j[1]**2 + diff_j[2]**2)
            R_minus = np.sqrt(diff_ii[0]**2 + diff_ii[1]**2 + diff_ii[2]**2)
            if R_plus + s_plus > 1e-15 and R_minus + s_minus > 1e-15:
                log_arg = (R_plus + s_plus) / (R_minus + s_minus)
                if log_arg > 0:
                    result += d * np.log(log_arg)
            if abs(h) > 1e-15:
                R0_sq = d**2 + h**2
                t1 = np.arctan2(d * s_plus, R0_sq + abs(h) * R_plus)
                t2 = np.arctan2(d * s_minus, R0_sq + abs(h) * R_minus)
                result -= abs(h) * (t1 - t2)
        return result

    @njit(cache=True)
    def _integral_R_nb(r_obs, v0, v1, v2, quad_pts, quad_wts):
        """∫_T |r_obs - r'| dS' (numerical quadrature), Numba version."""
        Nq = len(quad_wts)
        e10 = v1 - v0; e20 = v2 - v0
        cr = np.array([e10[1]*e20[2] - e10[2]*e20[1],
                       e10[2]*e20[0] - e10[0]*e20[2],
                       e10[0]*e20[1] - e10[1]*e20[0]])
        area = 0.5 * np.sqrt(cr[0]**2 + cr[1]**2 + cr[2]**2)
        result = 0.0
        for q in range(Nq):
            lam0 = 1.0 - quad_pts[q, 0] - quad_pts[q, 1]
            rr = lam0 * v0 + quad_pts[q, 0] * v1 + quad_pts[q, 1] * v2
            dr = rr - r_obs
            R = np.sqrt(dr[0]**2 + dr[1]**2 + dr[2]**2)
            result += quad_wts[q] * R
        return area * result

    @njit(cache=True)
    def _vector_potential_nb(r_obs, v0, v1, v2, quad_pts, quad_wts):
        """∫_T r'/|r_obs - r'| dS', Numba version."""
        P = _potential_integral_nb(r_obs, v0, v1, v2)
        eps = 1e-6
        grad_W = np.zeros(3)
        for d in range(3):
            rp = r_obs.copy(); rp[d] += eps
            rm = r_obs.copy(); rm[d] -= eps
            grad_W[d] = (_integral_R_nb(rp, v0, v1, v2, quad_pts, quad_wts) -
                         _integral_R_nb(rm, v0, v1, v2, quad_pts, quad_wts)) / (2 * eps)
        return r_obs * P - grad_W

    @njit(cache=True, parallel=True)
    def _singular_corrections_nb(L_re, L_im, N, tri_p, tri_m, div_p, div_m,
                                  f_p, f_m, jw_p, jw_m, length, area_p, area_m,
                                  free_p, free_m, tris, verts,
                                  quad_pts, quad_wts, ik_re, ik_im, ik_inv_re, ik_inv_im,
                                  inv4pi):
        """Apply singular corrections in parallel over test functions m."""
        Nq = len(quad_wts)
        for m in prange(N):
            for t_half in range(2):
                if t_half == 0:
                    t_f = f_p[m]; t_div = div_p[m]; t_jw = jw_p[m]; t_tri = tri_p[m]
                else:
                    t_f = f_m[m]; t_div = div_m[m]; t_jw = jw_m[m]; t_tri = tri_m[m]
                t0 = tris[t_tri, 0]; t1_idx = tris[t_tri, 1]; t2 = tris[t_tri, 2]
                tv0 = verts[t0]; tv1 = verts[t1_idx]; tv2 = verts[t2]
                # Compute P and V at quad points for this triangle
                lam0_arr = 1.0 - quad_pts[:, 0] - quad_pts[:, 1]
                P_vals = np.zeros(Nq)
                V_vals = np.zeros((Nq, 3))
                for iq in range(Nq):
                    rq = lam0_arr[iq] * tv0 + quad_pts[iq, 0] * tv1 + quad_pts[iq, 1] * tv2
                    P_vals[iq] = _potential_integral_nb(rq, tv0, tv1, tv2)
                    V_vals[iq] = _vector_potential_nb(rq, tv0, tv1, tv2, quad_pts, quad_wts)
                # scalar_base = dot(P_vals, t_jw) * inv4pi
                scalar_base = 0.0
                for iq in range(Nq):
                    scalar_base += P_vals[iq] * t_jw[iq]
                scalar_base *= inv4pi
                # Loop over source basis functions on same triangle
                for n in range(N):
                    for s_half in range(2):
                        if s_half == 0:
                            s_tri = tri_p[n]
                        else:
                            s_tri = tri_m[n]
                        if s_tri != t_tri:
                            continue
                        if s_half == 0:
                            s_div = div_p[n]
                            s_coeff = length[n] / (2.0 * area_p[n])
                            s_free = free_p[n]
                            s_sign = 1.0
                        else:
                            s_div = div_m[n]
                            s_coeff = length[n] / (2.0 * area_m[n])
                            s_free = free_m[n]
                            s_sign = -1.0
                        # L_sing_scalar = -ik_inv * t_div * s_div * scalar_base
                        ls_re = -(ik_inv_re * t_div * s_div * scalar_base)
                        ls_im = -(ik_inv_im * t_div * s_div * scalar_base)
                        # vec_integral = sum(sum(t_f * fn_over_R, axis=1) * t_jw) * inv4pi
                        vec_re = 0.0
                        for iq in range(Nq):
                            fn0 = s_sign * s_coeff * (V_vals[iq, 0] - s_free[0] * P_vals[iq])
                            fn1 = s_sign * s_coeff * (V_vals[iq, 1] - s_free[1] * P_vals[iq])
                            fn2 = s_sign * s_coeff * (V_vals[iq, 2] - s_free[2] * P_vals[iq])
                            dot_val = t_f[iq, 0]*fn0 + t_f[iq, 1]*fn1 + t_f[iq, 2]*fn2
                            vec_re += dot_val * t_jw[iq]
                        vec_re *= inv4pi
                        # L[m, n] += L_sing_scalar + ik * vec_integral
                        L_re[m, n] += ls_re + ik_re * vec_re
                        L_im[m, n] += ls_im + ik_im * vec_re
        return L_re, L_im


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

    if HAS_NUMBA:
        # Numba-accelerated parallel singular corrections
        L_re = np.real(L).copy()
        L_im = np.imag(L).copy()
        ik_val = complex(ik)
        ik_inv_val = complex(ik_inv)
        _singular_corrections_nb(
            L_re, L_im, N,
            rwg['tri_p'].astype(np.int64), rwg['tri_m'].astype(np.int64),
            div_p, div_m, f_p, f_m, jw_p, jw_m,
            rwg['length'], rwg['area_p'], rwg['area_m'],
            rwg['free_p'], rwg['free_m'],
            tris.astype(np.int64), verts,
            quad_pts, quad_wts,
            ik_val.real, ik_val.imag,
            ik_inv_val.real, ik_inv_val.imag,
            inv4pi)
        L = L_re + 1j * L_im
    else:
        # Pure Python fallback
        tri_verts_cache = {}
        for ti in range(len(tris)):
            tri_verts_cache[ti] = (verts[tris[ti, 0]].copy(),
                                    verts[tris[ti, 1]].copy(),
                                    verts[tris[ti, 2]].copy())
        tri_to_rwg_p = {}
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
        tri_PV_cache = {}
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
        for m in range(N):
            for t_f_h, t_div_h, t_jw_h, t_tri_idx in [
                (f_p[m], div_p[m], jw_p[m], rwg['tri_p'][m]),
                (f_m[m], div_m[m], jw_m[m], rwg['tri_m'][m]),
            ]:
                P_vals, V_vals = tri_PV_cache[t_tri_idx]
                scalar_base = np.dot(P_vals, t_jw_h) * inv4pi
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


# ============================================================
# 12. Multi-body BEM for particle clusters
# ============================================================

def merge_meshes(bodies):
    """Merge multiple particle meshes into one.

    Parameters
    ----------
    bodies : list of dict
        Each dict has keys:
        - 'verts': ndarray (V, 3) — vertex positions
        - 'tris': ndarray (T, 3) — triangle indices
        - 'k_int': float/complex — interior wavenumber
        - 'eta_int': float/complex — interior impedance

    Returns
    -------
    verts : ndarray — merged vertex array
    tris : ndarray — merged triangle array
    body_ranges : list of (tri_start, tri_end) — triangle ranges per body
    """
    all_verts = []
    all_tris = []
    body_ranges = []
    vert_offset = 0
    tri_offset = 0

    for body in bodies:
        v = body['verts']
        t = body['tris']
        all_verts.append(v)
        all_tris.append(t + vert_offset)
        body_ranges.append((tri_offset, tri_offset + len(t)))
        vert_offset += len(v)
        tri_offset += len(t)

    return np.vstack(all_verts), np.vstack(all_tris).astype(np.int32), body_ranges


def assemble_multibody_pmchwt(bodies, k_ext, eta_ext, quad_order=7):
    """Assemble PMCHWT system for a cluster of dielectric particles.

    Each particle has its own interior (k_int, eta_int), but they all share
    the same exterior medium (k_ext, eta_ext). The coupling between particles
    is through the exterior Green's function only.

    Parameters
    ----------
    bodies : list of dict
        Each dict has keys:
        - 'verts': ndarray (V, 3)
        - 'tris': ndarray (T, 3)
        - 'k_int': float/complex
        - 'eta_int': float/complex
    k_ext : float
        Exterior wavenumber.
    eta_ext : float
        Exterior impedance.
    quad_order : int
        Quadrature order.

    Returns
    -------
    Z : ndarray (2N_total, 2N_total) — PMCHWT system matrix
    rwg_list : list of dict — RWG data for each body
    rwg_merged : dict — merged RWG data
    verts : ndarray — merged vertices
    tris : ndarray — merged triangles
    body_rwg_ranges : list of (start, end) — RWG index ranges per body
    """
    import time as _t

    # Merge meshes
    verts, tris, body_tri_ranges = merge_meshes(bodies)

    # Build RWG for merged mesh
    rwg = build_rwg(verts, tris)
    N = rwg['N']
    print(f"  Multi-body: {len(bodies)} particles, {len(tris)} tris, {N} RWG total")

    # Determine which RWG basis belongs to which body
    # An RWG edge belongs to body i if its tri_p falls in body i's triangle range
    body_rwg_ranges = []
    rwg_body = np.zeros(N, dtype=int)
    for bi, (t_start, t_end) in enumerate(body_tri_ranges):
        mask = (rwg['tri_p'] >= t_start) & (rwg['tri_p'] < t_end)
        rwg_body[mask] = bi
        indices = np.where(mask)[0]
        if len(indices) > 0:
            body_rwg_ranges.append((indices[0], indices[-1] + 1))
        else:
            body_rwg_ranges.append((0, 0))

    # Assemble exterior operators (full N×N coupling)
    t0 = _t.time()
    print(f"  Assembling exterior operators (k_ext={k_ext:.4f})...")
    L_ext, K_ext = assemble_L_K(rwg, verts, tris, k_ext, quad_order=quad_order)
    print(f"  Exterior assembly: {_t.time()-t0:.1f}s")

    # Build system matrix
    Z = np.zeros((2*N, 2*N), dtype=complex)

    # Exterior contribution (couples ALL bodies)
    Z[:N, :N] += eta_ext * L_ext
    Z[:N, N:] += -K_ext
    Z[N:, :N] += K_ext
    Z[N:, N:] += L_ext / eta_ext

    # Interior contribution (block-diagonal, each body independent)
    for bi, body in enumerate(bodies):
        rng = body_rwg_ranges[bi]
        n_start, n_end = rng
        n_bi = n_end - n_start

        if n_bi == 0:
            continue

        k_int = body['k_int']
        eta_int = body['eta_int']

        # Build RWG for this body alone
        body_verts = body['verts']
        body_tris = body['tris']
        body_rwg = build_rwg(body_verts, body_tris)

        t1 = _t.time()
        print(f"  Assembling interior operators body {bi} "
              f"(k_int={k_int}, {n_bi} RWG)...")
        L_int, K_int = assemble_L_K(body_rwg, body_verts, body_tris, k_int,
                                      quad_order=quad_order)
        print(f"  Interior assembly body {bi}: {_t.time()-t1:.1f}s")

        # Add interior contribution (block-diagonal)
        s = slice(n_start, n_end)
        Z[s, s] += eta_int * L_int
        Z[s, N+n_start:N+n_end] += K_int
        Z[N+n_start:N+n_end, s] += -K_int
        Z[N+n_start:N+n_end, N+n_start:N+n_end] += L_int / eta_int

    return Z, rwg, verts, tris, body_rwg_ranges


def compute_rhs_multibody(rwg, verts, tris, k_ext, eta_ext,
                            E0=None, k_hat=None, quad_order=7):
    """Compute RHS for multi-body PMCHWT (same as single-body)."""
    return compute_rhs_planewave(rwg, verts, tris, k_ext, eta_ext,
                                  E0=E0, k_hat=k_hat, quad_order=quad_order)


# ============================================================
# 13. H-matrix compression with ACA for O(N log N) matvec
# ============================================================

class HMatrix:
    """Hierarchical matrix using Adaptive Cross Approximation (ACA).

    Splits the Z matrix into near-field (dense) and far-field (low-rank)
    blocks based on basis function separation. Far-field blocks are compressed
    using ACA to rank-k approximations U·V^H.

    This enables O(N log N) matrix-vector products instead of O(N²).
    """

    def __init__(self, Z, rwg, verts, tris, eta=3.0, aca_tol=1e-4, max_rank=50):
        """Build H-matrix from dense matrix Z.

        Parameters
        ----------
        Z : ndarray (M, M) — full system matrix (2N×2N for PMCHWT)
        rwg : dict — RWG basis function data
        verts, tris : ndarray — mesh data
        eta : float — admissibility parameter (block is far-field if
              dist(cluster_i, cluster_j) > eta * max(diam_i, diam_j))
        aca_tol : float — ACA approximation tolerance
        max_rank : int — maximum rank for low-rank blocks
        """
        self.M = Z.shape[0]
        self.dtype = Z.dtype
        N = rwg['N']

        # Compute RWG centroids for clustering
        centroids = np.zeros((N, 3))
        for n in range(N):
            tp = rwg['tri_p'][n]; tm = rwg['tri_m'][n]
            cp = verts[tris[tp]].mean(axis=0)
            cm = verts[tris[tm]].mean(axis=0)
            centroids[n] = 0.5 * (cp + cm)

        # For PMCHWT (2N×2N), J and M share same spatial locations
        is_pmchwt = (self.M == 2 * N)
        if is_pmchwt:
            full_centroids = np.vstack([centroids, centroids])
        else:
            full_centroids = centroids

        # Build block structure using bisection clustering
        block_size = max(32, N // 16)
        self.blocks = []
        self._build_blocks(Z, full_centroids, block_size, eta, aca_tol, max_rank)

        # Statistics
        dense_entries = sum(b['rows'] * b['cols'] for b in self.blocks if b['type'] == 'dense')
        lr_entries = sum(b['rank'] * (b['rows'] + b['cols']) for b in self.blocks if b['type'] == 'lr')
        total = self.M * self.M
        ratio = (dense_entries + lr_entries) / total
        avg_rank = np.mean([b['rank'] for b in self.blocks if b['type'] == 'lr']) if any(b['type'] == 'lr' for b in self.blocks) else 0
        print(f"  H-matrix: {len(self.blocks)} blocks, "
              f"compression {ratio:.1%}, avg rank {avg_rank:.0f}")

    def _build_blocks(self, Z, centroids, block_size, eta, aca_tol, max_rank):
        """Recursively partition matrix into near/far-field blocks."""
        M = self.M
        # Simple 1D partition by spatial coordinate (longest axis)
        indices = np.arange(M)
        self._partition_recursive(Z, centroids, indices, indices,
                                   block_size, eta, aca_tol, max_rank)

    def _partition_recursive(self, Z, centroids, row_idx, col_idx,
                              block_size, eta, aca_tol, max_rank):
        nr = len(row_idx); nc = len(col_idx)

        if nr <= block_size or nc <= block_size:
            # Leaf block — check admissibility
            if self._is_admissible(centroids, row_idx, col_idx, eta):
                # Far-field: ACA compression
                self._add_lr_block(Z, row_idx, col_idx, aca_tol, max_rank)
            else:
                # Near-field: store dense
                self._add_dense_block(Z, row_idx, col_idx)
            return

        # Check admissibility before splitting
        if self._is_admissible(centroids, row_idx, col_idx, eta):
            self._add_lr_block(Z, row_idx, col_idx, aca_tol, max_rank)
            return

        # Split along longest axis
        row_split = self._bisect(centroids, row_idx)
        col_split = self._bisect(centroids, col_idx)

        for ri in row_split:
            for ci in col_split:
                self._partition_recursive(Z, centroids, ri, ci,
                                           block_size, eta, aca_tol, max_rank)

    def _bisect(self, centroids, indices):
        """Split indices into two groups along longest spatial axis."""
        pts = centroids[indices]
        span = pts.max(axis=0) - pts.min(axis=0)
        axis = np.argmax(span)
        median = np.median(pts[:, axis])
        mask = pts[:, axis] <= median
        idx1 = indices[mask]
        idx2 = indices[~mask]
        if len(idx1) == 0: idx1 = indices[:1]; idx2 = indices[1:]
        if len(idx2) == 0: idx1 = indices[:-1]; idx2 = indices[-1:]
        return [idx1, idx2]

    def _is_admissible(self, centroids, row_idx, col_idx, eta):
        """Check if block is admissible (far-field)."""
        r_pts = centroids[row_idx]; c_pts = centroids[col_idx]
        r_center = r_pts.mean(axis=0); c_center = c_pts.mean(axis=0)
        dist = np.linalg.norm(r_center - c_center)
        r_diam = np.max(np.linalg.norm(r_pts - r_center, axis=1)) * 2
        c_diam = np.max(np.linalg.norm(c_pts - c_center, axis=1)) * 2
        return dist > eta * max(r_diam, c_diam, 1e-15)

    def _add_dense_block(self, Z, row_idx, col_idx):
        self.blocks.append({
            'type': 'dense',
            'row_idx': row_idx,
            'col_idx': col_idx,
            'data': Z[np.ix_(row_idx, col_idx)].copy(),
            'rows': len(row_idx),
            'cols': len(col_idx),
        })

    def _add_lr_block(self, Z, row_idx, col_idx, tol, max_rank):
        """ACA with partial pivoting."""
        nr = len(row_idx); nc = len(col_idx)
        max_k = min(max_rank, min(nr, nc))

        U = np.zeros((nr, max_k), dtype=self.dtype)
        V = np.zeros((max_k, nc), dtype=self.dtype)

        used_rows = set()
        pivot_row = 0
        rank = 0
        ref_norm = 0.0

        for k in range(max_k):
            # Get row of residual
            row = Z[row_idx[pivot_row], col_idx].copy()
            for j in range(rank):
                row -= U[pivot_row, j] * V[j, :]

            # Find pivot column
            pivot_col = np.argmax(np.abs(row))
            if abs(row[pivot_col]) < 1e-15:
                break

            # Get column of residual
            col = Z[row_idx, col_idx[pivot_col]].copy()
            for j in range(rank):
                col -= U[:, j] * V[j, pivot_col]

            # Update
            V[rank, :] = row / row[pivot_col]
            U[:, rank] = col
            rank += 1

            # Check convergence
            uv_norm = np.linalg.norm(col) * np.linalg.norm(row) / abs(row[pivot_col])
            ref_norm += uv_norm
            if uv_norm < tol * ref_norm and rank > 1:
                break

            # Next pivot row
            used_rows.add(pivot_row)
            residual_col = np.abs(col)
            for r in used_rows:
                residual_col[r] = 0
            pivot_row = np.argmax(residual_col)

        self.blocks.append({
            'type': 'lr',
            'row_idx': row_idx,
            'col_idx': col_idx,
            'U': U[:, :rank].copy(),
            'V': V[:rank, :].copy(),
            'rank': rank,
            'rows': nr,
            'cols': nc,
        })

    def matvec(self, x):
        """Matrix-vector product y = Z·x using H-matrix."""
        y = np.zeros(self.M, dtype=self.dtype)
        for block in self.blocks:
            xb = x[block['col_idx']]
            if block['type'] == 'dense':
                y[block['row_idx']] += block['data'] @ xb
            else:
                y[block['row_idx']] += block['U'] @ (block['V'] @ xb)
        return y

    def memory_bytes(self):
        """Estimate memory usage in bytes."""
        total = 0
        for b in self.blocks:
            if b['type'] == 'dense':
                total += b['rows'] * b['cols'] * 16  # complex128
            else:
                total += b['rank'] * (b['rows'] + b['cols']) * 16
        return total


def solve_hmatrix_gmres(Z_hmat, b, tol=1e-6, maxiter=200, Z_diag_blocks=None):
    """Solve Z·x = b using GMRES with H-matrix matvec.

    Parameters
    ----------
    Z_hmat : HMatrix
        H-matrix approximation of Z.
    b : ndarray
        Right-hand side.
    tol : float
        GMRES tolerance.
    maxiter : int
        Maximum iterations.
    Z_diag_blocks : list of ndarray, optional
        Dense diagonal blocks for block-diagonal preconditioner.

    Returns
    -------
    x : ndarray — solution vector
    """
    from scipy.sparse.linalg import LinearOperator, gmres

    M = Z_hmat.M
    A = LinearOperator((M, M), matvec=Z_hmat.matvec, dtype=Z_hmat.dtype)

    precond = None
    if Z_diag_blocks is not None:
        # Block-diagonal preconditioner from dense diagonal blocks
        lu_blocks = [linalg.lu_factor(blk) for blk in Z_diag_blocks]
        sizes = [blk.shape[0] for blk in Z_diag_blocks]
        offsets = np.cumsum([0] + sizes)

        def precond_matvec(r):
            y = np.empty(M, dtype=complex)
            for i, lu in enumerate(lu_blocks):
                s = slice(offsets[i], offsets[i+1])
                y[s] = linalg.lu_solve(lu, r[s])
            return y
        precond = LinearOperator((M, M), matvec=precond_matvec, dtype=complex)

    iters = [0]
    def callback(pr_norm):
        iters[0] += 1

    x, info = gmres(A, b, rtol=tol, maxiter=maxiter, M=precond,
                     callback=callback, callback_type='pr_norm')
    print(f"  H-matrix GMRES: {iters[0]} iterations, info={info}")
    return x
