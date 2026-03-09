"""
Quick FMM test: x=5, m=1.311, refine=4 (15360 DOFs).
At this size FMM should beat dense.
Compare: FMM GMRES vs ADDA.
"""
import numpy as np
import time
import subprocess, os, tempfile
import bempp_cl.api as bempp_api

bempp_api.set_logging_level("warning")
ADDA = "/home/serg/GO-for-ADDA/adda/src/seq/adda"


def mie_Qext(x, m_rel, n_max=None):
    from scipy.special import spherical_jn, spherical_yn
    if n_max is None:
        n_max = int(x + 4*x**(1/3) + 2) + 5
    def psi(n, z): return z * spherical_jn(n, z)
    def psi_d(n, z): return spherical_jn(n, z) + z * spherical_jn(n, z, derivative=True)
    def xi(n, z): return z * (spherical_jn(n, z) + 1j * spherical_yn(n, z))
    def xi_d(n, z):
        return (spherical_jn(n, z) + 1j * spherical_yn(n, z)) + \
               z * (spherical_jn(n, z, derivative=True) + 1j * spherical_yn(n, z, derivative=True))
    mx = m_rel * x; Q = 0.0
    for n in range(1, n_max + 1):
        a_n = (m_rel * psi(n, mx) * psi_d(n, x) - psi(n, x) * psi_d(n, mx)) / \
              (m_rel * psi(n, mx) * xi_d(n, x) - xi(n, x) * psi_d(n, mx))
        b_n = (psi(n, mx) * psi_d(n, x) - m_rel * psi(n, x) * psi_d(n, mx)) / \
              (psi(n, mx) * xi_d(n, x) - m_rel * xi(n, x) * psi_d(n, mx))
        Q += (2*n + 1) * np.real(a_n + b_n)
    return 2 * Q / x**2


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
            if key in edge_midpoints: return edge_midpoints[key]
            mid = (np.array(verts_list[v0]) + np.array(verts_list[v1])) / 2
            mid /= np.linalg.norm(mid)
            idx = len(verts_list); verts_list.append(mid); edge_midpoints[key] = idx
            return idx
        new_tris = []
        for tri in tris:
            a, b, c = tri
            ab = get_midpoint(a, b); bc = get_midpoint(b, c); ca = get_midpoint(c, a)
            new_tris.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]])
        verts = np.array(verts_list); tris = np.array(new_tris, dtype=np.int32)
    verts *= radius
    return verts, tris


x = 5.0; m = 1.311; refine = 4
k_ext = 1.0; k_int = k_ext * m
eps_r = m**2; mu_r = 1.0
radius = x
polarization = np.array([1.0, 0.0, 0.0])
direction = np.array([0.0, 0.0, 1.0])

Q_mie = mie_Qext(x, m)
print(f"Mie: Q_ext = {Q_mie:.6f}")

# ADDA
with tempfile.TemporaryDirectory() as tmpdir:
    cmd = [ADDA, "-shape", "sphere", "-eq_rad", str(x),
           "-m", str(m), "0", "-lambda", str(2*np.pi),
           "-dpl", "10", "-dir", tmpdir, "-no_vol_cor"]
    t0 = time.time()
    subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    t_a = time.time() - t0
    with open(os.path.join(tmpdir, "CrossSec-Y")) as f:
        for line in f:
            if line.strip().startswith("Qext"):
                Q_a = float(line.split('=')[1].strip())
                break
    n_dip = 0
    with open(os.path.join(tmpdir, "log")) as f:
        for line in f:
            if "Total number of occupied dipoles" in line:
                n_dip = int(line.split(':')[1].strip())
print(f"ADDA: Q_ext = {Q_a:.6f} ({abs(Q_a-Q_mie)/Q_mie*100:.2f}%), "
      f"dips={n_dip}, t={t_a:.2f}s")

# bempp FMM
print(f"\nbempp FMM (ref={refine})...")
t0 = time.time()
verts, tris = icosphere(radius, refinements=refine)
grid = bempp_api.Grid(verts.T, tris.T)
print(f"  mesh: {len(tris)} tris, {grid.number_of_vertices} verts")

from bempp_cl.api.operators.boundary.maxwell import multitrace_operator
A_int = multitrace_operator(grid, k_int, epsilon_r=eps_r, mu_r=mu_r,
                             space_type="all_rwg", assembler="fmm")
A_ext = multitrace_operator(grid, k_ext, space_type="all_rwg", assembler="fmm")
A = A_int + A_ext
n_dof = sum(s.global_dof_count for s in A.domain_spaces)
t_setup = time.time() - t0
print(f"  setup: {t_setup:.1f}s, DOFs={n_dof}")

@bempp_api.complex_callable
def tangential_trace(x, n, domain_index, result):
    e_inc = polarization * np.exp(1j * k_ext * (direction[0]*x[0] + direction[1]*x[1] + direction[2]*x[2]))
    result[0] = e_inc[1]*n[2] - e_inc[2]*n[1]
    result[1] = e_inc[2]*n[0] - e_inc[0]*n[2]
    result[2] = e_inc[0]*n[1] - e_inc[1]*n[0]

@bempp_api.complex_callable
def neumann_trace(x, n, domain_index, result):
    phase = np.exp(1j * k_ext * (direction[0]*x[0] + direction[1]*x[1] + direction[2]*x[2]))
    curl_E = np.array([
        1j * k_ext * (direction[1]*polarization[2] - direction[2]*polarization[1]),
        1j * k_ext * (direction[2]*polarization[0] - direction[0]*polarization[2]),
        1j * k_ext * (direction[0]*polarization[1] - direction[1]*polarization[0]),
    ]) * phase
    inv_ik = 1.0 / (1j * k_ext)
    result[0] = inv_ik * (curl_E[1]*n[2] - curl_E[2]*n[1])
    result[1] = inv_ik * (curl_E[2]*n[0] - curl_E[0]*n[2])
    result[2] = inv_ik * (curl_E[0]*n[1] - curl_E[1]*n[0])

rhs = [
    bempp_api.GridFunction(space=A.range_spaces[0], dual_space=A.dual_to_range_spaces[0], fun=tangential_trace),
    bempp_api.GridFunction(space=A.range_spaces[1], dual_space=A.dual_to_range_spaces[1], fun=neumann_trace),
]

print(f"  solving GMRES...")
t_s = time.time()
sol, info, residuals, n_iter = bempp_api.linalg.gmres(
    A, rhs, tol=1e-4, restart=300, maxiter=1000,
    use_strong_form=False,
    return_residuals=True, return_iteration_count=True)
t_solve = time.time() - t_s
print(f"  solve: {t_solve:.1f}s, iters={n_iter}, info={info}")

# Q_ext
fwd = np.array([[0.0], [0.0], [1.0]])
ef = bempp_api.operators.far_field.maxwell.electric_field(sol[1].space, fwd, k_ext)
mf = bempp_api.operators.far_field.maxwell.magnetic_field(sol[0].space, fwd, k_ext)
E_fwd = (-ef * sol[1] - mf * sol[0]).ravel()
Q_ext = (4*np.pi/k_ext) * np.imag(np.dot(E_fwd, polarization)) / (np.pi * radius**2)

t_tot = time.time() - t0
print(f"\n  Q_ext = {Q_ext:.6f} ({abs(Q_ext-Q_mie)/Q_mie*100:.2f}%)")
print(f"  total: {t_tot:.1f}s")
