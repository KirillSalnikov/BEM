"""
Test bempp-cl on PEC and dielectric sphere, compare with Mie.
Uses icosphere mesh (no gmsh dependency).
"""
import numpy as np
import time
import bempp_cl.api as bempp_api

# Suppress excessive logging
bempp_api.set_logging_level("warning")

# ============================================================
# Mie theory
# ============================================================

def mie_Qext_Qsca(x, m_rel, n_max=None):
    from scipy.special import spherical_jn, spherical_yn
    if n_max is None:
        n_max = int(x + 4*x**(1/3) + 2) + 5
    def psi(n, z): return z * spherical_jn(n, z)
    def psi_d(n, z): return spherical_jn(n, z) + z * spherical_jn(n, z, derivative=True)
    def xi(n, z): return z * (spherical_jn(n, z) + 1j * spherical_yn(n, z))
    def xi_d(n, z):
        return (spherical_jn(n, z) + 1j * spherical_yn(n, z)) + \
               z * (spherical_jn(n, z, derivative=True) + 1j * spherical_yn(n, z, derivative=True))
    mx = m_rel * x; Q_ext = 0.0; Q_sca = 0.0
    for n in range(1, n_max + 1):
        a_n = (m_rel * psi(n, mx) * psi_d(n, x) - psi(n, x) * psi_d(n, mx)) / \
              (m_rel * psi(n, mx) * xi_d(n, x) - xi(n, x) * psi_d(n, mx))
        b_n = (psi(n, mx) * psi_d(n, x) - m_rel * psi(n, x) * psi_d(n, mx)) / \
              (psi(n, mx) * xi_d(n, x) - m_rel * xi(n, x) * psi_d(n, mx))
        Q_ext += (2*n + 1) * np.real(a_n + b_n)
        Q_sca += (2*n + 1) * (abs(a_n)**2 + abs(b_n)**2)
    return 2 * Q_ext / x**2, 2 * Q_sca / x**2


def mie_pec_Q(x, n_max=30):
    from scipy.special import spherical_jn, spherical_yn
    def psi(n, z): return z * spherical_jn(n, z)
    def psi_d(n, z): return spherical_jn(n, z) + z * spherical_jn(n, z, derivative=True)
    def xi(n, z): return z * (spherical_jn(n, z) + 1j * spherical_yn(n, z))
    def xi_d(n, z):
        return (spherical_jn(n, z) + 1j * spherical_yn(n, z)) + \
               z * (spherical_jn(n, z, derivative=True) + 1j * spherical_yn(n, z, derivative=True))
    Q = 0.0
    for n in range(1, n_max + 1):
        a_n = psi(n, x) / xi(n, x)
        b_n = psi_d(n, x) / xi_d(n, x)
        Q += (2*n+1) * (abs(a_n)**2 + abs(b_n)**2)
    return 2 * Q / x**2


# ============================================================
# Icosphere mesh for bempp
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


def make_bempp_grid(verts, tris):
    """Convert icosphere mesh to bempp Grid."""
    return bempp_api.Grid(verts.T, tris.T)


# ============================================================
# PEC sphere test
# ============================================================

def test_pec(x=1.0, refine=3):
    radius = x; k = 1.0
    print(f"\n=== PEC Sphere (x={x}, refine={refine}) ===")
    Q_mie = mie_pec_Q(x)
    print(f"  Mie Q = {Q_mie:.6f}")

    verts, tris = icosphere(radius, refinements=refine)
    grid = make_bempp_grid(verts, tris)
    print(f"  Mesh: {grid.number_of_vertices} verts, {grid.number_of_elements} elements")

    # RWG space
    rwg = bempp_api.function_space(grid, "RWG", 0)
    snc = bempp_api.function_space(grid, "SNC", 0)
    print(f"  DOFs: {rwg.global_dof_count}")

    # EFIE
    efie = bempp_api.operators.boundary.maxwell.electric_field(rwg, rwg, snc, k)

    # Incident plane wave: E = x_hat * exp(ikz)
    # For EFIE: L[J] = -E_inc_tan
    # In bempp with RWG/SNC, provide the incident E-field
    @bempp_api.complex_callable
    def inc_E(x, n, domain_index, result):
        result[0] = np.exp(1j * k * x[2])
        result[1] = 0.0 + 0j
        result[2] = 0.0 + 0j

    # The EFIE is: efie @ J = rhs where rhs = -<test, E_inc>
    rhs = bempp_api.GridFunction(snc, fun=inc_E)

    # Solve
    t0 = time.time()
    sol, info = bempp_api.gmres(efie, rhs, tol=1e-5)
    t1 = time.time()
    print(f"  GMRES: info={info}, time={t1-t0:.2f}s")

    # Far field — integrate over both θ and φ
    n_theta = 91; n_phi = 72
    theta_1d = np.linspace(0.01, np.pi - 0.01, n_theta)
    phi_1d = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    theta_2d, phi_2d = np.meshgrid(theta_1d, phi_1d, indexing='ij')
    theta_flat = theta_2d.ravel()
    phi_flat = phi_2d.ravel()

    points = np.vstack([np.sin(theta_flat)*np.cos(phi_flat),
                         np.sin(theta_flat)*np.sin(phi_flat),
                         np.cos(theta_flat)])

    far_op = bempp_api.operators.far_field.maxwell.electric_field(rwg, points, k)
    E_far = far_op @ sol  # (3, N_directions)

    dsigma_flat = np.sum(np.abs(E_far)**2, axis=0)
    dsigma_2d = dsigma_flat.reshape(n_theta, n_phi)

    # Integrate: C_sca = ∫∫ |F|² sin(θ) dθ dφ
    dphi = 2 * np.pi / n_phi
    integrand_theta = np.sum(dsigma_2d, axis=1) * dphi  # integrate over phi
    C_sca = np.trapezoid(integrand_theta * np.sin(theta_1d), theta_1d)
    Q_bem = C_sca / (np.pi * radius**2)

    print(f"  Q_sca (BEM): {Q_bem:.6f}")
    print(f"  Q (Mie):     {Q_mie:.6f}")
    print(f"  Error:       {abs(Q_bem - Q_mie)/Q_mie*100:.2f}%")

    return Q_bem, Q_mie


if __name__ == "__main__":
    # Try different normalizations
    for refine in [2, 3]:
        test_pec(x=1.0, refine=refine)
