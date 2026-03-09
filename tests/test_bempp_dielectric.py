"""
Test bempp-cl on dielectric sphere with PMCHWT (multitrace) formulation.
Compare Q_sca with Mie theory.
"""
import numpy as np
import time
import bempp_cl.api as bempp_api

bempp_api.set_logging_level("warning")


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


def test_dielectric(x=1.0, m_rel=1.5, refine=3):
    radius = x  # k_ext = 1
    k_ext = 1.0
    k_int = k_ext * m_rel
    eps_r = m_rel**2
    mu_r = 1.0

    print(f"\n=== Dielectric Sphere (x={x}, m={m_rel}, refine={refine}) ===")
    Q_ext_mie, Q_sca_mie = mie_Qext_Qsca(x, m_rel)
    print(f"  Mie: Q_ext = {Q_ext_mie:.6f}, Q_sca = {Q_sca_mie:.6f}")

    verts, tris = icosphere(radius, refinements=refine)
    grid = bempp_api.Grid(verts.T, tris.T)
    print(f"  Mesh: {grid.number_of_vertices} verts, {grid.number_of_elements} elements")

    # Multitrace operator (PMCHWT formulation from tutorial)
    from bempp_cl.api.operators.boundary.maxwell import multitrace_operator
    A_int = multitrace_operator(grid, k_int, epsilon_r=eps_r, mu_r=mu_r, space_type="all_rwg")
    A_ext = multitrace_operator(grid, k_ext, space_type="all_rwg")
    A = A_int + A_ext

    print(f"  DOFs: {sum(s.global_dof_count for s in A.domain_spaces)}")

    # Incident plane wave: E = x_hat * exp(ikz)
    direction = np.array([0.0, 0.0, 1.0])
    polarization = np.array([1.0, 0.0, 0.0])

    @bempp_api.complex_callable
    def tangential_trace(x, n, domain_index, result):
        e_inc = polarization * np.exp(1j * k_ext * (direction[0]*x[0] + direction[1]*x[1] + direction[2]*x[2]))
        result[0] = e_inc[1]*n[2] - e_inc[2]*n[1]
        result[1] = e_inc[2]*n[0] - e_inc[0]*n[2]
        result[2] = e_inc[0]*n[1] - e_inc[1]*n[0]

    @bempp_api.complex_callable
    def neumann_trace(x, n, domain_index, result):
        phase = np.exp(1j * k_ext * (direction[0]*x[0] + direction[1]*x[1] + direction[2]*x[2]))
        # curl E = ik (direction × polarization) * exp(ik·r)
        curl_E = np.array([
            1j * k_ext * (direction[1]*polarization[2] - direction[2]*polarization[1]),
            1j * k_ext * (direction[2]*polarization[0] - direction[0]*polarization[2]),
            1j * k_ext * (direction[0]*polarization[1] - direction[1]*polarization[0]),
        ]) * phase
        # neumann_trace = (1/ik) * curl_E × n
        inv_ik = 1.0 / (1j * k_ext)
        result[0] = inv_ik * (curl_E[1]*n[2] - curl_E[2]*n[1])
        result[1] = inv_ik * (curl_E[2]*n[0] - curl_E[0]*n[2])
        result[2] = inv_ik * (curl_E[0]*n[1] - curl_E[1]*n[0])

    rhs = [
        bempp_api.GridFunction(space=A.range_spaces[0], dual_space=A.dual_to_range_spaces[0], fun=tangential_trace),
        bempp_api.GridFunction(space=A.range_spaces[1], dual_space=A.dual_to_range_spaces[1], fun=neumann_trace),
    ]

    # Solve
    print(f"  Solving...")
    t0 = time.time()
    sol = bempp_api.linalg.lu(A, rhs)
    t1 = time.time()
    print(f"  LU solve: time={t1-t0:.2f}s")

    # Far field — same convention as tutorial:
    # far_field = -electric_far * sol[1] - magnetic_far * sol[0]
    n_theta = 91; n_phi = 72
    theta_1d = np.linspace(0.01, np.pi - 0.01, n_theta)
    phi_1d = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    theta_2d, phi_2d = np.meshgrid(theta_1d, phi_1d, indexing='ij')
    theta_flat = theta_2d.ravel()
    phi_flat = phi_2d.ravel()

    points = np.vstack([np.sin(theta_flat)*np.cos(phi_flat),
                         np.sin(theta_flat)*np.sin(phi_flat),
                         np.cos(theta_flat)])

    electric_far = bempp_api.operators.far_field.maxwell.electric_field(sol[1].space, points, k_ext)
    magnetic_far = bempp_api.operators.far_field.maxwell.magnetic_field(sol[0].space, points, k_ext)
    E_far = -electric_far * sol[1] - magnetic_far * sol[0]

    # Forward direction for optical theorem
    fwd_point = np.array([[0.0], [0.0], [1.0]])
    efar_fwd = bempp_api.operators.far_field.maxwell.electric_field(sol[1].space, fwd_point, k_ext)
    mfar_fwd = bempp_api.operators.far_field.maxwell.magnetic_field(sol[0].space, fwd_point, k_ext)
    E_far_fwd = (-efar_fwd * sol[1] - mfar_fwd * sol[0]).ravel()
    print(f"  E_far_forward = {E_far_fwd}")

    # Optical theorem
    dot_fwd = np.dot(E_far_fwd, polarization)
    C_ext_ot = (4*np.pi/k_ext) * np.imag(dot_fwd)
    Q_ext_ot = C_ext_ot / (np.pi * radius**2)

    # Scattering cross section
    dsigma_flat = np.sum(np.abs(E_far)**2, axis=0)
    dsigma_2d = dsigma_flat.reshape(n_theta, n_phi)
    dphi = 2 * np.pi / n_phi
    integrand_theta = np.sum(dsigma_2d, axis=1) * dphi
    C_sca = np.trapezoid(integrand_theta * np.sin(theta_1d), theta_1d)
    Q_sca_bem = C_sca / (np.pi * radius**2)

    print(f"\n  Q_ext (opt. thm): {Q_ext_ot:.6f}")
    print(f"  Q_ext (Mie):      {Q_ext_mie:.6f}")
    print(f"  Q_ext error:      {abs(Q_ext_ot - Q_ext_mie)/Q_ext_mie*100:.2f}%")
    print(f"\n  Q_sca (BEM):      {Q_sca_bem:.6f}")
    print(f"  Q_sca (Mie):      {Q_sca_mie:.6f}")
    print(f"  Q_sca error:      {abs(Q_sca_bem - Q_sca_mie)/Q_sca_mie*100:.2f}%")

    return Q_sca_bem, Q_sca_mie


if __name__ == "__main__":
    for refine in [2, 3]:
        test_dielectric(x=1.0, m_rel=1.5, refine=refine)
