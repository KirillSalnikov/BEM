"""
Clean test: dielectric sphere PMCHWT, compare SNC vs RWG-RWG vs Mie.
Uses optical theorem for Q_ext (more reliable than angular integration for Q_sca).
"""
import numpy as np
import time
from bem_core import (icosphere, build_rwg, tri_quadrature,
                      assemble_L_K, assemble_pmchwt, compute_rhs_planewave,
                      assemble_L_K_snc, assemble_pmchwt_snc, compute_rhs_planewave_snc,
                      compute_far_field, compute_cross_sections)


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
    mx = m_rel * x
    Q_ext = 0.0; Q_sca = 0.0
    for n in range(1, n_max + 1):
        a_n = (m_rel * psi(n, mx) * psi_d(n, x) - psi(n, x) * psi_d(n, mx)) / \
              (m_rel * psi(n, mx) * xi_d(n, x) - xi(n, x) * psi_d(n, mx))
        b_n = (psi(n, mx) * psi_d(n, x) - m_rel * psi(n, x) * psi_d(n, mx)) / \
              (psi(n, mx) * xi_d(n, x) - m_rel * xi(n, x) * psi_d(n, mx))
        Q_ext += (2*n + 1) * np.real(a_n + b_n)
        Q_sca += (2*n + 1) * (abs(a_n)**2 + abs(b_n)**2)
    return 2 * Q_ext / x**2, 2 * Q_sca / x**2


def compute_Qext_optical_theorem(rwg, verts, tris, coeffs_J, coeffs_M,
                                  k_ext, eta_ext, radius, E0, k_hat, quad_order=7):
    """Q_ext via optical theorem: C_ext = (4π/k) Im[F(forward) · E0*]."""
    quad_pts, quad_wts = tri_quadrature(quad_order)
    N = rwg['N']
    lam0 = 1 - quad_pts[:,0] - quad_pts[:,1]

    def get_qpts(ti):
        t = tris[ti]; v0 = verts[t[:,0]]; v1 = verts[t[:,1]]; v2 = verts[t[:,2]]
        return np.einsum('q,ni->nqi', lam0, v0) + np.einsum('q,ni->nqi', quad_pts[:,0], v1) + \
               np.einsum('q,ni->nqi', quad_pts[:,1], v2)

    qp = get_qpts(rwg['tri_p']); qm = get_qpts(rwg['tri_m'])
    r_hat = k_hat  # forward direction

    J_int = np.zeros(3, dtype=complex)
    M_int = np.zeros(3, dtype=complex)
    for qpts, free, area, sign in [(qp, rwg['free_p'], rwg['area_p'], +1),
                                     (qm, rwg['free_m'], rwg['area_m'], -1)]:
        f = sign * (rwg['length'][:,None,None] / (2*area[:,None,None])) * (qpts - free[:,None,:])
        jw = area[:,None] * quad_wts[None,:]
        phase = np.exp(-1j * k_ext * np.einsum('i,nqi->nq', r_hat, qpts))
        weighted_J = f * (coeffs_J[:,None,None] * phase[:,:,None] * jw[:,:,None])
        weighted_M = f * (coeffs_M[:,None,None] * phase[:,:,None] * jw[:,:,None])
        J_int += weighted_J.sum(axis=(0, 1))
        M_int += weighted_M.sum(axis=(0, 1))

    # Far field pattern: F = -ik/(4π) * [η J_perp + r̂ × M]
    J_perp = J_int - r_hat * np.dot(r_hat, J_int)
    M_cross = np.cross(r_hat, M_int)
    F = -1j * k_ext / (4 * np.pi) * (eta_ext * J_perp + M_cross)

    # Optical theorem: C_ext = (4π/k) Im[F · E0*] / |E0|²
    S_fwd = np.dot(F, np.conj(E0))
    C_ext = 4 * np.pi / k_ext * np.imag(S_fwd) / np.abs(np.dot(E0, np.conj(E0)))
    Q_ext = C_ext / (np.pi * radius**2)
    return Q_ext, F


if __name__ == "__main__":
    radius = 1.0
    k_ext = 1.0
    m_rel = 1.5
    x = k_ext * radius
    k_int = k_ext * m_rel
    eta_ext = 1.0
    eta_int = 1.0 / m_rel
    E0 = np.array([1.0, 0, 0])
    k_hat = np.array([0, 0, 1.0])

    Q_ext_mie, Q_sca_mie = mie_Qext_Qsca(x, m_rel)
    print(f"Mie: Q_ext = {Q_ext_mie:.6f}, Q_sca = {Q_sca_mie:.6f}")

    for refine in [1, 2]:
        print(f"\n{'='*60}")
        print(f"Refinement = {refine}")
        verts, tris = icosphere(radius, refinements=refine)
        rwg = build_rwg(verts, tris)
        N = rwg['N']
        print(f"  Mesh: {len(verts)} verts, {len(tris)} tris, {N} RWG, system {2*N}x{2*N}")

        # --- SNC-tested PMCHWT ---
        print(f"\n  --- SNC-tested PMCHWT ---")
        t0 = time.time()
        Z_snc, L_ext_snc, K_ext_snc = assemble_pmchwt_snc(rwg, verts, tris,
                                                             k_ext, k_int, eta_ext, eta_int)
        b_snc = compute_rhs_planewave_snc(rwg, verts, tris, k_ext, eta_ext, E0, k_hat)
        t1 = time.time()
        print(f"  Assembly: {t1-t0:.1f}s")
        print(f"  cond(Z) = {np.linalg.cond(Z_snc):.2e}")
        print(f"  |b| = {np.linalg.norm(b_snc):.4e}")

        coeffs_snc = np.linalg.solve(Z_snc, b_snc)
        J_snc = coeffs_snc[:N]; M_snc = coeffs_snc[N:]
        print(f"  |J| = {np.linalg.norm(J_snc):.4e}, |M| = {np.linalg.norm(M_snc):.4e}")

        Q_ext_snc, F_fwd = compute_Qext_optical_theorem(
            rwg, verts, tris, J_snc, M_snc, k_ext, eta_ext, radius, E0, k_hat)
        err_snc = abs(Q_ext_snc - Q_ext_mie) / Q_ext_mie * 100
        print(f"  Q_ext = {Q_ext_snc:.6f}  (Mie: {Q_ext_mie:.6f}, err: {err_snc:.1f}%)")

        # Also compute Q_sca by angular integration
        theta_arr = np.linspace(0.01, np.pi - 0.01, 181)
        F_theta, F_phi = compute_far_field(rwg, verts, tris, J_snc, M_snc,
                                            k_ext, theta_arr, phi=0.0)
        dsigma, C_sca = compute_cross_sections(F_theta, F_phi, theta_arr, k_ext)
        Q_sca_snc = C_sca / (np.pi * radius**2)
        err_sca = abs(Q_sca_snc - Q_sca_mie) / Q_sca_mie * 100
        print(f"  Q_sca = {Q_sca_snc:.6f}  (Mie: {Q_sca_mie:.6f}, err: {err_sca:.1f}%)")

        # --- RWG-RWG PMCHWT (for comparison) ---
        print(f"\n  --- RWG-RWG PMCHWT ---")
        t0 = time.time()
        Z_rwg, L_ext_rwg, K_ext_rwg = assemble_pmchwt(rwg, verts, tris,
                                                         k_ext, k_int, eta_ext, eta_int)
        b_rwg = compute_rhs_planewave(rwg, verts, tris, k_ext, eta_ext, E0, k_hat)
        t1 = time.time()
        print(f"  Assembly: {t1-t0:.1f}s")
        print(f"  cond(Z) = {np.linalg.cond(Z_rwg):.2e}")

        coeffs_rwg = np.linalg.solve(Z_rwg, b_rwg)
        J_rwg = coeffs_rwg[:N]; M_rwg = coeffs_rwg[N:]

        Q_ext_rwg, _ = compute_Qext_optical_theorem(
            rwg, verts, tris, J_rwg, M_rwg, k_ext, eta_ext, radius, E0, k_hat)
        err_rwg = abs(Q_ext_rwg - Q_ext_mie) / Q_ext_mie * 100
        print(f"  Q_ext = {Q_ext_rwg:.6f}  (err: {err_rwg:.1f}%)")

        F_theta2, F_phi2 = compute_far_field(rwg, verts, tris, J_rwg, M_rwg,
                                              k_ext, theta_arr, phi=0.0)
        dsigma2, C_sca2 = compute_cross_sections(F_theta2, F_phi2, theta_arr, k_ext)
        Q_sca_rwg = C_sca2 / (np.pi * radius**2)
        err_sca2 = abs(Q_sca_rwg - Q_sca_mie) / Q_sca_mie * 100
        print(f"  Q_sca = {Q_sca_rwg:.6f}  (err: {err_sca2:.1f}%)")
