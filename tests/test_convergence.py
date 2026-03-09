"""
Test convergence of the standard PMCHWT with fixed L at ref=1,2,3.
Also verify Q_ext = Q_sca for lossless dielectric.
"""
import numpy as np
import time
from bem_core import (icosphere, build_rwg, tri_quadrature,
                      assemble_L_K, compute_far_field, compute_cross_sections)


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


if __name__ == "__main__":
    radius = 1.0; k_ext = 1.0; m_rel = 1.5; x = k_ext * radius
    k_int = k_ext * m_rel; eta_ext = 1.0; eta_int = 1.0 / m_rel
    E0 = np.array([1.0, 0, 0]); k_hat = np.array([0, 0, 1.0])

    Q_ext_mie, Q_sca_mie = mie_Qext_Qsca(x, m_rel)
    print(f"Mie: Q_ext = {Q_ext_mie:.6f}, Q_sca = {Q_sca_mie:.6f}")

    quad_pts_rhs, quad_wts_rhs = tri_quadrature(7)
    lam0_rhs = 1 - quad_pts_rhs[:, 0] - quad_pts_rhs[:, 1]
    theta_arr = np.linspace(0.01, np.pi - 0.01, 361)

    for refine in [1, 2, 3]:
        print(f"\n{'='*60}")
        t0 = time.time()
        verts, tris = icosphere(radius, refinements=refine)
        rwg = build_rwg(verts, tris)
        N = rwg['N']
        print(f"Refine={refine}: {len(tris)} tris, {N} RWG, system {2*N}x{2*N}")

        L_ext, K_ext = assemble_L_K(rwg, verts, tris, k_ext)
        L_int, K_int = assemble_L_K(rwg, verts, tris, k_int)

        # RHS
        def get_qpts(ti):
            t = tris[ti]; v0 = verts[t[:,0]]; v1 = verts[t[:,1]]; v2 = verts[t[:,2]]
            return np.einsum('q,ni->nqi', lam0_rhs, v0) + np.einsum('q,ni->nqi', quad_pts_rhs[:,0], v1) + \
                   np.einsum('q,ni->nqi', quad_pts_rhs[:,1], v2)
        qp = get_qpts(rwg['tri_p']); qm = get_qpts(rwg['tri_m'])
        H0 = np.cross(k_hat, E0) / eta_ext
        b = np.zeros(2*N, dtype=complex)
        for qpts, free, area, sign in [(qp, rwg['free_p'], rwg['area_p'], +1),
                                         (qm, rwg['free_m'], rwg['area_m'], -1)]:
            f = sign * (rwg['length'][:,None,None] / (2*area[:,None,None])) * (qpts - free[:,None,:])
            jw = area[:,None] * quad_wts_rhs[None,:]
            phase = np.exp(1j * k_ext * np.einsum('i,nqi->nq', k_hat, qpts))
            b[:N] += np.sum(np.einsum('nqi,i->nq', f, E0) * phase * jw, axis=1)
            b[N:] += np.sum(np.einsum('nqi,i->nq', f, H0) * phase * jw, axis=1)

        # Standard PMCHWT: [ηL, +K; -K, L/η]
        Z = np.zeros((2*N, 2*N), dtype=complex)
        Z[:N, :N] = eta_ext * L_ext + eta_int * L_int
        Z[:N, N:] = K_ext + K_int
        Z[N:, :N] = -(K_ext + K_int)
        Z[N:, N:] = L_ext / eta_ext + L_int / eta_int

        coeffs = np.linalg.solve(Z, b)
        J = coeffs[:N]; M = coeffs[N:]

        # Far field
        F_th, F_ph = compute_far_field(rwg, verts, tris, J, M, k_ext, theta_arr)
        ds, Cs = compute_cross_sections(F_th, F_ph, theta_arr, k_ext)
        Q_sca = Cs / (np.pi * radius**2)

        # Q_ext via optical theorem using the SAME far-field computation
        # Forward direction is theta=0 which is at the start of theta_arr
        # Actually let's compute forward separately for accuracy
        F_th_fwd, F_ph_fwd = compute_far_field(rwg, verts, tris, J, M, k_ext,
                                                 np.array([1e-6]))
        # For x-polarized, forward: F_theta * cos(0)*cos(0) - F_phi * sin(0) ≈ F_theta
        # Actually at theta≈0, phi=0: theta_hat = (cos0*cos0, cos0*sin0, -sin0) ≈ (1,0,0)
        # So F · E0 = F_theta (at theta≈0, phi=0)
        S_fwd = F_th_fwd[0]  # F(forward) · polarization
        C_ext = 4 * np.pi / k_ext * np.imag(S_fwd)
        Q_ext = C_ext / (np.pi * radius**2)

        t1 = time.time()
        eS = abs(Q_sca - Q_sca_mie) / Q_sca_mie * 100
        eE = abs(Q_ext - Q_ext_mie) / Q_ext_mie * 100
        print(f"  Q_sca = {Q_sca:.6f} (err={eS:.1f}%)  Q_ext = {Q_ext:.6f} (err={eE:.1f}%)  time={t1-t0:.1f}s")
        print(f"  F_fwd = {F_th_fwd[0]:.6f},  Im = {np.imag(F_th_fwd[0]):.6f}")
        print(f"  cond(Z) = {np.linalg.cond(Z):.2e}")
        print(f"  |J|={np.linalg.norm(J):.3e}, |M|={np.linalg.norm(M):.3e}")

        # Also test with sK1=+1, sK2=+1 (symmetric K)
        Z2 = np.zeros((2*N, 2*N), dtype=complex)
        Z2[:N, :N] = eta_ext * L_ext + eta_int * L_int
        Z2[:N, N:] = K_ext + K_int
        Z2[N:, :N] = K_ext + K_int  # +K instead of -K
        Z2[N:, N:] = L_ext / eta_ext + L_int / eta_int
        coeffs2 = np.linalg.solve(Z2, b)
        J2 = coeffs2[:N]; M2 = coeffs2[N:]
        F_th2, F_ph2 = compute_far_field(rwg, verts, tris, J2, M2, k_ext, theta_arr)
        ds2, Cs2 = compute_cross_sections(F_th2, F_ph2, theta_arr, k_ext)
        Q_sca2 = Cs2 / (np.pi * radius**2)
        eS2 = abs(Q_sca2 - Q_sca_mie) / Q_sca_mie * 100
        print(f"  Symmetric K: Q_sca = {Q_sca2:.6f} (err={eS2:.1f}%)")
