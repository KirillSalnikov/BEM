"""
Compute Q_sca with full 2D angular integration over (θ,φ).
"""
import numpy as np
from bem_core import (icosphere, build_rwg, tri_quadrature, assemble_L_K)


def mie_pec_Qsca(x, n_max=30):
    from scipy.special import spherical_jn, spherical_yn
    def psi(n, z): return z * spherical_jn(n, z)
    def psi_d(n, z): return spherical_jn(n, z) + z * spherical_jn(n, z, derivative=True)
    def xi(n, z): return z * (spherical_jn(n, z) + 1j * spherical_yn(n, z))
    def xi_d(n, z):
        return (spherical_jn(n, z) + 1j * spherical_yn(n, z)) + \
               z * (spherical_jn(n, z, derivative=True) + 1j * spherical_yn(n, z, derivative=True))
    Q = 0.0
    for n in range(1, n_max+1):
        a_n = psi(n, x) / xi(n, x)
        b_n = psi_d(n, x) / xi_d(n, x)
        Q += (2*n+1) * (abs(a_n)**2 + abs(b_n)**2)
    return 2 * Q / x**2


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


def compute_Qsca_2d(rwg, verts, tris, J, M, k, eta, radius, sM=+1, ntheta=91, nphi=72):
    """Full 2D integration over (theta, phi)."""
    quad_pts, quad_wts = tri_quadrature(7)
    N = rwg['N']
    lam0 = 1 - quad_pts[:,0] - quad_pts[:,1]
    def get_qpts(ti):
        t = tris[ti]; v0 = verts[t[:,0]]; v1 = verts[t[:,1]]; v2 = verts[t[:,2]]
        return np.einsum('q,ni->nqi', lam0, v0) + np.einsum('q,ni->nqi', quad_pts[:,0], v1) + \
               np.einsum('q,ni->nqi', quad_pts[:,1], v2)
    qp = get_qpts(rwg['tri_p']); qm = get_qpts(rwg['tri_m'])

    theta_arr = np.linspace(0.01, np.pi - 0.01, ntheta)
    phi_arr = np.linspace(0, 2*np.pi, nphi, endpoint=False)

    # Precompute current integrals for each source half
    # For each direction (theta, phi), we need ∫ f_n e^{-ikr̂·r'} dS
    # This is the bottleneck - let's compute it for all directions at once

    C_sca_total = 0.0
    Q_ext_fwd = None

    for ip, phi in enumerate(phi_arr):
        dsigma_phi = np.zeros(ntheta)
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
                phase = np.exp(-1j * k * np.einsum('i,nqi->nq', r_hat, qpts))
                integral = (f * (phase * jw)[:,:,None]).sum(axis=1)
                Jt += (integral * J[:,None]).sum(0)
                Mt += (integral * M[:,None]).sum(0)
            Jp = Jt - r_hat * np.dot(r_hat, Jt)
            Mc = np.cross(r_hat, Mt)
            Fv = -1j * k / (4*np.pi) * (eta * Jp + sM * Mc)
            dsigma_phi[it] = abs(np.dot(Fv, theta_hat))**2 + abs(np.dot(Fv, phi_hat))**2

            # Q_ext from forward direction (first phi only)
            if ip == 0 and it == 0:
                F_th_fwd = np.dot(Fv, theta_hat)
                Q_ext_fwd = 4 * np.pi / k * np.imag(F_th_fwd) / (np.pi * radius**2)

        dphi = 2 * np.pi / nphi
        C_sca_total += dphi * np.trapezoid(dsigma_phi * np.sin(theta_arr), theta_arr)

    Q_sca = C_sca_total / (np.pi * radius**2)
    return Q_ext_fwd, Q_sca


if __name__ == "__main__":
    radius = 1.0; k_ext = 1.0; eta_ext = 1.0
    E0 = np.array([1.0, 0, 0]); k_hat = np.array([0, 0, 1.0])

    # PEC test
    Q_mie_pec = mie_pec_Qsca(k_ext * radius)
    print(f"PEC Mie Q_sca = {Q_mie_pec:.6f}")

    for refine in [2]:
        verts, tris = icosphere(radius, refinements=refine)
        rwg = build_rwg(verts, tris)
        N = rwg['N']
        print(f"\nRefine={refine}: {len(tris)} tris, {N} RWG")
        L, K_op = assemble_L_K(rwg, verts, tris, k_ext)
        quad_pts, quad_wts = tri_quadrature(7)
        lam0 = 1 - quad_pts[:, 0] - quad_pts[:, 1]
        def get_qpts(ti):
            t = tris[ti]; v0 = verts[t[:,0]]; v1 = verts[t[:,1]]; v2 = verts[t[:,2]]
            return np.einsum('q,ni->nqi', lam0, v0) + np.einsum('q,ni->nqi', quad_pts[:,0], v1) + \
                   np.einsum('q,ni->nqi', quad_pts[:,1], v2)
        qp = get_qpts(rwg['tri_p']); qm = get_qpts(rwg['tri_m'])
        V_E = np.zeros(N, dtype=complex)
        for qpts, free, area, sign in [(qp, rwg['free_p'], rwg['area_p'], +1),
                                         (qm, rwg['free_m'], rwg['area_m'], -1)]:
            f = sign * (rwg['length'][:,None,None] / (2*area[:,None,None])) * (qpts - free[:,None,:])
            jw = area[:,None] * quad_wts[None,:]
            phase = np.exp(1j * k_ext * np.einsum('i,nqi->nq', k_hat, qpts))
            V_E += np.sum(np.einsum('nqi,i->nq', f, E0) * phase * jw, axis=1)
        J = np.linalg.solve(eta_ext * L, V_E)
        M = np.zeros(N, dtype=complex)
        print("  Computing 2D Q_sca for PEC...")
        Q_ext, Q_sca = compute_Qsca_2d(rwg, verts, tris, J, M, k_ext, eta_ext, radius, ntheta=91, nphi=72)
        print(f"  PEC: Q_ext(OT)={Q_ext:.6f} ({abs(Q_ext-Q_mie_pec)/Q_mie_pec*100:.1f}%)")
        print(f"  PEC: Q_sca(2D)={Q_sca:.6f} ({abs(Q_sca-Q_mie_pec)/Q_mie_pec*100:.1f}%)")

    # Dielectric test
    m_rel = 1.5; k_int = k_ext * m_rel; eta_int = 1.0 / m_rel
    Q_ext_mie, Q_sca_mie = mie_Qext_Qsca(k_ext * radius, m_rel)
    print(f"\nDielectric Mie: Q_ext = {Q_ext_mie:.6f}, Q_sca = {Q_sca_mie:.6f}")

    for refine in [2]:
        verts, tris = icosphere(radius, refinements=refine)
        rwg = build_rwg(verts, tris)
        N = rwg['N']
        print(f"\nRefine={refine}: {len(tris)} tris, {N} RWG")
        L_ext, K_ext = assemble_L_K(rwg, verts, tris, k_ext)
        L_int, K_int = assemble_L_K(rwg, verts, tris, k_int)
        quad_pts, quad_wts = tri_quadrature(7)
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
        Z = np.zeros((2*N, 2*N), dtype=complex)
        Z[:N, :N] = eta_ext * L_ext + eta_int * L_int
        Z[:N, N:] = -(K_ext + K_int)
        Z[N:, :N] = -(K_ext + K_int)
        Z[N:, N:] = L_ext / eta_ext + L_int / eta_int
        coeffs = np.linalg.solve(Z, -b)
        J = coeffs[:N]; M = coeffs[N:]
        print("  Computing 2D Q_sca for dielectric...")
        Q_ext, Q_sca = compute_Qsca_2d(rwg, verts, tris, J, M, k_ext, eta_ext, radius, sM=+1, ntheta=91, nphi=72)
        print(f"  Diel: Q_ext(OT)={Q_ext:.6f} ({abs(Q_ext-Q_ext_mie)/Q_ext_mie*100:.1f}%)")
        print(f"  Diel: Q_sca(2D)={Q_sca:.6f} ({abs(Q_sca-Q_sca_mie)/Q_sca_mie*100:.1f}%)")
