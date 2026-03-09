"""
Fix the far-field Q_sca formula: need to integrate over phi properly.

For x-polarized incidence on a sphere:
  F_θ(θ,φ) = cos(φ) · A(θ)
  F_φ(θ,φ) = -sin(φ) · B(θ)

∫₀²π (cos²φ|A|² + sin²φ|B|²) dφ = π(|A|² + |B|²)

So: C_sca = π ∫₀π [|A(θ)|² + |B(θ)|²] sinθ dθ
         = π ∫ [|F_θ(θ,0)|² + |F_φ(θ,π/2)|²] sinθ dθ
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


def compute_far_field_at_angles(rwg, verts, tris, J, M, k, eta, theta_arr, phi, sM=+1):
    """Compute far field at given theta array and fixed phi."""
    quad_pts, quad_wts = tri_quadrature(7)
    N = rwg['N']
    lam0 = 1 - quad_pts[:,0] - quad_pts[:,1]
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
            phase = np.exp(-1j * k * np.einsum('i,nqi->nq', r_hat, qpts))
            integral = (f * (phase * jw)[:,:,None]).sum(axis=1)
            Jt += (integral * J[:,None]).sum(0)
            Mt += (integral * M[:,None]).sum(0)
        Jp = Jt - r_hat * np.dot(r_hat, Jt)
        Mc = np.cross(r_hat, Mt)
        Fv = -1j * k / (4*np.pi) * (eta * Jp + sM * Mc)
        F_theta[it] = np.dot(Fv, theta_hat)
        F_phi[it] = np.dot(Fv, phi_hat)

    return F_theta, F_phi


def compute_Qsca_correct(rwg, verts, tris, J, M, k, eta, radius, sM=+1, ntheta=361):
    """Correct Q_sca: evaluate at phi=0 and phi=pi/2, then combine."""
    theta_arr = np.linspace(0.01, np.pi - 0.01, ntheta)

    # At phi=0: F_θ = A(θ), F_φ = 0
    F_th0, F_ph0 = compute_far_field_at_angles(rwg, verts, tris, J, M, k, eta, theta_arr, phi=0.0, sM=sM)
    # At phi=pi/2: F_θ = 0, F_φ = -B(θ)
    F_th90, F_ph90 = compute_far_field_at_angles(rwg, verts, tris, J, M, k, eta, theta_arr, phi=np.pi/2, sM=sM)

    # C_sca = π ∫ (|A|² + |B|²) sinθ dθ = π ∫ (|F_θ(φ=0)|² + |F_φ(φ=π/2)|²) sinθ dθ
    dsigma = np.abs(F_th0)**2 + np.abs(F_ph90)**2
    C_sca = np.pi * np.trapezoid(dsigma * np.sin(theta_arr), theta_arr)
    Q_sca = C_sca / (np.pi * radius**2)

    # Also compute old (wrong) formula for comparison
    dsigma_old = np.abs(F_th0)**2 + np.abs(F_ph0)**2  # F_ph0 ≈ 0
    C_sca_old = 2 * np.pi * np.trapezoid(dsigma_old * np.sin(theta_arr), theta_arr)
    Q_sca_old = C_sca_old / (np.pi * radius**2)

    # Q_ext from optical theorem (using phi=0 forward direction)
    Q_ext = 4 * np.pi / k * np.imag(F_th0[0]) / (np.pi * radius**2)

    return Q_ext, Q_sca, Q_sca_old


if __name__ == "__main__":
    radius = 1.0; k_ext = 1.0; eta_ext = 1.0
    E0 = np.array([1.0, 0, 0]); k_hat = np.array([0, 0, 1.0])

    # ===== PEC test =====
    Q_mie_pec = mie_pec_Qsca(k_ext * radius)
    print(f"PEC Mie Q_sca = {Q_mie_pec:.6f}")

    for refine in [1, 2, 3]:
        verts, tris = icosphere(radius, refinements=refine)
        rwg = build_rwg(verts, tris)
        N = rwg['N']
        print(f"\nRefine={refine}: {len(tris)} tris, {N} RWG")

        L, K_op = assemble_L_K(rwg, verts, tris, k_ext)

        # PEC EFIE RHS
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

        # PEC EFIE: η*L*J = V_E
        J = np.linalg.solve(eta_ext * L, V_E)
        M = np.zeros(N, dtype=complex)

        Q_ext, Q_sca, Q_sca_old = compute_Qsca_correct(rwg, verts, tris, J, M, k_ext, eta_ext, radius)
        print(f"  PEC EFIE:")
        print(f"    Q_ext(OT)   = {Q_ext:.6f} (err={abs(Q_ext-Q_mie_pec)/Q_mie_pec*100:.1f}%)")
        print(f"    Q_sca(new)  = {Q_sca:.6f} (err={abs(Q_sca-Q_mie_pec)/Q_mie_pec*100:.1f}%)")
        print(f"    Q_sca(old)  = {Q_sca_old:.6f} (err={abs(Q_sca_old-Q_mie_pec)/Q_mie_pec*100:.1f}%)")

    # ===== Dielectric test =====
    m_rel = 1.5; k_int = k_ext * m_rel; eta_int = 1.0 / m_rel
    Q_ext_mie, Q_sca_mie = mie_Qext_Qsca(k_ext * radius, m_rel)
    print(f"\nDielectric Mie: Q_ext = {Q_ext_mie:.6f}, Q_sca = {Q_sca_mie:.6f}")

    for refine in [2, 3]:
        verts, tris = icosphere(radius, refinements=refine)
        rwg = build_rwg(verts, tris)
        N = rwg['N']
        print(f"\nRefine={refine}: {len(tris)} tris, {N} RWG")

        L_ext, K_ext = assemble_L_K(rwg, verts, tris, k_ext)
        L_int, K_int = assemble_L_K(rwg, verts, tris, k_int)

        # RHS
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

        # PMCHWT: [ηL, -K; +K, L/η] · x = +b
        Z = np.zeros((2*N, 2*N), dtype=complex)
        Z[:N, :N] = eta_ext * L_ext + eta_int * L_int
        Z[:N, N:] = -(K_ext + K_int)
        Z[N:, :N] = +(K_ext + K_int)
        Z[N:, N:] = L_ext / eta_ext + L_int / eta_int
        coeffs = np.linalg.solve(Z, b)
        J = coeffs[:N]; M = coeffs[N:]

        Q_ext, Q_sca, Q_sca_old = compute_Qsca_correct(rwg, verts, tris, J, M, k_ext, eta_ext, radius, sM=-1)
        print(f"  PMCHWT (sM=-1):")
        print(f"    Q_ext(OT)   = {Q_ext:.6f} (err={abs(Q_ext-Q_ext_mie)/Q_ext_mie*100:.1f}%)")
        print(f"    Q_sca(new)  = {Q_sca:.6f} (err={abs(Q_sca-Q_sca_mie)/Q_sca_mie*100:.1f}%)")
        print(f"    Q_sca(old)  = {Q_sca_old:.6f} (err={abs(Q_sca_old-Q_sca_mie)/Q_sca_mie*100:.1f}%)")
