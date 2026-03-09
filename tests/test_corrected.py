"""
Test PMCHWT with the "corrected standard" sign convention:
If K_ours = -K_textbook (from PEC MFIE evidence), then the standard PMCHWT
[ηL, K_text; -K_text, L/η] becomes [ηL, -K; K, L/η] with our K.
Also test all sK1, sK2 combos with sB=+1 (don't negate RHS).
"""
import numpy as np
import time
from bem_core import (icosphere, build_rwg, tri_quadrature, assemble_L_K)


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


def compute_Qext_Qsca(rwg, verts, tris, J, M, k, eta, radius, E0, sM=+1):
    """Compute Q_ext and Q_sca using angular integration."""
    quad_pts, quad_wts = tri_quadrature(7)
    N = rwg['N']
    lam0 = 1 - quad_pts[:,0] - quad_pts[:,1]
    def get_qpts(ti):
        t = tris[ti]; v0 = verts[t[:,0]]; v1 = verts[t[:,1]]; v2 = verts[t[:,2]]
        return np.einsum('q,ni->nqi', lam0, v0) + np.einsum('q,ni->nqi', quad_pts[:,0], v1) + \
               np.einsum('q,ni->nqi', quad_pts[:,1], v2)
    qp = get_qpts(rwg['tri_p']); qm = get_qpts(rwg['tri_m'])

    theta_arr = np.linspace(0.01, np.pi - 0.01, 361)
    F_theta = np.zeros(len(theta_arr), dtype=complex)
    F_phi = np.zeros(len(theta_arr), dtype=complex)

    for it, theta in enumerate(theta_arr):
        r_hat = np.array([np.sin(theta), 0, np.cos(theta)])
        theta_hat = np.array([np.cos(theta), 0, -np.sin(theta)])
        phi_hat = np.array([0, 1.0, 0])
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

    dsigma = np.abs(F_theta)**2 + np.abs(F_phi)**2
    C_sca = 2 * np.pi * np.trapezoid(dsigma * np.sin(theta_arr), theta_arr)
    Q_sca = C_sca / (np.pi * radius**2)
    Q_ext = 4 * np.pi / k * np.imag(F_theta[0]) / (np.pi * radius**2)

    return Q_ext, Q_sca


if __name__ == "__main__":
    radius = 1.0; k_ext = 1.0; m_rel = 1.5; x = k_ext * radius
    k_int = k_ext * m_rel; eta_ext = 1.0; eta_int = 1.0 / m_rel
    E0 = np.array([1.0, 0, 0]); k_hat = np.array([0, 0, 1.0])

    Q_ext_mie, Q_sca_mie = mie_Qext_Qsca(x, m_rel)
    print(f"Mie: Q_ext = {Q_ext_mie:.6f}, Q_sca = {Q_sca_mie:.6f}")

    # Test all 16 combinations at ref=2 and 3
    configs = {}
    for sK1 in [+1, -1]:
        for sK2 in [+1, -1]:
            for sB in [+1, -1]:
                label = f"sK1={sK1:+d},sK2={sK2:+d},sB={sB:+d}"
                configs[label] = (sK1, sK2, sB)

    for refine in [2, 3]:
        print(f"\n{'='*80}")
        t0 = time.time()
        verts, tris = icosphere(radius, refinements=refine)
        rwg = build_rwg(verts, tris)
        N = rwg['N']
        print(f"Refine={refine}: {len(tris)} tris, {N} RWG")

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

        t_asm = time.time() - t0

        print(f"  {'config':>35s} | {'sM':>3s} {'Q_ext':>8s} {'eE%':>6s} | {'Q_sca':>8s} {'eS%':>6s}")
        print(f"  {'-'*75}")

        for name, (sK1, sK2, sB) in configs.items():
            Z = np.zeros((2*N, 2*N), dtype=complex)
            Z[:N, :N] = eta_ext * L_ext + eta_int * L_int
            Z[:N, N:] = sK1 * (K_ext + K_int)
            Z[N:, :N] = sK2 * (K_ext + K_int)
            Z[N:, N:] = L_ext / eta_ext + L_int / eta_int
            coeffs = np.linalg.solve(Z, sB * b)
            J = coeffs[:N]; M = coeffs[N:]

            for sM in [+1, -1]:
                Q_ext_c, Q_sca_c = compute_Qext_Qsca(rwg, verts, tris, J, M,
                                                        k_ext, eta_ext, radius, E0, sM=sM)
                eE = abs(Q_ext_c - Q_ext_mie) / Q_ext_mie * 100
                eS = abs(Q_sca_c - Q_sca_mie) / Q_sca_mie * 100
                mark = " ***" if eE < 15 and eS < 5 else (" <-" if eE < 30 and eS < 10 else "")
                print(f"  {name:>35s} | {sM:+d}  {Q_ext_c:>8.4f} {eE:>5.1f}% | {Q_sca_c:>8.4f} {eS:>5.1f}%{mark}")

        print(f"  time={time.time()-t0:.1f}s")
