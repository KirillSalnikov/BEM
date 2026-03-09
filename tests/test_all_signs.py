"""
Brute-force test of ALL sign combinations:
- PMCHWT K blocks (sK1, sK2)
- RHS sign (sB)
- Far-field M sign (sM)
Find the combination where Q_ext ≈ Q_sca ≈ Mie.
"""
import numpy as np
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


def compute_far(rwg, verts, tris, J, M, k, eta, theta_arr, sM=+1):
    """Compute far field with configurable M sign."""
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
    return F_theta, F_phi


if __name__ == "__main__":
    radius = 1.0; k_ext = 1.0; m_rel = 1.5; x = k_ext * radius
    k_int = k_ext * m_rel; eta_ext = 1.0; eta_int = 1.0 / m_rel
    E0 = np.array([1.0, 0, 0]); k_hat = np.array([0, 0, 1.0])

    Q_ext_mie, Q_sca_mie = mie_Qext_Qsca(x, m_rel)
    print(f"Mie: Q_ext = {Q_ext_mie:.6f}, Q_sca = {Q_sca_mie:.6f}")

    for refine in [1, 2]:
        print(f"\n{'='*60}")
        print(f"Refinement = {refine}")
        verts, tris = icosphere(radius, refinements=refine)
        rwg = build_rwg(verts, tris)
        N = rwg['N']

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

        theta_arr = np.linspace(0.01, np.pi - 0.01, 181)

        print(f"  {'sK1':>4s} {'sK2':>4s} {'sB':>3s} {'sM':>3s} | {'Q_ext':>9s} {'eE%':>6s} | {'Q_sca':>9s} {'eS%':>6s} | {'diff%':>6s}")
        print(f"  {'-'*70}")

        best_err = 1e10
        best_params = None

        for sK1 in [+1, -1]:
            for sK2 in [+1, -1]:
                for sB in [+1, -1]:
                    Z = np.zeros((2*N, 2*N), dtype=complex)
                    Z[:N, :N] = eta_ext * L_ext + eta_int * L_int
                    Z[:N, N:] = sK1 * (K_ext + K_int)
                    Z[N:, :N] = sK2 * (K_ext + K_int)
                    Z[N:, N:] = L_ext / eta_ext + L_int / eta_int
                    coeffs = np.linalg.solve(Z, sB * b)
                    J = coeffs[:N]; M = coeffs[N:]

                    for sM in [+1, -1]:
                        F_th, F_ph = compute_far(rwg, verts, tris, J, M, k_ext, eta_ext,
                                                  theta_arr, sM=sM)
                        dsigma = np.abs(F_th)**2 + np.abs(F_ph)**2
                        C_sca = 2*np.pi * np.trapezoid(dsigma * np.sin(theta_arr), theta_arr)
                        Q_sca = C_sca / (np.pi * radius**2)

                        # Q_ext via optical theorem (using same far-field at theta=0)
                        F_fwd_th, _ = compute_far(rwg, verts, tris, J, M, k_ext, eta_ext,
                                                   np.array([1e-6]), sM=sM)
                        Q_ext = 4*np.pi / k_ext * np.imag(F_fwd_th[0]) / (np.pi * radius**2)

                        eE = abs(Q_ext - Q_ext_mie) / Q_ext_mie * 100
                        eS = abs(Q_sca - Q_sca_mie) / Q_sca_mie * 100
                        diff = abs(Q_ext - Q_sca) / max(abs(Q_ext), abs(Q_sca)) * 100

                        total_err = eE + eS
                        if total_err < best_err:
                            best_err = total_err
                            best_params = (sK1, sK2, sB, sM, Q_ext, Q_sca, eE, eS)

                        mark = " <--" if eE < 30 and eS < 15 else ""
                        print(f"  {sK1:+d}   {sK2:+d}   {sB:+d}  {sM:+d} | {Q_ext:>9.4f} {eE:>5.1f}% | {Q_sca:>9.4f} {eS:>5.1f}% | {diff:>5.1f}%{mark}")

        sK1, sK2, sB, sM, Q_ext, Q_sca, eE, eS = best_params
        print(f"\n  BEST: sK1={sK1:+d}, sK2={sK2:+d}, sB={sB:+d}, sM={sM:+d}")
        print(f"        Q_ext={Q_ext:.6f} ({eE:.1f}%), Q_sca={Q_sca:.6f} ({eS:.1f}%)")
