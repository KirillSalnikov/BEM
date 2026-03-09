"""
Investigate Q_ext accuracy. Try:
1. Optical theorem at different theta values near 0
2. Direct power extraction: P_ext = Re(b^H · x) / 2
3. Energy balance: Q_abs = Q_ext - Q_sca should be ~0 for lossless
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


if __name__ == "__main__":
    radius = 1.0; k_ext = 1.0; m_rel = 1.5; x = k_ext * radius
    k_int = k_ext * m_rel; eta_ext = 1.0; eta_int = 1.0 / m_rel
    E0 = np.array([1.0, 0, 0]); k_hat = np.array([0, 0, 1.0])

    Q_ext_mie, Q_sca_mie = mie_Qext_Qsca(x, m_rel)
    print(f"Mie: Q_ext = {Q_ext_mie:.6f}, Q_sca = {Q_sca_mie:.6f}")

    for refine in [2, 3]:
        print(f"\n{'='*70}")
        verts, tris = icosphere(radius, refinements=refine)
        rwg = build_rwg(verts, tris)
        N = rwg['N']
        print(f"Refine={refine}: {len(tris)} tris, {N} RWG")

        L_ext, K_ext = assemble_L_K(rwg, verts, tris, k_ext)
        L_int, K_int = assemble_L_K(rwg, verts, tris, k_int)

        # Check K symmetry
        K_sum = K_ext + K_int
        print(f"  ||K - K^T|| / ||K|| = {np.linalg.norm(K_sum - K_sum.T) / np.linalg.norm(K_sum):.4e}")

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

        # Config B: [ηL, -K; -K, L/η] · x = -b
        Z = np.zeros((2*N, 2*N), dtype=complex)
        Z[:N, :N] = eta_ext * L_ext + eta_int * L_int
        Z[:N, N:] = -(K_ext + K_int)
        Z[N:, :N] = -(K_ext + K_int)
        Z[N:, N:] = L_ext / eta_ext + L_int / eta_int
        coeffs = np.linalg.solve(Z, -b)
        J = coeffs[:N]; M = coeffs[N:]

        print(f"\n  Config B solution: |J| = {np.linalg.norm(J):.4e}, |M| = {np.linalg.norm(M):.4e}")

        # --- Method 1: Optical theorem at various theta ---
        print("\n  --- Optical theorem at various theta ---")
        for theta_fwd in [1e-6, 1e-4, 0.001, 0.01, 0.05]:
            r_hat = np.array([np.sin(theta_fwd), 0, np.cos(theta_fwd)])
            theta_hat = np.array([np.cos(theta_fwd), 0, -np.sin(theta_fwd)])
            Jt = np.zeros(3, dtype=complex); Mt = np.zeros(3, dtype=complex)
            for qpts, free, area, sign in [(qp, rwg['free_p'], rwg['area_p'], +1),
                                             (qm, rwg['free_m'], rwg['area_m'], -1)]:
                f = sign * (rwg['length'][:,None,None] / (2*area[:,None,None])) * (qpts - free[:,None,:])
                jw = area[:,None] * quad_wts[None,:]
                phase = np.exp(-1j * k_ext * np.einsum('i,nqi->nq', r_hat, qpts))
                integral = (f * (phase * jw)[:,:,None]).sum(axis=1)
                Jt += (integral * J[:,None]).sum(0)
                Mt += (integral * M[:,None]).sum(0)
            Jp = Jt - r_hat * np.dot(r_hat, Jt)
            Mc = np.cross(r_hat, Mt)

            # Try different far-field formulas
            for sM_val in [+1, -1]:
                Fv = -1j * k_ext / (4*np.pi) * (eta_ext * Jp + sM_val * Mc)
                F_th = np.dot(Fv, theta_hat)
                Q_ext_ot = 4 * np.pi / k_ext * np.imag(F_th) / (np.pi * radius**2)
                err = abs(Q_ext_ot - Q_ext_mie) / Q_ext_mie * 100
                if sM_val == +1:
                    print(f"    theta={theta_fwd:.1e}, sM=+1: Q_ext={Q_ext_ot:.6f} ({err:.1f}%)")

        # --- Decompose the forward amplitude ---
        print("\n  --- Forward amplitude decomposition ---")
        theta_fwd = 1e-6
        r_hat = np.array([np.sin(theta_fwd), 0, np.cos(theta_fwd)])
        theta_hat = np.array([np.cos(theta_fwd), 0, -np.sin(theta_fwd)])
        Jt = np.zeros(3, dtype=complex); Mt = np.zeros(3, dtype=complex)
        for qpts, free, area, sign in [(qp, rwg['free_p'], rwg['area_p'], +1),
                                         (qm, rwg['free_m'], rwg['area_m'], -1)]:
            f = sign * (rwg['length'][:,None,None] / (2*area[:,None,None])) * (qpts - free[:,None,:])
            jw = area[:,None] * quad_wts[None,:]
            phase = np.exp(-1j * k_ext * np.einsum('i,nqi->nq', r_hat, qpts))
            integral = (f * (phase * jw)[:,:,None]).sum(axis=1)
            Jt += (integral * J[:,None]).sum(0)
            Mt += (integral * M[:,None]).sum(0)
        Jp = Jt - r_hat * np.dot(r_hat, Jt)
        Mc = np.cross(r_hat, Mt)

        # F = -ik/(4π) [η J_⊥ + sM * r̂×M]
        J_contrib = -1j * k_ext / (4*np.pi) * eta_ext * Jp
        M_contrib = -1j * k_ext / (4*np.pi) * Mc
        print(f"    J̃_⊥ = {Jp}")
        print(f"    r̂×M̃ = {Mc}")
        print(f"    J contrib to F_θ: {np.dot(J_contrib, theta_hat):.6f}")
        print(f"    M contrib to F_θ: {np.dot(M_contrib, theta_hat):.6f}")
        F_th_p = np.dot(J_contrib + M_contrib, theta_hat)
        F_th_m = np.dot(J_contrib - M_contrib, theta_hat)
        print(f"    Im(F_θ) with sM=+1: {np.imag(F_th_p):.6f}")
        print(f"    Im(F_θ) with sM=-1: {np.imag(F_th_m):.6f}")
        Q_ext_needed = Q_sca_mie  # For lossless
        Im_needed = Q_ext_needed * (np.pi * radius**2) * k_ext / (4 * np.pi)
        print(f"    Im(F_θ) needed for Q_ext=Mie: {Im_needed:.6f}")

        # --- Method 2: Direct power extraction ---
        # P_ext = Re(b_physical · conj(x)) / 2
        # But need to be careful about signs
        print("\n  --- Direct power extraction ---")
        # b_physical = [<f_m, E_inc>, <f_m, H_inc>]
        # x = [J, M] from Config B (Z·x = -b_physical)
        # Physical: J_physical corresponds to n̂ × H on surface
        # C_ext = ½ Re ∫ (E_inc · J* - H_inc · M*) dS  [with PMCHWT convention]
        # = ½ Re (b_E^T conj(J) - b_H^T conj(M))  ... or is it +?
        for s1, s2 in [(+1, +1), (+1, -1), (-1, +1), (-1, -1)]:
            P = 0.5 * np.real(s1 * np.dot(b[:N], np.conj(coeffs[:N])) +
                               s2 * np.dot(b[N:], np.conj(coeffs[N:])))
            Q = P / (np.pi * radius**2)
            err = abs(Q - Q_ext_mie) / Q_ext_mie * 100
            print(f"    s_J={s1:+d}, s_M={s2:+d}: Q_ext={Q:.6f} ({err:.1f}%)")

        # Also try b_physical^H · x (Hermitian inner product)
        print("  --- b^H · x variants ---")
        for s1, s2 in [(+1, +1), (+1, -1), (-1, +1), (-1, -1)]:
            P = 0.5 * np.real(s1 * np.dot(np.conj(b[:N]), coeffs[:N]) +
                               s2 * np.dot(np.conj(b[N:]), coeffs[N:]))
            Q = P / (np.pi * radius**2)
            err = abs(Q - Q_ext_mie) / Q_ext_mie * 100
            print(f"    s_J={s1:+d}, s_M={s2:+d}: Q_ext={Q:.6f} ({err:.1f}%)")

        # --- Q_sca for reference ---
        theta_arr = np.linspace(0.01, np.pi - 0.01, 361)
        F_theta_arr = np.zeros(len(theta_arr), dtype=complex)
        F_phi_arr = np.zeros(len(theta_arr), dtype=complex)
        for it, theta in enumerate(theta_arr):
            r_hat = np.array([np.sin(theta), 0, np.cos(theta)])
            theta_hat = np.array([np.cos(theta), 0, -np.sin(theta)])
            phi_hat = np.array([0, 1.0, 0])
            Jt = np.zeros(3, dtype=complex); Mt = np.zeros(3, dtype=complex)
            for qpts, free, area, sign in [(qp, rwg['free_p'], rwg['area_p'], +1),
                                             (qm, rwg['free_m'], rwg['area_m'], -1)]:
                f = sign * (rwg['length'][:,None,None] / (2*area[:,None,None])) * (qpts - free[:,None,:])
                jw = area[:,None] * quad_wts[None,:]
                phase = np.exp(-1j * k_ext * np.einsum('i,nqi->nq', r_hat, qpts))
                integral = (f * (phase * jw)[:,:,None]).sum(axis=1)
                Jt += (integral * J[:,None]).sum(0)
                Mt += (integral * M[:,None]).sum(0)
            Jp = Jt - r_hat * np.dot(r_hat, Jt)
            Mc = np.cross(r_hat, Mt)
            Fv = -1j * k_ext / (4*np.pi) * (eta_ext * Jp + Mc)
            F_theta_arr[it] = np.dot(Fv, theta_hat)
            F_phi_arr[it] = np.dot(Fv, phi_hat)
        dsigma = np.abs(F_theta_arr)**2 + np.abs(F_phi_arr)**2
        C_sca = 2 * np.pi * np.trapezoid(dsigma * np.sin(theta_arr), theta_arr)
        Q_sca = C_sca / (np.pi * radius**2)
        print(f"\n  Q_sca = {Q_sca:.6f} (err={abs(Q_sca-Q_sca_mie)/Q_sca_mie*100:.1f}%)")
