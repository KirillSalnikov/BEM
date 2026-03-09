"""
Test PEC MFIE to verify the K operator independently.
PEC MFIE: (½G - K) J = V_H
where G_mn = <f_m, f_n> is the mass matrix.
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


if __name__ == "__main__":
    radius = 1.0; k = 1.0; x = k * radius; eta = 1.0
    E0 = np.array([1.0, 0, 0]); k_hat = np.array([0, 0, 1.0])
    H0 = np.cross(k_hat, E0) / eta

    Q_mie = mie_pec_Qsca(x)
    print(f"Mie PEC Q_sca = {Q_mie:.6f}")

    for refine in [1, 2, 3]:
        verts, tris = icosphere(radius, refinements=refine)
        rwg = build_rwg(verts, tris)
        N = rwg['N']
        print(f"\nRefine={refine}: {len(tris)} tris, {N} RWG")

        L, K_op = assemble_L_K(rwg, verts, tris, k)

        # Compute RWG mass matrix G_mn = <f_m, f_n>
        quad_pts, quad_wts = tri_quadrature(7)
        Nq = len(quad_wts)
        lam0 = 1 - quad_pts[:, 0] - quad_pts[:, 1]

        def get_qpts(ti):
            t = tris[ti]; v0 = verts[t[:,0]]; v1 = verts[t[:,1]]; v2 = verts[t[:,2]]
            return np.einsum('q,ni->nqi', lam0, v0) + np.einsum('q,ni->nqi', quad_pts[:,0], v1) + \
                   np.einsum('q,ni->nqi', quad_pts[:,1], v2)

        qp = get_qpts(rwg['tri_p']); qm = get_qpts(rwg['tri_m'])

        # RWG basis function values
        f_p = (rwg['length'][:, None, None] / (2 * rwg['area_p'][:, None, None])) * \
              (qp - rwg['free_p'][:, None, :])
        f_m_basis = -(rwg['length'][:, None, None] / (2 * rwg['area_m'][:, None, None])) * \
              (qm - rwg['free_m'][:, None, :])
        jw_p = rwg['area_p'][:, None] * quad_wts[None, :]
        jw_m = rwg['area_m'][:, None] * quad_wts[None, :]

        # Mass matrix G_mn = Σ_h <f_m^h, f_n^h> (only same-triangle pairs)
        G = np.zeros((N, N))
        # p-half contributions
        for m in range(N):
            for n in range(N):
                if rwg['tri_p'][m] == rwg['tri_p'][n]:
                    G[m, n] += np.sum(np.einsum('qi,qi->q', f_p[m], f_p[n]) * jw_p[m])
                if rwg['tri_p'][m] == rwg['tri_m'][n]:
                    G[m, n] += np.sum(np.einsum('qi,qi->q', f_p[m], f_m_basis[n]) * jw_p[m])
                if rwg['tri_m'][m] == rwg['tri_p'][n]:
                    G[m, n] += np.sum(np.einsum('qi,qi->q', f_m_basis[m], f_p[n]) * jw_m[m])
                if rwg['tri_m'][m] == rwg['tri_m'][n]:
                    G[m, n] += np.sum(np.einsum('qi,qi->q', f_m_basis[m], f_m_basis[n]) * jw_m[m])

        # PEC EFIE: η*L*I = V_E
        V_E = np.zeros(N, dtype=complex)
        for qpts, free, area, sign in [
            (qp, rwg['free_p'], rwg['area_p'], +1),
            (qm, rwg['free_m'], rwg['area_m'], -1),
        ]:
            f = sign * (rwg['length'][:,None,None] / (2*area[:,None,None])) * (qpts - free[:,None,:])
            jw = area[:,None] * quad_wts[None,:]
            phase = np.exp(1j * k * np.einsum('i,nqi->nq', k_hat, qpts))
            V_E += np.sum(np.einsum('nqi,i->nq', f, E0) * phase * jw, axis=1)

        # PEC MFIE: (½G - K)*I = V_H
        V_H = np.zeros(N, dtype=complex)
        for qpts, free, area, sign in [
            (qp, rwg['free_p'], rwg['area_p'], +1),
            (qm, rwg['free_m'], rwg['area_m'], -1),
        ]:
            f = sign * (rwg['length'][:,None,None] / (2*area[:,None,None])) * (qpts - free[:,None,:])
            jw = area[:,None] * quad_wts[None,:]
            phase = np.exp(1j * k * np.einsum('i,nqi->nq', k_hat, qpts))
            V_H += np.sum(np.einsum('nqi,i->nq', f, H0) * phase * jw, axis=1)

        # Solve EFIE
        Z_efie = eta * L
        I_efie = np.linalg.solve(Z_efie, V_E)

        # Solve MFIE
        Z_mfie = 0.5 * G - K_op
        I_mfie = np.linalg.solve(Z_mfie, V_H)

        # Q_ext via optical theorem for both
        for label, I_sol in [("EFIE", I_efie), ("MFIE", I_mfie)]:
            r_hat_fwd = k_hat
            J_int = np.zeros(3, dtype=complex)
            for qpts, free, area, sign in [
                (qp, rwg['free_p'], rwg['area_p'], +1),
                (qm, rwg['free_m'], rwg['area_m'], -1),
            ]:
                f = sign * (rwg['length'][:,None,None] / (2*area[:,None,None])) * (qpts - free[:,None,:])
                jw = area[:,None] * quad_wts[None,:]
                phase = np.exp(-1j * k * np.einsum('i,nqi->nq', r_hat_fwd, qpts))
                weighted = f * (I_sol[:, None, None] * phase[:, :, None] * jw[:, :, None])
                J_int += weighted.sum(axis=(0, 1))

            J_perp = J_int - r_hat_fwd * np.dot(r_hat_fwd, J_int)
            F_fwd = -1j * k * eta / (4 * np.pi) * J_perp
            S_fwd = np.dot(F_fwd, E0)
            C_ext = 4 * np.pi / k * np.imag(S_fwd)
            Q_ext = C_ext / (np.pi * radius**2)
            err = abs(Q_ext - Q_mie) / Q_mie * 100
            cond = np.linalg.cond(Z_efie if label == "EFIE" else Z_mfie)
            print(f"  {label}: Q_ext={Q_ext:.6f}, err={err:.1f}%, cond={cond:.2e}")
