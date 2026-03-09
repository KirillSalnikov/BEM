"""
Test PEC MFIE with different K signs to find the correct one.
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

    refine = 2
    verts, tris = icosphere(radius, refinements=refine)
    rwg = build_rwg(verts, tris)
    N = rwg['N']
    print(f"Mesh: {len(tris)} tris, {N} RWG")

    L, K_op = assemble_L_K(rwg, verts, tris, k)

    quad_pts, quad_wts = tri_quadrature(7)
    Nq = len(quad_wts)
    lam0 = 1 - quad_pts[:, 0] - quad_pts[:, 1]

    def get_qpts(ti):
        t = tris[ti]; v0 = verts[t[:,0]]; v1 = verts[t[:,1]]; v2 = verts[t[:,2]]
        return np.einsum('q,ni->nqi', lam0, v0) + np.einsum('q,ni->nqi', quad_pts[:,0], v1) + \
               np.einsum('q,ni->nqi', quad_pts[:,1], v2)

    qp = get_qpts(rwg['tri_p']); qm = get_qpts(rwg['tri_m'])

    f_p = (rwg['length'][:, None, None] / (2 * rwg['area_p'][:, None, None])) * \
          (qp - rwg['free_p'][:, None, :])
    f_m_b = -(rwg['length'][:, None, None] / (2 * rwg['area_m'][:, None, None])) * \
          (qm - rwg['free_m'][:, None, :])
    jw_p = rwg['area_p'][:, None] * quad_wts[None, :]
    jw_m = rwg['area_m'][:, None] * quad_wts[None, :]

    # Mass matrix
    G = np.zeros((N, N))
    for m in range(N):
        for n in range(N):
            if rwg['tri_p'][m] == rwg['tri_p'][n]:
                G[m, n] += np.sum(np.einsum('qi,qi->q', f_p[m], f_p[n]) * jw_p[m])
            if rwg['tri_p'][m] == rwg['tri_m'][n]:
                G[m, n] += np.sum(np.einsum('qi,qi->q', f_p[m], f_m_b[n]) * jw_p[m])
            if rwg['tri_m'][m] == rwg['tri_p'][n]:
                G[m, n] += np.sum(np.einsum('qi,qi->q', f_m_b[m], f_p[n]) * jw_m[m])
            if rwg['tri_m'][m] == rwg['tri_m'][n]:
                G[m, n] += np.sum(np.einsum('qi,qi->q', f_m_b[m], f_m_b[n]) * jw_m[m])

    # RHS
    V_E = np.zeros(N, dtype=complex)
    V_H = np.zeros(N, dtype=complex)
    for qpts, free, area, sign in [
        (qp, rwg['free_p'], rwg['area_p'], +1),
        (qm, rwg['free_m'], rwg['area_m'], -1),
    ]:
        f = sign * (rwg['length'][:,None,None] / (2*area[:,None,None])) * (qpts - free[:,None,:])
        jw = area[:,None] * quad_wts[None,:]
        phase = np.exp(1j * k * np.einsum('i,nqi->nq', k_hat, qpts))
        V_E += np.sum(np.einsum('nqi,i->nq', f, E0) * phase * jw, axis=1)
        V_H += np.sum(np.einsum('nqi,i->nq', f, H0) * phase * jw, axis=1)

    def Q_ext_ot(I_sol):
        r_hat = k_hat
        J_int = np.zeros(3, dtype=complex)
        for qpts, free, area, sign in [
            (qp, rwg['free_p'], rwg['area_p'], +1),
            (qm, rwg['free_m'], rwg['area_m'], -1),
        ]:
            f = sign * (rwg['length'][:,None,None] / (2*area[:,None,None])) * (qpts - free[:,None,:])
            jw = area[:,None] * quad_wts[None,:]
            phase = np.exp(-1j * k * np.einsum('i,nqi->nq', r_hat, qpts))
            weighted = f * (I_sol[:, None, None] * phase[:, :, None] * jw[:, :, None])
            J_int += weighted.sum(axis=(0, 1))
        J_perp = J_int - r_hat * np.dot(r_hat, J_int)
        F_fwd = -1j * k * eta / (4 * np.pi) * J_perp
        S = np.dot(F_fwd, E0)
        return 4 * np.pi / k * np.imag(S) / (np.pi * radius**2)

    print(f"\n  EFIE (η*L): Q_ext = {Q_ext_ot(np.linalg.solve(eta*L, V_E)):.6f}")

    # Test all MFIE sign combinations
    for sK in [+1, -1]:
        for sB in [+1, -1]:
            Z = 0.5 * G + sK * K_op
            I = np.linalg.solve(Z, sB * V_H)
            Q = Q_ext_ot(I)
            err = abs(Q - Q_mie) / Q_mie * 100
            print(f"  MFIE (½G {'+' if sK>0 else '-'} K, sB={sB:+d}): Q_ext = {Q:.6f}, err={err:.1f}%")
