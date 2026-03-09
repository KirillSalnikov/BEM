"""Quick check: does div fix improve OT consistency?"""
import numpy as np
from bem_core import icosphere, build_rwg, tri_quadrature, assemble_L_K
from test_farfield_fix import compute_Qsca_correct, mie_pec_Qsca

radius = 1.0; k = 1.0; eta = 1.0; E0 = np.array([1.0, 0, 0]); k_hat = np.array([0, 0, 1.0])
Q_mie = mie_pec_Qsca(k * radius)

for refine in [1, 2]:
    verts, tris = icosphere(radius, refinements=refine)
    rwg = build_rwg(verts, tris)
    N = rwg['N']
    L, _ = assemble_L_K(rwg, verts, tris, k)
    quad_pts, quad_wts = tri_quadrature(7)
    lam0 = 1 - quad_pts[:,0] - quad_pts[:,1]
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
        phase = np.exp(1j * k * np.einsum('i,nqi->nq', k_hat, qpts))
        V_E += np.sum(np.einsum('nqi,i->nq', f, E0) * phase * jw, axis=1)
    J = np.linalg.solve(eta * L, V_E)
    M = np.zeros(N, dtype=complex)
    Q_ext, Q_sca, _ = compute_Qsca_correct(rwg, verts, tris, J, M, k, eta, radius, ntheta=181)
    print(f"Ref={refine}: Q_ext={Q_ext:.6f}, Q_sca={Q_sca:.6f}, ratio={Q_ext/Q_sca:.4f}, Mie={Q_mie:.6f}")
