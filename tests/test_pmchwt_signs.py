"""Test different PMCHWT sign conventions with the corrected L operator."""
import numpy as np
from bem_core import icosphere, build_rwg, tri_quadrature, assemble_L_K
from test_farfield_fix import mie_Qext_Qsca, compute_Qsca_correct

radius = 1.0; k_ext = 1.0; eta_ext = 1.0
m_rel = 1.5; k_int = k_ext * m_rel; eta_int = 1.0 / m_rel
E0 = np.array([1.0, 0, 0]); k_hat = np.array([0, 0, 1.0])
Q_ext_mie, Q_sca_mie = mie_Qext_Qsca(k_ext * radius, m_rel)
print(f"Mie: Q_ext={Q_ext_mie:.6f}, Q_sca={Q_sca_mie:.6f}")

verts, tris = icosphere(radius, refinements=2)
rwg = build_rwg(verts, tris)
N = rwg['N']
print(f"Refine=2: {len(tris)} tris, {N} RWG")

L_ext, K_ext = assemble_L_K(rwg, verts, tris, k_ext)
L_int, K_int = assemble_L_K(rwg, verts, tris, k_int)

quad_pts, quad_wts = tri_quadrature(7)
lam0 = 1 - quad_pts[:,0] - quad_pts[:,1]
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

K_sum = K_ext + K_int
LL = eta_ext*L_ext + eta_int*L_int
LL2 = L_ext/eta_ext + L_int/eta_int

configs = [
    ("B: [ηL,-K;-K,L/η]·x=-b", LL, -K_sum, -K_sum, LL2, -b),
    ("C: [ηL,-K;+K,L/η]·x=+b", LL, -K_sum, +K_sum, LL2, +b),
    ("C': [ηL,-K;+K,L/η]·x=-b", LL, -K_sum, +K_sum, LL2, -b),
    ("D: [ηL,+K;-K,L/η]·x=-b", LL, +K_sum, -K_sum, LL2, -b),
    ("D': [ηL,+K;-K,L/η]·x=+b", LL, +K_sum, -K_sum, LL2, +b),
    ("E: [ηL,+K;+K,L/η]·x=-b", LL, +K_sum, +K_sum, LL2, -b),
]

for name, Z11, Z12, Z21, Z22, rhs in configs:
    Z = np.block([[Z11, Z12], [Z21, Z22]])
    try:
        coeffs = np.linalg.solve(Z, rhs)
        J = coeffs[:N]; M = coeffs[N:]
        if np.max(np.abs(J)) > 1e6:
            print(f"  {name}: DIVERGED")
            continue
        for sM in [+1, -1]:
            Q_ext, Q_sca, _ = compute_Qsca_correct(rwg, verts, tris, J, M, k_ext, eta_ext, radius, sM=sM, ntheta=91)
            err_ext = abs(Q_ext - Q_ext_mie) / Q_ext_mie * 100
            err_sca = abs(Q_sca - Q_sca_mie) / Q_sca_mie * 100
            tag = " <<<" if err_sca < 20 else ""
            print(f"  {name} sM={sM:+d}: Qe={Q_ext:.4f}({err_ext:.1f}%) Qs={Q_sca:.4f}({err_sca:.1f}%){tag}")
    except:
        print(f"  {name}: FAILED")
