"""Test: skip singular extraction entirely to check if optical theorem holds."""
import numpy as np
from bem_core import icosphere, build_rwg, tri_quadrature

def assemble_L_nosingular(rwg, verts, tris, k, quad_order=7):
    """Assemble L using G_full everywhere (no singularity extraction)."""
    N = rwg['N']
    quad_pts, quad_wts = tri_quadrature(quad_order)
    Nq = len(quad_wts)
    lam0 = 1 - quad_pts[:, 0] - quad_pts[:, 1]

    def get_qpts(ti):
        t = tris[ti]; v0 = verts[t[:,0]]; v1 = verts[t[:,1]]; v2 = verts[t[:,2]]
        return np.einsum('q,ni->nqi', lam0, v0) + np.einsum('q,ni->nqi', quad_pts[:,0], v1) + \
               np.einsum('q,ni->nqi', quad_pts[:,1], v2)

    qpts_p = get_qpts(rwg['tri_p'])
    qpts_m = get_qpts(rwg['tri_m'])
    f_p = (rwg['length'][:,None,None] / (2*rwg['area_p'][:,None,None])) * (qpts_p - rwg['free_p'][:,None,:])
    f_m = -(rwg['length'][:,None,None] / (2*rwg['area_m'][:,None,None])) * (qpts_m - rwg['free_m'][:,None,:])
    div_p = rwg['length'] / (2*rwg['area_p'])
    div_m = -rwg['length'] / (2*rwg['area_m'])
    jw_p = rwg['area_p'][:,None] * quad_wts[None,:]
    jw_m = rwg['area_m'][:,None] * quad_wts[None,:]

    L = np.zeros((N,N), dtype=complex)
    ik = 1j*k; ik_inv = 1j/k

    halves = [(qpts_p, f_p, div_p, jw_p), (qpts_m, f_m, div_m, jw_m)]

    for tq, tf, td, tw in halves:
        for sq, sf, sd, sw in halves:
            for m in range(N):
                for n in range(N):
                    for iq in range(Nq):
                        for js in range(Nq):
                            R_vec = tq[m,iq] - sq[n,js]
                            R = np.linalg.norm(R_vec)
                            if R < 1e-12:
                                continue
                            G = np.exp(1j*k*R) / (4*np.pi*R)
                            fdot = np.dot(tf[m,iq], sf[n,js])
                            dprod = td[m] * sd[n]
                            jw = tw[m,iq] * sw[n,js]
                            L[m,n] += (ik * fdot + ik_inv * dprod) * G * jw
    return L

radius = 1.0; k = 1.0; eta = 1.0
E0 = np.array([1.0, 0, 0]); k_hat = np.array([0, 0, 1.0])

verts, tris = icosphere(radius, refinements=1)
rwg = build_rwg(verts, tris)
N = rwg['N']
print(f"Refine=1: {len(tris)} tris, {N} RWG")

print("Assembling L (no singular extraction)...")
L_nosing = assemble_L_nosingular(rwg, verts, tris, k)
L_nosing = (L_nosing + L_nosing.T) / 2

print("Assembling L (with singular extraction)...")
from bem_core import assemble_L_K
L_sing, K_sing = assemble_L_K(rwg, verts, tris, k)

# Compare a few entries
print("\nL entries comparison:")
for m, n in [(0,0), (1,1), (0,1), (0, N//2)]:
    print(f"  ({m},{n}): nosing={L_nosing[m,n]:.6f}, sing={L_sing[m,n]:.6f}")

# Solve EFIE with both
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

from test_farfield_fix import compute_Qsca_correct, mie_pec_Qsca
Q_mie = mie_pec_Qsca(k * radius)

for label, L_mat in [("no-sing", L_nosing), ("with-sing", L_sing)]:
    J = np.linalg.solve(eta * L_mat, V_E)
    M = np.zeros(N, dtype=complex)
    Q_ext, Q_sca, Q_sca_old = compute_Qsca_correct(rwg, verts, tris, J, M, k, eta, radius, ntheta=361)
    P_ext = 0.5 * np.real(np.conj(V_E) @ J)
    print(f"\n{label}:")
    print(f"  Q_ext(OT) = {Q_ext:.6f} (err={abs(Q_ext-Q_mie)/Q_mie*100:.1f}%)")
    print(f"  Q_sca     = {Q_sca:.6f} (err={abs(Q_sca-Q_mie)/Q_mie*100:.1f}%)")
    print(f"  P_ext     = {P_ext:.6f}")
    print(f"  Q_ext/Q_sca = {Q_ext/Q_sca:.6f}")
    print(f"  Mie Q_sca = {Q_mie:.6f}")
