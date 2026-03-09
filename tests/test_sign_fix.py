"""Test both signs of the div-div term to see which satisfies the optical theorem."""
import numpy as np
from bem_core import icosphere, build_rwg, tri_quadrature, assemble_L_K
from test_farfield_fix import mie_pec_Qsca

radius = 1.0; k = 1.0; eta = 1.0
E0 = np.array([1.0, 0, 0]); k_hat = np.array([0, 0, 1.0])
Q_mie = mie_pec_Qsca(k * radius)
print(f"Mie Q_sca = {Q_mie:.6f}")

verts, tris = icosphere(radius, refinements=2)
rwg = build_rwg(verts, tris)
N = rwg['N']

# Get L from code (plus sign)
L_plus, _ = assemble_L_K(rwg, verts, tris, k)

# Construct L with minus sign by subtracting twice the D part
# L_plus = ikA + (i/k)D → L_minus = ikA - (i/k)D = L_plus - 2*(i/k)D
# But we need D separately. Let's assemble with k→0 limit? No, that doesn't work.
# Instead, let's compute the div-div contribution separately.

# The div-div part of L is: (i/k) * div_m * div_n * ∫∫ G dS dS' (smooth + singular)
# We can extract it: for each (m,n), the div-div contribution is proportional to div_m * div_n

# Actually, simpler: just modify the assembly to flip the sign
# Let's do it by computing D_mn = div_m * div_n * G_integral separately

quad_pts, quad_wts = tri_quadrature(7)
Nq = len(quad_wts)
lam0 = 1 - quad_pts[:, 0] - quad_pts[:, 1]

def get_qpts(ti):
    t = tris[ti]; v0 = verts[t[:,0]]; v1 = verts[t[:,1]]; v2 = verts[t[:,2]]
    return np.einsum('q,ni->nqi', lam0, v0) + np.einsum('q,ni->nqi', quad_pts[:,0], v1) + \
           np.einsum('q,ni->nqi', quad_pts[:,1], v2)

qp = get_qpts(rwg['tri_p']); qm = get_qpts(rwg['tri_m'])

div_p = rwg['length'] / (2 * rwg['area_p'])
div_m = -rwg['length'] / (2 * rwg['area_m'])
jw_p = rwg['area_p'][:,None] * quad_wts[None,:]
jw_m = rwg['area_m'][:,None] * quad_wts[None,:]

# Compute the scalar G integral for each (m,n) pair: S_mn = ∫∫ G dS_test dS_src
# Then L_plus = ik*A + (i/k)*D where D_mn = div_prod * S_mn
# And L_minus = ik*A - (i/k)*D = L_plus - 2*(i/k)*D

# Compute S_mn (scalar G integral between test/src halves)
S = np.zeros((N, N), dtype=complex)
halves_q = [(qp, jw_p, rwg['tri_p']), (qm, jw_m, rwg['tri_m'])]
halves_d = [div_p, div_m]

inv4pi = 1.0 / (4*np.pi)
ik = 1j * k

for tq, tw, tt in halves_q:
    for sq, sw, st in halves_q:
        for m in range(N):
            for n in range(N):
                val = 0.0 + 0j
                for iq in range(Nq):
                    for js in range(Nq):
                        R = np.linalg.norm(tq[m,iq] - sq[n,js])
                        if R < 1e-12:
                            G = ik * inv4pi  # G_smooth limit
                        else:
                            G = np.exp(ik * R) * inv4pi / R
                        val += G * tw[m,iq] * sw[n,js]
                S[m,n] += val

# D_mn = div_prod * S_mn
div_prod = np.outer(np.concatenate([div_p, div_m]).reshape(-1), np.concatenate([div_p, div_m]).reshape(-1))
# Actually, D involves all 4 combinations of test/src halves
# Let me compute D directly
D = np.zeros((N,N), dtype=complex)
for td, (tq, tw, tt) in zip([div_p, div_m], halves_q):
    for sd, (sq, sw, st) in zip([div_p, div_m], halves_q):
        dp = td[:, None] * sd[None, :]  # (N, N)
        # Compute S for this half-pair
        for m in range(N):
            for n in range(N):
                val = 0.0 + 0j
                same_tri = (tt[m] == st[n])
                for iq in range(Nq):
                    for js in range(Nq):
                        R = np.linalg.norm(tq[m,iq] - sq[n,js])
                        if R < 1e-12:
                            G = ik * inv4pi  # G_smooth at R=0
                        else:
                            G = np.exp(ik * R) * inv4pi / R
                        val += G * tw[m,iq] * sw[n,js]
                D[m,n] += dp[m,n] * val

print("Computing D done.")

# L_plus = ik*A + (i/k)*D
# So A = (L_plus - (i/k)*D) / (ik)
# And L_minus = ik*A - (i/k)*D = L_plus - 2*(i/k)*D
L_minus = L_plus - 2 * (1j/k) * D

# RHS
V_E = np.zeros(N, dtype=complex)
for qpts, free, area, sign in [(qp, rwg['free_p'], rwg['area_p'], +1),
                                 (qm, rwg['free_m'], rwg['area_m'], -1)]:
    f = sign * (rwg['length'][:,None,None] / (2*area[:,None,None])) * (qpts - free[:,None,:])
    jw = area[:,None] * quad_wts[None,:]
    phase = np.exp(1j * k * np.einsum('i,nqi->nq', k_hat, qpts))
    V_E += np.sum(np.einsum('nqi,i->nq', f, E0) * phase * jw, axis=1)

from test_farfield_fix import compute_Qsca_correct

for label, L_mat in [("L_plus (ikA + i/k D)", L_plus),
                      ("L_minus (ikA - i/k D)", L_minus)]:
    J = np.linalg.solve(eta * L_mat, V_E)
    M = np.zeros(N, dtype=complex)
    Q_ext, Q_sca, Q_sca_old = compute_Qsca_correct(rwg, verts, tris, J, M, k, eta, radius, ntheta=181)
    P_ext_val = 0.5 * np.real(np.conj(V_E) @ J)
    print(f"\n{label}:")
    print(f"  Q_ext(OT) = {Q_ext:.6f} (err={abs(Q_ext-Q_mie)/Q_mie*100:.1f}%)")
    print(f"  Q_sca     = {Q_sca:.6f} (err={abs(Q_sca-Q_mie)/Q_mie*100:.1f}%)")
    print(f"  Q_ext/Q_sca = {Q_ext/Q_sca:.4f}")
    print(f"  P_ext = {P_ext_val:.6f}")
