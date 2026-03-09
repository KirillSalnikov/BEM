"""Check if F_theta ~ cos(phi), F_phi ~ sin(phi) symmetry holds for PEC sphere."""
import numpy as np
from bem_core import icosphere, build_rwg, tri_quadrature, assemble_L_K

radius = 1.0; k = 1.0; eta = 1.0
E0 = np.array([1.0, 0, 0]); k_hat = np.array([0, 0, 1.0])

verts, tris = icosphere(radius, refinements=2)
rwg = build_rwg(verts, tris)
N = rwg['N']
L, K_op = assemble_L_K(rwg, verts, tris, k)

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
    phase = np.exp(1j * k * np.einsum('i,nqi->nq', k_hat, qpts))
    V_E += np.sum(np.einsum('nqi,i->nq', f, E0) * phase * jw, axis=1)
J = np.linalg.solve(eta * L, V_E)

# Compute far field at various theta and phi
def compute_F(theta, phi):
    ct, st = np.cos(theta), np.sin(theta)
    cp, sp = np.cos(phi), np.sin(phi)
    r_hat = np.array([st*cp, st*sp, ct])
    theta_hat = np.array([ct*cp, ct*sp, -st])
    phi_hat = np.array([-sp, cp, 0.0])
    Jt = np.zeros(3, dtype=complex)
    for qpts_h, free, area, sign in [(qp, rwg['free_p'], rwg['area_p'], +1),
                                      (qm, rwg['free_m'], rwg['area_m'], -1)]:
        f = sign * (rwg['length'][:,None,None] / (2*area[:,None,None])) * (qpts_h - free[:,None,:])
        jw = area[:,None] * quad_wts[None,:]
        phase = np.exp(-1j * k * np.einsum('i,nqi->nq', r_hat, qpts_h))
        integral = (f * (phase * jw)[:,:,None]).sum(axis=1)
        Jt += (integral * J[:,None]).sum(0)
    Jp = Jt - r_hat * np.dot(r_hat, Jt)
    Fv = -1j * k / (4*np.pi) * eta * Jp
    return np.dot(Fv, theta_hat), np.dot(Fv, phi_hat)

# Test at theta=pi/3 for various phi
theta_test = np.pi / 3
print("At theta=pi/3, testing phi dependence:")
print(f"{'phi':>6s} {'|F_th|':>10s} {'|F_ph|':>10s} {'F_th/cos':>12s} {'F_ph/sin':>12s}")
for phi in np.linspace(0, 2*np.pi, 13)[:-1]:
    Ft, Fp = compute_F(theta_test, phi)
    cp = np.cos(phi); sp = np.sin(phi)
    ratio_t = Ft / cp if abs(cp) > 0.01 else float('nan')
    ratio_p = Fp / sp if abs(sp) > 0.01 else float('nan')
    print(f"  {phi:.3f} {abs(Ft):>10.6f} {abs(Fp):>10.6f}  {ratio_t:>12.6f}  {ratio_p:>12.6f}")

# Also compute total dσ/dΩ as function of phi at fixed theta
print(f"\ndσ/dΩ at theta=pi/3 vs phi:")
for phi in [0, np.pi/6, np.pi/4, np.pi/3, np.pi/2]:
    Ft, Fp = compute_F(theta_test, phi)
    dsigma = abs(Ft)**2 + abs(Fp)**2
    print(f"  phi={phi:.4f}: dσ/dΩ={dsigma:.8f}")

# Full 2D vs two-cut comparison
print("\n=== Full 2D Q_sca vs two-cut formula ===")
theta_arr = np.linspace(0.01, np.pi-0.01, 181)

# Two-cut formula
Fth0 = np.array([compute_F(th, 0.0)[0] for th in theta_arr])
Fph90 = np.array([compute_F(th, np.pi/2)[1] for th in theta_arr])
dsigma_twocut = np.abs(Fth0)**2 + np.abs(Fph90)**2
C_sca_twocut = np.pi * np.trapezoid(dsigma_twocut * np.sin(theta_arr), theta_arr)

# Full 2D
nphi = 36
phi_arr = np.linspace(0, 2*np.pi, nphi, endpoint=False)
C_sca_2d = 0.0
for phi in phi_arr:
    dsigma_line = np.zeros(len(theta_arr))
    for it, th in enumerate(theta_arr):
        Ft, Fp = compute_F(th, phi)
        dsigma_line[it] = abs(Ft)**2 + abs(Fp)**2
    C_sca_2d += (2*np.pi/nphi) * np.trapezoid(dsigma_line * np.sin(theta_arr), theta_arr)

Q_twocut = C_sca_twocut / (np.pi * radius**2)
Q_2d = C_sca_2d / (np.pi * radius**2)
print(f"Q_sca (two-cut):  {Q_twocut:.6f}")
print(f"Q_sca (full 2D):  {Q_2d:.6f}")
print(f"Ratio 2D/twocut:  {Q_2d/Q_twocut:.6f}")

# Also compute P_ext and check
P_ext = 0.5 * np.real(np.conj(V_E) @ J)
print(f"\nP_ext = 0.5*Re(V_E^H J) = {P_ext:.6f}")
print(f"Q_ext(power) = -2*P_ext/(πa²) = {-2*P_ext/(np.pi*radius**2):.6f}")  # try negative
Ft0, _ = compute_F(0.01, 0.0)
Q_ext_OT = 4*np.pi / k * np.imag(Ft0) / (np.pi * radius**2)
print(f"Q_ext(OT) = {Q_ext_OT:.6f}")
