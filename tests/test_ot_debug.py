"""Debug optical theorem violation by computing both sides from the same J."""
import numpy as np
from bem_core import icosphere, build_rwg, tri_quadrature, assemble_L_K

radius = 1.0; k = 1.0; eta = 1.0
E0 = np.array([1.0, 0, 0]); k_hat = np.array([0, 0, 1.0])

verts, tris = icosphere(radius, refinements=2)
rwg = build_rwg(verts, tris)
N = rwg['N']
L, _ = assemble_L_K(rwg, verts, tris, k)

quad_pts, quad_wts = tri_quadrature(7)
Nq = len(quad_wts)
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

# Compute N(r̂) = ∫ J(r') e^{-ik r̂·r'} dS' for arbitrary direction
def compute_N(r_hat):
    """Compute N = ∫ J e^{-ik r̂·r'} dS' (3-vector, complex)."""
    Nt = np.zeros(3, dtype=complex)
    for qpts_h, free, area, sign in [(qp, rwg['free_p'], rwg['area_p'], +1),
                                      (qm, rwg['free_m'], rwg['area_m'], -1)]:
        f = sign * (rwg['length'][:,None,None] / (2*area[:,None,None])) * (qpts_h - free[:,None,:])
        jw = area[:,None] * quad_wts[None,:]
        phase = np.exp(-1j * k * np.einsum('i,nqi->nq', r_hat, qpts_h))
        # ∫ f_n e^{-ik r̂·r'} dS' for each n: sum over quad pts
        integral = np.sum(f * (phase * jw)[:,:,None], axis=1)  # (N, 3)
        Nt += np.sum(integral * J[:,None], axis=0)  # sum over n
    return Nt

# Forward direction
r_fwd = k_hat
N_fwd = compute_N(r_fwd)
N_fwd_perp = N_fwd - r_fwd * np.dot(r_fwd, N_fwd)

# Forward scattering amplitude
f_fwd = -1j * k * eta / (4*np.pi) * N_fwd_perp
print(f"N(forward) = {N_fwd}")
print(f"N_perp(forward) = {N_fwd_perp}")
print(f"f(forward) = {f_fwd}")
print(f"Im(f_x) = {np.imag(f_fwd[0]):.8f}")

sigma_ext = (4*np.pi/k) * np.imag(f_fwd[0])
print(f"\nsigma_ext (OT) = {sigma_ext:.6f}")
Q_ext = sigma_ext / (np.pi * radius**2)
print(f"Q_ext (OT) = {Q_ext:.6f}")

# Also from V^H J
sigma_ext2 = eta * np.abs(np.real(np.conj(V_E) @ J))
print(f"sigma_ext (V^H J) = {sigma_ext2:.6f}")
print(f"Re(V_E^H J) = {np.real(np.conj(V_E) @ J):.6f}")

# Now compute sigma_sca
ntheta = 361
theta_arr = np.linspace(0.01, np.pi-0.01, ntheta)

# Full 2D integration
nphi = 72
phi_arr = np.linspace(0, 2*np.pi, nphi, endpoint=False)
sigma_sca = 0.0
for phi in phi_arr:
    dsigma = np.zeros(ntheta)
    for it, theta in enumerate(theta_arr):
        ct, st = np.cos(theta), np.sin(theta)
        cp, sp = np.cos(phi), np.sin(phi)
        r_hat = np.array([st*cp, st*sp, ct])
        N_r = compute_N(r_hat)
        N_perp = N_r - r_hat * np.dot(r_hat, N_r)
        f_r = -1j * k * eta / (4*np.pi) * N_perp
        dsigma[it] = np.sum(np.abs(f_r)**2)
    sigma_sca += (2*np.pi/nphi) * np.trapezoid(dsigma * np.sin(theta_arr), theta_arr)

print(f"\nsigma_sca (2D) = {sigma_sca:.6f}")
Q_sca = sigma_sca / (np.pi * radius**2)
print(f"Q_sca (2D) = {Q_sca:.6f}")
print(f"Q_ext/Q_sca = {Q_ext/Q_sca:.6f}")

# Decompose: what fraction of |f|^2 is in f_theta vs f_phi?
# At theta=pi/2, phi=0: r_hat=[1,0,0], theta_hat=[0,0,-1], phi_hat=[0,1,0]
theta_test = np.pi/2
for phi_test in [0, np.pi/4, np.pi/2]:
    ct, st = np.cos(theta_test), np.sin(theta_test)
    cp, sp = np.cos(phi_test), np.sin(phi_test)
    r_hat = np.array([st*cp, st*sp, ct])
    theta_hat = np.array([ct*cp, ct*sp, -st])
    phi_hat = np.array([-sp, cp, 0.0])
    N_r = compute_N(r_hat)
    N_perp = N_r - r_hat * np.dot(r_hat, N_r)
    f_r = -1j * k * eta / (4*np.pi) * N_perp
    print(f"\nAt theta=pi/2, phi={phi_test:.4f}:")
    print(f"  |f|^2 = {np.sum(np.abs(f_r)**2):.8f}")
    print(f"  |f_th|^2 = {np.abs(np.dot(f_r, theta_hat))**2:.8f}")
    print(f"  |f_ph|^2 = {np.abs(np.dot(f_r, phi_hat))**2:.8f}")
    print(f"  |f_r|^2 = {np.abs(np.dot(f_r, r_hat))**2:.8f}")  # should be ~0
