"""Compare bistatic RCS with Mie theory to check if shape is correct."""
import numpy as np
from bem_core import icosphere, build_rwg, tri_quadrature, assemble_L_K
from scipy.special import spherical_jn, spherical_yn

def mie_pec_far_field(x, theta_arr, n_max=30):
    """Mie series for PEC sphere far field (x-polarized incidence).
    Returns |f_theta(theta, phi=0)|^2 and |f_phi(theta, phi=pi/2)|^2.
    These correspond to A(theta) and B(theta) in the symmetry decomposition.
    """
    from scipy.special import lpmv

    def psi(n, z): return z * spherical_jn(n, z)
    def psi_d(n, z): return spherical_jn(n, z) + z * spherical_jn(n, z, derivative=True)
    def xi(n, z): return z * (spherical_jn(n, z) + 1j * spherical_yn(n, z))
    def xi_d(n, z):
        return (spherical_jn(n, z) + 1j * spherical_yn(n, z)) + \
               z * (spherical_jn(n, z, derivative=True) + 1j * spherical_yn(n, z, derivative=True))

    # Mie coefficients
    a_n = np.zeros(n_max+1, dtype=complex)
    b_n = np.zeros(n_max+1, dtype=complex)
    for n in range(1, n_max+1):
        a_n[n] = psi(n, x) / xi(n, x)
        b_n[n] = psi_d(n, x) / xi_d(n, x)

    # Angular functions pi_n and tau_n
    # pi_n = P_n^1(cos theta) / sin(theta)
    # tau_n = d/dtheta P_n^1(cos theta)
    S1 = np.zeros(len(theta_arr), dtype=complex)  # S1 = sum (2n+1)/(n(n+1)) [a_n pi_n + b_n tau_n]
    S2 = np.zeros(len(theta_arr), dtype=complex)  # S2 = sum (2n+1)/(n(n+1)) [a_n tau_n + b_n pi_n]

    for it, theta in enumerate(theta_arr):
        mu = np.cos(theta)
        st = np.sin(theta)
        for n in range(1, n_max+1):
            # pi_n(cos theta) = P_n^1 / sin(theta)
            if abs(st) > 1e-10:
                pi_n = lpmv(1, n, mu) / st
            else:
                # At theta=0: pi_n = n(n+1)/2
                # At theta=pi: pi_n = (-1)^{n+1} n(n+1)/2
                if abs(theta) < 0.01:
                    pi_n = n*(n+1)/2
                else:
                    pi_n = (-1)**(n+1) * n*(n+1)/2

            # tau_n = d/dtheta P_n^1(cos theta) = cos(theta) * pi_n - (n+1) * pi_{n-1}
            # Using recurrence: tau_n = n*mu*pi_n - (n+1)*pi_{n-1}
            if n == 1:
                pi_prev = 0.0
            else:
                if abs(st) > 1e-10:
                    pi_prev = lpmv(1, n-1, mu) / st
                else:
                    if abs(theta) < 0.01:
                        pi_prev = (n-1)*n/2
                    else:
                        pi_prev = (-1)**n * (n-1)*n/2
            tau_n = n * mu * pi_n - (n+1) * pi_prev

            coeff = (2*n+1) / (n*(n+1))
            S1[it] += coeff * (a_n[n] * pi_n + b_n[n] * tau_n)
            S2[it] += coeff * (a_n[n] * tau_n + b_n[n] * pi_n)

    # f_theta(theta, phi=0) = (i/k) * S2(theta) * cos(phi=0) = (i/k) * S2
    # f_phi(theta, phi=pi/2) = -(i/k) * S1(theta) * sin(phi=pi/2) = -(i/k) * S1
    # dsigma = |f_theta(phi=0)|^2 + |f_phi(phi=pi/2)|^2
    return np.abs(S2 / x)**2, np.abs(S1 / x)**2


radius = 1.0; k = 1.0; eta = 1.0
x = k * radius
E0 = np.array([1.0, 0, 0]); k_hat = np.array([0, 0, 1.0])

verts, tris = icosphere(radius, refinements=2)
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

# Compute BEM far field
from test_farfield_fix import compute_far_field_at_angles

theta_arr = np.linspace(0.01, np.pi-0.01, 91)
M = np.zeros(N, dtype=complex)
F_th0, F_ph0 = compute_far_field_at_angles(rwg, verts, tris, J, M, k, eta, theta_arr, phi=0.0)
F_th90, F_ph90 = compute_far_field_at_angles(rwg, verts, tris, J, M, k, eta, theta_arr, phi=np.pi/2)

# Mie far field
A_mie2, B_mie2 = mie_pec_far_field(x, theta_arr)

# BEM: |A(theta)|^2 = |F_theta(phi=0)|^2, |B(theta)|^2 = |F_phi(phi=pi/2)|^2
A_bem2 = np.abs(F_th0)**2
B_bem2 = np.abs(F_ph90)**2

# Compare shape (normalized)
A_mie_norm = A_mie2 / np.max(A_mie2) if np.max(A_mie2) > 0 else A_mie2
A_bem_norm = A_bem2 / np.max(A_bem2) if np.max(A_bem2) > 0 else A_bem2
B_mie_norm = B_mie2 / np.max(B_mie2) if np.max(B_mie2) > 0 else B_mie2
B_bem_norm = B_bem2 / np.max(B_bem2) if np.max(B_bem2) > 0 else B_bem2

print("Theta[deg]  |A_mie|^2   |A_bem|^2   ratio    |B_mie|^2   |B_bem|^2   ratio")
for i in range(0, len(theta_arr), 10):
    th_deg = np.degrees(theta_arr[i])
    ra = A_bem2[i] / A_mie2[i] if A_mie2[i] > 1e-15 else float('nan')
    rb = B_bem2[i] / B_mie2[i] if B_mie2[i] > 1e-15 else float('nan')
    print(f"  {th_deg:6.1f}  {A_mie2[i]:10.6f}  {A_bem2[i]:10.6f}  {ra:6.3f}  {B_mie2[i]:10.6f}  {B_bem2[i]:10.6f}  {rb:6.3f}")

# Overall magnitude ratio
total_mie = np.trapezoid((A_mie2 + B_mie2) * np.sin(theta_arr), theta_arr)
total_bem = np.trapezoid((A_bem2 + B_bem2) * np.sin(theta_arr), theta_arr)
print(f"\n∫(A²+B²)sinθ dθ: Mie={total_mie:.6f}, BEM={total_bem:.6f}, ratio={total_bem/total_mie:.6f}")

# L matrix properties
print(f"\nL matrix properties:")
print(f"  ||L||_F = {np.linalg.norm(L):.4f}")
print(f"  cond(L) = {np.linalg.cond(L):.2f}")
eigvals = np.linalg.eigvals(L)
print(f"  eigenvalue range: Re=[{np.min(np.real(eigvals)):.4f}, {np.max(np.real(eigvals)):.4f}]")
print(f"                    Im=[{np.min(np.imag(eigvals)):.4f}, {np.max(np.imag(eigvals)):.4f}]")
