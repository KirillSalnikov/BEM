"""
Test EFIE for PEC sphere — simplest possible BEM test.
Only L operator, only J current, no M, no interior.
"""
import numpy as np
import time
from bem_core import (icosphere, build_rwg, assemble_L_K,
                      tri_quadrature, compute_far_field, compute_cross_sections)


def mie_pec_Qsca(x, n_max=30):
    """Scattering efficiency for PEC sphere from Mie theory."""
    from scipy.special import spherical_jn, spherical_yn

    def psi(n, z):
        return z * spherical_jn(n, z)
    def psi_d(n, z):
        return spherical_jn(n, z) + z * spherical_jn(n, z, derivative=True)
    def xi(n, z):
        return z * (spherical_jn(n, z) + 1j * spherical_yn(n, z))
    def xi_d(n, z):
        sj = spherical_jn(n, z)
        sjd = spherical_jn(n, z, derivative=True)
        sy = spherical_yn(n, z)
        syd = spherical_yn(n, z, derivative=True)
        return (sj + 1j*sy) + z*(sjd + 1j*syd)

    Q = 0.0
    for n in range(1, n_max+1):
        a_n = psi(n, x) / xi(n, x)  # TE (PEC: m_rel→∞)
        b_n = psi_d(n, x) / xi_d(n, x)  # TM
        Q += (2*n+1) * (abs(a_n)**2 + abs(b_n)**2)
    return 2 * Q / x**2


if __name__ == "__main__":
    radius = 1.0
    k = 1.0  # x = k*a = 1
    x = k * radius
    eta = 1.0

    print(f"=== PEC Sphere EFIE Test ===")
    print(f"  x = {x:.2f}")

    Q_mie = mie_pec_Qsca(x)
    print(f"  Mie Q_sca = {Q_mie:.6f}")

    # Mesh
    refine = 1
    verts, tris = icosphere(radius, refinements=refine)
    rwg = build_rwg(verts, tris)
    N = rwg['N']
    print(f"  Mesh: {len(verts)} verts, {len(tris)} tris, {N} RWG")

    # EFIE: Z*I = V
    # Z_mn = jkη <f_m, ∫G f_n dS'> + η/(jk) <∇·f_m, ∫G ∇'·f_n dS'>
    # In my convention: Z = η * L
    L, K = assemble_L_K(rwg, verts, tris, k)
    Z = eta * L
    print(f"  Z assembled. Cond = {np.linalg.cond(Z):.2e}")

    # RHS: V_m = <f_m, E^inc>
    # E_inc = x_hat * exp(ikz)
    E0 = np.array([1.0, 0, 0])
    k_hat = np.array([0, 0, 1.0])

    quad_pts, quad_wts = tri_quadrature(4)
    Nq = len(quad_wts)
    lam0 = 1 - quad_pts[:, 0] - quad_pts[:, 1]

    def get_qpts(tri_indices):
        t = tris[tri_indices]
        v0 = verts[t[:, 0]]; v1 = verts[t[:, 1]]; v2 = verts[t[:, 2]]
        return (np.einsum('q,ni->nqi', lam0, v0) +
                np.einsum('q,ni->nqi', quad_pts[:, 0], v1) +
                np.einsum('q,ni->nqi', quad_pts[:, 1], v2))

    qpts_p = get_qpts(rwg['tri_p'])
    qpts_m = get_qpts(rwg['tri_m'])

    V = np.zeros(N, dtype=complex)
    for sign_idx, (qpts, free, area, sign) in enumerate([
        (qpts_p, rwg['free_p'], rwg['area_p'], +1),
        (qpts_m, rwg['free_m'], rwg['area_m'], -1),
    ]):
        f = sign * (rwg['length'][:, None, None] / (2 * area[:, None, None])) * \
            (qpts - free[:, None, :])
        jw = 2 * area[:, None] * quad_wts[None, :]
        phase = np.exp(1j * k * np.einsum('i,nqi->nq', k_hat, qpts))
        f_dot_E0 = np.einsum('nqi,i->nq', f, E0)
        V += np.sum(f_dot_E0 * phase * jw, axis=1)

    # Solve
    I = np.linalg.solve(Z, V)
    print(f"  |I| range: {abs(I).min():.4e} to {abs(I).max():.4e}")

    # Far field for PEC: only J, no M
    theta_arr = np.linspace(0.01, np.pi - 0.01, 181)
    F_theta = np.zeros(len(theta_arr), dtype=complex)
    F_phi = np.zeros(len(theta_arr), dtype=complex)

    for it, theta in enumerate(theta_arr):
        r_hat = np.array([np.sin(theta), 0, np.cos(theta)])
        theta_hat = np.array([np.cos(theta), 0, -np.sin(theta)])
        phi_hat = np.array([0, 1.0, 0])

        J_integral = np.zeros(3, dtype=complex)
        for sign_idx, (qpts, free, area, sign) in enumerate([
            (qpts_p, rwg['free_p'], rwg['area_p'], +1),
            (qpts_m, rwg['free_m'], rwg['area_m'], -1),
        ]):
            f = sign * (rwg['length'][:, None, None] / (2 * area[:, None, None])) * \
                (qpts - free[:, None, :])
            jw = 2 * area[:, None] * quad_wts[None, :]
            phase = np.exp(-1j * k * np.einsum('i,nqi->nq', r_hat, qpts))
            # weighted integral: sum over n,q of f[n,q,:] * I[n] * phase * jw
            weighted = f * (I[:, None, None] * phase[:, :, None] * jw[:, :, None])
            J_integral += weighted.sum(axis=(0, 1))

        # F = -ik/(4pi) * eta * [J - r_hat*(r_hat·J)]
        J_perp = J_integral - r_hat * np.dot(r_hat, J_integral)
        F_vec = -1j * k * eta / (4 * np.pi) * J_perp

        F_theta[it] = np.dot(F_vec, theta_hat)
        F_phi[it] = np.dot(F_vec, phi_hat)

    dsigma = np.abs(F_theta)**2 + np.abs(F_phi)**2
    C_sca = 2 * np.pi * np.trapezoid(dsigma * np.sin(theta_arr), theta_arr)
    Q_bem = C_sca / (np.pi * radius**2)

    print(f"\n=== Results ===")
    print(f"  Q_sca (BEM):  {Q_bem:.6f}")
    print(f"  Q_sca (Mie):  {Q_mie:.6f}")
    print(f"  Error: {abs(Q_bem - Q_mie)/Q_mie * 100:.1f}%")

    # Also check extinction via optical theorem: C_ext = (4pi/k) * Im[F(0)]
    # Forward direction theta=0: F should have only theta component
    r_hat_fwd = np.array([0, 0, 1.0])
    J_int_fwd = np.zeros(3, dtype=complex)
    for sign_idx, (qpts, free, area, sign) in enumerate([
        (qpts_p, rwg['free_p'], rwg['area_p'], +1),
        (qpts_m, rwg['free_m'], rwg['area_m'], -1),
    ]):
        f = sign * (rwg['length'][:, None, None] / (2 * area[:, None, None])) * \
            (qpts - free[:, None, :])
        jw = 2 * area[:, None] * quad_wts[None, :]
        phase = np.exp(-1j * k * np.einsum('i,nqi->nq', r_hat_fwd, qpts))
        weighted = f * (I[:, None, None] * phase[:, :, None] * jw[:, :, None])
        J_int_fwd += weighted.sum(axis=(0, 1))

    J_perp_fwd = J_int_fwd - r_hat_fwd * np.dot(r_hat_fwd, J_int_fwd)
    F_fwd = -1j * k * eta / (4 * np.pi) * J_perp_fwd
    # For x-polarized inc, forward scattering amplitude is F·x_hat
    S_fwd = np.dot(F_fwd, E0)
    C_ext_ot = 4 * np.pi / k * np.imag(S_fwd)
    Q_ext_ot = C_ext_ot / (np.pi * radius**2)
    print(f"  Q_ext (opt. theorem): {Q_ext_ot:.6f}")
    print(f"  Q_ext (Mie):          {Q_mie:.6f}")
