"""
Quick test: is the sign on the div-div term in L correct?
Standard EFIE impedance (e^{-iwt} convention, G = e^{ikR}/(4piR)):
  Z_mn = jωμ <f,f>_G + j/(ωε) <∇·f,∇·f>_G     (PLUS on div-div)
Our L:
  L_mn = jk <f,f>_G - j/k <∇·f,∇·f>_G           (MINUS on div-div)
  η*L = jkη <f,f>_G - jη/k <∇·f,∇·f>_G
If Z should have PLUS, then our L has the WRONG sign.

Test: flip the div-div sign in assemble_L_K and check PEC + dielectric.
"""
import numpy as np
from bem_core import (icosphere, build_rwg, tri_quadrature,
                      assemble_L_K, potential_integral_triangle,
                      vector_potential_integral_triangle,
                      _compute_tri_normals)


def assemble_L_K_plus(rwg, verts, tris, k, quad_order=7):
    """Same as assemble_L_K but with PLUS on div-div term."""
    N = rwg['N']
    quad_pts, quad_wts = tri_quadrature(quad_order)
    Nq = len(quad_wts)
    lam0 = 1 - quad_pts[:, 0] - quad_pts[:, 1]

    def get_qpts_batch(tri_indices):
        t = tris[tri_indices]
        v0 = verts[t[:, 0]]; v1 = verts[t[:, 1]]; v2 = verts[t[:, 2]]
        return np.einsum('q,ni->nqi', lam0, v0) + \
               np.einsum('q,ni->nqi', quad_pts[:, 0], v1) + \
               np.einsum('q,ni->nqi', quad_pts[:, 1], v2)

    qpts_p = get_qpts_batch(rwg['tri_p'])
    qpts_m = get_qpts_batch(rwg['tri_m'])

    f_p = (rwg['length'][:, None, None] / (2 * rwg['area_p'][:, None, None])) * \
          (qpts_p - rwg['free_p'][:, None, :])
    f_m = -(rwg['length'][:, None, None] / (2 * rwg['area_m'][:, None, None])) * \
          (qpts_m - rwg['free_m'][:, None, :])

    div_p = rwg['length'] / (2 * rwg['area_p'])
    div_m = -rwg['length'] / (2 * rwg['area_m'])

    jw_p = rwg['area_p'][:, None] * quad_wts[None, :]
    jw_m = rwg['area_m'][:, None] * quad_wts[None, :]

    all_qpts = np.concatenate([qpts_p, qpts_m], axis=1)
    all_f = np.concatenate([f_p, f_m], axis=1)
    all_div = np.concatenate([np.broadcast_to(div_p[:, None], (N, Nq)),
                               np.broadcast_to(div_m[:, None], (N, Nq))], axis=1)
    all_jw = np.concatenate([jw_p, jw_m], axis=1)

    L = np.zeros((N, N), dtype=complex)
    K = np.zeros((N, N), dtype=complex)

    tri_verts_cache = {}
    for ti in range(len(tris)):
        tri_verts_cache[ti] = (verts[tris[ti, 0]].copy(),
                                verts[tris[ti, 1]].copy(),
                                verts[tris[ti, 2]].copy())

    for m in range(N):
        test_halves = [
            (qpts_p[m], f_p[m], div_p[m], jw_p[m], rwg['tri_p'][m]),
            (qpts_m[m], f_m[m], div_m[m], jw_m[m], rwg['tri_m'][m]),
        ]
        for test_pts, test_f_h, test_div_h, test_jw_h, test_tri_idx in test_halves:
            R_vec = test_pts[None, :, None, :] - all_qpts[:, None, :, :]
            R = np.linalg.norm(R_vec, axis=-1)
            R_safe = np.where(R < 1e-15, 1.0, R)

            G_full = np.exp(1j * k * R) / (4 * np.pi * R_safe)
            G_full = np.where(R < 1e-15, 0.0, G_full)

            G_smooth = (np.exp(1j * k * R) - 1.0) / (4 * np.pi * R_safe)
            G_smooth = np.where(R < 1e-15, 1j * k / (4 * np.pi), G_smooth)

            singular_mask_p = (rwg['tri_p'] == test_tri_idx)
            singular_mask_m = (rwg['tri_m'] == test_tri_idx)
            sing_mask = np.zeros((N, 2*Nq), dtype=bool)
            sing_mask[:, :Nq] = singular_mask_p[:, None]
            sing_mask[:, Nq:] = singular_mask_m[:, None]

            G_use = np.where(sing_mask[:, None, :], G_smooth, G_full)

            f_dot = np.einsum('qi,nji->nqj', test_f_h, all_f)
            div_prod = test_div_h * all_div

            jw_outer = test_jw_h[None, :, None] * all_jw[:, None, :]

            # KEY CHANGE: + instead of - on div_prod
            L_contrib = (1j * k * f_dot + (1j / k) * div_prod[:, None, :]) * G_use * jw_outer
            L[m, :] += L_contrib.sum(axis=(1, 2))

            # Singular extraction (analytical inner integral)
            if np.any(sing_mask):
                tv = tri_verts_cache[test_tri_idx]
                for iq in range(Nq):
                    P_val = potential_integral_triangle(test_pts[iq], *tv)
                    V_r = vector_potential_integral_triangle(test_pts[iq], *tv)

                    for n in range(N):
                        for half_s in range(2):
                            if half_s == 0 and not singular_mask_p[n]: continue
                            if half_s == 1 and not singular_mask_m[n]: continue
                            if half_s == 0:
                                s_c = rwg['length'][n] / (2*rwg['area_p'][n])
                                s_f = rwg['free_p'][n]; s_d = div_p[n]; s_s = +1
                            else:
                                s_c = rwg['length'][n] / (2*rwg['area_m'][n])
                                s_f = rwg['free_m'][n]; s_d = div_m[n]; s_s = -1
                            fn_R = s_s * s_c * (V_r - s_f * P_val)
                            w = test_jw_h[iq]
                            # Vector part: + (same as original)
                            L[m, n] += 1j * k * np.dot(test_f_h[iq], fn_R) / (4*np.pi) * w
                            # Scalar part: + instead of - (FLIPPED)
                            L[m, n] += (1j / k) * test_div_h * s_d * P_val / (4*np.pi) * w

            # K operator (same as original)
            gradG_coeff = (1j * k - 1.0 / R_safe) / R_safe
            gradG_coeff = np.where(R < 1e-12, 0.0, gradG_coeff)
            gradG_coeff = np.where(sing_mask[:, None, :], 0.0, gradG_coeff)
            gradG = G_full[:, :, :, None] * gradG_coeff[:, :, :, None] * R_vec
            cross = np.cross(gradG, all_f[:, None, :, :])
            K_integrand = np.einsum('qi,nqji->nqj', test_f_h, cross)
            K[m, :] += (K_integrand * jw_outer).sum(axis=(1, 2))

    return L, K


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

    Q_mie = mie_pec_Qsca(x)
    print(f"Mie PEC Q_sca = {Q_mie:.6f}")

    for refine in [1, 2]:
        verts, tris = icosphere(radius, refinements=refine)
        rwg = build_rwg(verts, tris)
        N = rwg['N']
        print(f"\nRefine={refine}: {len(tris)} tris, {N} RWG")

        # Original L (minus on div-div)
        L_orig, K_orig = assemble_L_K(rwg, verts, tris, k)
        # New L (plus on div-div)
        L_plus, K_plus = assemble_L_K_plus(rwg, verts, tris, k)

        # PEC EFIE: Z*I = V, Z = η*L
        quad_pts, quad_wts = tri_quadrature(4)
        Nq = len(quad_wts)
        lam0 = 1 - quad_pts[:, 0] - quad_pts[:, 1]

        def get_qpts(tri_indices):
            t = tris[tri_indices]
            v0 = verts[t[:, 0]]; v1 = verts[t[:, 1]]; v2 = verts[t[:, 2]]
            return np.einsum('q,ni->nqi', lam0, v0) + \
                   np.einsum('q,ni->nqi', quad_pts[:, 0], v1) + \
                   np.einsum('q,ni->nqi', quad_pts[:, 1], v2)

        qpts_p = get_qpts(rwg['tri_p']); qpts_m = get_qpts(rwg['tri_m'])
        E0 = np.array([1.0, 0, 0]); k_hat = np.array([0, 0, 1.0])

        V = np.zeros(N, dtype=complex)
        for qpts, free, area, sign in [
            (qpts_p, rwg['free_p'], rwg['area_p'], +1),
            (qpts_m, rwg['free_m'], rwg['area_m'], -1),
        ]:
            f = sign * (rwg['length'][:,None,None] / (2*area[:,None,None])) * (qpts - free[:,None,:])
            jw = area[:,None] * quad_wts[None,:]
            phase = np.exp(1j * k * np.einsum('i,nqi->nq', k_hat, qpts))
            V += np.sum(np.einsum('nqi,i->nq', f, E0) * phase * jw, axis=1)

        for label, L_mat in [("L_orig (- div)", L_orig), ("L_plus (+ div)", L_plus)]:
            Z = eta * L_mat
            I = np.linalg.solve(Z, V)

            # Q_ext via optical theorem
            r_hat_fwd = k_hat
            J_int = np.zeros(3, dtype=complex)
            for qpts, free, area, sign in [
                (qpts_p, rwg['free_p'], rwg['area_p'], +1),
                (qpts_m, rwg['free_m'], rwg['area_m'], -1),
            ]:
                f = sign * (rwg['length'][:,None,None] / (2*area[:,None,None])) * (qpts - free[:,None,:])
                jw = area[:,None] * quad_wts[None,:]
                phase = np.exp(-1j * k * np.einsum('i,nqi->nq', r_hat_fwd, qpts))
                weighted = f * (I[:, None, None] * phase[:, :, None] * jw[:, :, None])
                J_int += weighted.sum(axis=(0, 1))

            J_perp = J_int - r_hat_fwd * np.dot(r_hat_fwd, J_int)
            F_fwd = -1j * k * eta / (4 * np.pi) * J_perp
            S_fwd = np.dot(F_fwd, E0)
            C_ext = 4 * np.pi / k * np.imag(S_fwd)
            Q_ext = C_ext / (np.pi * radius**2)
            err = abs(Q_ext - Q_mie) / Q_mie * 100

            cond = np.linalg.cond(Z)
            print(f"  {label}: cond={cond:.2e}, Q_ext={Q_ext:.6f}, err={err:.1f}%")
