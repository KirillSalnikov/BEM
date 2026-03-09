"""
Test both div-div signs for L operator in BOTH PEC EFIE AND dielectric PMCHWT.

Theory: L_mn = ik <f·f>_G ± (i/k) <div·div>_G
The sign of the div-div term is critical.

Test hypothesis: the "correct" theoretical sign is MINUS (from E = -iωA - ∇Φ),
but maybe we had it wrong and need to recheck.
"""
import numpy as np
from bem_core import (icosphere, build_rwg, tri_quadrature,
                       potential_integral_triangle, vector_potential_integral_triangle)


def assemble_L_both(rwg, verts, tris, k, quad_order=7):
    """Assemble L with both + and - on div-div, returning L_plus and L_minus."""
    N = rwg['N']
    quad_pts, quad_wts = tri_quadrature(quad_order)
    Nq = len(quad_wts)
    lam0 = 1 - quad_pts[:, 0] - quad_pts[:, 1]

    def get_qpts(ti):
        t = tris[ti]; v0 = verts[t[:,0]]; v1 = verts[t[:,1]]; v2 = verts[t[:,2]]
        return np.einsum('q,ni->nqi', lam0, v0) + np.einsum('q,ni->nqi', quad_pts[:,0], v1) + \
               np.einsum('q,ni->nqi', quad_pts[:,1], v2)

    qpts_p = get_qpts(rwg['tri_p']); qpts_m = get_qpts(rwg['tri_m'])
    f_p = (rwg['length'][:,None,None] / (2*rwg['area_p'][:,None,None])) * (qpts_p - rwg['free_p'][:,None,:])
    f_m = -(rwg['length'][:,None,None] / (2*rwg['area_m'][:,None,None])) * (qpts_m - rwg['free_m'][:,None,:])
    div_p = rwg['length'] / (2 * rwg['area_p'])
    div_m = -rwg['length'] / (2 * rwg['area_m'])
    jw_p = rwg['area_p'][:,None] * quad_wts[None,:]
    jw_m = rwg['area_m'][:,None] * quad_wts[None,:]

    all_qpts = np.concatenate([qpts_p, qpts_m], axis=1)
    all_f = np.concatenate([f_p, f_m], axis=1)
    all_div = np.concatenate([div_p[:,None].repeat(Nq, axis=1),
                               div_m[:,None].repeat(Nq, axis=1)], axis=1)
    all_jw = np.concatenate([jw_p, jw_m], axis=1)

    # Separate f·f and div·div contributions
    L_ff = np.zeros((N, N), dtype=complex)
    L_dd = np.zeros((N, N), dtype=complex)

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

            # f·f contribution: ik * f_dot * G
            ff_contrib = (1j * k * f_dot) * G_use * jw_outer
            L_ff[m, :] += ff_contrib.sum(axis=(1, 2))

            # div·div contribution: (i/k) * div_prod * G
            dd_contrib = ((1j / k) * div_prod[:, None, :]) * G_use * jw_outer
            L_dd[m, :] += dd_contrib.sum(axis=(1, 2))

            # Singular corrections
            if np.any(sing_mask):
                tv = tri_verts_cache[test_tri_idx]
                P_vals = np.zeros(Nq)
                for iq in range(Nq):
                    P_vals[iq] = potential_integral_triangle(test_pts[iq], *tv)

                for n in range(N):
                    for half_s in range(2):
                        if half_s == 0 and not singular_mask_p[n]: continue
                        if half_s == 1 and not singular_mask_m[n]: continue

                        if half_s == 0:
                            src_div = div_p[n]
                            src_coeff = rwg['length'][n] / (2 * rwg['area_p'][n])
                            src_free = rwg['free_p'][n]
                            src_sign = +1
                        else:
                            src_div = div_m[n]
                            src_coeff = rwg['length'][n] / (2 * rwg['area_m'][n])
                            src_free = rwg['free_m'][n]
                            src_sign = -1

                        scalar_integral = np.dot(P_vals, test_jw_h) / (4 * np.pi)
                        L_dd[m, n] += (1j / k) * test_div_h * src_div * scalar_integral

                        vec_integral = 0.0
                        for iq in range(Nq):
                            V_r = vector_potential_integral_triangle(test_pts[iq], *tv)
                            fn_over_R = src_sign * src_coeff * (V_r - src_free * P_vals[iq])
                            vec_integral += np.dot(test_f_h[iq], fn_over_R) * test_jw_h[iq]
                        vec_integral /= (4 * np.pi)
                        L_ff[m, n] += 1j * k * vec_integral

    return L_ff, L_dd


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


def mie_Qext_Qsca(x, m_rel, n_max=None):
    from scipy.special import spherical_jn, spherical_yn
    if n_max is None:
        n_max = int(x + 4*x**(1/3) + 2) + 5
    def psi(n, z): return z * spherical_jn(n, z)
    def psi_d(n, z): return spherical_jn(n, z) + z * spherical_jn(n, z, derivative=True)
    def xi(n, z): return z * (spherical_jn(n, z) + 1j * spherical_yn(n, z))
    def xi_d(n, z):
        return (spherical_jn(n, z) + 1j * spherical_yn(n, z)) + \
               z * (spherical_jn(n, z, derivative=True) + 1j * spherical_yn(n, z, derivative=True))
    mx = m_rel * x
    Q_ext = 0.0; Q_sca = 0.0
    for n in range(1, n_max + 1):
        a_n = (m_rel * psi(n, mx) * psi_d(n, x) - psi(n, x) * psi_d(n, mx)) / \
              (m_rel * psi(n, mx) * xi_d(n, x) - xi(n, x) * psi_d(n, mx))
        b_n = (psi(n, mx) * psi_d(n, x) - m_rel * psi(n, x) * psi_d(n, mx)) / \
              (psi(n, mx) * xi_d(n, x) - m_rel * xi(n, x) * psi_d(n, mx))
        Q_ext += (2*n + 1) * np.real(a_n + b_n)
        Q_sca += (2*n + 1) * (abs(a_n)**2 + abs(b_n)**2)
    return 2 * Q_ext / x**2, 2 * Q_sca / x**2


def compute_Qsca(rwg, verts, tris, J, M, k, eta, radius, E0, sM=+1):
    quad_pts, quad_wts = tri_quadrature(7)
    N = rwg['N']
    lam0 = 1 - quad_pts[:,0] - quad_pts[:,1]
    def get_qpts(ti):
        t = tris[ti]; v0 = verts[t[:,0]]; v1 = verts[t[:,1]]; v2 = verts[t[:,2]]
        return np.einsum('q,ni->nqi', lam0, v0) + np.einsum('q,ni->nqi', quad_pts[:,0], v1) + \
               np.einsum('q,ni->nqi', quad_pts[:,1], v2)
    qp = get_qpts(rwg['tri_p']); qm = get_qpts(rwg['tri_m'])
    theta_arr = np.linspace(0.01, np.pi - 0.01, 181)
    dsigma_total = np.zeros(len(theta_arr))
    for it, theta in enumerate(theta_arr):
        r_hat = np.array([np.sin(theta), 0, np.cos(theta)])
        theta_hat = np.array([np.cos(theta), 0, -np.sin(theta)])
        phi_hat = np.array([0, 1.0, 0])
        Jt = np.zeros(3, dtype=complex); Mt = np.zeros(3, dtype=complex)
        for qpts, free, area, sign in [(qp, rwg['free_p'], rwg['area_p'], +1),
                                         (qm, rwg['free_m'], rwg['area_m'], -1)]:
            f = sign * (rwg['length'][:,None,None] / (2*area[:,None,None])) * (qpts - free[:,None,:])
            jw = area[:,None] * quad_wts[None,:]
            phase = np.exp(-1j * k * np.einsum('i,nqi->nq', r_hat, qpts))
            integral = (f * (phase * jw)[:,:,None]).sum(axis=1)
            Jt += (integral * J[:,None]).sum(0)
            Mt += (integral * M[:,None]).sum(0)
        Jp = Jt - r_hat * np.dot(r_hat, Jt)
        Mc = np.cross(r_hat, Mt)
        Fv = -1j * k / (4*np.pi) * (eta * Jp + sM * Mc)
        dsigma_total[it] = abs(np.dot(Fv, theta_hat))**2 + abs(np.dot(Fv, phi_hat))**2
    C_sca = 2 * np.pi * np.trapezoid(dsigma_total * np.sin(theta_arr), theta_arr)
    return C_sca / (np.pi * radius**2)


if __name__ == "__main__":
    radius = 1.0; k_ext = 1.0; eta_ext = 1.0
    E0 = np.array([1.0, 0, 0]); k_hat = np.array([0, 0, 1.0])

    # ===== PEC test =====
    Q_mie_pec = mie_pec_Qsca(k_ext * radius)
    print(f"PEC Mie Q_sca = {Q_mie_pec:.6f}")

    for refine in [1, 2, 3]:
        verts, tris = icosphere(radius, refinements=refine)
        rwg = build_rwg(verts, tris)
        N = rwg['N']
        print(f"\nRefine={refine}: {len(tris)} tris, {N} RWG")

        print("  Assembling L (separate f·f and div·div)...")
        L_ff, L_dd = assemble_L_both(rwg, verts, tris, k_ext)

        # RHS for PEC EFIE
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
            phase = np.exp(1j * k_ext * np.einsum('i,nqi->nq', k_hat, qpts))
            V_E += np.sum(np.einsum('nqi,i->nq', f, E0) * phase * jw, axis=1)

        # PEC EFIE: η * L * J = V_E
        # Q_ext via far field
        for dd_sign, label in [(+1, "PLUS"), (-1, "MINUS")]:
            L = L_ff + dd_sign * L_dd
            J = np.linalg.solve(eta_ext * L, V_E)
            M = np.zeros(N, dtype=complex)

            Q_sca = compute_Qsca(rwg, verts, tris, J, M, k_ext, eta_ext, radius, E0, sM=+1)
            err = abs(Q_sca - Q_mie_pec) / Q_mie_pec * 100
            cond_L = np.linalg.cond(L)
            print(f"  PEC EFIE ({label}): Q_sca={Q_sca:.6f} ({err:.1f}%), cond={cond_L:.2e}")

    # ===== Dielectric test =====
    m_rel = 1.5; k_int = k_ext * m_rel; eta_int = 1.0 / m_rel
    Q_ext_mie, Q_sca_mie = mie_Qext_Qsca(k_ext * radius, m_rel)
    print(f"\nDielectric Mie: Q_ext = {Q_ext_mie:.6f}, Q_sca = {Q_sca_mie:.6f}")

    for refine in [2, 3]:
        verts, tris = icosphere(radius, refinements=refine)
        rwg = build_rwg(verts, tris)
        N = rwg['N']
        print(f"\nRefine={refine}: {len(tris)} tris, {N} RWG")

        print("  Assembling L_ext...")
        L_ff_ext, L_dd_ext = assemble_L_both(rwg, verts, tris, k_ext)
        print("  Assembling L_int...")
        L_ff_int, L_dd_int = assemble_L_both(rwg, verts, tris, k_int)

        # K operator (from assemble_L_K, extract K only)
        from bem_core import assemble_L_K
        _, K_ext = assemble_L_K(rwg, verts, tris, k_ext)
        _, K_int = assemble_L_K(rwg, verts, tris, k_int)
        K = K_ext + K_int

        # RHS
        quad_pts, quad_wts = tri_quadrature(7)
        lam0 = 1 - quad_pts[:, 0] - quad_pts[:, 1]
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

        # Test all combinations of div-div sign AND PMCHWT K signs
        print(f"\n  {'dd_sign':>8s} {'K_config':>12s} | {'Q_sca':>8s} {'eS%':>6s}")
        print(f"  {'-'*50}")

        for dd_sign in [+1, -1]:
            L_ext = L_ff_ext + dd_sign * L_dd_ext
            L_int = L_ff_int + dd_sign * L_dd_int

            for sK, sB, config_label in [
                (-1, -1, "B: [-K,-K,-b]"),
                (+1, +1, "C: [+K,-K,+b]"),  # standard
            ]:
                Z = np.zeros((2*N, 2*N), dtype=complex)
                Z[:N, :N] = eta_ext * L_ext + eta_int * L_int
                Z[:N, N:] = sK * K  # sK1
                Z[N:, :N] = -K  # sK2 always -1 for standard; for B, both are sK
                if config_label.startswith("B"):
                    Z[N:, :N] = sK * K  # Both same sign for Config B
                Z[N:, N:] = L_ext / eta_ext + L_int / eta_int
                coeffs = np.linalg.solve(Z, sB * b)
                J = coeffs[:N]; M = coeffs[N:]
                Q_sca = compute_Qsca(rwg, verts, tris, J, M, k_ext, eta_ext, radius, E0, sM=+1)
                err = abs(Q_sca - Q_sca_mie) / Q_sca_mie * 100
                print(f"  {'PLUS' if dd_sign>0 else 'MINUS':>8s} {config_label:>12s} | {Q_sca:>8.4f} {err:>5.1f}%")

        # Also test: MINUS div-div with standard PMCHWT [ηL, +K; -K, L/η]·x = b
        # This is the theoretically correct combination
        for dd_sign in [+1, -1]:
            L_ext = L_ff_ext + dd_sign * L_dd_ext
            L_int = L_ff_int + dd_sign * L_dd_int
            Z = np.zeros((2*N, 2*N), dtype=complex)
            Z[:N, :N] = eta_ext * L_ext + eta_int * L_int
            Z[:N, N:] = K  # +K
            Z[N:, :N] = -K  # -K
            Z[N:, N:] = L_ext / eta_ext + L_int / eta_int
            coeffs = np.linalg.solve(Z, b)
            J = coeffs[:N]; M = coeffs[N:]
            Q_sca = compute_Qsca(rwg, verts, tris, J, M, k_ext, eta_ext, radius, E0, sM=+1)
            err = abs(Q_sca - Q_sca_mie) / Q_sca_mie * 100
            dd_label = "PLUS" if dd_sign > 0 else "MINUS"
            print(f"  Standard PMCHWT with dd={dd_label}: Q_sca={Q_sca:.4f} ({err:.1f}%)")
