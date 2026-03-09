"""
Validate L operator entries by comparing with high-order brute-force quadrature.
"""
import numpy as np
from bem_core import (icosphere, build_rwg, tri_quadrature, assemble_L_K)


def compute_L_entry_bruteforce(m, n, rwg, verts, tris, k, quad_order_test=7, quad_order_src=7):
    """Compute L[m,n] using brute-force quadrature (no singularity extraction)."""
    from scipy.special import roots_legendre

    quad_pts_t, quad_wts_t = tri_quadrature(quad_order_test)
    quad_pts_s, quad_wts_s = tri_quadrature(quad_order_src)
    Nq_t = len(quad_wts_t); Nq_s = len(quad_wts_s)

    lam0_t = 1 - quad_pts_t[:,0] - quad_pts_t[:,1]
    lam0_s = 1 - quad_pts_s[:,0] - quad_pts_s[:,1]

    def get_qpts_one(idx, quad_pts_local, lam0_local):
        t = tris[idx]
        v0 = verts[t[0]]; v1 = verts[t[1]]; v2 = verts[t[2]]
        return lam0_local[:,None]*v0 + quad_pts_local[:,0:1]*v1 + quad_pts_local[:,1:2]*v2

    # Test basis function halves
    L_val = 0.0 + 0j
    test_div = [rwg['length'][m] / (2*rwg['area_p'][m]),
                -rwg['length'][m] / (2*rwg['area_m'][m])]
    src_div = [rwg['length'][n] / (2*rwg['area_p'][n]),
               -rwg['length'][n] / (2*rwg['area_m'][n])]

    for th, (test_tri, test_sign) in enumerate([(rwg['tri_p'][m], +1), (rwg['tri_m'][m], -1)]):
        test_pts = get_qpts_one(test_tri, quad_pts_t, lam0_t)
        test_free = rwg['free_p'][m] if test_sign == +1 else rwg['free_m'][m]
        test_area = rwg['area_p'][m] if test_sign == +1 else rwg['area_m'][m]
        test_coeff = test_sign * rwg['length'][m] / (2 * test_area)
        test_f = test_coeff * (test_pts - test_free)
        test_jw = test_area * quad_wts_t

        for sh, (src_tri, src_sign) in enumerate([(rwg['tri_p'][n], +1), (rwg['tri_m'][n], -1)]):
            src_pts = get_qpts_one(src_tri, quad_pts_s, lam0_s)
            src_free = rwg['free_p'][n] if src_sign == +1 else rwg['free_m'][n]
            src_area = rwg['area_p'][n] if src_sign == +1 else rwg['area_m'][n]
            src_coeff = src_sign * rwg['length'][n] / (2 * src_area)
            src_f = src_coeff * (src_pts - src_free)
            src_jw = src_area * quad_wts_s

            is_same_tri = (test_tri == src_tri)

            for it in range(Nq_t):
                for js in range(Nq_s):
                    R_vec = test_pts[it] - src_pts[js]
                    R = np.linalg.norm(R_vec)

                    if R < 1e-12:
                        continue  # Skip coincident points

                    G = np.exp(1j * k * R) / (4 * np.pi * R)
                    f_dot = np.dot(test_f[it], src_f[js])
                    d_prod = test_div[th] * src_div[sh]

                    jw = test_jw[it] * src_jw[js]
                    L_val += (1j * k * f_dot + 1j / k * d_prod) * G * jw

    return L_val


if __name__ == "__main__":
    radius = 1.0; k = 1.0
    verts, tris = icosphere(radius, refinements=1)
    rwg = build_rwg(verts, tris)
    N = rwg['N']

    print(f"Mesh: {len(tris)} tris, {N} RWG")
    print("Assembling L with singularity extraction...")
    L_full, K_full = assemble_L_K(rwg, verts, tris, k)

    # Compare a few entries
    print("\nComparing L entries (code vs brute-force):")
    print(f"{'(m,n)':>8s} {'L_code':>20s} {'L_brute':>20s} {'rel_err':>10s} {'same_tri':>10s}")

    test_pairs = []
    # Diagonal entries
    test_pairs.extend([(i, i) for i in range(5)])
    # Near entries (sharing a triangle)
    for m in range(5):
        for n in range(m+1, min(m+10, N)):
            if (rwg['tri_p'][m] == rwg['tri_p'][n] or rwg['tri_p'][m] == rwg['tri_m'][n] or
                rwg['tri_m'][m] == rwg['tri_p'][n] or rwg['tri_m'][m] == rwg['tri_m'][n]):
                test_pairs.append((m, n))
                if len(test_pairs) >= 15:
                    break
        if len(test_pairs) >= 15:
            break
    # Far entries
    test_pairs.extend([(0, N//2), (0, N-1), (1, N//3)])

    for m, n in test_pairs:
        L_code = L_full[m, n]
        L_brute = compute_L_entry_bruteforce(m, n, rwg, verts, tris, k)
        rel_err = abs(L_code - L_brute) / max(abs(L_code), abs(L_brute), 1e-15)
        same = (rwg['tri_p'][m] == rwg['tri_p'][n] or rwg['tri_p'][m] == rwg['tri_m'][n] or
                rwg['tri_m'][m] == rwg['tri_p'][n] or rwg['tri_m'][m] == rwg['tri_m'][n])
        print(f"  ({m},{n}){'' if m>=10 else ' ':s} {L_code.real:>9.6f}+{L_code.imag:>9.6f}j"
              f"  {L_brute.real:>9.6f}+{L_brute.imag:>9.6f}j  {rel_err:>9.2e}  {'SING' if same else 'FAR'}")

    # Also check if L is symmetric
    print(f"\n||L - L^T|| / ||L|| = {np.linalg.norm(L_full - L_full.T) / np.linalg.norm(L_full):.2e}")

    # Check a few K entries too
    print("\nComparing K entries:")
    for m, n in [(0, 1), (0, N//2), (0, 0)]:
        K_code = K_full[m, n]
        print(f"  K[{m},{n}] = {K_code.real:.6f}+{K_code.imag:.6f}j")
    print(f"||K - K^T|| / ||K|| = {np.linalg.norm(K_full - K_full.T) / np.linalg.norm(K_full):.2e}")
