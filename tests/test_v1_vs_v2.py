"""
Compare bem_core (v1) vs bem_core_v2 on dielectric sphere.
"""
import numpy as np
import time
import sys

# Mie reference
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
    mx = m_rel * x; Q_ext = 0.0; Q_sca = 0.0
    for n in range(1, n_max + 1):
        a_n = (m_rel * psi(n, mx) * psi_d(n, x) - psi(n, x) * psi_d(n, mx)) / \
              (m_rel * psi(n, mx) * xi_d(n, x) - xi(n, x) * psi_d(n, mx))
        b_n = (psi(n, mx) * psi_d(n, x) - m_rel * psi(n, x) * psi_d(n, mx)) / \
              (psi(n, mx) * xi_d(n, x) - m_rel * xi(n, x) * psi_d(n, mx))
        Q_ext += (2*n + 1) * np.real(a_n + b_n)
        Q_sca += (2*n + 1) * (abs(a_n)**2 + abs(b_n)**2)
    return 2 * Q_ext / x**2, 2 * Q_sca / x**2


def run_test(module, label, refine=2):
    wavelength = 2 * np.pi
    k_ext = 1.0
    radius = 1.0
    x = k_ext * radius
    m_rel = 1.5
    k_int = k_ext * m_rel
    eta_ext = 1.0
    eta_int = 1.0 / m_rel

    Q_ext_mie, Q_sca_mie = mie_Qext_Qsca(x, m_rel)

    print(f"\n{'='*60}")
    print(f"  {label}, refine={refine}")
    print(f"  Mie: Q_ext={Q_ext_mie:.6f}, Q_sca={Q_sca_mie:.6f}")
    print(f"{'='*60}")

    t0 = time.time()
    verts, tris = module.icosphere(radius, refinements=refine)
    rwg = module.build_rwg(verts, tris)
    N = rwg['N']
    print(f"  Mesh: {len(tris)} tris, {N} RWG, system {2*N}x{2*N}")

    t1 = time.time()
    Z, L_ext, K_ext = module.assemble_pmchwt(
        rwg, verts, tris, k_ext, k_int, eta_ext, eta_int)
    t_asm = time.time() - t1
    print(f"  Assembly: {t_asm:.1f}s")

    b = module.compute_rhs_planewave(rwg, verts, tris, k_ext, eta_ext)
    coeffs = np.linalg.solve(Z, b)
    coeffs_J = coeffs[:N]
    coeffs_M = coeffs[N:]

    theta_arr = np.linspace(0.01, np.pi - 0.01, 181)
    F_theta, F_phi = module.compute_far_field(
        rwg, verts, tris, coeffs_J, coeffs_M, k_ext, theta_arr)
    dsigma, C_sca = module.compute_cross_sections(F_theta, F_phi, theta_arr, k_ext)
    Q_sca = C_sca / (np.pi * radius**2)

    # Extinction via optical theorem
    r_fwd = np.array([0, 0, 1.0])
    F_t_fwd, F_p_fwd = module.compute_far_field(
        rwg, verts, tris, coeffs_J, coeffs_M, k_ext,
        np.array([0.001]), phi=0.0)
    # Simpler: use forward scattering
    E0 = np.array([1.0, 0, 0])
    # F at theta~0: F_theta ~ F_x (for x-pol, phi=0)
    S_fwd = F_t_fwd[0]  # approximately F·x_hat
    Q_ext = 4 * np.pi / k_ext * np.imag(S_fwd) / (np.pi * radius**2)

    t_total = time.time() - t0

    err_sca = abs(Q_sca - Q_sca_mie) / Q_sca_mie * 100
    err_ext = abs(Q_ext - Q_ext_mie) / Q_ext_mie * 100

    print(f"\n  Q_sca: {Q_sca:.6f}  (Mie: {Q_sca_mie:.6f}, err: {err_sca:.1f}%)")
    print(f"  Q_ext: {Q_ext:.6f}  (Mie: {Q_ext_mie:.6f}, err: {err_ext:.1f}%)")
    print(f"  Total time: {t_total:.1f}s")
    print(f"  Assembly time: {t_asm:.1f}s")

    return Q_sca, Q_ext, t_asm, t_total


if __name__ == "__main__":
    refine = 2
    if len(sys.argv) > 1:
        refine = int(sys.argv[1])

    import bem_core as v1
    import bem_core_v2 as v2

    r1 = run_test(v1, "v1 (original)", refine=refine)
    r2 = run_test(v2, "v2 (improved)", refine=refine)

    print(f"\n{'='*60}")
    print(f"  SUMMARY (refine={refine})")
    print(f"{'='*60}")
    print(f"  {'':20s} {'Q_sca':>10s} {'Q_ext':>10s} {'asm(s)':>8s} {'total(s)':>8s}")
    print(f"  {'v1 (original)':20s} {r1[0]:>10.6f} {r1[1]:>10.6f} {r1[2]:>8.1f} {r1[3]:>8.1f}")
    print(f"  {'v2 (improved)':20s} {r2[0]:>10.6f} {r2[1]:>10.6f} {r2[2]:>8.1f} {r2[3]:>8.1f}")
