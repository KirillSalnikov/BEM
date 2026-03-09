"""
Mueller matrix comparison: BEM vs Mie theory
ka=1, m=1.5, ref=3 (1920 RWG basis functions)
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy.linalg import lu_factor
from scipy.special import spherical_jn, spherical_yn, lpmv

from bem_core import (icosphere, build_rwg, assemble_pmchwt,
                      compute_amplitude_matrix, amplitude_to_mueller)


# ============================================================
# Mie theory
# ============================================================
def mie_mueller(x, m_rel, theta_arr, n_max=None):
    """Mie S1, S2 and Mueller elements for a sphere."""
    if n_max is None:
        n_max = int(x + 4*x**(1/3) + 2) + 10

    def psi(n, z): return z * spherical_jn(n, z)
    def psi_d(n, z): return spherical_jn(n, z) + z * spherical_jn(n, z, derivative=True)
    def xi(n, z): return z * (spherical_jn(n, z) + 1j * spherical_yn(n, z))
    def xi_d(n, z):
        return (spherical_jn(n, z) + 1j * spherical_yn(n, z)) + \
               z * (spherical_jn(n, z, derivative=True) + 1j * spherical_yn(n, z, derivative=True))

    mx = m_rel * x
    a_coeffs = []; b_coeffs = []
    for n in range(1, n_max + 1):
        a_n = (m_rel * psi(n, mx) * psi_d(n, x) - psi(n, x) * psi_d(n, mx)) / \
              (m_rel * psi(n, mx) * xi_d(n, x) - xi(n, x) * psi_d(n, mx))
        b_n = (psi(n, mx) * psi_d(n, x) - m_rel * psi(n, x) * psi_d(n, mx)) / \
              (psi(n, mx) * xi_d(n, x) - m_rel * xi(n, x) * psi_d(n, mx))
        a_coeffs.append(a_n); b_coeffs.append(b_n)

    S1 = np.zeros(len(theta_arr), dtype=complex)
    S2 = np.zeros(len(theta_arr), dtype=complex)
    for n in range(1, n_max + 1):
        for it, th in enumerate(theta_arr):
            ct = np.cos(th); st = np.sin(th)
            if abs(st) < 1e-15:
                pi_n = n*(n+1)/2 if ct > 0 else ((-1)**(n+1))*n*(n+1)/2
                tau_n = pi_n * ct
            else:
                pi_n = lpmv(1, n, ct) / (-st)
                dt = 1e-6
                tau_n = -(lpmv(1, n, np.cos(th+dt)) - lpmv(1, n, np.cos(th-dt))) / (2*dt)
            coeff = (2*n+1) / (n*(n+1))
            S1[it] += coeff * (a_coeffs[n-1]*pi_n + b_coeffs[n-1]*tau_n)
            S2[it] += coeff * (a_coeffs[n-1]*tau_n + b_coeffs[n-1]*pi_n)

    M11 = 0.5 * (np.abs(S1)**2 + np.abs(S2)**2) / x**2
    M12 = 0.5 * (np.abs(S2)**2 - np.abs(S1)**2) / x**2
    M33 = np.real(S1 * np.conj(S2)) / x**2
    M34 = np.imag(S2 * np.conj(S1)) / x**2
    return {'S1': S1, 'S2': S2, 'M11': M11, 'M12': M12, 'M33': M33, 'M34': M34}


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    ka = 1.0
    m_rel = 1.5
    ref = 3
    radius = 1.0
    k_ext = ka
    eta_ext = 1.0
    k_int = k_ext * m_rel
    eta_int = eta_ext / m_rel

    theta_arr = np.linspace(0.01, np.pi - 0.01, 181)
    theta_deg = np.degrees(theta_arr)

    # --- Mie ---
    print("Computing Mie reference...")
    mie = mie_mueller(ka, m_rel, theta_arr)

    # --- BEM ---
    print(f"BEM assembly (ref={ref})...")
    t0 = time.time()
    verts, tris = icosphere(radius, refinements=ref)
    rwg = build_rwg(verts, tris)
    N = rwg['N']
    print(f"  Mesh: {len(tris)} triangles, {N} RWG basis functions")
    Z, _, _ = assemble_pmchwt(rwg, verts, tris, k_ext, k_int, eta_ext, eta_int)
    Z_lu = lu_factor(Z)
    t_assembly = time.time() - t0
    print(f"  Assembly + LU: {t_assembly:.1f}s")

    print("Computing BEM amplitude matrix...")
    t0 = time.time()
    S_bem = compute_amplitude_matrix(rwg, verts, tris, k_ext, eta_ext, theta_arr,
                                      Z_lu=Z_lu, sM=-1)
    M_bem = amplitude_to_mueller(S_bem['S1'], S_bem['S2'],
                                  S_bem['S3'], S_bem['S4']) / k_ext**2
    t_ff = time.time() - t0
    print(f"  Far-field: {t_ff:.1f}s")

    # --- Compare ---
    bem_M11 = M_bem[0, 0]
    bem_M12 = M_bem[0, 1]
    bem_M33 = M_bem[2, 2]
    bem_M34 = M_bem[2, 3]

    mie_M11 = mie['M11']
    mie_M12 = mie['M12']
    mie_M33 = mie['M33']
    mie_M34 = mie['M34']

    # Relative errors (RMS over angles, weighted by M11 for M12/M33/M34)
    def rms_rel_error(bem, mie_ref, denom):
        """RMS relative error with given denominator."""
        return np.sqrt(np.mean(((bem - mie_ref) / np.maximum(np.abs(denom), 1e-30))**2))

    err_M11 = rms_rel_error(bem_M11, mie_M11, mie_M11)
    err_M12 = rms_rel_error(bem_M12, mie_M12, mie_M11)  # normalize by M11
    err_M33 = rms_rel_error(bem_M33, mie_M33, mie_M11)
    err_M34 = rms_rel_error(bem_M34, mie_M34, mie_M11)

    print(f"\n{'='*60}")
    print(f"  Mueller matrix comparison: ka={ka}, m={m_rel}, ref={ref}")
    print(f"  N_rwg = {N}")
    print(f"{'='*60}")
    print(f"  Element   RMS relative error")
    print(f"  -------   ------------------")
    print(f"  M11       {err_M11*100:.2f}%")
    print(f"  M12       {err_M12*100:.2f}%  (normalized by M11)")
    print(f"  M33       {err_M33*100:.2f}%  (normalized by M11)")
    print(f"  M34       {err_M34*100:.2f}%  (normalized by M11)")
    print(f"{'='*60}")

    # Also print max errors
    maxerr_M11 = np.max(np.abs(bem_M11 - mie_M11) / np.maximum(np.abs(mie_M11), 1e-30))
    print(f"\n  Max relative errors:")
    print(f"  M11: {maxerr_M11*100:.2f}%")

    # Print sample values at a few angles
    sample_idx = [0, 45, 90, 135, 180]
    print(f"\n  Sample M11 values:")
    print(f"  {'theta':>6s}  {'BEM':>12s}  {'Mie':>12s}  {'rel_err':>8s}")
    for i in sample_idx:
        if i < len(theta_deg):
            rel = abs(bem_M11[i] - mie_M11[i]) / max(abs(mie_M11[i]), 1e-30)
            print(f"  {theta_deg[i]:6.1f}  {bem_M11[i]:12.6f}  {mie_M11[i]:12.6f}  {rel*100:7.2f}%")

    print("\nDone.")
