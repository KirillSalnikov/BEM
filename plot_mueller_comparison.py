"""
Compare Mueller matrix M11(θ) from BEM vs ADDA vs Mie for dielectric sphere.
Generates PDF with plots.
"""
import numpy as np
import subprocess
import os
import time
import tempfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.linalg import lu_factor
from scipy.special import spherical_jn, spherical_yn, lpmv

from bem_core import (icosphere, build_rwg, assemble_pmchwt,
                      compute_amplitude_matrix, amplitude_to_mueller,
                      compute_mueller_matrix, orientation_average_mueller)

ADDA = "/home/serg/GO-for-ADDA/adda/src/seq/adda"


# ============================================================
# Mie theory Mueller matrix
# ============================================================

def mie_mueller(x, m_rel, theta_arr, n_max=None):
    """Compute Mie S1, S2 and Mueller matrix for a sphere."""
    if n_max is None:
        n_max = int(x + 4*x**(1/3) + 2) + 10

    # Mie coefficients
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

    # Mueller matrix: M11 = (|S1|² + |S2|²) / (2k²) (normalized)
    M11 = 0.5 * (np.abs(S1)**2 + np.abs(S2)**2) / x**2  # k=x/a, a=1 => k²=x²
    M12 = 0.5 * (np.abs(S2)**2 - np.abs(S1)**2) / x**2
    M33 = np.real(S1 * np.conj(S2)) / x**2
    M34 = np.imag(S2 * np.conj(S1)) / x**2
    return {'S1': S1, 'S2': S2, 'M11': M11, 'M12': M12, 'M33': M33, 'M34': M34}


# ============================================================
# ADDA Mueller runner
# ============================================================

def run_adda_mueller(x, m_rel, dpl=20):
    """Run ADDA and parse Mueller matrix file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        lam = 2 * np.pi; a_eq = x
        cmd = [ADDA, "-shape", "sphere", "-eq_rad", str(a_eq),
               "-m", str(m_rel), "0", "-lambda", str(lam),
               "-dpl", str(dpl), "-dir", tmpdir, "-no_vol_cor",
               "-scat_matr", "muel"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                                cwd=tmpdir)
        if result.returncode != 0:
            print(f"ADDA error: {result.stderr[:200]}")
            return None

        muel_file = os.path.join(tmpdir, "mueller")
        if not os.path.exists(muel_file):
            print("No mueller file found")
            return None

        data = np.loadtxt(muel_file, skiprows=1)
        # columns: theta s11 s12 s13 s14 s21 s22 s23 s24 s31 s32 s33 s34 s41 s42 s43 s44
        theta_deg = data[:, 0]
        # ADDA s11 is the unnormalized Mueller element
        # ADDA convention: s11 = (|S2|² + |S1|²) / 2  (already divided by k²)
        # Actually ADDA outputs s_ij as in C_sca * M_ij / (4π) or similar...
        # Let's check: for forward scattering, s11 should relate to our M11
        s11 = data[:, 1]
        s12 = data[:, 2]
        s33 = data[:, 11]
        s34 = data[:, 12]

        return {'theta_deg': theta_deg, 's11': s11, 's12': s12,
                's33': s33, 's34': s34}


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    pdf_path = "/home/serg/bem_solver/mueller_comparison.pdf"

    # Test cases
    cases = [
        {'ka': 1.0, 'm': 1.5, 'bem_ref': 3, 'adda_dpl': 20},
        {'ka': 2.0, 'm': 1.5, 'bem_ref': 3, 'adda_dpl': 20},
        {'ka': 3.0, 'm': 1.5, 'bem_ref': 3, 'adda_dpl': 20},
    ]

    with PdfPages(pdf_path) as pdf:
        for case in cases:
            ka = case['ka']; m_rel = case['m']
            ref = case['bem_ref']; dpl = case['adda_dpl']

            radius = 1.0; k_ext = ka; eta_ext = 1.0
            k_int = k_ext * m_rel; eta_int = eta_ext / m_rel

            print(f"\n{'='*60}")
            print(f"  ka={ka}, m={m_rel}, BEM ref={ref}, ADDA dpl={dpl}")
            print(f"{'='*60}")

            # --- Mie ---
            theta_arr = np.linspace(0.01, np.pi - 0.01, 181)
            theta_deg = np.degrees(theta_arr)
            print("  Mie...")
            mie = mie_mueller(ka, m_rel, theta_arr)

            # --- BEM ---
            print("  BEM assembly...")
            verts, tris = icosphere(radius, refinements=ref)
            rwg = build_rwg(verts, tris)
            N = rwg['N']
            Z, _, _ = assemble_pmchwt(rwg, verts, tris, k_ext, k_int, eta_ext, eta_int)
            Z_lu = lu_factor(Z)

            print("  BEM amplitude matrix...")
            t0 = time.time()
            S_bem = compute_amplitude_matrix(rwg, verts, tris, k_ext, eta_ext, theta_arr,
                                              Z_lu=Z_lu, sM=-1)
            M_bem = amplitude_to_mueller(S_bem['S1'], S_bem['S2'],
                                          S_bem['S3'], S_bem['S4']) / k_ext**2
            t_bem = time.time() - t0
            print(f"  BEM Mueller: {t_bem:.1f}s")

            # --- ADDA ---
            print("  ADDA...")
            adda = run_adda_mueller(ka, m_rel, dpl=dpl)

            # --- Normalize ADDA to match our convention ---
            # ADDA outputs: s_ij where s11 at θ=0 = |S1(0)|²/k² = |S2(0)|²/k²
            # Our M11 = (|S1|² + |S2|²) / (2k²)
            # At θ=0: S1=S2, so ADDA s11(0) = |S1(0)|²/k² = 2*M11(0)
            # Actually let's check the scale factor by comparing forward values
            if adda is not None:
                # ADDA s11 normalization: ADDA outputs s11 = (|S2|²+|S1|²)/2 in some units
                # Let's find scale by matching at θ=0
                scale = mie['M11'][0] / adda['s11'][0] if adda['s11'][0] > 0 else 1.0
                print(f"  ADDA/Mie scale factor at θ=0: {scale:.6f}")
                adda_M11 = adda['s11'] * scale
                adda_M12 = adda['s12'] * scale
                adda_M33 = adda['s33'] * scale
                adda_M34 = adda['s34'] * scale

            # ========== Plot: M11 ==========
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f'Матрица Мюллера: ka = {ka}, m = {m_rel}',
                         fontsize=16, fontweight='bold')

            # M11 (log scale)
            ax = axes[0, 0]
            ax.semilogy(theta_deg, mie['M11'], 'k-', lw=2, label='Mie')
            ax.semilogy(theta_deg, M_bem[0, 0], 'b--', lw=2, label=f'BEM (N={N})')
            if adda is not None:
                ax.semilogy(adda['theta_deg'], adda_M11, 'r:', lw=2, label=f'ADDA (dpl={dpl})')
            ax.set_xlabel('θ (deg)'); ax.set_ylabel('$M_{11}$')
            ax.set_title('$M_{11}$ (дифференциальное сечение)')
            ax.legend(); ax.grid(True, alpha=0.3)

            # -M12/M11 (degree of polarization)
            ax = axes[0, 1]
            pol_mie = -mie['M12'] / np.maximum(mie['M11'], 1e-30)
            pol_bem = -M_bem[1, 0] / np.maximum(M_bem[0, 0], 1e-30)
            ax.plot(theta_deg, pol_mie, 'k-', lw=2, label='Mie')
            ax.plot(theta_deg, pol_bem, 'b--', lw=2, label='BEM')
            if adda is not None:
                pol_adda = -adda_M12 / np.maximum(adda_M11, 1e-30)
                ax.plot(adda['theta_deg'], pol_adda, 'r:', lw=2, label='ADDA')
            ax.set_xlabel('θ (deg)'); ax.set_ylabel('$-M_{12}/M_{11}$')
            ax.set_title('Степень поляризации')
            ax.set_ylim(-1.1, 1.1)
            ax.legend(); ax.grid(True, alpha=0.3)

            # M33/M11
            ax = axes[1, 0]
            r33_mie = mie['M33'] / np.maximum(mie['M11'], 1e-30)
            r33_bem = M_bem[2, 2] / np.maximum(M_bem[0, 0], 1e-30)
            ax.plot(theta_deg, r33_mie, 'k-', lw=2, label='Mie')
            ax.plot(theta_deg, r33_bem, 'b--', lw=2, label='BEM')
            if adda is not None:
                r33_adda = adda_M33 / np.maximum(adda_M11, 1e-30)
                ax.plot(adda['theta_deg'], r33_adda, 'r:', lw=2, label='ADDA')
            ax.set_xlabel('θ (deg)'); ax.set_ylabel('$M_{33}/M_{11}$')
            ax.set_title('$M_{33}/M_{11}$')
            ax.set_ylim(-1.1, 1.1)
            ax.legend(); ax.grid(True, alpha=0.3)

            # M34/M11
            ax = axes[1, 1]
            r34_mie = mie['M34'] / np.maximum(mie['M11'], 1e-30)
            r34_bem = M_bem[2, 3] / np.maximum(M_bem[0, 0], 1e-30)
            ax.plot(theta_deg, r34_mie, 'k-', lw=2, label='Mie')
            ax.plot(theta_deg, r34_bem, 'b--', lw=2, label='BEM')
            if adda is not None:
                r34_adda = adda_M34 / np.maximum(adda_M11, 1e-30)
                ax.plot(adda['theta_deg'], r34_adda, 'r:', lw=2, label='ADDA')
            ax.set_xlabel('θ (deg)'); ax.set_ylabel('$M_{34}/M_{11}$')
            ax.set_title('$M_{34}/M_{11}$')
            ax.set_ylim(-1.1, 1.1)
            ax.legend(); ax.grid(True, alpha=0.3)

            plt.tight_layout()
            pdf.savefig(fig); plt.close(fig)

            # ========== Page 2: M11 ratio ==========
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(f'Отношение M11: ka = {ka}, m = {m_rel}',
                         fontsize=14, fontweight='bold')

            ax = axes[0]
            ratio_bem = M_bem[0, 0] / np.maximum(mie['M11'], 1e-30)
            ax.plot(theta_deg, ratio_bem, 'b-', lw=2, label='BEM/Mie')
            if adda is not None:
                # Interpolate ADDA to our theta grid
                from scipy.interpolate import interp1d
                f_adda = interp1d(adda['theta_deg'], adda_M11, kind='linear',
                                   fill_value='extrapolate')
                ratio_adda = f_adda(theta_deg) / np.maximum(mie['M11'], 1e-30)
                ax.plot(theta_deg, ratio_adda, 'r--', lw=2, label='ADDA/Mie')
            ax.axhline(1.0, color='k', ls='--', alpha=0.5)
            ax.set_xlabel('θ (deg)'); ax.set_ylabel('Ratio')
            ax.set_title('$M_{11}$ / Mie')
            ax.set_ylim(0.8, 1.2)
            ax.legend(); ax.grid(True, alpha=0.3)

            # M11 relative error
            ax = axes[1]
            err_bem = np.abs(M_bem[0, 0] - mie['M11']) / np.maximum(mie['M11'], 1e-30) * 100
            ax.semilogy(theta_deg, err_bem, 'b-', lw=2, label='BEM error')
            if adda is not None:
                err_adda = np.abs(f_adda(theta_deg) - mie['M11']) / np.maximum(mie['M11'], 1e-30) * 100
                ax.semilogy(theta_deg, err_adda, 'r--', lw=2, label='ADDA error')
            ax.set_xlabel('θ (deg)'); ax.set_ylabel('Relative error (%)')
            ax.set_title('$M_{11}$ error vs Mie')
            ax.legend(); ax.grid(True, alpha=0.3)

            plt.tight_layout()
            pdf.savefig(fig); plt.close(fig)

    print(f"\nDone! Saved to {pdf_path}")
