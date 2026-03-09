"""
Generate comparison plots: BEM solver vs ADDA vs Mie theory.
Saves results to PDF.
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

from bem_core import (icosphere, build_rwg, tri_quadrature,
                      assemble_L_K, assemble_pmchwt,
                      compute_rhs_planewave, compute_far_field,
                      compute_cross_sections)

ADDA = "/home/serg/GO-for-ADDA/adda/src/seq/adda"


# ============================================================
# Mie theory
# ============================================================

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


def mie_S1S2(x, m_rel, theta_arr, n_max=None):
    """Mie amplitude functions S1(theta), S2(theta)."""
    from scipy.special import spherical_jn, spherical_yn, lpmv
    if n_max is None:
        n_max = int(x + 4*x**(1/3) + 2) + 5
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
    return S1, S2


# ============================================================
# ADDA runner
# ============================================================

def run_adda(x, m_rel, dpl=10):
    with tempfile.TemporaryDirectory() as tmpdir:
        lam = 2 * np.pi; a_eq = x
        cmd = [ADDA, "-shape", "sphere", "-eq_rad", str(a_eq),
               "-m", str(m_rel), "0", "-lambda", str(lam),
               "-dpl", str(dpl), "-dir", tmpdir, "-no_vol_cor"]
        t0 = time.time()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        except Exception as e:
            print(f"  ADDA failed: {e}")
            return None, None, 0, 0
        t_adda = time.time() - t0
        if result.returncode != 0:
            print(f"  ADDA error: {result.stderr[:200]}")
            return None, None, t_adda, 0
        cs_file = os.path.join(tmpdir, "CrossSec-Y")
        if not os.path.exists(cs_file):
            return None, None, t_adda, 0
        Q_ext = Q_sca = Q_abs = None
        with open(cs_file) as f:
            for line in f:
                if '=' in line:
                    key, val = line.split('=', 1)
                    key = key.strip(); val = float(val.strip())
                    if key == "Qext" and Q_ext is None: Q_ext = val
                    elif key == "Qsca" and Q_sca is None: Q_sca = val
                    elif key == "Qabs" and Q_abs is None: Q_abs = val
        if Q_sca is None and Q_ext is not None:
            Q_sca = Q_ext - (Q_abs or 0)
        n_dip = 0
        log_file = os.path.join(tmpdir, "log")
        if os.path.exists(log_file):
            with open(log_file) as f:
                for line in f:
                    if "Total number of occupied dipoles" in line:
                        n_dip = int(line.split(':')[1].strip())
        return Q_ext, Q_sca, t_adda, n_dip


# ============================================================
# BEM solver (PEC EFIE)
# ============================================================

def run_bem_pec(radius, k, eta, refine):
    E0 = np.array([1.0, 0, 0]); k_hat = np.array([0, 0, 1.0])
    t0 = time.time()
    verts, tris = icosphere(radius, refinements=refine)
    rwg = build_rwg(verts, tris)
    N = rwg['N']

    L, K_op = assemble_L_K(rwg, verts, tris, k)

    # RHS
    quad_pts, quad_wts = tri_quadrature(7)
    lam0 = 1 - quad_pts[:,0] - quad_pts[:,1]
    def get_qpts(ti):
        t = tris[ti]; v0 = verts[t[:,0]]; v1 = verts[t[:,1]]; v2 = verts[t[:,2]]
        return (np.einsum('q,ni->nqi', lam0, v0) +
                np.einsum('q,ni->nqi', quad_pts[:,0], v1) +
                np.einsum('q,ni->nqi', quad_pts[:,1], v2))
    qp = get_qpts(rwg['tri_p']); qm = get_qpts(rwg['tri_m'])
    V_E = np.zeros(N, dtype=complex)
    for qpts, free, area, sign in [(qp, rwg['free_p'], rwg['area_p'], +1),
                                     (qm, rwg['free_m'], rwg['area_m'], -1)]:
        f = sign * (rwg['length'][:,None,None] / (2*area[:,None,None])) * (qpts - free[:,None,:])
        jw = area[:,None] * quad_wts[None,:]
        phase = np.exp(1j * k * np.einsum('i,nqi->nq', k_hat, qpts))
        V_E += np.sum(np.einsum('nqi,i->nq', f, E0) * phase * jw, axis=1)

    J = np.linalg.solve(eta * L, V_E)
    M = np.zeros(N, dtype=complex)

    Q_ext, Q_sca = compute_cross_sections(rwg, verts, tris, J, M, k, eta, radius, sM=+1)
    t_total = time.time() - t0

    # Far field for bistatic pattern
    theta_arr = np.linspace(0.01, np.pi - 0.01, 181)
    F_th, F_ph = compute_far_field(rwg, verts, tris, J, M, k, eta, theta_arr, phi=0.0, sM=+1)

    return Q_ext, Q_sca, t_total, N, len(tris), theta_arr, F_th, F_ph


# ============================================================
# BEM solver (dielectric PMCHWT)
# ============================================================

def run_bem_dielectric(radius, k_ext, k_int, eta_ext, eta_int, refine):
    t0 = time.time()
    verts, tris = icosphere(radius, refinements=refine)
    rwg = build_rwg(verts, tris)
    N = rwg['N']

    Z, _, _ = assemble_pmchwt(rwg, verts, tris, k_ext, k_int, eta_ext, eta_int)
    b = compute_rhs_planewave(rwg, verts, tris, k_ext, eta_ext)
    coeffs = np.linalg.solve(Z, b)
    J = coeffs[:N]; M = coeffs[N:]

    Q_ext, Q_sca = compute_cross_sections(rwg, verts, tris, J, M, k_ext, eta_ext, radius, sM=-1)
    t_total = time.time() - t0

    # Far field
    theta_arr = np.linspace(0.01, np.pi - 0.01, 181)
    F_th, F_ph = compute_far_field(rwg, verts, tris, J, M, k_ext, eta_ext, theta_arr, phi=0.0, sM=-1)

    return Q_ext, Q_sca, t_total, N, len(tris), theta_arr, F_th, F_ph


# ============================================================
# Main: run all cases and generate PDF
# ============================================================

if __name__ == "__main__":
    pdf_path = "/home/serg/bem_solver/comparison_results.pdf"

    # ---- Parameters ----
    radius = 1.0
    k_ext = 1.0
    eta_ext = 1.0

    # Dielectric
    m_rel = 1.5
    k_int = k_ext * m_rel
    eta_int = eta_ext / m_rel
    x = k_ext * radius

    # Mie references
    Q_mie_pec = mie_pec_Qsca(x)
    Q_ext_mie_d, Q_sca_mie_d = mie_Qext_Qsca(x, m_rel)

    print(f"Mie PEC Q_sca = {Q_mie_pec:.6f}")
    print(f"Mie dielectric Q_ext = {Q_ext_mie_d:.6f}, Q_sca = {Q_sca_mie_d:.6f}")

    # ---- Run BEM for multiple refinements ----
    pec_results = []
    die_results = []

    for ref in [1, 2, 3]:
        print(f"\n--- PEC refine={ref} ---")
        res = run_bem_pec(radius, k_ext, eta_ext, ref)
        pec_results.append({'ref': ref, 'Q_ext': res[0], 'Q_sca': res[1],
                            'time': res[2], 'N': res[3], 'ntri': res[4],
                            'theta': res[5], 'F_th': res[6], 'F_ph': res[7]})
        print(f"  N={res[3]}, Q_ext={res[0]:.6f}, Q_sca={res[1]:.6f}, time={res[2]:.1f}s")

    for ref in [1, 2, 3]:
        print(f"\n--- Dielectric refine={ref} ---")
        res = run_bem_dielectric(radius, k_ext, k_int, eta_ext, eta_int, ref)
        die_results.append({'ref': ref, 'Q_ext': res[0], 'Q_sca': res[1],
                            'time': res[2], 'N': res[3], 'ntri': res[4],
                            'theta': res[5], 'F_th': res[6], 'F_ph': res[7]})
        print(f"  N={res[3]}, Q_ext={res[0]:.6f}, Q_sca={res[1]:.6f}, time={res[2]:.1f}s")

    # ---- Run ADDA ----
    print("\n--- ADDA ---")
    adda_results = []
    for dpl in [10, 15, 20]:
        print(f"  dpl={dpl}...")
        Qe, Qs, ta, nd = run_adda(x, m_rel, dpl=dpl)
        if Qe is not None:
            adda_results.append({'dpl': dpl, 'Q_ext': Qe, 'Q_sca': Qs,
                                 'time': ta, 'n_dip': nd})
            print(f"    Q_ext={Qe:.6f}, Q_sca={Qs:.6f}, ndip={nd}, time={ta:.1f}s")

    # ---- Generate PDF ----
    print(f"\nGenerating {pdf_path}...")

    with PdfPages(pdf_path) as pdf:

        # ========== Page 1: Convergence PEC ==========
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'PEC Sphere, ka = {x:.1f}', fontsize=14, fontweight='bold')

        Ns = [r['N'] for r in pec_results]
        Q_ext_pec = [r['Q_ext'] for r in pec_results]
        Q_sca_pec = [r['Q_sca'] for r in pec_results]
        err_ext_pec = [abs(q - Q_mie_pec)/Q_mie_pec*100 for q in Q_ext_pec]
        err_sca_pec = [abs(q - Q_mie_pec)/Q_mie_pec*100 for q in Q_sca_pec]

        ax = axes[0]
        ax.plot(Ns, Q_ext_pec, 'bo-', label='BEM $Q_{ext}$ (OT)', markersize=8)
        ax.plot(Ns, Q_sca_pec, 'rs-', label='BEM $Q_{sca}$', markersize=8)
        ax.axhline(Q_mie_pec, color='k', ls='--', label=f'Mie = {Q_mie_pec:.4f}')
        ax.set_xlabel('N (RWG DOFs)')
        ax.set_ylabel('Q')
        ax.set_title('Convergence to Mie')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.semilogy(Ns, err_ext_pec, 'bo-', label='$Q_{ext}$ error', markersize=8)
        ax.semilogy(Ns, err_sca_pec, 'rs-', label='$Q_{sca}$ error', markersize=8)
        ax.set_xlabel('N (RWG DOFs)')
        ax.set_ylabel('Relative error (%)')
        ax.set_title('Error vs mesh refinement')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ========== Page 2: Convergence Dielectric ==========
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'Dielectric Sphere, ka = {x:.1f}, m = {m_rel}', fontsize=14, fontweight='bold')

        Ns_d = [r['N'] for r in die_results]
        Q_ext_d = [r['Q_ext'] for r in die_results]
        Q_sca_d = [r['Q_sca'] for r in die_results]
        err_ext_d = [abs(q - Q_ext_mie_d)/Q_ext_mie_d*100 for q in Q_ext_d]
        err_sca_d = [abs(q - Q_sca_mie_d)/Q_sca_mie_d*100 for q in Q_sca_d]

        # Add ADDA points
        adda_ext = [r['Q_ext'] for r in adda_results]
        adda_sca = [r['Q_sca'] for r in adda_results]
        adda_dpls = [r['dpl'] for r in adda_results]
        adda_err_ext = [abs(q - Q_ext_mie_d)/Q_ext_mie_d*100 for q in adda_ext]
        adda_err_sca = [abs(q - Q_sca_mie_d)/Q_sca_mie_d*100 for q in adda_sca]

        ax = axes[0]
        ax.plot(Ns_d, Q_ext_d, 'bo-', label='BEM $Q_{ext}$', markersize=8)
        ax.plot(Ns_d, Q_sca_d, 'rs-', label='BEM $Q_{sca}$', markersize=8)
        ax.axhline(Q_ext_mie_d, color='b', ls='--', alpha=0.5, label=f'Mie $Q_{{ext}}$ = {Q_ext_mie_d:.4f}')
        ax.axhline(Q_sca_mie_d, color='r', ls='--', alpha=0.5, label=f'Mie $Q_{{sca}}$ = {Q_sca_mie_d:.4f}')
        if adda_results:
            # Plot ADDA at rightmost x position + offset for visibility
            x_adda = max(Ns_d) * 1.3
            ax.plot([x_adda]*len(adda_ext), adda_ext, 'b^', markersize=10, label='ADDA $Q_{ext}$')
            ax.plot([x_adda]*len(adda_sca), adda_sca, 'r^', markersize=10, label='ADDA $Q_{sca}$')
        ax.set_xlabel('N (RWG DOFs)')
        ax.set_ylabel('Q')
        ax.set_title('Convergence to Mie')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.semilogy(Ns_d, err_ext_d, 'bo-', label='BEM $Q_{ext}$ err', markersize=8)
        ax.semilogy(Ns_d, err_sca_d, 'rs-', label='BEM $Q_{sca}$ err', markersize=8)
        if adda_results:
            ax.semilogy(adda_dpls, adda_err_ext, 'b^--', label='ADDA $Q_{ext}$ err', markersize=10)
            ax.semilogy(adda_dpls, adda_err_sca, 'r^--', label='ADDA $Q_{sca}$ err', markersize=10)
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim())
            ax2.set_xlabel('ADDA dpl', color='gray')
        ax.set_xlabel('N (RWG DOFs) / ADDA dpl')
        ax.set_ylabel('Relative error (%)')
        ax.set_title('Error convergence')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ========== Page 3: BEM vs Mie far-field pattern (dielectric, best mesh) ==========
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'Far-field pattern: BEM vs Mie (dielectric, m={m_rel}, ka={x:.1f})',
                     fontsize=14, fontweight='bold')

        best = die_results[-1]  # finest mesh
        theta_deg = np.degrees(best['theta'])

        # Mie S1, S2
        S1, S2 = mie_S1S2(x, m_rel, best['theta'])
        # Mie far-field: dsigma/dOmega = (|S1|^2 + |S2|^2) / (2*k^2) for unpolarized?
        # Actually for x-pol at phi=0: F_theta ~ S2, F_phi ~ 0
        # dsigma = |S2(theta)|^2 / k^2 at phi=0
        dsigma_mie = np.abs(S2)**2 / k_ext**2

        dsigma_bem = np.abs(best['F_th'])**2 + np.abs(best['F_ph'])**2

        ax = axes[0]
        ax.semilogy(theta_deg, dsigma_bem, 'b-', linewidth=2, label=f'BEM (N={best["N"]})')
        ax.semilogy(theta_deg, dsigma_mie, 'r--', linewidth=2, label='Mie')
        ax.set_xlabel('Scattering angle (deg)')
        ax.set_ylabel('$d\\sigma/d\\Omega$')
        ax.set_title('Differential cross section ($\\varphi=0$)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Ratio
        ax = axes[1]
        ratio = dsigma_bem / np.maximum(dsigma_mie, 1e-30)
        ax.plot(theta_deg, ratio, 'b-', linewidth=2)
        ax.axhline(1.0, color='k', ls='--', alpha=0.5)
        ax.set_xlabel('Scattering angle (deg)')
        ax.set_ylabel('BEM / Mie')
        ax.set_title('Ratio BEM/Mie')
        ax.set_ylim(0, 2)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ========== Page 4: Optical theorem check ==========
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        fig.suptitle('Optical theorem: $Q_{ext}$ vs $Q_{sca}$ (PEC, should be equal)',
                     fontsize=14, fontweight='bold')

        ratio_ot = [qe/qs for qe, qs in zip(Q_ext_pec, Q_sca_pec)]
        ax.plot(Ns, ratio_ot, 'go-', markersize=10, linewidth=2)
        ax.axhline(1.0, color='k', ls='--', alpha=0.5, label='$Q_{ext}/Q_{sca} = 1$ (exact)')
        ax.set_xlabel('N (RWG DOFs)')
        ax.set_ylabel('$Q_{ext} / Q_{sca}$')
        ax.set_ylim(0.95, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)

        for i, (n, r) in enumerate(zip(Ns, ratio_ot)):
            ax.annotate(f'ref={pec_results[i]["ref"]}\n{r:.4f}', (n, r),
                        textcoords="offset points", xytext=(10, 10), fontsize=10)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ========== Page 5: Summary table ==========
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.axis('off')
        fig.suptitle('Summary: BEM solver vs ADDA vs Mie theory', fontsize=14, fontweight='bold')

        table_data = []
        headers = ['Method', 'DOFs', '$Q_{ext}$', 'err %', '$Q_{sca}$', 'err %', 'Time (s)']

        # PEC section
        table_data.append(['', '', '', '', '', '', ''])
        table_data.append(['PEC sphere, ka=1', '', f'Mie: {Q_mie_pec:.6f}', '', f'Mie: {Q_mie_pec:.6f}', '', ''])
        for r in pec_results:
            e1 = abs(r['Q_ext'] - Q_mie_pec)/Q_mie_pec*100
            e2 = abs(r['Q_sca'] - Q_mie_pec)/Q_mie_pec*100
            table_data.append([
                f'BEM ref={r["ref"]} ({r["ntri"]} tri)',
                str(r['N']),
                f'{r["Q_ext"]:.6f}', f'{e1:.1f}%',
                f'{r["Q_sca"]:.6f}', f'{e2:.1f}%',
                f'{r["time"]:.1f}'
            ])

        # Dielectric section
        table_data.append(['', '', '', '', '', '', ''])
        table_data.append([f'Dielectric m={m_rel}, ka=1', '',
                           f'Mie: {Q_ext_mie_d:.6f}', '',
                           f'Mie: {Q_sca_mie_d:.6f}', '', ''])
        for r in die_results:
            e1 = abs(r['Q_ext'] - Q_ext_mie_d)/Q_ext_mie_d*100
            e2 = abs(r['Q_sca'] - Q_sca_mie_d)/Q_sca_mie_d*100
            table_data.append([
                f'BEM ref={r["ref"]} ({r["ntri"]} tri)',
                str(r['N']),
                f'{r["Q_ext"]:.6f}', f'{e1:.1f}%',
                f'{r["Q_sca"]:.6f}', f'{e2:.1f}%',
                f'{r["time"]:.1f}'
            ])
        for r in adda_results:
            e1 = abs(r['Q_ext'] - Q_ext_mie_d)/Q_ext_mie_d*100
            e2 = abs(r['Q_sca'] - Q_sca_mie_d)/Q_sca_mie_d*100
            table_data.append([
                f'ADDA dpl={r["dpl"]}',
                str(r['n_dip']),
                f'{r["Q_ext"]:.6f}', f'{e1:.1f}%',
                f'{r["Q_sca"]:.6f}', f'{e2:.1f}%',
                f'{r["time"]:.1f}'
            ])

        table = ax.table(cellText=table_data, colLabels=headers,
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.4)

        # Style header
        for j in range(len(headers)):
            table[0, j].set_facecolor('#4472C4')
            table[0, j].set_text_props(color='white', fontweight='bold')

        # Style section headers
        for i, row in enumerate(table_data, 1):
            if row[1] == '' and row[0] != '':
                for j in range(len(headers)):
                    table[i, j].set_facecolor('#D9E2F3')
                    table[i, j].set_text_props(fontweight='bold')

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"\nDone! Results saved to {pdf_path}")
