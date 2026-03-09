"""
Convergence plots: Q_ext, Q_sca, Q_abs vs size parameter ka.
BEM solver vs ADDA vs Mie theory.
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
        n_max = int(x + 4*x**(1/3) + 2) + 10
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


def mie_pec_Qsca(x, n_max=None):
    from scipy.special import spherical_jn, spherical_yn
    if n_max is None:
        n_max = int(x + 4*x**(1/3) + 2) + 10
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


# ============================================================
# ADDA runner
# ============================================================

def run_adda(x, m_rel, dpl=15):
    with tempfile.TemporaryDirectory() as tmpdir:
        lam = 2 * np.pi; a_eq = x
        cmd = [ADDA, "-shape", "sphere", "-eq_rad", str(a_eq),
               "-m", str(m_rel), "0", "-lambda", str(lam),
               "-dpl", str(dpl), "-dir", tmpdir, "-no_vol_cor"]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        except Exception:
            return None, None, 0
        if result.returncode != 0:
            return None, None, 0
        cs_file = os.path.join(tmpdir, "CrossSec-Y")
        if not os.path.exists(cs_file):
            return None, None, 0
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
        return Q_ext, Q_sca, n_dip


# ============================================================
# BEM runners
# ============================================================

def run_bem_pec(ka, refine):
    """PEC EFIE for given ka and mesh refinement."""
    radius = 1.0; k = ka / radius; eta = 1.0
    E0 = np.array([1.0, 0, 0]); k_hat = np.array([0, 0, 1.0])

    verts, tris = icosphere(radius, refinements=refine)
    rwg = build_rwg(verts, tris)
    N = rwg['N']

    L, K_op = assemble_L_K(rwg, verts, tris, k)

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
    return Q_ext, Q_sca, N


def run_bem_dielectric(ka, m_rel, refine):
    """Dielectric PMCHWT for given ka, m_rel and mesh refinement."""
    radius = 1.0; k_ext = ka / radius; eta_ext = 1.0
    k_int = k_ext * m_rel; eta_int = eta_ext / m_rel

    verts, tris = icosphere(radius, refinements=refine)
    rwg = build_rwg(verts, tris)
    N = rwg['N']

    Z, _, _ = assemble_pmchwt(rwg, verts, tris, k_ext, k_int, eta_ext, eta_int)
    b = compute_rhs_planewave(rwg, verts, tris, k_ext, eta_ext)
    coeffs = np.linalg.solve(Z, b)
    J = coeffs[:N]; M = coeffs[N:]

    Q_ext, Q_sca = compute_cross_sections(rwg, verts, tris, J, M, k_ext, eta_ext, radius, sM=-1)
    return Q_ext, Q_sca, N


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    pdf_path = "/home/serg/bem_solver/convergence_vs_ka.pdf"

    # ---- ka sweep for dielectric sphere ----
    m_rel = 1.5
    ka_values = np.array([0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0])

    # Mie reference (dense for smooth curve)
    ka_dense = np.linspace(0.1, 3.5, 200)
    mie_ext_dense = np.zeros(len(ka_dense))
    mie_sca_dense = np.zeros(len(ka_dense))
    for i, ka in enumerate(ka_dense):
        mie_ext_dense[i], mie_sca_dense[i] = mie_Qext_Qsca(ka, m_rel)
    mie_abs_dense = mie_ext_dense - mie_sca_dense

    # Mie at test points
    mie_ext = np.zeros(len(ka_values))
    mie_sca = np.zeros(len(ka_values))
    for i, ka in enumerate(ka_values):
        mie_ext[i], mie_sca[i] = mie_Qext_Qsca(ka, m_rel)
    mie_abs = mie_ext - mie_sca

    # PEC Mie
    ka_pec_dense = np.linspace(0.1, 3.5, 200)
    mie_pec_dense = np.array([mie_pec_Qsca(ka) for ka in ka_pec_dense])
    mie_pec_pts = np.array([mie_pec_Qsca(ka) for ka in ka_values])

    # ---- Run BEM dielectric for multiple refinements ----
    bem_die = {}  # ref -> {Q_ext, Q_sca} arrays
    for ref in [2, 3]:
        print(f"\n=== Dielectric BEM refine={ref} ===")
        qe = np.zeros(len(ka_values))
        qs = np.zeros(len(ka_values))
        ns = np.zeros(len(ka_values), dtype=int)
        for i, ka in enumerate(ka_values):
            t0 = time.time()
            qe[i], qs[i], ns[i] = run_bem_dielectric(ka, m_rel, ref)
            dt = time.time() - t0
            print(f"  ka={ka:.2f}: Q_ext={qe[i]:.6f}, Q_sca={qs[i]:.6f}, N={ns[i]}, t={dt:.1f}s")
        bem_die[ref] = {'Q_ext': qe, 'Q_sca': qs, 'N': ns}

    # ---- Run BEM PEC for multiple refinements ----
    bem_pec = {}
    for ref in [2, 3]:
        print(f"\n=== PEC BEM refine={ref} ===")
        qe = np.zeros(len(ka_values))
        qs = np.zeros(len(ka_values))
        ns = np.zeros(len(ka_values), dtype=int)
        for i, ka in enumerate(ka_values):
            t0 = time.time()
            qe[i], qs[i], ns[i] = run_bem_pec(ka, ref)
            dt = time.time() - t0
            print(f"  ka={ka:.2f}: Q_ext={qe[i]:.6f}, Q_sca={qs[i]:.6f}, N={ns[i]}, t={dt:.1f}s")
        bem_pec[ref] = {'Q_ext': qe, 'Q_sca': qs, 'N': ns}

    # ---- Run ADDA ----
    print(f"\n=== ADDA (dielectric, m={m_rel}) ===")
    adda_die = {}
    for dpl in [10, 20]:
        qe = np.zeros(len(ka_values))
        qs = np.zeros(len(ka_values))
        nd = np.zeros(len(ka_values), dtype=int)
        for i, ka in enumerate(ka_values):
            qe[i], qs[i], nd[i] = run_adda(ka, m_rel, dpl=dpl)
            if qe[i] is None:
                qe[i] = np.nan; qs[i] = np.nan
            print(f"  dpl={dpl}, ka={ka:.2f}: Q_ext={qe[i]:.6f}, Q_sca={qs[i]:.6f}, ndip={nd[i]}")
        adda_die[dpl] = {'Q_ext': qe, 'Q_sca': qs, 'n_dip': nd}

    # ============================================================
    # Generate PDF
    # ============================================================
    print(f"\nGenerating {pdf_path}...")

    with PdfPages(pdf_path) as pdf:

        # ========== Page 1: Q_ext, Q_sca vs ka — Dielectric ==========
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Диэлектрическая сфера, m = {m_rel}: Q vs ka', fontsize=16, fontweight='bold')

        # Q_ext vs ka
        ax = axes[0, 0]
        ax.plot(ka_dense, mie_ext_dense, 'k-', linewidth=2, label='Mie (exact)', zorder=0)
        for ref, style in [(2, 'bo-'), (3, 'rs-')]:
            ax.plot(ka_values, bem_die[ref]['Q_ext'], style, markersize=7,
                    label=f'BEM ref={ref} (N={bem_die[ref]["N"][0]})', zorder=2)
        for dpl, style in [(10, 'g^--'), (20, 'mv--')]:
            mask = ~np.isnan(adda_die[dpl]['Q_ext'])
            ax.plot(ka_values[mask], adda_die[dpl]['Q_ext'][mask], style, markersize=7,
                    label=f'ADDA dpl={dpl}', zorder=1)
        ax.set_xlabel('ka'); ax.set_ylabel('$Q_{ext}$')
        ax.set_title('Extinction efficiency')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        # Q_sca vs ka
        ax = axes[0, 1]
        ax.plot(ka_dense, mie_sca_dense, 'k-', linewidth=2, label='Mie (exact)', zorder=0)
        for ref, style in [(2, 'bo-'), (3, 'rs-')]:
            ax.plot(ka_values, bem_die[ref]['Q_sca'], style, markersize=7,
                    label=f'BEM ref={ref}', zorder=2)
        for dpl, style in [(10, 'g^--'), (20, 'mv--')]:
            mask = ~np.isnan(adda_die[dpl]['Q_sca'])
            ax.plot(ka_values[mask], adda_die[dpl]['Q_sca'][mask], style, markersize=7,
                    label=f'ADDA dpl={dpl}', zorder=1)
        ax.set_xlabel('ka'); ax.set_ylabel('$Q_{sca}$')
        ax.set_title('Scattering efficiency')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        # Q_ext relative error vs ka
        ax = axes[1, 0]
        for ref, style in [(2, 'bo-'), (3, 'rs-')]:
            err = np.abs(bem_die[ref]['Q_ext'] - mie_ext) / np.maximum(np.abs(mie_ext), 1e-15) * 100
            ax.semilogy(ka_values, err, style, markersize=7, label=f'BEM ref={ref}')
        for dpl, style in [(10, 'g^--'), (20, 'mv--')]:
            err = np.abs(adda_die[dpl]['Q_ext'] - mie_ext) / np.maximum(np.abs(mie_ext), 1e-15) * 100
            mask = ~np.isnan(err)
            ax.semilogy(ka_values[mask], err[mask], style, markersize=7, label=f'ADDA dpl={dpl}')
        ax.set_xlabel('ka'); ax.set_ylabel('$Q_{ext}$ error (%)')
        ax.set_title('$Q_{ext}$ relative error')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        # Q_sca relative error vs ka
        ax = axes[1, 1]
        for ref, style in [(2, 'bo-'), (3, 'rs-')]:
            err = np.abs(bem_die[ref]['Q_sca'] - mie_sca) / np.maximum(np.abs(mie_sca), 1e-15) * 100
            ax.semilogy(ka_values, err, style, markersize=7, label=f'BEM ref={ref}')
        for dpl, style in [(10, 'g^--'), (20, 'mv--')]:
            err = np.abs(adda_die[dpl]['Q_sca'] - mie_sca) / np.maximum(np.abs(mie_sca), 1e-15) * 100
            mask = ~np.isnan(err)
            ax.semilogy(ka_values[mask], err[mask], style, markersize=7, label=f'ADDA dpl={dpl}')
        ax.set_xlabel('ka'); ax.set_ylabel('$Q_{sca}$ error (%)')
        ax.set_title('$Q_{sca}$ relative error')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # ========== Page 2: PEC sphere ==========
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('PEC сфера: $Q_{sca}$ vs ka', fontsize=16, fontweight='bold')

        # Q_sca vs ka
        ax = axes[0]
        ax.plot(ka_pec_dense, mie_pec_dense, 'k-', linewidth=2, label='Mie (exact)')
        for ref, style in [(2, 'bo-'), (3, 'rs-')]:
            ax.plot(ka_values, bem_pec[ref]['Q_sca'], style, markersize=7,
                    label=f'BEM ref={ref} (N={bem_pec[ref]["N"][0]})')
        ax.set_xlabel('ka'); ax.set_ylabel('$Q_{sca}$')
        ax.set_title('Scattering efficiency (PEC)')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        # Error vs ka
        ax = axes[1]
        for ref, style in [(2, 'bo-'), (3, 'rs-')]:
            err = np.abs(bem_pec[ref]['Q_sca'] - mie_pec_pts) / mie_pec_pts * 100
            ax.semilogy(ka_values, err, style, markersize=7, label=f'BEM ref={ref}')
        ax.set_xlabel('ka'); ax.set_ylabel('$Q_{sca}$ error (%)')
        ax.set_title('Relative error (PEC)')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # ========== Page 3: Optical theorem PEC ==========
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.suptitle('Оптическая теорема: $Q_{ext}/Q_{sca}$ vs ka (PEC)', fontsize=14, fontweight='bold')

        for ref, style in [(2, 'bo-'), (3, 'rs-')]:
            ratio = bem_pec[ref]['Q_ext'] / bem_pec[ref]['Q_sca']
            ax.plot(ka_values, ratio, style, markersize=8, label=f'BEM ref={ref}')
        ax.axhline(1.0, color='k', ls='--', alpha=0.5)
        ax.set_xlabel('ka'); ax.set_ylabel('$Q_{ext} / Q_{sca}$')
        ax.set_ylim(0.95, 1.05)
        ax.legend(); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # ========== Page 4: Mesh convergence at fixed ka ==========
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Сходимость по мешу при фиксированном ka', fontsize=16, fontweight='bold')

        # Pick ka=1 and ka=2 for mesh convergence
        for ax_idx, ka_test in enumerate([1.0, 2.0]):
            ax = axes[ax_idx]
            idx = np.argmin(np.abs(ka_values - ka_test))
            mie_ref_ext, mie_ref_sca = mie_Qext_Qsca(ka_test, m_rel)

            Ns = [bem_die[ref]['N'][idx] for ref in [2, 3]]
            errs_ext = [abs(bem_die[ref]['Q_ext'][idx] - mie_ref_ext)/abs(mie_ref_ext)*100 for ref in [2, 3]]
            errs_sca = [abs(bem_die[ref]['Q_sca'][idx] - mie_ref_sca)/abs(mie_ref_sca)*100 for ref in [2, 3]]

            ax.loglog(Ns, errs_ext, 'bo-', markersize=8, label='$Q_{ext}$ err')
            ax.loglog(Ns, errs_sca, 'rs-', markersize=8, label='$Q_{sca}$ err')

            # Add ADDA points
            for dpl, marker in [(10, '^'), (20, 'v')]:
                if not np.isnan(adda_die[dpl]['Q_ext'][idx]):
                    e_ext = abs(adda_die[dpl]['Q_ext'][idx] - mie_ref_ext)/abs(mie_ref_ext)*100
                    e_sca = abs(adda_die[dpl]['Q_sca'][idx] - mie_ref_sca)/abs(mie_ref_sca)*100
                    n_d = adda_die[dpl]['n_dip'][idx]
                    ax.loglog([n_d], [e_ext], f'g{marker}', markersize=10, label=f'ADDA dpl={dpl} $Q_{{ext}}$')
                    ax.loglog([n_d], [e_sca], f'm{marker}', markersize=10, label=f'ADDA dpl={dpl} $Q_{{sca}}$')

            ax.set_xlabel('DOFs (BEM: N_rwg, ADDA: N_dip)')
            ax.set_ylabel('Relative error (%)')
            ax.set_title(f'ka = {ka_test:.1f}, m = {m_rel}')
            ax.legend(fontsize=7); ax.grid(True, alpha=0.3, which='both')

        plt.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # ========== Page 5: Dielectric Q_ext, Q_sca, Q_abs on one plot ==========
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle(f'Диэлектрическая сфера m={m_rel}: полный спектр vs ka', fontsize=14, fontweight='bold')

        ax.plot(ka_dense, mie_ext_dense, 'b-', linewidth=2, label='$Q_{ext}$ (Mie)')
        ax.plot(ka_dense, mie_sca_dense, 'r-', linewidth=2, label='$Q_{sca}$ (Mie)')
        ax.plot(ka_dense, mie_abs_dense, 'g-', linewidth=2, label='$Q_{abs}$ (Mie)')

        ref = 3
        ax.plot(ka_values, bem_die[ref]['Q_ext'], 'bs', markersize=9, markerfacecolor='none',
                markeredgewidth=2, label=f'$Q_{{ext}}$ BEM ref={ref}')
        ax.plot(ka_values, bem_die[ref]['Q_sca'], 'ro', markersize=9, markerfacecolor='none',
                markeredgewidth=2, label=f'$Q_{{sca}}$ BEM ref={ref}')
        q_abs_bem = np.array(bem_die[ref]['Q_ext']) - np.array(bem_die[ref]['Q_sca'])
        ax.plot(ka_values, q_abs_bem, 'g^', markersize=9, markerfacecolor='none',
                markeredgewidth=2, label=f'$Q_{{abs}}$ BEM ref={ref}')

        dpl = 20
        mask = ~np.isnan(adda_die[dpl]['Q_ext'])
        ax.plot(ka_values[mask], adda_die[dpl]['Q_ext'][mask], 'bx', markersize=10,
                markeredgewidth=2, label=f'$Q_{{ext}}$ ADDA dpl={dpl}')
        ax.plot(ka_values[mask], adda_die[dpl]['Q_sca'][mask], 'r+', markersize=10,
                markeredgewidth=2, label=f'$Q_{{sca}}$ ADDA dpl={dpl}')

        ax.set_xlabel('ka', fontsize=12)
        ax.set_ylabel('Efficiency Q', fontsize=12)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # ========== Page 6: Summary table ==========
        fig, ax = plt.subplots(figsize=(14, 9))
        ax.axis('off')
        fig.suptitle(f'Сводная таблица: BEM vs ADDA vs Mie (m = {m_rel})', fontsize=14, fontweight='bold')

        headers = ['ka', 'Mie $Q_{ext}$', 'BEM r2 $Q_{ext}$', 'err%',
                   'BEM r3 $Q_{ext}$', 'err%', 'ADDA20 $Q_{ext}$', 'err%']
        table_data = []
        for i, ka in enumerate(ka_values):
            row = [f'{ka:.2f}', f'{mie_ext[i]:.6f}']
            for ref in [2, 3]:
                q = bem_die[ref]['Q_ext'][i]
                e = abs(q - mie_ext[i])/max(abs(mie_ext[i]), 1e-15)*100
                row.extend([f'{q:.6f}', f'{e:.1f}%'])
            q_a = adda_die[20]['Q_ext'][i]
            if np.isnan(q_a):
                row.extend(['—', '—'])
            else:
                e_a = abs(q_a - mie_ext[i])/max(abs(mie_ext[i]), 1e-15)*100
                row.extend([f'{q_a:.6f}', f'{e_a:.1f}%'])
            table_data.append(row)

        table = ax.table(cellText=table_data, colLabels=headers,
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        for j in range(len(headers)):
            table[0, j].set_facecolor('#4472C4')
            table[0, j].set_text_props(color='white', fontweight='bold')

        plt.tight_layout()
        pdf.savefig(fig); plt.close(fig)

    print(f"\nDone! Saved to {pdf_path}")
