"""Generate comparison_results.pdf: BEM vs ADDA orientation-averaged Mueller matrix."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import subprocess, tempfile, shutil, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.linalg import lu_factor
from scipy.interpolate import interp1d
from bem_core import (icosphere, build_rwg, assemble_pmchwt,
                      orientation_average_mueller_batched,
                      amplitude_to_mueller)

ADDA = "/home/serg/GO-for-ADDA/adda/src/seq/adda"
AVG_PARAMS = "/home/serg/GO-for-ADDA/adda/input/avg_params.dat"


def make_ellipsoid(radius_eq, axis_ratios, refinements):
    verts, tris = icosphere(1.0, refinements=refinements)
    ry, rz = axis_ratios
    ax = radius_eq / (ry * rz) ** (1.0/3)
    ay = ry * ax; az = rz * ax
    verts = verts * np.array([ax, ay, az])
    return verts, tris, (ax, ay, az)


def run_adda_orient_avg(ka, m_rel, axis_ratios, dpl=15):
    tmpdir = tempfile.mkdtemp()
    try:
        shutil.copy(AVG_PARAMS, os.path.join(tmpdir, "avg_params.dat"))
        lam = 2 * np.pi; ry, rz = axis_ratios
        cmd = [ADDA, "-shape", "ellipsoid", str(ry), str(rz),
               "-eq_rad", str(ka), "-m", str(m_rel), "0",
               "-lambda", str(lam), "-dpl", str(dpl),
               "-orient", "avg", "-scat_matr", "muel",
               "-dir", tmpdir, "-no_vol_cor"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1200, cwd=tmpdir)
        if r.returncode != 0:
            print(f"  ADDA error: {r.stderr[:200]}")
            return None
        Qext = Qsca = None
        for line in r.stdout.split('\n'):
            if line.startswith('Qext'): Qext = float(line.split('=')[1].strip())
            elif line.startswith('Qsca'): Qsca = float(line.split('=')[1].strip())
        muel_file = os.path.join(tmpdir, "mueller")
        if not os.path.exists(muel_file): return None
        data = np.loadtxt(muel_file, skiprows=1)
        theta_deg = data[:, 0]
        M = np.zeros((4, 4, len(theta_deg)))
        for i in range(4):
            for j in range(4):
                M[i, j] = data[:, 1 + i*4 + j]
        return {'M': M, 'theta_deg': theta_deg, 'Qext': Qext, 'Qsca': Qsca}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def run_bem_orient_avg(ka, m_rel, axis_ratios, ref, n_alpha=8, n_beta=5, n_gamma=8):
    radius_eq = ka; k_ext = 1.0; eta_ext = 1.0
    k_int = k_ext * m_rel; eta_int = eta_ext / m_rel
    verts, tris, axes = make_ellipsoid(radius_eq, axis_ratios, ref)
    rwg = build_rwg(verts, tris)
    N = rwg['N']
    print(f"    BEM: {len(tris)} tris, {N} RWG")
    Z, _, _ = assemble_pmchwt(rwg, verts, tris, k_ext, k_int, eta_ext, eta_int)
    Z_lu = lu_factor(Z)
    theta_arr = np.linspace(0.01, np.pi - 0.01, 181)
    M_avg = orientation_average_mueller_batched(
        rwg, verts, tris, k_ext, eta_ext, theta_arr,
        Z_lu=Z_lu, sM=-1, n_alpha=n_alpha, n_beta=n_beta, n_gamma=n_gamma)
    C_sca = 2 * np.pi * np.trapezoid(M_avg[0, 0] * np.sin(theta_arr), theta_arr)
    Q_sca = C_sca / (np.pi * radius_eq**2)
    return {'M': M_avg, 'theta_deg': np.degrees(theta_arr), 'Q_sca': Q_sca, 'N': N}


if __name__ == "__main__":
    pdf_path = "/home/serg/bem_solver/pdf/comparison_results.pdf"
    m_rel = 1.5
    axis_ratios = (1.0, 0.5)

    cases = [
        {'ka': 1.0, 'ref': 2, 'dpl': 15, 'na': 8, 'nb': 5, 'ng': 8},
        {'ka': 2.0, 'ref': 3, 'dpl': 15, 'na': 8, 'nb': 5, 'ng': 8},
        {'ka': 3.0, 'ref': 3, 'dpl': 20, 'na': 10, 'nb': 6, 'ng': 10},
    ]

    results = []
    for c in cases:
        ka = c['ka']
        print(f"\n=== ka={ka} ===")
        print("  Running BEM...")
        bem = run_bem_orient_avg(ka, m_rel, axis_ratios, c['ref'],
                                  c['na'], c['nb'], c['ng'])
        print("  Running ADDA...")
        adda = run_adda_orient_avg(ka, m_rel, axis_ratios, c['dpl'])
        results.append({'ka': ka, 'bem': bem, 'adda': adda})

    # ================================================================
    # Generate PDF
    # ================================================================
    with PdfPages(pdf_path) as pdf:
        # --- Page 1: Title + Q_sca table ---
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.axis('off')
        title = (f"Orientation-averaged Mueller matrix: BEM vs ADDA\n"
                 f"Oblate ellipsoid az/ax = 0.5, m = {m_rel}")
        ax.text(0.5, 0.95, title, transform=ax.transAxes,
                fontsize=16, fontweight='bold', ha='center', va='top')

        table_data = [['ka', 'N_RWG', 'N_orient', 'Q_sca BEM', 'Q_ext ADDA', 'Diff %']]
        for r in results:
            q_bem = r['bem']['Q_sca']
            q_adda = r['adda']['Qext'] if r['adda'] else 0
            diff = abs(q_bem - q_adda) / max(abs(q_adda), 1e-15) * 100
            table_data.append([
                f"{r['ka']:.1f}",
                str(r['bem']['N']),
                str(320 if r['ka'] <= 2 else 600),
                f"{q_bem:.4f}",
                f"{q_adda:.4f}" if q_adda else "N/A",
                f"{diff:.1f}%"
            ])
        table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.8)
        plt.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # --- Pages 2+: Mueller elements for each ka ---
        for r in results:
            ka = r['ka']
            bem = r['bem']; adda = r['adda']
            if adda is None:
                continue

            # Normalize ADDA
            k_ext = 1.0
            M_adda = adda['M'] / k_ext**2
            M_bem = bem['M']
            theta_bem = bem['theta_deg']

            # Interpolate ADDA to BEM angles
            M_adda_i = np.zeros((4, 4, len(theta_bem)))
            for i in range(4):
                for j in range(4):
                    f = interp1d(adda['theta_deg'], M_adda[i, j],
                                 kind='linear', fill_value='extrapolate')
                    M_adda_i[i, j] = f(theta_bem)

            # --- 4-panel Mueller elements ---
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f'Orientation-averaged Mueller: ka={ka}, m={m_rel}, '
                         f'oblate ellipsoid (az/ax=0.5)',
                         fontsize=14, fontweight='bold')

            # M11 (log)
            ax = axes[0, 0]
            ax.semilogy(theta_bem, M_bem[0, 0], 'b-', lw=2, label=f'BEM (N={bem["N"]})')
            ax.semilogy(adda['theta_deg'], M_adda[0, 0], 'r--', lw=2, label='ADDA')
            ax.set_xlabel('θ (°)'); ax.set_ylabel('$\\langle M_{11} \\rangle$')
            ax.set_title('$M_{11}$ (дифф. сечение)')
            ax.legend(); ax.grid(True, alpha=0.3)

            # -M12/M11
            ax = axes[0, 1]
            p_bem = -M_bem[0, 1] / np.maximum(M_bem[0, 0], 1e-30)
            p_adda = -M_adda_i[0, 1] / np.maximum(M_adda_i[0, 0], 1e-30)
            ax.plot(theta_bem, p_bem, 'b-', lw=2, label='BEM')
            ax.plot(theta_bem, p_adda, 'r--', lw=2, label='ADDA')
            ax.set_xlabel('θ (°)'); ax.set_ylabel('$-M_{12}/M_{11}$')
            ax.set_title('Степень линейной поляризации')
            ax.set_ylim(-1.1, 1.1)
            ax.legend(); ax.grid(True, alpha=0.3)

            # M22/M11 (depolarization)
            ax = axes[1, 0]
            d_bem = M_bem[1, 1] / np.maximum(M_bem[0, 0], 1e-30)
            d_adda = M_adda_i[1, 1] / np.maximum(M_adda_i[0, 0], 1e-30)
            ax.plot(theta_bem, d_bem, 'b-', lw=2, label='BEM')
            ax.plot(theta_bem, d_adda, 'r--', lw=2, label='ADDA')
            ax.axhline(1.0, color='k', ls=':', alpha=0.3)
            ax.set_xlabel('θ (°)'); ax.set_ylabel('$M_{22}/M_{11}$')
            ax.set_title('$M_{22}/M_{11}$ (=1 для сферы, <1 деполяризация)')
            ax.set_ylim(0.5, 1.05)
            ax.legend(); ax.grid(True, alpha=0.3)

            # M33/M11
            ax = axes[1, 1]
            r33_bem = M_bem[2, 2] / np.maximum(M_bem[0, 0], 1e-30)
            r33_adda = M_adda_i[2, 2] / np.maximum(M_adda_i[0, 0], 1e-30)
            ax.plot(theta_bem, r33_bem, 'b-', lw=2, label='BEM')
            ax.plot(theta_bem, r33_adda, 'r--', lw=2, label='ADDA')
            ax.set_xlabel('θ (°)'); ax.set_ylabel('$M_{33}/M_{11}$')
            ax.set_title('$M_{33}/M_{11}$')
            ax.set_ylim(-1.1, 1.1)
            ax.legend(); ax.grid(True, alpha=0.3)

            plt.tight_layout()
            pdf.savefig(fig); plt.close(fig)

            # --- Page: M34/M11 + M11 ratio + depolarization 1-M22/M11 ---
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            fig.suptitle(f'ka={ka}: дополнительные элементы', fontsize=14)

            # M34/M11
            ax = axes[0]
            r34_bem = M_bem[2, 3] / np.maximum(M_bem[0, 0], 1e-30)
            r34_adda = M_adda_i[2, 3] / np.maximum(M_adda_i[0, 0], 1e-30)
            ax.plot(theta_bem, r34_bem, 'b-', lw=2, label='BEM')
            ax.plot(theta_bem, r34_adda, 'r--', lw=2, label='ADDA')
            ax.set_xlabel('θ (°)'); ax.set_ylabel('$M_{34}/M_{11}$')
            ax.set_title('$M_{34}/M_{11}$')
            ax.set_ylim(-1.1, 1.1)
            ax.legend(); ax.grid(True, alpha=0.3)

            # M11 ratio BEM/ADDA
            ax = axes[1]
            ratio = M_bem[0, 0] / np.maximum(M_adda_i[0, 0], 1e-30)
            ax.plot(theta_bem, ratio, 'b-', lw=2)
            ax.axhline(1.0, color='k', ls='--', alpha=0.5)
            ax.set_xlabel('θ (°)'); ax.set_ylabel('BEM / ADDA')
            ax.set_title('$M_{11}$ ratio')
            ax.set_ylim(0.7, 1.3)
            ax.grid(True, alpha=0.3)

            # 1 - M22/M11 (depolarization)
            ax = axes[2]
            dep_bem = 1.0 - d_bem
            dep_adda = 1.0 - d_adda
            ax.plot(theta_bem, dep_bem, 'b-', lw=2, label='BEM')
            ax.plot(theta_bem, dep_adda, 'r--', lw=2, label='ADDA')
            ax.set_xlabel('θ (°)'); ax.set_ylabel('$1 - M_{22}/M_{11}$')
            ax.set_title('Деполяризация')
            ax.legend(); ax.grid(True, alpha=0.3)

            plt.tight_layout()
            pdf.savefig(fig); plt.close(fig)

    print(f"\nDone! Saved to {pdf_path}")
