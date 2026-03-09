"""Compare orientation-averaged Mueller matrix: BEM vs ADDA for ellipsoids at different ka."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import subprocess, tempfile, shutil, time
from scipy.linalg import lu_factor
from bem_core import (icosphere, build_rwg, assemble_pmchwt,
                      compute_rhs_planewave, compute_cross_sections,
                      orientation_average_mueller_batched,
                      amplitude_to_mueller)

ADDA = "/home/serg/GO-for-ADDA/adda/src/seq/adda"
AVG_PARAMS = "/home/serg/GO-for-ADDA/adda/input/avg_params.dat"


def make_ellipsoid(radius_eq, axis_ratios, refinements):
    """Create ellipsoid mesh by scaling icosphere.
    axis_ratios = (ay/ax, az/ax), ax is set so volume = (4/3)pi*r_eq^3.
    """
    verts, tris = icosphere(1.0, refinements=refinements)
    # Scale: ax * ay * az = r_eq^3, ay = ry*ax, az = rz*ax
    ry, rz = axis_ratios
    ax = radius_eq / (ry * rz) ** (1.0/3)
    ay = ry * ax
    az = rz * ax
    verts = verts * np.array([ax, ay, az])
    return verts, tris, (ax, ay, az)


def run_adda_orient_avg(ka, m_rel, axis_ratios, dpl=15):
    """Run ADDA with orientation averaging for an ellipsoid."""
    tmpdir = tempfile.mkdtemp()
    try:
        # Copy avg_params.dat
        shutil.copy(AVG_PARAMS, os.path.join(tmpdir, "avg_params.dat"))

        lam = 2 * np.pi
        eq_rad = ka  # ka = k*a_eq, k = 2pi/lam = 1, so a_eq = ka
        ry, rz = axis_ratios

        cmd = [ADDA,
               "-shape", "ellipsoid", str(ry), str(rz),
               "-eq_rad", str(eq_rad),
               "-m", str(m_rel), "0",
               "-lambda", str(lam),
               "-dpl", str(dpl),
               "-orient", "avg",
               "-scat_matr", "muel",
               "-dir", tmpdir,
               "-no_vol_cor"]

        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1200, cwd=tmpdir)
        t_adda = time.time() - t0

        if r.returncode != 0:
            print(f"  ADDA error: {r.stderr[:200]}")
            return None, t_adda

        # Parse cross sections from stdout
        Cext = Csca = Qext = Qsca = None
        for line in r.stdout.split('\n'):
            if line.startswith('Cext'):
                Cext = float(line.split('=')[1].strip())
            elif line.startswith('Qext'):
                Qext = float(line.split('=')[1].strip())
            elif line.startswith('Csca'):
                Csca = float(line.split('=')[1].strip())
            elif line.startswith('Qsca'):
                Qsca = float(line.split('=')[1].strip())

        # Parse Mueller matrix
        muel_file = os.path.join(tmpdir, "mueller")
        if not os.path.exists(muel_file):
            print("  No mueller file")
            return None, t_adda

        data = np.loadtxt(muel_file, skiprows=1)
        theta_deg = data[:, 0]
        M_adda = np.zeros((4, 4, len(theta_deg)))
        for i in range(4):
            for j in range(4):
                M_adda[i, j] = data[:, 1 + i*4 + j]

        return {'M': M_adda, 'theta_deg': theta_deg,
                'Cext': Cext, 'Csca': Csca, 'Qext': Qext, 'Qsca': Qsca,
                'time': t_adda}, t_adda
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def run_bem_orient_avg(ka, m_rel, axis_ratios, ref, n_alpha=8, n_beta=5, n_gamma=8):
    """Run BEM with orientation averaging for an ellipsoid."""
    # k = 2pi/lam = 1 when lambda=2pi; radius_eq = ka so that k*a = ka
    radius_eq = ka
    k_ext_val = 1.0
    eta_ext = 1.0
    k_int = k_ext_val * m_rel
    eta_int = eta_ext / m_rel

    verts, tris, (ax, ay, az) = make_ellipsoid(radius_eq, axis_ratios, ref)
    rwg = build_rwg(verts, tris)
    N = rwg['N']

    print(f"  BEM: {len(tris)} tris, {N} RWG, ellipsoid ({ax:.2f}, {ay:.2f}, {az:.2f})")

    t0 = time.time()
    Z, L_ext, K_ext = assemble_pmchwt(rwg, verts, tris, k_ext_val, k_int, eta_ext, eta_int)
    t_assemble = time.time() - t0
    print(f"  Assembly: {t_assemble:.1f}s")

    Z_lu = lu_factor(Z)

    theta_arr = np.linspace(0.01, np.pi - 0.01, 181)

    t0 = time.time()
    M_avg = orientation_average_mueller_batched(
        rwg, verts, tris, k_ext_val, eta_ext, theta_arr,
        Z_lu=Z_lu, sM=-1,
        n_alpha=n_alpha, n_beta=n_beta, n_gamma=n_gamma)
    t_orient = time.time() - t0
    n_or = n_alpha * n_beta * n_gamma
    print(f"  Orientation avg ({n_or} orientations): {t_orient:.1f}s")

    t_total = t_assemble + t_orient

    # Cross sections from orientation-averaged M11
    # C_sca = 2pi * int M11 sin(theta) dtheta (axial symmetry after averaging)
    C_sca = 2 * np.pi * np.trapezoid(M_avg[0, 0] * np.sin(theta_arr), theta_arr)
    geom_cs = np.pi * radius_eq**2  # equivalent sphere cross section
    Q_sca = C_sca / geom_cs

    return {'M': M_avg, 'theta_deg': np.degrees(theta_arr),
            'C_sca': C_sca, 'Q_sca': Q_sca,
            'time': t_total, 'N': N, 'n_orient': n_or}


# ================================================================
# Main comparison
# ================================================================
if __name__ == "__main__":
    m_rel = 1.5
    axis_ratios = (1.0, 0.5)  # oblate ellipsoid, az = 0.5*ax

    cases = [
        {'ka': 0.5, 'bem_ref': 2, 'adda_dpl': 15, 'n_alpha': 8, 'n_beta': 5, 'n_gamma': 8},
        {'ka': 1.0, 'bem_ref': 2, 'adda_dpl': 15, 'n_alpha': 8, 'n_beta': 5, 'n_gamma': 8},
        {'ka': 2.0, 'bem_ref': 3, 'adda_dpl': 15, 'n_alpha': 8, 'n_beta': 5, 'n_gamma': 8},
        {'ka': 3.0, 'bem_ref': 3, 'adda_dpl': 20, 'n_alpha': 10, 'n_beta': 6, 'n_gamma': 10},
    ]

    print("=" * 90)
    print(f"Orientation-averaged Mueller: BEM vs ADDA")
    print(f"Oblate ellipsoid ay/ax={axis_ratios[0]}, az/ax={axis_ratios[1]}, m={m_rel}")
    print("=" * 90)

    results = []

    for c in cases:
        ka = c['ka']
        print(f"\n{'─'*90}")
        print(f"  ka = {ka}")
        print(f"{'─'*90}")

        # BEM
        bem = run_bem_orient_avg(ka, m_rel, axis_ratios, c['bem_ref'],
                                  c['n_alpha'], c['n_beta'], c['n_gamma'])

        # ADDA
        print(f"  ADDA (dpl={c['adda_dpl']})...")
        adda, t_adda = run_adda_orient_avg(ka, m_rel, axis_ratios, c['adda_dpl'])
        if adda:
            print(f"  ADDA done: {adda['time']:.1f}s")

        results.append({'ka': ka, 'bem': bem, 'adda': adda})

    # ================================================================
    # Summary table
    # ================================================================
    print(f"\n\n{'='*90}")
    print("SUMMARY: Cross sections (orientation-averaged)")
    print(f"{'='*90}")
    print(f"{'ka':>4}  {'N_RWG':>6}  {'N_or':>5}  "
          f"{'Q_sca BEM':>10}  {'Q_ext ADDA':>10}  {'Q_sca ADDA':>10}  "
          f"{'diff%':>6}  {'t_BEM':>7}  {'t_ADDA':>7}")
    print("-" * 90)

    for r in results:
        ka = r['ka']
        bem = r['bem']
        adda = r['adda']
        q_bem = bem['Q_sca']
        q_adda = adda['Qext'] if adda and adda['Qext'] else 0
        q_sca_adda = adda['Qsca'] if adda and adda.get('Qsca') else q_adda
        diff = abs(q_bem - q_adda) / max(abs(q_adda), 1e-15) * 100 if q_adda else 0
        t_bem = bem['time']
        t_adda = adda['time'] if adda else 0
        print(f"{ka:>4.1f}  {bem['N']:>6}  {bem['n_orient']:>5}  "
              f"{q_bem:>10.6f}  {q_adda:>10.6f}  {q_sca_adda:>10.6f}  "
              f"{diff:>5.1f}%  {t_bem:>6.1f}s  {t_adda:>6.1f}s")

    # ================================================================
    # Mueller element comparison at key angles
    # ================================================================
    print(f"\n\n{'='*90}")
    print("Mueller elements: BEM vs ADDA at selected angles")
    print(f"{'='*90}")

    for r in results:
        ka = r['ka']
        bem = r['bem']
        adda = r['adda']
        if adda is None:
            continue

        # Normalize ADDA by 1/k²
        k_ext_val = 1.0
        M_adda = adda['M'] / k_ext_val**2
        M_bem = bem['M']

        # Interpolate ADDA to BEM angles
        from scipy.interpolate import interp1d
        M_adda_interp = np.zeros((4, 4, len(bem['theta_deg'])))
        for i in range(4):
            for j in range(4):
                f = interp1d(adda['theta_deg'], M_adda[i, j],
                             kind='linear', fill_value='extrapolate')
                M_adda_interp[i, j] = f(bem['theta_deg'])

        print(f"\n  ka={ka}:")
        print(f"  {'θ':>6}  {'M11 BEM':>10}  {'M11 ADDA':>10}  {'err%':>6}  "
              f"{'M12/M11 B':>10}  {'M12/M11 A':>10}  "
              f"{'M33/M11 B':>10}  {'M33/M11 A':>10}  "
              f"{'M34/M11 B':>10}  {'M34/M11 A':>10}")

        for deg in [0, 30, 60, 90, 120, 150, 180]:
            idx = np.argmin(np.abs(bem['theta_deg'] - deg))
            m11b = M_bem[0,0,idx]; m11a = M_adda_interp[0,0,idx]
            err = abs(m11b - m11a) / max(abs(m11a), 1e-15) * 100

            def ratio(M, i, j, idx):
                return M[i,j,idx] / M[0,0,idx] if abs(M[0,0,idx]) > 1e-15 else 0

            print(f"  {deg:>5}°  {m11b:>10.6f}  {m11a:>10.6f}  {err:>5.1f}%  "
                  f"{ratio(M_bem,0,1,idx):>+10.6f}  {ratio(M_adda_interp,0,1,idx):>+10.6f}  "
                  f"{ratio(M_bem,2,2,idx):>+10.6f}  {ratio(M_adda_interp,2,2,idx):>+10.6f}  "
                  f"{ratio(M_bem,2,3,idx):>+10.6f}  {ratio(M_adda_interp,2,3,idx):>+10.6f}")

    # ================================================================
    # Depolarization check: M22/M11 (should be <1 for non-spherical after averaging)
    # ================================================================
    print(f"\n\n{'='*90}")
    print("Depolarization: M22/M11 (=1 for sphere, <1 for averaged non-sphere)")
    print(f"{'='*90}")

    for r in results:
        ka = r['ka']
        bem = r['bem']
        adda = r['adda']
        if adda is None:
            continue

        M_adda_n = adda['M'] / k_ext_val**2
        from scipy.interpolate import interp1d
        M_adda_i = np.zeros((4, 4, len(bem['theta_deg'])))
        for i in range(4):
            for j in range(4):
                f = interp1d(adda['theta_deg'], M_adda_n[i, j],
                             kind='linear', fill_value='extrapolate')
                M_adda_i[i, j] = f(bem['theta_deg'])

        M_bem = bem['M']

        print(f"\n  ka={ka}:")
        print(f"  {'θ':>6}  {'M22/M11 BEM':>12}  {'M22/M11 ADDA':>13}  "
              f"{'1-M22/M11 B':>12}  {'1-M22/M11 A':>12}")
        for deg in [0, 30, 60, 90, 120, 150, 180]:
            idx = np.argmin(np.abs(bem['theta_deg'] - deg))
            r22b = M_bem[1,1,idx] / M_bem[0,0,idx] if abs(M_bem[0,0,idx]) > 1e-15 else 0
            r22a = M_adda_i[1,1,idx] / M_adda_i[0,0,idx] if abs(M_adda_i[0,0,idx]) > 1e-15 else 0
            print(f"  {deg:>5}°  {r22b:>12.6f}  {r22a:>13.6f}  "
                  f"{1-r22b:>12.6f}  {1-r22a:>12.6f}")
