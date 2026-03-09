"""
Benchmark: BEM vs ADDA wall time at comparable accuracy (~1-2%).
For each ka, pick mesh/dpl that gives ~1% error, measure total time.
"""
import numpy as np
import subprocess
import os
import time
import tempfile

from bem_core import (icosphere, build_rwg, tri_quadrature,
                      assemble_L_K, assemble_pmchwt,
                      compute_rhs_planewave, compute_cross_sections)


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


ADDA = "/home/serg/GO-for-ADDA/adda/src/seq/adda"

def run_adda(x, m_rel, dpl):
    with tempfile.TemporaryDirectory() as tmpdir:
        lam = 2 * np.pi; a_eq = x
        cmd = [ADDA, "-shape", "sphere", "-eq_rad", str(a_eq),
               "-m", str(m_rel), "0", "-lambda", str(lam),
               "-dpl", str(dpl), "-dir", tmpdir, "-no_vol_cor"]
        t0 = time.time()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        except Exception:
            return None, None, 999, 0
        t_adda = time.time() - t0
        if result.returncode != 0:
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


def run_bem(ka, m_rel, refine):
    radius = 1.0; k_ext = ka; eta_ext = 1.0
    k_int = k_ext * m_rel; eta_int = eta_ext / m_rel

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
    return Q_ext, Q_sca, t_total, N, len(tris)


if __name__ == "__main__":
    m_rel = 1.5

    # For each ka, determine what refinement/dpl gives ~1-2% accuracy
    # BEM: need ~10 elements per wavelength => refine such that edge_length < lambda/10
    #   lambda = 2*pi/k, edge ~ 2*pi*R / sqrt(N_tri), N_tri = 20*4^ref
    #   For ref=2 (320 tri): edge ~ 2*pi / sqrt(320) ~ 0.35 => ok for ka < ~2
    #   For ref=3 (1280 tri): edge ~ 2*pi / sqrt(1280) ~ 0.18 => ok for ka < ~4
    #   For ref=4 (5120 tri): edge ~ 2*pi / sqrt(5120) ~ 0.09 => ok for ka < ~8
    # ADDA: dpl=20 gives ~1% for moderate ka

    print("=" * 90)
    print(f"  BEM vs ADDA benchmark (dielectric sphere, m = {m_rel})")
    print(f"  Target: ~1-2% accuracy")
    print("=" * 90)

    # Test cases: (ka, bem_refine, adda_dpl)
    cases = [
        (0.5, 2, 20),
        (1.0, 2, 20),
        (1.5, 3, 20),
        (2.0, 3, 20),
        (2.5, 3, 20),
        (3.0, 3, 20),
        (4.0, 4, 20),
        (5.0, 4, 20),
        (6.0, 4, 25),
        (8.0, 4, 25),
    ]

    print(f"\n  {'ka':>4s}  {'Mie_Qext':>10s}  "
          f"{'BEM_Qext':>10s} {'err%':>6s} {'BEM_N':>7s} {'BEM_t':>7s}  "
          f"{'ADDA_Qext':>10s} {'err%':>6s} {'ADDA_N':>7s} {'ADDA_t':>7s}  "
          f"{'winner':>8s}")
    print(f"  {'-'*85}")

    for ka, ref, dpl in cases:
        Q_ext_mie, Q_sca_mie = mie_Qext_Qsca(ka, m_rel)

        # BEM
        print(f"  ka={ka:.1f}: running BEM ref={ref}...", end='', flush=True)
        try:
            qe_b, qs_b, t_b, N_b, nt_b = run_bem(ka, m_rel, ref)
            err_b = abs(qe_b - Q_ext_mie) / max(abs(Q_ext_mie), 1e-15) * 100
        except Exception as e:
            print(f" FAILED: {e}")
            qe_b = np.nan; err_b = 999; t_b = 999; N_b = 0
        print(f" ADDA dpl={dpl}...", end='', flush=True)

        # ADDA
        qe_a, qs_a, t_a, n_a = run_adda(ka, m_rel, dpl)
        if qe_a is None:
            qe_a = np.nan; err_a = 999; n_a = 0
        else:
            err_a = abs(qe_a - Q_ext_mie) / max(abs(Q_ext_mie), 1e-15) * 100

        winner = "BEM" if t_b < t_a else "ADDA"
        if np.isnan(qe_b): winner = "ADDA"
        if np.isnan(qe_a): winner = "BEM"

        print(f"\r  {ka:4.1f}  {Q_ext_mie:10.6f}  "
              f"{qe_b:10.6f} {err_b:5.1f}% {N_b:7d} {t_b:6.1f}s  "
              f"{qe_a:10.6f} {err_a:5.1f}% {n_a:7d} {t_a:6.1f}s  "
              f"{'<< '+winner:>8s}")

    print(f"\n{'='*90}")
    print("  Scaling:")
    print("    BEM:  N_dof ~ (ka)^2,  time ~ N^3 (LU) => time ~ (ka)^6")
    print("    ADDA: N_dip ~ (ka)^3,  time ~ N*log(N) (FFT) => time ~ (ka)^3 * log")
    print("    Crossover: BEM wins when N_surface << N_volume (large ka)")
    print("    But BEM+LU is O(N^3) — needs FMM/GMRES for large N")
    print(f"{'='*90}")
