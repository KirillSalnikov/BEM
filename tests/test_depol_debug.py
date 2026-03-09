"""Debug: compare all 16 Mueller matrix elements BEM vs ADDA vs Mie."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import subprocess, tempfile
from scipy.linalg import lu_factor
from scipy.special import spherical_jn, spherical_yn, lpmv
from bem_core import (icosphere, build_rwg, assemble_pmchwt,
                      compute_amplitude_matrix, amplitude_to_mueller)

ADDA = "/home/serg/GO-for-ADDA/adda/src/seq/adda"

# --- Mie full Mueller ---
def mie_full_mueller(x, m_rel, theta_arr, n_max=None):
    if n_max is None:
        n_max = int(x + 4*x**(1/3) + 2) + 10
    def psi(n, z): return z * spherical_jn(n, z)
    def psi_d(n, z): return spherical_jn(n, z) + z * spherical_jn(n, z, derivative=True)
    def xi(n, z): return z * (spherical_jn(n, z) + 1j * spherical_yn(n, z))
    def xi_d(n, z):
        return (spherical_jn(n, z) + 1j * spherical_yn(n, z)) + \
               z * (spherical_jn(n, z, derivative=True) + 1j * spherical_yn(n, z, derivative=True))
    mx = m_rel * x
    a_c = []; b_c = []
    for n in range(1, n_max + 1):
        a_n = (m_rel * psi(n, mx) * psi_d(n, x) - psi(n, x) * psi_d(n, mx)) / \
              (m_rel * psi(n, mx) * xi_d(n, x) - xi(n, x) * psi_d(n, mx))
        b_n = (psi(n, mx) * psi_d(n, x) - m_rel * psi(n, x) * psi_d(n, mx)) / \
              (psi(n, mx) * xi_d(n, x) - m_rel * xi(n, x) * psi_d(n, mx))
        a_c.append(a_n); b_c.append(b_n)
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
            S1[it] += coeff * (a_c[n-1]*pi_n + b_c[n-1]*tau_n)
            S2[it] += coeff * (a_c[n-1]*tau_n + b_c[n-1]*pi_n)
    # For sphere: S3=S4=0
    M = amplitude_to_mueller(S1, S2, np.zeros_like(S1), np.zeros_like(S1))
    return M / x**2, S1, S2

# --- ADDA ---
def run_adda(x, m_rel, dpl=20):
    with tempfile.TemporaryDirectory() as tmpdir:
        lam = 2 * np.pi; a_eq = x
        cmd = [ADDA, "-shape", "sphere", "-eq_rad", str(a_eq),
               "-m", str(m_rel), "0", "-lambda", str(lam),
               "-dpl", str(dpl), "-dir", tmpdir, "-no_vol_cor",
               "-scat_matr", "muel"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=tmpdir)
        if r.returncode != 0:
            print(f"ADDA error: {r.stderr[:300]}")
            return None, None
        muel_file = os.path.join(tmpdir, "mueller")
        data = np.loadtxt(muel_file, skiprows=1)
        theta_deg = data[:, 0]
        # cols: theta s11 s12 s13 s14 s21 s22 s23 s24 s31 s32 s33 s34 s41 s42 s43 s44
        M_adda = np.zeros((4, 4, len(theta_deg)))
        for i in range(4):
            for j in range(4):
                col = 1 + i*4 + j
                M_adda[i, j] = data[:, col]
        return M_adda, theta_deg

# =============================================================
ka = 2.0; m_rel = 1.5; ref = 3
radius = 1.0; k_ext = ka; eta_ext = 1.0
k_int = k_ext * m_rel; eta_int = eta_ext / m_rel

theta_arr = np.linspace(0.01, np.pi - 0.01, 37)
theta_deg = np.degrees(theta_arr)

# Mie
print("Computing Mie...")
M_mie, S1_mie, S2_mie = mie_full_mueller(ka, m_rel, theta_arr)

# BEM
print("Computing BEM...")
verts, tris = icosphere(radius, refinements=ref)
rwg = build_rwg(verts, tris)
N = rwg['N']
Z, _, _ = assemble_pmchwt(rwg, verts, tris, k_ext, k_int, eta_ext, eta_int)
Z_lu = lu_factor(Z)
S_bem = compute_amplitude_matrix(rwg, verts, tris, k_ext, eta_ext, theta_arr,
                                  Z_lu=Z_lu, sM=-1)
M_bem = amplitude_to_mueller(S_bem['S1'], S_bem['S2'], S_bem['S3'], S_bem['S4']) / k_ext**2

# ADDA
print("Computing ADDA...")
M_adda_raw, theta_adda = run_adda(ka, m_rel, dpl=20)

if M_adda_raw is not None:
    # Normalize ADDA: ADDA outputs s_ij, need to divide by k² for dσ/dΩ
    M_adda = M_adda_raw / k_ext**2
    # Interpolate ADDA to our theta grid
    from scipy.interpolate import interp1d
    M_adda_interp = np.zeros((4, 4, len(theta_arr)))
    for i in range(4):
        for j in range(4):
            f = interp1d(theta_adda, M_adda[i,j], kind='linear', fill_value='extrapolate')
            M_adda_interp[i,j] = f(theta_deg)

# Compare all elements at theta=90 deg
idx90 = len(theta_arr) // 2
print(f"\n{'='*80}")
print(f"All Mueller elements at θ = {theta_deg[idx90]:.1f}° (ka={ka}, m={m_rel})")
print(f"{'='*80}")
print(f"{'Elem':>6}  {'Mie':>12}  {'BEM':>12}  {'BEM err%':>9}", end="")
if M_adda_raw is not None:
    print(f"  {'ADDA':>12}  {'ADDA err%':>10}", end="")
print()
print("-" * 80)

for i in range(4):
    for j in range(4):
        mie_val = M_mie[i,j,idx90]
        bem_val = M_bem[i,j,idx90]
        ref_val = max(abs(mie_val), 1e-15)
        bem_err = abs(bem_val - mie_val) / ref_val * 100
        line = f"M{i+1}{j+1}  {mie_val:>12.6f}  {bem_val:>12.6f}  {bem_err:>8.2f}%"
        if M_adda_raw is not None:
            adda_val = M_adda_interp[i,j,idx90]
            adda_err = abs(adda_val - mie_val) / ref_val * 100
            line += f"  {adda_val:>12.6f}  {adda_err:>9.2f}%"
        print(line)

# Depolarization metrics
print(f"\n{'='*80}")
print(f"Depolarization ratios at θ = {theta_deg[idx90]:.1f}°")
print(f"{'='*80}")

# M22/M11 - linear depolarization
for label, M in [("Mie", M_mie), ("BEM", M_bem)] + ([("ADDA", M_adda_interp)] if M_adda_raw is not None else []):
    m11 = M[0,0,idx90]
    m22 = M[1,1,idx90]
    m12 = M[0,1,idx90]
    m33 = M[2,2,idx90]
    m34 = M[2,3,idx90]
    m44 = M[3,3,idx90]
    pol = -m12/m11 if abs(m11) > 1e-15 else 0
    depol_lin = 1 - m22/m11 if abs(m11) > 1e-15 else 0
    print(f"  {label:>5}: -M12/M11={pol:+.6f}  1-M22/M11={depol_lin:.6f}  "
          f"M33/M11={m33/m11:.6f}  M34/M11={m34/m11:.6f}  M44/M11={m44/m11:.6f}")

# Sphere checks: M11=M22, M33=M44, off-diag blocks = 0
print(f"\nSphere symmetry checks (should be 0 for sphere):")
for label, M in [("Mie", M_mie), ("BEM", M_bem)] + ([("ADDA", M_adda_interp)] if M_adda_raw is not None else []):
    d1 = np.max(np.abs(M[0,0] - M[1,1]))
    d2 = np.max(np.abs(M[2,2] - M[3,3]))
    d3 = np.max(np.abs(M[0,1] + M[1,0]))  # M12 = -M21 for sphere? No, M12=M21
    off = np.max([np.max(np.abs(M[0,2])), np.max(np.abs(M[0,3])),
                  np.max(np.abs(M[1,2])), np.max(np.abs(M[1,3]))])
    print(f"  {label:>5}: |M11-M22|={d1:.2e}  |M33-M44|={d2:.2e}  "
          f"max|off-diag|={off:.2e}")

# Full angle comparison for -M12/M11 and M22/M11
print(f"\n{'='*80}")
print(f"Angle scan: -M12/M11 and M22/M11")
print(f"{'='*80}")
print(f"{'θ':>6}  {'Mie -M12/M11':>13}  {'BEM':>13}  {'ADDA':>13}  |  {'Mie M22/M11':>13}  {'BEM':>13}  {'ADDA':>13}")
for k in range(0, len(theta_arr), 3):
    m11_mie = M_mie[0,0,k]; m12_mie = M_mie[0,1,k]; m22_mie = M_mie[1,1,k]
    m11_bem = M_bem[0,0,k]; m12_bem = M_bem[0,1,k]; m22_bem = M_bem[1,1,k]
    p_mie = -m12_mie/m11_mie; r22_mie = m22_mie/m11_mie
    p_bem = -m12_bem/m11_bem; r22_bem = m22_bem/m11_bem
    line = f"{theta_deg[k]:>5.1f}°  {p_mie:>13.6f}  {p_bem:>13.6f}"
    if M_adda_raw is not None:
        m11_a = M_adda_interp[0,0,k]; m12_a = M_adda_interp[0,1,k]; m22_a = M_adda_interp[1,1,k]
        p_a = -m12_a/m11_a; r22_a = m22_a/m11_a
        line += f"  {p_a:>13.6f}"
    else:
        line += f"  {'N/A':>13}"
    line += f"  |  {r22_mie:>13.6f}  {r22_bem:>13.6f}"
    if M_adda_raw is not None:
        line += f"  {r22_a:>13.6f}"
    else:
        line += f"  {'N/A':>13}"
    print(line)
