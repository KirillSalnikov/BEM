"""Test adaptive orientation averaging with Halton/Sobol vs fixed grid."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import time
from scipy.linalg import lu_factor
from bem_core import (icosphere, build_rwg, assemble_pmchwt,
                      orientation_average_mueller_batched,
                      orientation_average_mueller_adaptive)

# Oblate ellipsoid
def make_ellipsoid(radius_eq, axis_ratios, refinements):
    verts, tris = icosphere(1.0, refinements=refinements)
    ry, rz = axis_ratios
    ax = radius_eq / (ry * rz) ** (1.0/3)
    verts = verts * np.array([ax, ry*ax, rz*ax])
    return verts, tris

ka = 2.0; m_rel = 1.5; ref = 3
axis_ratios = (1.0, 0.5)
radius_eq = ka; k_ext = 1.0; eta_ext = 1.0
k_int = k_ext * m_rel; eta_int = eta_ext / m_rel

verts, tris = make_ellipsoid(radius_eq, axis_ratios, ref)
rwg = build_rwg(verts, tris)
N = rwg['N']
print(f"Mesh: {len(tris)} tris, {N} RWG")

print("\nAssembling PMCHWT...")
Z, _, _ = assemble_pmchwt(rwg, verts, tris, k_ext, k_int, eta_ext, eta_int)
Z_lu = lu_factor(Z)

theta_arr = np.linspace(0.01, np.pi - 0.01, 181)

# 1. Fixed grid (reference)
print("\n" + "="*70)
print("Fixed grid: 8×5×8 = 320 orientations")
print("="*70)
t0 = time.time()
M_grid = orientation_average_mueller_batched(
    rwg, verts, tris, k_ext, eta_ext, theta_arr,
    Z_lu=Z_lu, sM=-1, n_alpha=8, n_beta=5, n_gamma=8)
t_grid = time.time() - t0
C_grid = 2*np.pi*np.trapezoid(M_grid[0,0]*np.sin(theta_arr), theta_arr)
Q_grid = C_grid / (np.pi * radius_eq**2)
print(f"  Q_sca = {Q_grid:.6f}, time = {t_grid:.1f}s")

# 2. Halton adaptive
for rtol in [0.05, 0.02, 0.01, 0.005]:
    print(f"\n{'='*70}")
    print(f"Halton adaptive, rtol={rtol}")
    print("="*70)
    t0 = time.time()
    result = orientation_average_mueller_adaptive(
        rwg, verts, tris, k_ext, eta_ext, theta_arr,
        Z_lu=Z_lu, sM=-1, sampling='halton',
        max_orient=2000, min_orient=30, batch_size=20,
        rtol=rtol, monitor_param='Q_sca')
    t_halt = time.time() - t0
    M_halt = result['M']
    C_halt = 2*np.pi*np.trapezoid(M_halt[0,0]*np.sin(theta_arr), theta_arr)
    Q_halt = C_halt / (np.pi * radius_eq**2)
    diff = abs(Q_halt - Q_grid) / abs(Q_grid) * 100
    print(f"  Q_sca = {Q_halt:.6f} (vs grid: {diff:.2f}%), "
          f"n={result['n_orient']}, converged={result['converged']}, "
          f"time={t_halt:.1f}s")

# 3. Sobol adaptive
print(f"\n{'='*70}")
print("Sobol adaptive, rtol=0.01")
print("="*70)
t0 = time.time()
result_sob = orientation_average_mueller_adaptive(
    rwg, verts, tris, k_ext, eta_ext, theta_arr,
    Z_lu=Z_lu, sM=-1, sampling='sobol',
    max_orient=2000, min_orient=30, batch_size=20,
    rtol=0.01, monitor_param='Q_sca')
t_sob = time.time() - t0
M_sob = result_sob['M']
C_sob = 2*np.pi*np.trapezoid(M_sob[0,0]*np.sin(theta_arr), theta_arr)
Q_sob = C_sob / (np.pi * radius_eq**2)
diff_sob = abs(Q_sob - Q_grid) / abs(Q_grid) * 100
print(f"  Q_sca = {Q_sob:.6f} (vs grid: {diff_sob:.2f}%), "
      f"n={result_sob['n_orient']}, converged={result_sob['converged']}, "
      f"time={t_sob:.1f}s")

# 4. Random adaptive
print(f"\n{'='*70}")
print("Random adaptive, rtol=0.01")
print("="*70)
t0 = time.time()
result_rnd = orientation_average_mueller_adaptive(
    rwg, verts, tris, k_ext, eta_ext, theta_arr,
    Z_lu=Z_lu, sM=-1, sampling='random',
    max_orient=2000, min_orient=30, batch_size=20,
    rtol=0.01, monitor_param='Q_sca')
t_rnd = time.time() - t0
M_rnd = result_rnd['M']
C_rnd = 2*np.pi*np.trapezoid(M_rnd[0,0]*np.sin(theta_arr), theta_arr)
Q_rnd = C_rnd / (np.pi * radius_eq**2)
diff_rnd = abs(Q_rnd - Q_grid) / abs(Q_grid) * 100
print(f"  Q_sca = {Q_rnd:.6f} (vs grid: {diff_rnd:.2f}%), "
      f"n={result_rnd['n_orient']}, converged={result_rnd['converged']}, "
      f"time={t_rnd:.1f}s")

# Summary
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"{'Method':<25} {'N_orient':>8} {'Q_sca':>10} {'diff%':>7} {'Time':>8} {'Conv':>5}")
print("-"*70)
print(f"{'Fixed grid 8×5×8':<25} {320:>8} {Q_grid:>10.6f} {'ref':>7} {t_grid:>7.1f}s")
for name, res, Q, t in [
    ('Halton rtol=0.01', result, Q_halt, t_halt),
    ('Sobol rtol=0.01', result_sob, Q_sob, t_sob),
    ('Random rtol=0.01', result_rnd, Q_rnd, t_rnd),
]:
    d = abs(Q - Q_grid)/abs(Q_grid)*100
    print(f"{name:<25} {res['n_orient']:>8} {Q:>10.6f} {d:>6.2f}% {t:>7.1f}s {str(res['converged']):>5}")
