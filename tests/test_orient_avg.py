"""Test orientation averaging on sphere (result should equal single-orientation)."""
import numpy as np
from scipy.linalg import lu_factor
from bem_core import (icosphere, build_rwg, assemble_pmchwt,
                      compute_mueller_matrix, orientation_average_mueller)

radius = 1.0; m_rel = 1.5
k_ext = 1.0; eta_ext = 1.0
k_int = k_ext * m_rel; eta_int = eta_ext / m_rel

verts, tris = icosphere(radius, refinements=1)  # coarse for speed
rwg = build_rwg(verts, tris)
N = rwg['N']
print(f"Mesh: {len(tris)} tris, {N} RWG")

Z, _, _ = assemble_pmchwt(rwg, verts, tris, k_ext, k_int, eta_ext, eta_int)
Z_lu = lu_factor(Z)

theta_arr = np.linspace(0.01, np.pi - 0.01, 19)

# Single orientation
print("\nSingle orientation Mueller matrix...")
M_single = compute_mueller_matrix(rwg, verts, tris, k_ext, eta_ext, theta_arr,
                                   Z_lu=Z_lu, sM=-1)

# Orientation-averaged (small n for speed)
print("\nOrientation-averaged Mueller matrix (4×4×2 = 32 orientations)...")
M_avg = orientation_average_mueller(rwg, verts, tris, k_ext, eta_ext, theta_arr,
                                     Z_lu=Z_lu, sM=-1, n_alpha=4, n_beta=4, n_gamma=2)

# For a sphere, M_avg should equal M_single
print(f"\nM11 single vs averaged at θ=0°, 90°, 180°:")
for idx, deg in [(0, 0), (9, 90), (-1, 180)]:
    print(f"  θ={deg:3d}°: single={M_single[0,0,idx]:.6f}, avg={M_avg[0,0,idx]:.6f}, "
          f"ratio={M_avg[0,0,idx]/max(M_single[0,0,idx],1e-30):.4f}")

# Relative difference
diff = np.abs(M_avg[0,0] - M_single[0,0]) / np.maximum(np.abs(M_single[0,0]), 1e-30)
print(f"\nmax|M11_avg - M11_single| / M11_single = {np.max(diff):.2e}")
print(f"(Should be small — sphere is rotationally invariant)")

# Check that off-diagonal blocks remain ~0
off_diag_avg = np.max([np.max(np.abs(M_avg[0,2])), np.max(np.abs(M_avg[0,3])),
                        np.max(np.abs(M_avg[2,0])), np.max(np.abs(M_avg[3,0]))])
print(f"Off-diagonal (should be ~0 for sphere): max = {off_diag_avg:.2e}")
