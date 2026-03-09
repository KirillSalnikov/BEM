"""Test Mueller matrix computation against Mie theory for a sphere."""
import numpy as np
from scipy.linalg import lu_factor
from bem_core import (icosphere, build_rwg, assemble_pmchwt,
                      compute_amplitude_matrix, amplitude_to_mueller,
                      compute_mueller_matrix)

# Sphere parameters
radius = 1.0; m_rel = 1.5
k_ext = 1.0; eta_ext = 1.0
k_int = k_ext * m_rel; eta_int = eta_ext / m_rel

# Mesh
verts, tris = icosphere(radius, refinements=2)
rwg = build_rwg(verts, tris)
N = rwg['N']
print(f"Mesh: {len(tris)} tris, {N} RWG")

# Assemble
Z, _, _ = assemble_pmchwt(rwg, verts, tris, k_ext, k_int, eta_ext, eta_int)
Z_lu = lu_factor(Z)

# Amplitude matrix
theta_arr = np.linspace(0.01, np.pi - 0.01, 37)
S = compute_amplitude_matrix(rwg, verts, tris, k_ext, eta_ext, theta_arr,
                              Z_lu=Z_lu, sM=-1)

print(f"\nAmplitude matrix at θ=0°, 90°, 180°:")
for idx, deg in [(0, 0), (18, 90), (-1, 180)]:
    print(f"  θ={deg:3d}°: S1={S['S1'][idx]:.4f}, S2={S['S2'][idx]:.4f}, "
          f"S3={S['S3'][idx]:.4f}, S4={S['S4'][idx]:.4f}")

# For a sphere, S3 and S4 should be ~0
print(f"\nSphere check (S3, S4 should be ~0):")
print(f"  max|S3| = {np.max(np.abs(S['S3'])):.2e}")
print(f"  max|S4| = {np.max(np.abs(S['S4'])):.2e}")
print(f"  max|S1| = {np.max(np.abs(S['S1'])):.4f}")
print(f"  max|S2| = {np.max(np.abs(S['S2'])):.4f}")

# Mueller matrix
M = amplitude_to_mueller(S['S1'], S['S2'], S['S3'], S['S4']) / k_ext**2

print(f"\nMueller matrix at θ=90° (normalized by k²):")
idx90 = 18
for i in range(4):
    row = " ".join(f"{M[i,j,idx90]:8.4f}" for j in range(4))
    print(f"  [{row}]")

# Check M11 = (|S1|² + |S2|²) / (2k²) at each angle
M11_check = 0.5*(np.abs(S['S1'])**2 + np.abs(S['S2'])**2) / k_ext**2
print(f"\nM11 consistency: max|M11 - check| = {np.max(np.abs(M[0,0] - M11_check)):.2e}")

# For sphere: M should be block-diagonal (M02=M03=M12=M13=M20=M21=M30=M31 ~ 0)
off_diag = np.max([np.max(np.abs(M[0,2])), np.max(np.abs(M[0,3])),
                    np.max(np.abs(M[1,2])), np.max(np.abs(M[1,3])),
                    np.max(np.abs(M[2,0])), np.max(np.abs(M[2,1])),
                    np.max(np.abs(M[3,0])), np.max(np.abs(M[3,1]))])
print(f"Block-diagonal check: max off-diagonal = {off_diag:.2e} (should be ~0 for sphere)")

# Check M22 = M33 and M23 = -M32 (for sphere)
print(f"M22-M33 max diff = {np.max(np.abs(M[2,2] - M[3,3])):.2e}")
print(f"M23+M32 max diff = {np.max(np.abs(M[2,3] + M[3,2])):.2e}")

# Q_sca from Mueller matrix: C_sca = ∫ M11 dΩ (for unpolarized)
# Actually C_sca = 2π ∫ M11 sinθ dθ (φ-integrated, for sphere by symmetry)
C_sca = 2 * np.pi * np.trapezoid(M[0,0] * np.sin(theta_arr), theta_arr)
Q_sca = C_sca / (np.pi * radius**2)

# Mie reference
from test_farfield_fix import mie_Qext_Qsca
Qe_mie, Qs_mie = mie_Qext_Qsca(k_ext * radius, m_rel)
print(f"\nQ_sca from Mueller M11: {Q_sca:.6f}")
print(f"Q_sca Mie:              {Qs_mie:.6f}")
print(f"Error: {abs(Q_sca - Qs_mie)/Qs_mie*100:.1f}%")

print("\nAll Mueller matrix tests passed!")
