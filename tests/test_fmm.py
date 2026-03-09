"""Test FMM-accelerated matrix-vector product against dense matvec."""
import numpy as np
import sys
sys.path.insert(0, '/home/serg/bem_solver')

import bem_core as bc

# Build a small test case: PEC sphere, ka=1
print("=== FMM Operator Test ===")
print()

# Mesh
radius = 1.0
k_ext = 1.0
eta_ext = 1.0
k_int = 1.5  # dielectric m=1.5
eta_int = eta_ext / 1.5

verts, tris = bc.icosphere(radius=radius, refinements=2)
rwg = bc.build_rwg(verts, tris)
N = rwg['N']
print(f"Mesh: {len(verts)} verts, {len(tris)} tris, {N} RWG functions")
print()

# Assemble PMCHWT
print("Assembling PMCHWT system...")
Z, L_ext, K_ext = bc.assemble_pmchwt(rwg, verts, tris, k_ext, k_int,
                                       eta_ext, eta_int, parallel=False)
print(f"Z shape: {Z.shape}")
print()

# Build FMM operator
print("Building FMM operator...")
fmm = bc.FMMOperator(rwg, verts, tris, k_ext, eta_ext,
                       None, None, N, tree_depth=3, Z_full=Z)
print(f"FMM memory: {fmm.memory_bytes() / 1e6:.2f} MB")
print(f"Dense memory: {Z.nbytes / 1e6:.2f} MB")
print()

# Test matvec accuracy
print("Testing matvec accuracy...")
np.random.seed(42)
x = np.random.randn(2 * N) + 1j * np.random.randn(2 * N)

y_dense = Z @ x
y_fmm = fmm.matvec(x)

abs_err = np.linalg.norm(y_fmm - y_dense)
rel_err = abs_err / np.linalg.norm(y_dense)
print(f"  |y_fmm - y_dense| = {abs_err:.6e}")
print(f"  Relative error    = {rel_err:.6e}")

assert rel_err < 0.01, f"FMM matvec error too large: {rel_err:.4e}"
print("  PASS: relative error < 1%")
print()

# Test solve_fmm_gmres
print("Testing solve_fmm_gmres...")
# Build RHS for plane wave
E0 = np.array([0, 0, 1.0])
k_hat = np.array([1, 0, 0.0])
b = bc.compute_rhs_planewave(rwg, verts, tris, k_ext, eta_ext, E0, k_hat)

# Dense solve for reference
print("  Dense solve...")
x_dense = np.linalg.solve(Z, b)

# FMM solve
print("  FMM GMRES solve...")
x_fmm = bc.solve_fmm_gmres(rwg, verts, tris, Z, b, k_ext, eta_ext,
                              tol=1e-4, maxiter=200)

sol_err = np.linalg.norm(x_fmm - x_dense) / np.linalg.norm(x_dense)
print(f"  Solution relative error = {sol_err:.6e}")
assert sol_err < 0.01, f"FMM solution error too large: {sol_err:.4e}"
print("  PASS: solution error < 1%")
print()

print("=== All FMM tests passed ===")
