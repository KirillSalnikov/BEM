"""Validation tests for BEM solver with non-spherical particles.

Tests prolate/oblate spheroids, cube-like shapes, and symmetry properties.
All tests use PMCHWT formulation for dielectric particles.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from bem_core import (icosphere, build_rwg, assemble_pmchwt,
                      compute_rhs_planewave, compute_cross_sections,
                      compute_far_field, compute_amplitude_matrix,
                      amplitude_to_mueller)

# Common parameters
k_ext = 1.0
m_rel = 1.5
k_int = k_ext * m_rel
eta_ext = 1.0
eta_int = eta_ext / m_rel
E0 = np.array([1.0, 0, 0.])
k_hat = np.array([0, 0, 1.])
sM = -1


def equivalent_radius(verts, tris):
    """Compute volume-equivalent sphere radius from a triangulated surface.

    Uses the divergence theorem: V = (1/6) |sum_i (v0 . (v1 x v2))|
    """
    vol = 0.0
    for tri in tris:
        v0, v1, v2 = verts[tri[0]], verts[tri[1]], verts[tri[2]]
        vol += np.dot(v0, np.cross(v1, v2))
    vol = abs(vol) / 6.0
    return (3.0 * vol / (4.0 * np.pi)) ** (1.0 / 3.0)


def geometric_cross_section(verts, tris, k_hat):
    """Compute geometric cross-section projected along k_hat direction."""
    # For Q normalization, use the projected area
    # For simplicity, compute bounding area in plane perpendicular to k_hat
    # Project all vertices onto plane perpendicular to k_hat
    # Use convex hull area as approximation
    from scipy.spatial import ConvexHull

    # Build orthonormal basis perpendicular to k_hat
    if abs(k_hat[2]) > 0.9:
        u = np.array([1.0, 0.0, 0.0])
    else:
        u = np.array([0.0, 0.0, 1.0])
    e1 = u - k_hat * np.dot(u, k_hat)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(k_hat, e1)

    proj = np.column_stack([verts @ e1, verts @ e2])
    hull = ConvexHull(proj)
    return hull.volume  # In 2D, ConvexHull.volume = area


def solve_pmchwt(verts, tris, refinements=None):
    """Assemble and solve PMCHWT system, return J, M, rwg, and other info."""
    rwg = build_rwg(verts, tris)
    N = rwg['N']

    Z, _, _ = assemble_pmchwt(rwg, verts, tris, k_ext, k_int, eta_ext, eta_int)
    b = compute_rhs_planewave(rwg, verts, tris, k_ext, eta_ext, E0=E0, k_hat=k_hat)
    coeffs = np.linalg.solve(Z, b)
    J = coeffs[:N]
    M = coeffs[N:]
    return rwg, J, M, Z


def compute_Q_optical_theorem(rwg, verts, tris, J, M, k, eta, geom_area):
    """Compute Q_ext from the optical theorem (forward scattering)."""
    theta_fwd = np.array([0.01])  # near-forward
    F_th, F_ph = compute_far_field(rwg, verts, tris, J, M, k, eta,
                                    theta_fwd, phi=0.0, sM=sM)
    C_ext = 4 * np.pi / k * np.imag(F_th[0])
    return C_ext / geom_area


# =============================================================================
# Test 1: Prolate spheroid (aspect ratio 2:1)
# =============================================================================
def test_prolate_spheroid():
    print("=" * 70)
    print("TEST 1: Prolate spheroid (aspect ratio 2:1, z-stretched)")
    print("=" * 70)

    verts, tris = icosphere(1.0, refinements=3)
    verts[:, 2] *= 2.0  # Stretch along z

    r_eq = equivalent_radius(verts, tris)
    geom_area = geometric_cross_section(verts, tris, k_hat)
    print(f"  Volume-equivalent radius: {r_eq:.4f}")
    print(f"  Geometric cross-section:  {geom_area:.4f}")
    print(f"  Number of triangles:      {len(tris)}")

    rwg, J, M, Z = solve_pmchwt(verts, tris)
    N = rwg['N']
    print(f"  Number of RWG functions:  {N}")

    # Compute cross sections
    Q_ext, Q_sca = compute_cross_sections(rwg, verts, tris, J, M, k_ext, eta_ext,
                                           np.sqrt(geom_area / np.pi), sM=sM,
                                           ntheta=91, nphi=18)
    Q_abs = Q_ext - Q_sca

    # Also compute Q_ext from optical theorem directly
    Q_ext_ot = compute_Q_optical_theorem(rwg, verts, tris, J, M, k_ext, eta_ext, geom_area)

    print(f"\n  Results (normalized by geometric cross-section):")
    print(f"    Q_ext (integrated) = {Q_ext:.6f}")
    print(f"    Q_ext (opt.thm)    = {Q_ext_ot:.6f}")
    print(f"    Q_sca              = {Q_sca:.6f}")
    print(f"    Q_abs              = {Q_abs:.6f}")

    # Checks
    passed = True
    if Q_ext <= 0:
        print("  FAIL: Q_ext <= 0")
        passed = False
    else:
        print(f"  PASS: Q_ext > 0")

    if Q_sca <= 0:
        print("  FAIL: Q_sca <= 0")
        passed = False
    else:
        print(f"  PASS: Q_sca > 0")

    if Q_abs < -0.01 * Q_ext:
        print(f"  FAIL: Q_abs = {Q_abs:.6f} < 0 (violates energy conservation)")
        passed = False
    else:
        print(f"  PASS: Q_abs >= 0 (energy conservation)")

    if Q_sca > Q_ext * 1.01:
        print(f"  FAIL: Q_sca > Q_ext (impossible for real refractive index)")
        passed = False
    else:
        print(f"  PASS: Q_sca <= Q_ext")

    return passed, {'Q_ext': Q_ext, 'Q_ext_ot': Q_ext_ot, 'Q_sca': Q_sca, 'Q_abs': Q_abs}


# =============================================================================
# Test 2: Oblate spheroid (aspect ratio 1:2)
# =============================================================================
def test_oblate_spheroid():
    print("\n" + "=" * 70)
    print("TEST 2: Oblate spheroid (aspect ratio 1:2, z-compressed)")
    print("=" * 70)

    verts, tris = icosphere(1.0, refinements=3)
    verts[:, 2] *= 0.5  # Compress along z

    r_eq = equivalent_radius(verts, tris)
    geom_area = geometric_cross_section(verts, tris, k_hat)
    print(f"  Volume-equivalent radius: {r_eq:.4f}")
    print(f"  Geometric cross-section:  {geom_area:.4f}")
    print(f"  Number of triangles:      {len(tris)}")

    rwg, J, M, Z = solve_pmchwt(verts, tris)
    N = rwg['N']
    print(f"  Number of RWG functions:  {N}")

    Q_ext, Q_sca = compute_cross_sections(rwg, verts, tris, J, M, k_ext, eta_ext,
                                           np.sqrt(geom_area / np.pi), sM=sM,
                                           ntheta=91, nphi=18)
    Q_abs = Q_ext - Q_sca

    Q_ext_ot = compute_Q_optical_theorem(rwg, verts, tris, J, M, k_ext, eta_ext, geom_area)

    print(f"\n  Results (normalized by geometric cross-section):")
    print(f"    Q_ext (integrated) = {Q_ext:.6f}")
    print(f"    Q_ext (opt.thm)    = {Q_ext_ot:.6f}")
    print(f"    Q_sca              = {Q_sca:.6f}")
    print(f"    Q_abs              = {Q_abs:.6f}")

    passed = True
    if Q_ext <= 0:
        print("  FAIL: Q_ext <= 0")
        passed = False
    else:
        print(f"  PASS: Q_ext > 0")

    if Q_sca <= 0:
        print("  FAIL: Q_sca <= 0")
        passed = False
    else:
        print(f"  PASS: Q_sca > 0")

    if Q_abs < -0.01 * Q_ext:
        print(f"  FAIL: Q_abs = {Q_abs:.6f} < 0 (violates energy conservation)")
        passed = False
    else:
        print(f"  PASS: Q_abs >= 0 (energy conservation)")

    if Q_sca > Q_ext * 1.01:
        print(f"  FAIL: Q_sca > Q_ext (impossible for real refractive index)")
        passed = False
    else:
        print(f"  PASS: Q_sca <= Q_ext")

    return passed, {'Q_ext': Q_ext, 'Q_ext_ot': Q_ext_ot, 'Q_sca': Q_sca, 'Q_abs': Q_abs}


# =============================================================================
# Test 3: Cube-like shape
# =============================================================================
def test_cube_like():
    print("\n" + "=" * 70)
    print("TEST 3: Cube-like shape (max-norm projection)")
    print("=" * 70)

    verts, tris = icosphere(1.0, refinements=2)
    # Deform to cube-like shape: normalize each vertex by its max-norm
    for i in range(len(verts)):
        max_abs = np.max(np.abs(verts[i]))
        if max_abs > 0:
            verts[i] = verts[i] / max_abs  # project to L-infinity unit sphere
    # Scale to desired size
    verts *= 0.5  # half-edge length = 0.5, so edge = 1.0

    r_eq = equivalent_radius(verts, tris)
    geom_area = geometric_cross_section(verts, tris, k_hat)
    print(f"  Volume-equivalent radius: {r_eq:.4f}")
    print(f"  Geometric cross-section:  {geom_area:.4f}")
    print(f"  Number of triangles:      {len(tris)}")

    rwg, J, M, Z = solve_pmchwt(verts, tris)
    N = rwg['N']
    print(f"  Number of RWG functions:  {N}")

    Q_ext, Q_sca = compute_cross_sections(rwg, verts, tris, J, M, k_ext, eta_ext,
                                           np.sqrt(geom_area / np.pi), sM=sM,
                                           ntheta=91, nphi=18)
    Q_abs = Q_ext - Q_sca

    Q_ext_ot = compute_Q_optical_theorem(rwg, verts, tris, J, M, k_ext, eta_ext, geom_area)

    print(f"\n  Results (normalized by geometric cross-section):")
    print(f"    Q_ext (integrated) = {Q_ext:.6f}")
    print(f"    Q_ext (opt.thm)    = {Q_ext_ot:.6f}")
    print(f"    Q_sca              = {Q_sca:.6f}")
    print(f"    Q_abs              = {Q_abs:.6f}")

    passed = True
    if Q_ext <= 0:
        print("  FAIL: Q_ext <= 0")
        passed = False
    else:
        print(f"  PASS: Q_ext > 0")

    if Q_sca <= 0:
        print("  FAIL: Q_sca <= 0")
        passed = False
    else:
        print(f"  PASS: Q_sca > 0")

    if Q_abs < -0.01 * Q_ext:
        print(f"  FAIL: Q_abs = {Q_abs:.6f} < 0 (violates energy conservation)")
        passed = False
    else:
        print(f"  PASS: Q_abs >= 0 (energy conservation)")

    # Check that the solve actually converged (residual is small)
    b = compute_rhs_planewave(rwg, verts, tris, k_ext, eta_ext, E0=E0, k_hat=k_hat)
    coeffs = np.concatenate([J, M])
    residual = np.linalg.norm(Z @ coeffs - b) / np.linalg.norm(b)
    print(f"  Relative residual: {residual:.2e}")
    if residual < 1e-10:
        print(f"  PASS: Solve converged (residual < 1e-10)")
    else:
        print(f"  WARN: Residual = {residual:.2e}")
        if residual > 1e-6:
            passed = False

    return passed, {'Q_ext': Q_ext, 'Q_ext_ot': Q_ext_ot, 'Q_sca': Q_sca, 'Q_abs': Q_abs}


# =============================================================================
# Test 4: Symmetry test for spheroid (Mueller matrix properties)
# =============================================================================
def test_spheroid_symmetry():
    print("\n" + "=" * 70)
    print("TEST 4: Spheroid symmetry (Mueller matrix, axial symmetry)")
    print("=" * 70)

    verts, tris = icosphere(1.0, refinements=3)
    verts[:, 2] *= 2.0  # Prolate spheroid

    rwg = build_rwg(verts, tris)
    N = rwg['N']
    print(f"  Mesh: {len(tris)} tris, {N} RWG")

    Z, _, _ = assemble_pmchwt(rwg, verts, tris, k_ext, k_int, eta_ext, eta_int)
    Z_lu = lu_factor(Z)

    # Compute amplitude matrix
    theta_arr = np.linspace(0.01, np.pi - 0.01, 37)
    S = compute_amplitude_matrix(rwg, verts, tris, k_ext, eta_ext, theta_arr,
                                  Z_lu=Z_lu, k_hat=k_hat, sM=sM)

    # For axial symmetry with incidence along the symmetry axis:
    # S3 and S4 should be zero (no cross-polarization)
    max_S3 = np.max(np.abs(S['S3']))
    max_S4 = np.max(np.abs(S['S4']))
    max_S1 = np.max(np.abs(S['S1']))
    max_S2 = np.max(np.abs(S['S2']))

    print(f"\n  Amplitude matrix elements:")
    print(f"    max|S1| = {max_S1:.6f}")
    print(f"    max|S2| = {max_S2:.6f}")
    print(f"    max|S3| = {max_S3:.2e}  (should be ~0)")
    print(f"    max|S4| = {max_S4:.2e}  (should be ~0)")

    # Mueller matrix
    Muel = amplitude_to_mueller(S['S1'], S['S2'], S['S3'], S['S4']) / k_ext**2

    print(f"\n  Mueller matrix at theta=90 deg (normalized):")
    idx90 = 18
    M11_90 = Muel[0, 0, idx90]
    for i in range(4):
        row = " ".join(f"{Muel[i,j,idx90]/M11_90:8.4f}" for j in range(4))
        print(f"    [{row}]")

    # Symmetry checks
    passed = True

    # S3, S4 ~ 0 for axially symmetric particle with on-axis incidence
    tol_cross = 0.02 * max(max_S1, max_S2)
    if max_S3 < tol_cross:
        print(f"\n  PASS: |S3| < {tol_cross:.4f} (axial symmetry)")
    else:
        print(f"\n  FAIL: |S3| = {max_S3:.4f} > {tol_cross:.4f}")
        passed = False

    if max_S4 < tol_cross:
        print(f"  PASS: |S4| < {tol_cross:.4f} (axial symmetry)")
    else:
        print(f"  FAIL: |S4| = {max_S4:.4f} > {tol_cross:.4f}")
        passed = False

    # For axial symmetry: M12/M11 should be well-defined,
    # Mueller matrix should be block-diagonal (off-diagonal blocks ~ 0)
    # M02, M03, M12, M13, M20, M21, M30, M31 should all be ~0
    off_block = []
    for i, j in [(0,2), (0,3), (1,2), (1,3), (2,0), (2,1), (3,0), (3,1)]:
        off_block.append(np.max(np.abs(Muel[i, j])))
    max_off_block = max(off_block)
    max_M11 = np.max(Muel[0, 0])

    tol_block = 0.02 * max_M11
    if max_off_block < tol_block:
        print(f"  PASS: Mueller off-diagonal blocks < {tol_block:.4f} (axial symmetry)")
    else:
        print(f"  FAIL: Mueller off-diagonal max = {max_off_block:.4f} > {tol_block:.4f}")
        passed = False

    # M22 = M33 (for axial symmetry)
    diff_M22_M33 = np.max(np.abs(Muel[2, 2] - Muel[3, 3]))
    tol_diag = 0.02 * max_M11
    if diff_M22_M33 < tol_diag:
        print(f"  PASS: |M22 - M33| < {tol_diag:.4f}")
    else:
        print(f"  FAIL: |M22 - M33| = {diff_M22_M33:.4f}")
        passed = False

    # M23 = -M32 (for axial symmetry)
    diff_M23_M32 = np.max(np.abs(Muel[2, 3] + Muel[3, 2]))
    if diff_M23_M32 < tol_diag:
        print(f"  PASS: |M23 + M32| < {tol_diag:.4f}")
    else:
        print(f"  FAIL: |M23 + M32| = {diff_M23_M32:.4f}")
        passed = False

    return passed, {'max_S3': max_S3, 'max_S4': max_S4}


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    results = {}

    p1, r1 = test_prolate_spheroid()
    results['Prolate spheroid'] = (p1, r1)

    p2, r2 = test_oblate_spheroid()
    results['Oblate spheroid'] = (p2, r2)

    p3, r3 = test_cube_like()
    results['Cube-like'] = (p3, r3)

    p4, r4 = test_spheroid_symmetry()
    results['Spheroid symmetry'] = (p4, r4)

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Test':<25s} {'Status':<8s} {'Q_ext':>10s} {'Q_sca':>10s} {'Q_abs':>10s}")
    print("-" * 70)
    for name, (passed, data) in results.items():
        status = "PASS" if passed else "FAIL"
        q_ext = f"{data.get('Q_ext', 0):.4f}" if 'Q_ext' in data else "N/A"
        q_sca = f"{data.get('Q_sca', 0):.4f}" if 'Q_sca' in data else "N/A"
        q_abs = f"{data.get('Q_abs', 0):.4f}" if 'Q_abs' in data else "N/A"
        print(f"{name:<25s} {status:<8s} {q_ext:>10s} {q_sca:>10s} {q_abs:>10s}")
    print("-" * 70)

    all_passed = all(p for p, _ in results.values())
    if all_passed:
        print("All tests PASSED.")
    else:
        print("Some tests FAILED.")
