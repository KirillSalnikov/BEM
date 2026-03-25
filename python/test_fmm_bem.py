"""Test FMM-accelerated BEM matvec against dense BEM matvec on a sphere."""

import numpy as np
import sys
import time

sys.path.insert(0, '/home/user/BEM')
from bem_core import (icosphere, build_rwg, assemble_pmchwt,
                       compute_rhs_planewave, compute_far_field,
                       solve_gmres, solve_gmres_aext, build_aext_preconditioner)
from bem_fmm import BEM_FMM_Operator, solve_gmres_fmm


def test_fmm_matvec(ka=1.0, m_re=1.3116, refinements=2):
    """Compare FMM matvec with dense matvec for dielectric sphere."""
    k_ext = 2 * np.pi  # wavelength = 1
    radius = ka / k_ext
    k_int = k_ext * m_re
    eta_ext = 1.0
    eta_int = 1.0 / m_re

    print(f"=== FMM BEM test: ka={ka}, m={m_re}, ref={refinements} ===")

    # Mesh
    verts, tris = icosphere(radius=radius, refinements=refinements)
    rwg = build_rwg(verts, tris)
    N = rwg['N']
    print(f"Mesh: {len(tris)} tris, {N} RWG, system size {2*N}")

    # Dense assembly
    t0 = time.time()
    Z, L_ext, K_ext = assemble_pmchwt(rwg, verts, tris, k_ext, k_int,
                                       eta_ext, eta_int)
    t_asm = time.time() - t0
    print(f"Dense assembly: {t_asm:.1f}s")

    # FMM operator
    t0 = time.time()
    fmm_op = BEM_FMM_Operator(rwg, verts, tris, k_ext, k_int,
                                eta_ext, eta_int, fmm_digits=3, max_leaf=32)
    t_fmm_init = time.time() - t0
    print(f"FMM init: {t_fmm_init:.1f}s")

    # Test matvec with random vector
    rng = np.random.RandomState(42)
    x = rng.randn(2*N) + 1j * rng.randn(2*N)

    t0 = time.time()
    y_dense = Z @ x
    t_dense_mv = time.time() - t0

    t0 = time.time()
    y_fmm = fmm_op.matvec(x)
    t_fmm_mv = time.time() - t0

    rel_err = np.linalg.norm(y_fmm - y_dense) / np.linalg.norm(y_dense)
    print(f"\nMatvec comparison:")
    print(f"  Dense matvec: {t_dense_mv:.3f}s")
    print(f"  FMM matvec:   {t_fmm_mv:.3f}s")
    print(f"  Relative error: {rel_err:.2e}")

    # Test with actual RHS
    b = compute_rhs_planewave(rwg, verts, tris, k_ext, eta_ext)
    y_dense_b = Z @ b
    y_fmm_b = fmm_op.matvec(b)
    rel_err_b = np.linalg.norm(y_fmm_b - y_dense_b) / np.linalg.norm(y_dense_b)
    print(f"  RHS matvec error: {rel_err_b:.2e}")

    # Solve with FMM GMRES
    print(f"\nSolving with FMM GMRES...")
    t0 = time.time()
    x_fmm = solve_gmres_fmm(fmm_op, b, tol=1e-3, maxiter=200)
    t_fmm_solve = time.time() - t0

    # Solve with dense LU for comparison
    print(f"\nSolving with dense LU...")
    from scipy.linalg import lu_factor, lu_solve
    t0 = time.time()
    Z_lu = lu_factor(Z)
    x_lu = lu_solve(Z_lu, b)
    t_lu = time.time() - t0
    print(f"  LU solve: {t_lu:.1f}s")

    # Compare solutions
    sol_err = np.linalg.norm(x_fmm - x_lu) / np.linalg.norm(x_lu)
    print(f"\nSolution comparison:")
    print(f"  FMM GMRES time: {t_fmm_solve:.1f}s")
    print(f"  Dense LU time:  {t_lu:.1f}s")
    print(f"  Solution rel error: {sol_err:.2e}")

    # Compare far fields
    theta = np.linspace(0, np.pi, 37)
    J_fmm = x_fmm[:N]; M_fmm = x_fmm[N:]
    J_lu = x_lu[:N]; M_lu = x_lu[N:]
    F_th_fmm, F_ph_fmm = compute_far_field(rwg, verts, tris, J_fmm, M_fmm,
                                              k_ext, eta_ext, theta)
    F_th_lu, F_ph_lu = compute_far_field(rwg, verts, tris, J_lu, M_lu,
                                           k_ext, eta_ext, theta)
    dsigma_fmm = np.abs(F_th_fmm)**2 + np.abs(F_ph_fmm)**2
    dsigma_lu = np.abs(F_th_lu)**2 + np.abs(F_ph_lu)**2

    mask = dsigma_lu > 1e-10 * np.max(dsigma_lu)
    ff_err = np.mean(np.abs(dsigma_fmm[mask] - dsigma_lu[mask]) / dsigma_lu[mask])
    print(f"  Far-field mean error: {ff_err:.2e}")

    return rel_err, sol_err, ff_err


if __name__ == '__main__':
    import sys
    ka = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    ref = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    test_fmm_matvec(ka=ka, refinements=ref)
