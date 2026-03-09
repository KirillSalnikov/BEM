"""
Direct comparison: extract bempp Galerkin matrices and compare with ours.
Build PMCHWT the same way bempp does.
"""
import numpy as np
import time
import bempp_cl.api as bempp_api
from bempp_cl.api.assembly.blocked_operator import (
    projections_from_grid_functions_list,
    grid_function_list_from_coefficients,
)

bempp_api.set_logging_level("warning")

# Import our code
from bem_core import icosphere, build_rwg, assemble_L_K, tri_quadrature


def mie_Qext_Qsca(x, m_rel, n_max=None):
    from scipy.special import spherical_jn, spherical_yn
    if n_max is None:
        n_max = int(x + 4*x**(1/3) + 2) + 5
    def psi(n, z): return z * spherical_jn(n, z)
    def psi_d(n, z): return spherical_jn(n, z) + z * spherical_jn(n, z, derivative=True)
    def xi(n, z): return z * (spherical_jn(n, z) + 1j * spherical_yn(n, z))
    def xi_d(n, z):
        return (spherical_jn(n, z) + 1j * spherical_yn(n, z)) + \
               z * (spherical_jn(n, z, derivative=True) + 1j * spherical_yn(n, z, derivative=True))
    mx = m_rel * x
    Q_ext = 0.0; Q_sca = 0.0
    for n in range(1, n_max + 1):
        a_n = (m_rel * psi(n, mx) * psi_d(n, x) - psi(n, x) * psi_d(n, mx)) / \
              (m_rel * psi(n, mx) * xi_d(n, x) - xi(n, x) * psi_d(n, mx))
        b_n = (psi(n, mx) * psi_d(n, x) - m_rel * psi(n, x) * psi_d(n, mx)) / \
              (psi(n, mx) * xi_d(n, x) - m_rel * xi(n, x) * psi_d(n, mx))
        Q_ext += (2*n + 1) * np.real(a_n + b_n)
        Q_sca += (2*n + 1) * (abs(a_n)**2 + abs(b_n)**2)
    return 2 * Q_ext / x**2, 2 * Q_sca / x**2


if __name__ == "__main__":
    radius = 1.0; k_ext = 1.0; m_rel = 1.5; x = k_ext * radius
    k_int = k_ext * m_rel; eta_ext = 1.0; eta_int = 1.0 / m_rel
    eps_r = m_rel**2; mu_r = 1.0
    E0 = np.array([1.0, 0, 0]); k_hat = np.array([0, 0, 1.0])

    Q_ext_mie, Q_sca_mie = mie_Qext_Qsca(x, m_rel)
    print(f"Mie: Q_ext = {Q_ext_mie:.6f}")

    refine = 2
    verts, tris = icosphere(radius, refinements=refine)
    rwg = build_rwg(verts, tris)
    N = rwg['N']
    print(f"Mesh: {len(tris)} tris, {N} RWG")

    # ======= BEMPP SOLVE =======
    grid = bempp_api.Grid(verts.T, tris.T)
    from bempp_cl.api.operators.boundary.maxwell import multitrace_operator
    A_int = multitrace_operator(grid, k_int, epsilon_r=eps_r, mu_r=mu_r, space_type="all_rwg")
    A_ext = multitrace_operator(grid, k_ext, space_type="all_rwg")
    A = A_int + A_ext

    @bempp_api.complex_callable
    def tangential_trace(x, n, domain_index, result):
        e_inc = E0 * np.exp(1j * k_ext * np.dot(k_hat, x))
        result[:] = np.cross(e_inc, n)  # n × E, then × n again... actually tangential trace = n × E × n

    @bempp_api.complex_callable
    def tangential_trace_correct(x, n, domain_index, result):
        e_inc = E0 * np.exp(1j * k_ext * np.dot(k_hat, x))
        nxE = np.cross(n, e_inc)
        result[0] = nxE[0]; result[1] = nxE[1]; result[2] = nxE[2]

    @bempp_api.complex_callable
    def neumann_trace(x, n, domain_index, result):
        phase = np.exp(1j * k_ext * np.dot(k_hat, x))
        curl_E = 1j * k_ext * np.cross(k_hat, E0) * phase
        inv_ik = 1.0 / (1j * k_ext)
        nxcurl = np.cross(n, inv_ik * curl_E)
        result[0] = nxcurl[0]; result[1] = nxcurl[1]; result[2] = nxcurl[2]

    rhs = [
        bempp_api.GridFunction(space=A.range_spaces[0], dual_space=A.dual_to_range_spaces[0], fun=tangential_trace_correct),
        bempp_api.GridFunction(space=A.range_spaces[1], dual_space=A.dual_to_range_spaces[1], fun=neumann_trace),
    ]

    A_dense = bempp_api.as_matrix(A.weak_form())
    b_vec = projections_from_grid_functions_list(rhs, A.dual_to_range_spaces)
    x_sol = np.linalg.solve(A_dense, b_vec)
    sol = grid_function_list_from_coefficients(x_sol, A.domain_spaces)

    # Far field
    fwd = np.array([[0.0], [0.0], [1.0]])
    ef = bempp_api.operators.far_field.maxwell.electric_field(sol[1].space, fwd, k_ext)
    mf = bempp_api.operators.far_field.maxwell.magnetic_field(sol[0].space, fwd, k_ext)
    E_fwd = (-ef * sol[1] - mf * sol[0]).ravel()
    Q_ext_bempp = (4*np.pi/k_ext) * np.imag(np.dot(E_fwd, E0)) / (np.pi * radius**2)
    print(f"\nbempp Q_ext = {Q_ext_bempp:.6f} (err={abs(Q_ext_bempp - Q_ext_mie)/Q_ext_mie*100:.1f}%)")
    print(f"bempp E_fwd = {E_fwd}")
    print(f"bempp |x_sol| = {np.linalg.norm(x_sol):.4e}")
    print(f"bempp A_dense shape = {A_dense.shape}")

    # ======= Extract bempp's individual operator matrices =======
    print("\n--- Extracting bempp operator blocks ---")
    # The multitrace operator A has a 2x2 block structure
    # Let's extract each block
    n_dof = A.domain_spaces[0].global_dof_count
    print(f"bempp DOFs per block: {n_dof}")

    # Extract the 4 blocks of the Galerkin matrix
    # A_dense = M^{-1} @ G where G is the SNC-tested Galerkin matrix
    # The blocks are:
    # A[0,0] = block (0,0) of multitrace
    # etc.
    A00 = A_dense[:n_dof, :n_dof]
    A01 = A_dense[:n_dof, n_dof:]
    A10 = A_dense[n_dof:, :n_dof]
    A11 = A_dense[n_dof:, n_dof:]

    print(f"  |A00| = {np.linalg.norm(A00):.4e} (should be L-type)")
    print(f"  |A01| = {np.linalg.norm(A01):.4e} (should be K-type)")
    print(f"  |A10| = {np.linalg.norm(A10):.4e} (should be K-type)")
    print(f"  |A11| = {np.linalg.norm(A11):.4e} (should be L-type)")

    # Check symmetry of K blocks
    print(f"  ||A01 + A10|| / ||A01|| = {np.linalg.norm(A01 + A10) / np.linalg.norm(A01):.4e}")
    print(f"  ||A01 - A10|| / ||A01|| = {np.linalg.norm(A01 - A10) / np.linalg.norm(A01):.4e}")

    # Check symmetry of L blocks
    print(f"  ||A00 - A00.T|| / ||A00|| = {np.linalg.norm(A00 - A00.T) / np.linalg.norm(A00):.4e}")
    print(f"  ||A11 - A11.T|| / ||A11|| = {np.linalg.norm(A11 - A11.T) / np.linalg.norm(A11):.4e}")

    # ======= OUR OPERATORS =======
    print("\n--- Our operators ---")
    L_ext, K_ext = assemble_L_K(rwg, verts, tris, k_ext)
    L_int, K_int = assemble_L_K(rwg, verts, tris, k_int)

    # Our PMCHWT blocks
    Z00 = eta_ext * L_ext + eta_int * L_int
    Z01 = K_ext + K_int
    Z10 = -(K_ext + K_int)
    Z11 = L_ext / eta_ext + L_int / eta_int

    print(f"  |Z00| = {np.linalg.norm(Z00):.4e}")
    print(f"  |Z01| = {np.linalg.norm(Z01):.4e}")
    print(f"  |Z10| = {np.linalg.norm(Z10):.4e}")
    print(f"  |Z11| = {np.linalg.norm(Z11):.4e}")
    print(f"  ||Z01 + Z10|| / ||Z01|| = {np.linalg.norm(Z01 + Z10) / np.linalg.norm(Z01):.4e}")

    # Check if L is symmetric
    print(f"  ||L_ext - L_ext.T|| / ||L_ext|| = {np.linalg.norm(L_ext - L_ext.T) / np.linalg.norm(L_ext):.4e}")

    # Check eigenvalue structure
    eigvals_A00 = np.linalg.eigvals(A00)
    eigvals_Z00 = np.linalg.eigvals(Z00)
    print(f"\n  bempp A00 eigenvalues: real [{eigvals_A00.real.min():.4f}, {eigvals_A00.real.max():.4f}], "
          f"imag [{eigvals_A00.imag.min():.4f}, {eigvals_A00.imag.max():.4f}]")
    print(f"  our Z00 eigenvalues:   real [{eigvals_Z00.real.min():.4f}, {eigvals_Z00.real.max():.4f}], "
          f"imag [{eigvals_Z00.imag.min():.4f}, {eigvals_Z00.imag.max():.4f}]")

    # Check if bempp K blocks are antisymmetric (A01 = -A10)
    # If so, bempp uses same convention as standard PMCHWT
    if np.linalg.norm(A01 + A10) < 0.1 * np.linalg.norm(A01):
        print("\n  bempp K blocks: A01 ≈ -A10 (antisymmetric, standard PMCHWT)")
    elif np.linalg.norm(A01 - A10) < 0.1 * np.linalg.norm(A01):
        print("\n  bempp K blocks: A01 ≈ A10 (symmetric)")
    else:
        print(f"\n  bempp K blocks: neither symmetric nor antisymmetric")

    # Direct ratio comparison (after accounting for DOF permutation)
    # Just compare norms and eigenvalue ranges for now
    print(f"\n  Ratio of L-block norms: |A00|/|Z00| = {np.linalg.norm(A00)/np.linalg.norm(Z00):.4f}")
    print(f"  Ratio of K-block norms: |A01|/|Z01| = {np.linalg.norm(A01)/np.linalg.norm(Z01):.4f}")
    print(f"  Ratio of L-block norms: |A11|/|Z11| = {np.linalg.norm(A11)/np.linalg.norm(Z11):.4f}")
