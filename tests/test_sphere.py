"""
Test BEM solver on a dielectric sphere, compare with Mie theory.
"""

import numpy as np
import sys
import time

from bem_core import (icosphere, build_rwg,
                      assemble_pmchwt, compute_rhs_planewave,
                      compute_far_field, compute_cross_sections)


# ============================================================
# Mie theory (simple implementation for comparison)
# ============================================================

def mie_coefficients(x, m_rel, n_max):
    """Compute Mie coefficients a_n, b_n for a sphere."""
    from scipy.special import spherical_jn, spherical_yn

    def psi(n, z):
        return z * spherical_jn(n, z)

    def psi_d(n, z):
        return spherical_jn(n, z) + z * spherical_jn(n, z, derivative=True)

    def xi(n, z):
        return z * (spherical_jn(n, z) + 1j * spherical_yn(n, z))

    def xi_d(n, z):
        sj = spherical_jn(n, z)
        sjd = spherical_jn(n, z, derivative=True)
        sy = spherical_yn(n, z)
        syd = spherical_yn(n, z, derivative=True)
        return (sj + 1j * sy) + z * (sjd + 1j * syd)

    mx = m_rel * x
    a_n = []
    b_n = []

    for n in range(1, n_max + 1):
        # a_n
        num_a = m_rel * psi(n, mx) * psi_d(n, x) - psi(n, x) * psi_d(n, mx)
        den_a = m_rel * psi(n, mx) * xi_d(n, x) - xi(n, x) * psi_d(n, mx)
        a_n.append(num_a / den_a)

        # b_n
        num_b = psi(n, mx) * psi_d(n, x) - m_rel * psi(n, x) * psi_d(n, mx)
        den_b = psi(n, mx) * xi_d(n, x) - m_rel * xi(n, x) * psi_d(n, mx)
        b_n.append(num_b / den_b)

    return np.array(a_n), np.array(b_n)


def mie_Csca(x, m_rel, n_max):
    """Scattering cross section from Mie theory."""
    a, b = mie_coefficients(x, m_rel, n_max)
    ns = np.arange(1, n_max + 1)
    Q_sca = (2 / x**2) * np.sum((2 * ns + 1) * (np.abs(a)**2 + np.abs(b)**2))
    return Q_sca * np.pi * (x / (2 * np.pi))**2  # Wait, need to be careful with units


def mie_Qsca(x, m_rel, n_max):
    """Scattering efficiency from Mie theory."""
    a, b = mie_coefficients(x, m_rel, n_max)
    ns = np.arange(1, n_max + 1)
    Q_sca = (2 / x**2) * np.sum((2 * ns + 1) * (np.abs(a)**2 + np.abs(b)**2))
    return Q_sca


def mie_Qext(x, m_rel, n_max):
    """Extinction efficiency from Mie theory."""
    a, b = mie_coefficients(x, m_rel, n_max)
    ns = np.arange(1, n_max + 1)
    Q_ext = (2 / x**2) * np.sum((2 * ns + 1) * np.real(a + b))
    return Q_ext


def mie_S1S2(x, m_rel, n_max, theta):
    """Mie amplitude functions S1, S2."""
    from scipy.special import lpmv
    a, b = mie_coefficients(x, m_rel, n_max)
    cos_t = np.cos(theta)

    S1 = np.zeros_like(theta, dtype=complex)
    S2 = np.zeros_like(theta, dtype=complex)

    for n in range(1, n_max + 1):
        # pi_n and tau_n
        if np.isscalar(theta):
            theta = np.array([theta])

        # Associated Legendre
        pi_n = np.zeros_like(theta)
        tau_n = np.zeros_like(theta)

        for it, t in enumerate(theta):
            ct = np.cos(t)
            st = np.sin(t)

            if abs(st) < 1e-15:
                pi_n[it] = n * (n + 1) / 2 if abs(ct - 1) < 1e-10 else ((-1)**(n+1)) * n * (n + 1) / 2
                tau_n[it] = pi_n[it] * ct
            else:
                # pi_n = P_n^1 / sin(theta)
                pi_n[it] = lpmv(1, n, ct) / (-st)  # Note sign convention
                # tau_n = dP_n^1/d(theta)
                # Use recurrence or numerical derivative
                dt = 1e-6
                p1 = lpmv(1, n, np.cos(t + dt))
                p2 = lpmv(1, n, np.cos(t - dt))
                tau_n[it] = -(p1 - p2) / (2 * dt)

        coeff = (2*n + 1) / (n * (n + 1))
        S1 += coeff * (a[n-1] * pi_n + b[n-1] * tau_n)
        S2 += coeff * (a[n-1] * tau_n + b[n-1] * pi_n)

    return S1, S2


# ============================================================
# Main test
# ============================================================

if __name__ == "__main__":
    # Parameters
    wavelength = 2 * np.pi  # lambda = 2*pi, so k=1
    k_ext = 2 * np.pi / wavelength  # k = 1

    # Sphere parameters
    radius = 1.0  # x = k*a = 1.0
    x = k_ext * radius
    m_rel = 1.5  # refractive index

    eps_ext = 1.0
    eps_int = m_rel**2
    mu = 1.0

    k_int = k_ext * m_rel
    eta_ext = np.sqrt(mu / eps_ext)  # = 1
    eta_int = np.sqrt(mu / eps_int)  # = 1/m_rel

    print(f"=== BEM Solver Test: Dielectric Sphere ===")
    print(f"  x = k*a = {x:.2f}")
    print(f"  m = {m_rel}")
    print(f"  k_ext = {k_ext:.4f}, k_int = {k_int:.4f}")
    print(f"  eta_ext = {eta_ext:.4f}, eta_int = {eta_int:.4f}")

    # Mie reference
    n_max = int(x + 4*x**(1/3) + 2) + 5
    Q_sca_mie = mie_Qsca(x, m_rel, n_max)
    Q_ext_mie = mie_Qext(x, m_rel, n_max)
    print(f"\n  Mie reference: Q_ext = {Q_ext_mie:.6f}, Q_sca = {Q_sca_mie:.6f}")

    # Mesh (icosphere)
    refine = 1  # 42 verts, 80 tris, 120 RWG — fast for validation
    print(f"\n  Generating icosphere (refinements={refine})...")
    t0 = time.time()
    verts, tris = icosphere(radius, refinements=refine)
    print(f"  Mesh: {len(verts)} vertices, {len(tris)} triangles")

    # Build RWG
    rwg = build_rwg(verts, tris)
    N = rwg['N']
    print(f"  RWG basis functions: {N}")
    print(f"  System size: {2*N} x {2*N}")

    # Assemble
    print(f"\n  Assembling PMCHWT system...")
    t1 = time.time()
    Z, L_ext, K_ext = assemble_pmchwt(rwg, verts, tris, k_ext, k_int, eta_ext, eta_int, quad_order=4)
    t2 = time.time()
    print(f"  Assembly time: {t2-t1:.1f} s")

    # RHS
    print(f"  Computing RHS (plane wave)...")
    b = compute_rhs_planewave(rwg, verts, tris, k_ext, eta_ext,
                               E0=np.array([1.0, 0, 0]),
                               k_hat=np.array([0, 0, 1.0]))
    t3 = time.time()

    # Solve
    print(f"  Solving {2*N}x{2*N} linear system...")
    coeffs = np.linalg.solve(Z, b)
    coeffs_J = coeffs[:N]
    coeffs_M = coeffs[N:]
    t4 = time.time()
    print(f"  Solve time: {t4-t3:.1f} s")

    # Far field
    print(f"\n  Computing far field...")
    theta_arr = np.linspace(0.01, np.pi - 0.01, 181)
    F_theta, F_phi = compute_far_field(rwg, verts, tris, coeffs_J, coeffs_M,
                                        k_ext, theta_arr, phi=0.0)

    # Cross sections
    dsigma, C_sca_bem = compute_cross_sections(F_theta, F_phi, theta_arr, k_ext)
    geom_cross = np.pi * radius**2
    Q_sca_bem = C_sca_bem / geom_cross

    print(f"\n=== Results ===")
    print(f"  Q_sca (BEM):  {Q_sca_bem:.6f}")
    print(f"  Q_sca (Mie):  {Q_sca_mie:.6f}")
    print(f"  Relative error: {abs(Q_sca_bem - Q_sca_mie)/Q_sca_mie * 100:.1f}%")

    total_time = time.time() - t0
    print(f"\n  Total time: {total_time:.1f} s")

    # Save results for plotting
    np.savez("/home/serg/bem_solver/results.npz",
             theta=theta_arr, F_theta=F_theta, F_phi=F_phi,
             dsigma=dsigma, Q_sca_bem=Q_sca_bem, Q_sca_mie=Q_sca_mie,
             x=x, m_rel=m_rel)
    print("  Results saved to results.npz")
