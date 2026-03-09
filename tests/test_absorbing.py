"""Test BEM solver with absorbing particles (complex refractive index) vs Mie theory."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from bem_core import (icosphere, build_rwg, assemble_pmchwt,
                      compute_rhs_planewave, compute_cross_sections)


def mie_Qext_Qsca(x, m_rel, n_max=None):
    """Mie theory for sphere with complex refractive index."""
    from scipy.special import spherical_jn, spherical_yn
    if n_max is None:
        n_max = int(abs(x) + 4*abs(x)**(1/3) + 2) + 10
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


cases = [
    {'name': 'Weak absorber',     'm': 1.5 + 0.01j,  'ka': 2.0, 'ref': 3},
    {'name': 'Moderate absorber', 'm': 1.5 + 0.1j,   'ka': 2.0, 'ref': 3},
    {'name': 'Strong absorber',   'm': 1.5 + 0.5j,   'ka': 2.0, 'ref': 3},
    {'name': 'Metal-like',        'm': 1.33 + 1.0j,  'ka': 1.0, 'ref': 3},
    {'name': 'Water droplet',     'm': 1.33 + 1e-6j, 'ka': 2.0, 'ref': 3},
]

print(f"{'Case':<22} {'m':>14} {'ka':>4}  "
      f"{'Q_ext BEM':>10} {'Q_ext Mie':>10} {'err%':>6}  "
      f"{'Q_sca BEM':>10} {'Q_sca Mie':>10} {'err%':>6}  "
      f"{'Q_abs BEM':>10} {'Q_abs Mie':>10} {'err%':>6}")
print('-' * 155)

for c in cases:
    m_rel = c['m']
    ka = c['ka']
    radius = 1.0
    k_ext = ka / radius
    k_int = k_ext * m_rel
    eta_ext = 1.0
    eta_int = eta_ext / m_rel

    verts, tris = icosphere(radius, c['ref'])
    rwg = build_rwg(verts, tris)
    N = rwg['N']

    Z, _, _ = assemble_pmchwt(rwg, verts, tris, k_ext, k_int, eta_ext, eta_int)
    b = compute_rhs_planewave(rwg, verts, tris, k_ext, eta_ext)
    coeffs = np.linalg.solve(Z, b)
    J, M = coeffs[:N], coeffs[N:]

    Q_ext_bem, Q_sca_bem = compute_cross_sections(
        rwg, verts, tris, J, M, k_ext, eta_ext, radius, sM=-1)
    Q_abs_bem = Q_ext_bem - Q_sca_bem

    Q_ext_mie, Q_sca_mie = mie_Qext_Qsca(ka, m_rel)
    Q_abs_mie = Q_ext_mie - Q_sca_mie

    err_ext = abs(Q_ext_bem - Q_ext_mie) / Q_ext_mie * 100
    err_sca = abs(Q_sca_bem - Q_sca_mie) / Q_sca_mie * 100
    err_abs = abs(Q_abs_bem - Q_abs_mie) / max(abs(Q_abs_mie), 1e-10) * 100

    print(f"{c['name']:<22} {str(c['m']):>14} {ka:>4.1f}  "
          f"{Q_ext_bem:>10.4f} {Q_ext_mie:>10.4f} {err_ext:>5.1f}%  "
          f"{Q_sca_bem:>10.4f} {Q_sca_mie:>10.4f} {err_sca:>5.1f}%  "
          f"{Q_abs_bem:>10.4f} {Q_abs_mie:>10.4f} {err_abs:>5.1f}%")
