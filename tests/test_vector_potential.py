"""Test vector_potential_integral_triangle accuracy."""
import numpy as np
from bem_core import potential_integral_triangle, vector_potential_integral_triangle, tri_quadrature
from scipy import integrate

v0 = np.array([0.0, 0.0, 0.0])
v1 = np.array([1.0, 0.0, 0.0])
v2 = np.array([0.0, 1.0, 0.0])
area2 = np.linalg.norm(np.cross(v1 - v0, v2 - v0))  # = 1.0

def V_scipy(r_obs):
    """∫_T r'/|r_obs - r'| dS' using scipy adaptive quadrature."""
    result = np.zeros(3)
    for comp in range(3):
        def integrand(u, v):
            pt = (1 - u - v) * v0 + u * v1 + v * v2
            R = np.linalg.norm(pt - r_obs)
            if R < 1e-15:
                return 0.0
            return pt[comp] / R
        val, _ = integrate.dblquad(integrand, 0, 1, 0, lambda u: 1 - u)
        result[comp] = val * area2
    return result


print("=== Vector potential integral: code vs scipy ===")
centroid = (v0 + v1 + v2) / 3.0

for h in [1.0, 0.1, 0.01, 0.001]:
    r_obs = centroid + np.array([0, 0, h])
    V_code = vector_potential_integral_triangle(r_obs, v0, v1, v2)
    V_ref = V_scipy(r_obs)
    err = np.linalg.norm(V_code - V_ref) / max(np.linalg.norm(V_ref), 1e-15)
    print(f"  h={h:.3f}: code={V_code}, ref={V_ref}, rel_err={err:.2e}")

# Test at centroid (ON the triangle)
r_obs = centroid
V_code = vector_potential_integral_triangle(r_obs, v0, v1, v2)
V_ref = V_scipy(r_obs)
err = np.linalg.norm(V_code - V_ref) / max(np.linalg.norm(V_ref), 1e-15)
print(f"  h=0.000 (on tri): code={V_code}, ref={V_ref}, rel_err={err:.2e}")

# Test at a quadrature point (what the actual assembly uses)
qpts, qwts = tri_quadrature(7)
lam0 = 1 - qpts[:, 0] - qpts[:, 1]
tri_qpts = lam0[:, None] * v0 + qpts[:, 0:1] * v1 + qpts[:, 1:2] * v2
print("\n=== At actual quadrature points ===")
for iq in range(min(4, len(qwts))):
    r_obs = tri_qpts[iq]
    V_code = vector_potential_integral_triangle(r_obs, v0, v1, v2)
    V_ref = V_scipy(r_obs)
    err = np.linalg.norm(V_code - V_ref) / max(np.linalg.norm(V_ref), 1e-15)
    print(f"  qpt[{iq}]={r_obs}: rel_err={err:.2e}")
    if err > 0.01:
        print(f"    code={V_code}")
        print(f"     ref={V_ref}")

# Now check how this error affects L singular corrections
# The singular correction uses:
# fn/R = sign * coeff * (V - free * P) where V = ∫ r'/R dS', P = ∫ 1/R dS'
# f_test · (fn/R) integrated over test triangle = vec_integral
print("\n=== Impact on L singular correction ===")
# Simulate for a single RWG function
free_p = v2  # free vertex of T+
coeff = 1.0 / (2 * 0.5)  # length/(2*area), assume length=1, area=0.5
for iq in range(min(4, len(qwts))):
    r_obs = tri_qpts[iq]
    P_code = potential_integral_triangle(r_obs, v0, v1, v2)
    V_code = vector_potential_integral_triangle(r_obs, v0, v1, v2)
    fn_over_R_code = coeff * (V_code - free_p * P_code)

    # Reference
    P_ref = P_code  # P is exact (Graglia)
    V_ref = V_scipy(r_obs)
    fn_over_R_ref = coeff * (V_ref - free_p * P_ref)

    err = np.linalg.norm(fn_over_R_code - fn_over_R_ref) / max(np.linalg.norm(fn_over_R_ref), 1e-15)
    print(f"  qpt[{iq}]: fn/R err={err:.2e}, |fn/R_code|={np.linalg.norm(fn_over_R_code):.4f}, |fn/R_ref|={np.linalg.norm(fn_over_R_ref):.4f}")
