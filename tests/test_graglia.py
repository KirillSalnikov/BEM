"""Test potential_integral_triangle against known analytical results and high-order quadrature."""
import numpy as np
from bem_core import potential_integral_triangle, tri_quadrature

# Simple right triangle in the xy-plane
v0 = np.array([0.0, 0.0, 0.0])
v1 = np.array([1.0, 0.0, 0.0])
v2 = np.array([0.0, 1.0, 0.0])

# Test 1: Point above the centroid at various heights
centroid = (v0 + v1 + v2) / 3.0
print("=== Test 1: Point above centroid of unit right triangle ===")
for h in [2.0, 1.0, 0.5, 0.1, 0.01, 0.001]:
    r_obs = centroid + np.array([0, 0, h])
    P_anal = potential_integral_triangle(r_obs, v0, v1, v2)

    # High-order numerical quadrature
    qpts, qwts = tri_quadrature(7)
    lam0 = 1 - qpts[:, 0] - qpts[:, 1]
    pts = lam0[:, None] * v0 + qpts[:, 0:1] * v1 + qpts[:, 1:2] * v2
    area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
    R_vals = np.linalg.norm(pts - r_obs, axis=1)
    P_num = area * np.sum(qwts / R_vals)

    print(f"  h={h:.3f}: analytical={P_anal:.8f}, numerical={P_num:.8f}, ratio={P_anal/P_num:.6f}")

# Test 2: Known exact result - integral of 1/R over a square [0,1]×[0,1] at height h
# For a square, this is known. For a triangle, let's verify consistency.
print("\n=== Test 2: Point at (0.5, 0.25, h) ===")
for h in [1.0, 0.1, 0.01]:
    r_obs = np.array([0.5, 0.25, h])
    P_anal = potential_integral_triangle(r_obs, v0, v1, v2)
    qpts, qwts = tri_quadrature(7)
    lam0 = 1 - qpts[:, 0] - qpts[:, 1]
    pts = lam0[:, None] * v0 + qpts[:, 0:1] * v1 + qpts[:, 1:2] * v2
    area = 0.5
    R_vals = np.linalg.norm(pts - r_obs, axis=1)
    P_num = area * np.sum(qwts / R_vals)
    print(f"  h={h:.3f}: analytical={P_anal:.8f}, numerical={P_num:.8f}, ratio={P_anal/P_num:.6f}")

# Test 3: Point far away (should converge to area/R)
print("\n=== Test 3: Far field (should → area/R) ===")
for dist in [10.0, 100.0]:
    r_obs = np.array([0.0, 0.0, dist])
    P_anal = potential_integral_triangle(r_obs, v0, v1, v2)
    area = 0.5
    P_approx = area / dist
    print(f"  dist={dist:.0f}: analytical={P_anal:.8f}, area/R={P_approx:.8f}, ratio={P_anal/P_approx:.6f}")

# Test 4: Compare with Wilton et al. reference implementation
# The formula should be: sum over edges of
#   d_i * ln((R+ + s+)/(R- + s-)) - |h| * (atan2(d*s+, R0^2+|h|*R+) - atan2(d*s-, R0^2+|h|*R-))
print("\n=== Test 4: Manual edge-by-edge computation ===")
r_obs = centroid + np.array([0, 0, 0.01])
n_tri = np.cross(v1 - v0, v2 - v0)
n_norm = np.linalg.norm(n_tri)
n_hat = n_tri / n_norm
h = np.dot(r_obs - v0, n_hat)
print(f"  h = {h:.6f}")
print(f"  n_hat = {n_hat}")

vertices = [v0, v1, v2]
total = 0.0
for i in range(3):
    vi = vertices[i]
    vj = vertices[(i + 1) % 3]
    edge = vj - vi
    l_edge = np.linalg.norm(edge)
    t_hat = edge / l_edge
    m_hat = np.cross(n_hat, t_hat)

    d = np.dot(r_obs - vi, m_hat)
    s_plus = np.dot(vj - r_obs, t_hat)
    s_minus = np.dot(vi - r_obs, t_hat)
    R_plus = np.linalg.norm(vj - r_obs)
    R_minus = np.linalg.norm(vi - r_obs)
    R0_sq = d**2 + h**2

    log_term = d * np.log((R_plus + s_plus) / (R_minus + s_minus))
    atan_term = abs(h) * (np.arctan2(d * s_plus, R0_sq + abs(h) * R_plus) -
                           np.arctan2(d * s_minus, R0_sq + abs(h) * R_minus))

    print(f"  Edge {i}: vi={vi}, vj={vj}")
    print(f"    t_hat={t_hat}, m_hat={m_hat}")
    print(f"    d={d:.6f}, s-={s_minus:.6f}, s+={s_plus:.6f}")
    print(f"    R-={R_minus:.6f}, R+={R_plus:.6f}")
    print(f"    log_term={log_term:.8f}, atan_term={atan_term:.8f}")
    print(f"    contribution={log_term - atan_term:.8f}")
    total += log_term - atan_term

print(f"  Total: {total:.8f}")
print(f"  Function: {potential_integral_triangle(r_obs, v0, v1, v2):.8f}")

# Test 5: Use very high-order quadrature for h=0.01
print("\n=== Test 5: High-order numerical for h=0.01 above centroid ===")
r_obs = centroid + np.array([0, 0, 0.01])
for order in [4, 7]:
    qpts, qwts = tri_quadrature(order)
    lam0 = 1 - qpts[:, 0] - qpts[:, 1]
    pts = lam0[:, None] * v0 + qpts[:, 0:1] * v1 + qpts[:, 1:2] * v2
    area = 0.5
    R_vals = np.linalg.norm(pts - r_obs, axis=1)
    P_num = area * np.sum(qwts / R_vals)
    print(f"  order={order} ({len(qwts)} pts): P_num={P_num:.8f}")

# Also try adaptive numerical integration with scipy
from scipy import integrate

def integrand(u, v):
    pt = (1 - u - v) * v0 + u * v1 + v * v2
    R = np.linalg.norm(pt - r_obs)
    return 1.0 / R if R > 1e-15 else 0.0

# Area element = 2*area (Jacobian of (u,v) -> triangle)
P_scipy, err = integrate.dblquad(integrand, 0, 1, 0, lambda u: 1 - u)
P_scipy *= 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))  # Wait, the Jacobian IS 2*area
# Actually for parametric triangle with r = (1-u-v)*v0 + u*v1 + v*v2,
# dS = |∂r/∂u × ∂r/∂v| du dv = |(v1-v0)×(v2-v0)| du dv = 2*area du dv
# So ∫_T f dS = 2*area ∫∫ f du dv
area2 = np.linalg.norm(np.cross(v1 - v0, v2 - v0))  # = 2*area
P_scipy2, err2 = integrate.dblquad(integrand, 0, 1, 0, lambda u: 1 - u)
P_scipy2 *= area2
print(f"  scipy adaptive: P={P_scipy2:.8f} (err_est={err2*area2:.2e})")
P_anal = potential_integral_triangle(r_obs, v0, v1, v2)
print(f"  analytical:     P={P_anal:.8f}")
print(f"  ratio (anal/scipy): {P_anal / P_scipy2:.6f}")
