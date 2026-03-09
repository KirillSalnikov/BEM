"""Numerically verify the surface divergence of RWG basis functions."""
import numpy as np

# Simple triangle
V0 = np.array([0.0, 0.0, 0.0])
V1 = np.array([1.0, 0.0, 0.0])
V2 = np.array([0.0, 1.0, 0.0])
area = 0.5

# RWG basis: edge V1-V2, free vertex V0
# f(r) = l/(2A) * (r - V0) on this triangle
l_edge = np.linalg.norm(V2 - V1)  # sqrt(2)
print(f"Edge length: {l_edge:.6f}")
print(f"Area: {area:.6f}")
print(f"l/(2A) = {l_edge / (2*area):.6f}")
print(f"l/A = {l_edge / area:.6f}")

# Numerical surface divergence using finite differences
# On the triangle surface (z=0), r = (x, y, 0)
# f(x,y) = l/(2A) * (x, y, 0)
# div_s(f) = l/(2A) * (∂x/∂x + ∂y/∂y) = l/(2A) * 2 = l/A
print(f"\nAnalytical ∇_s·f = l/(2A) × 2 = l/A = {l_edge / area:.6f}")

# Verify with numerical divergence
eps = 1e-6
r = np.array([0.3, 0.2, 0.0])
coeff = l_edge / (2 * area)

# Surface divergence = ∂f_x/∂x + ∂f_y/∂y (for flat triangle in xy-plane)
f_center = coeff * (r - V0)
fx_plus = coeff * (r + np.array([eps, 0, 0]) - V0)
fx_minus = coeff * (r - np.array([eps, 0, 0]) - V0)
fy_plus = coeff * (r + np.array([0, eps, 0]) - V0)
fy_minus = coeff * (r - np.array([0, eps, 0]) - V0)

dfx_dx = (fx_plus[0] - fx_minus[0]) / (2*eps)
dfy_dy = (fy_plus[1] - fy_minus[1]) / (2*eps)
div_numerical = dfx_dx + dfy_dy
print(f"Numerical ∇_s·f = {div_numerical:.6f}")
print(f"= l/(2A) × {div_numerical/coeff:.1f}")

# Now check: what does charge conservation require?
# For a current J = J_0 * f_n flowing on the triangle:
# The total charge flowing out of the edge per unit time = ∫_edge J · n̂ dl
# The total charge change rate = -∫_T ∇·J dS = -J_0 * ∫_T ∇·f dS

# For the RWG function, the normal current through the edge should be:
# ∫_edge f · m̂ dl = l/(2A) * ∫_edge (r-V0) · m̂ dl
# where m̂ is the edge outward normal

# Edge from V1 to V2: tangent t = (V2-V1)/|V2-V1| = (-1,1,0)/sqrt(2)
# Outward normal (in triangle plane): m = (t × n̂) = cross((−1,1,0)/√2, (0,0,1)) = (1,1,0)/√2
# Wait: outward normal should point AWAY from V0
t_hat = (V2 - V1) / l_edge
n_hat = np.array([0, 0, 1.0])  # triangle normal
m_hat = np.cross(t_hat, n_hat)  # = cross((-1,1,0)/sqrt2, (0,0,1))
print(f"\nEdge tangent: {t_hat}")
print(f"Edge outward normal: {m_hat}")
# Check: does m_hat point away from V0?
print(f"m_hat · (centroid_edge - V0) = {np.dot(m_hat, (V1+V2)/2 - V0):.4f}")

# Normal current through edge:
# ∫₀¹ f(V1+t*(V2-V1)) · m̂ * l dt
# = l/(2A) * ∫₀¹ (V1+t*(V2-V1)-V0) · m̂ * l dt
# At V1: (V1-V0) · m̂ = (1,0,0) · (1,1,0)/√2 = 1/√2
# At V2: (V2-V0) · m̂ = (0,1,0) · (1,1,0)/√2 = 1/√2
# So the integrand is constant = 1/√2
# ∫ = l²/(2A) × 1/√2 = 2/(2×0.5) × 1/√2 = 2/√2 = √2

# But wait, f · m̂ is the normal component at each point on the edge
npts = 100
edge_pts = np.array([V1 + t*(V2-V1) for t in np.linspace(0, 1, npts)])
f_at_edge = coeff * (edge_pts - V0)
fn_at_edge = np.sum(f_at_edge * m_hat, axis=1)
flux_numerical = l_edge * np.trapz(fn_at_edge, np.linspace(0, 1, npts))
print(f"\nNormal flux through edge (numerical): {flux_numerical:.6f}")

# From divergence theorem: ∫_T ∇·f dS = Σ_edges ∫_edge f·n̂ dl
# For the RWG function, the flux through the OPPOSITE edge (V1-V2) should equal ∫_T ∇·f dS
# since f=0 on the other two edges (but actually f is NOT zero on other edges!)
# Let's compute flux through all three edges

edges = [(V0, V1), (V1, V2), (V2, V0)]
edge_names = ["V0-V1", "V1-V2", "V2-V0"]
total_flux = 0
for (va, vb), name in zip(edges, edge_names):
    e = vb - va
    le = np.linalg.norm(e)
    te = e / le
    me = np.cross(te, n_hat)  # outward normal (need to verify direction)
    # Check if outward: for edge V0-V1, outward should point away from V2
    opp_vert = [V2, V0, V1]  # opposite vertex for each edge
    pts = np.array([va + t*(vb-va) for t in np.linspace(0, 1, npts)])
    f_pts = coeff * (pts - V0)
    fn_pts = np.sum(f_pts * me, axis=1)
    flux = le * np.trapz(fn_pts, np.linspace(0, 1, npts))
    print(f"  Flux through {name}: {flux:.6f}")
    total_flux += flux

div_from_flux = total_flux / area
print(f"\nTotal flux / area = {div_from_flux:.6f}")
print(f"Expected ∇_s·f = l/A = {l_edge/area:.6f}")
print(f"Code's div = l/(2A) = {l_edge/(2*area):.6f}")
