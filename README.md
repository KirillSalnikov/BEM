# BEM Solver

Boundary Element Method (BEM) solver for electromagnetic scattering problems on 3D surfaces. Implements the PMCHWT (Poggio–Miller–Chang–Harrington–Wu–Tsai) formulation with RWG (Rao–Wilton–Glisson) basis functions.

## Installation

**Requirements:** Python 3.8+, NumPy, SciPy.

```bash
pip install numpy scipy
```

No further installation needed — just place `bem_core.py` in your project and import it:

```python
from bem_core import icosphere, build_rwg, assemble_L_K, assemble_pmchwt, ...
```

## Quick Start

### Arbitrary Shape from File

```python
from bem_core import load_mesh, build_rwg, assemble_pmchwt, compute_rhs_planewave
import numpy as np

# Load mesh from STL, OBJ, or Gmsh .msh file
verts, tris = load_mesh("particle.stl")  # or .obj, .msh
rwg = build_rwg(verts, tris)

# Set up physics
k_ext = 2*np.pi / wavelength
m_rel = 1.5  # refractive index
k_int = k_ext * m_rel
eta_ext = 1.0; eta_int = eta_ext / m_rel

# Assemble and solve
Z, _, _ = assemble_pmchwt(rwg, verts, tris, k_ext, k_int, eta_ext, eta_int)
b = compute_rhs_planewave(rwg, verts, tris, k_ext, eta_ext)
coeffs = np.linalg.solve(Z, b)
```

The mesh must be a **closed** triangular surface (no boundary edges). You can generate meshes with [Gmsh](https://gmsh.info/), [MeshLab](https://meshlab.net/), or any CAD tool that exports STL/OBJ.

### PEC Sphere (EFIE)

```python
import numpy as np
from bem_core import (icosphere, build_rwg, tri_quadrature,
                      assemble_L_K, compute_far_field, compute_cross_sections)

radius = 1.0; k = 2*np.pi; eta = 377.0
E0 = np.array([1, 0, 0]); k_hat = np.array([0, 0, 1])

# Mesh and basis
verts, tris = icosphere(radius, refinements=3)
rwg = build_rwg(verts, tris)
N = rwg['N']

# Assemble L operator
L, K_op = assemble_L_K(rwg, verts, tris, k)

# RHS (plane wave excitation)
quad_pts, quad_wts = tri_quadrature(7)
lam0 = 1 - quad_pts[:,0] - quad_pts[:,1]
def get_qpts(ti):
    t = tris[ti]; v0 = verts[t[:,0]]; v1 = verts[t[:,1]]; v2 = verts[t[:,2]]
    return (np.einsum('q,ni->nqi', lam0, v0) +
            np.einsum('q,ni->nqi', quad_pts[:,0], v1) +
            np.einsum('q,ni->nqi', quad_pts[:,1], v2))
qp = get_qpts(rwg['tri_p']); qm = get_qpts(rwg['tri_m'])
V_E = np.zeros(N, dtype=complex)
for qpts, free, area, sign in [(qp, rwg['free_p'], rwg['area_p'], +1),
                                 (qm, rwg['free_m'], rwg['area_m'], -1)]:
    f = sign * (rwg['length'][:,None,None] / (2*area[:,None,None])) * (qpts - free[:,None,:])
    jw = area[:,None] * quad_wts[None,:]
    phase = np.exp(1j * k * np.einsum('i,nqi->nq', k_hat, qpts))
    V_E += np.sum(np.einsum('nqi,i->nq', f, E0) * phase * jw, axis=1)

# Solve EFIE: η·L·J = V_E
J = np.linalg.solve(eta * L, V_E)
M = np.zeros(N, dtype=complex)

# Cross sections
Q_ext, Q_sca = compute_cross_sections(rwg, verts, tris, J, M, k, eta, radius)
print(f"Q_ext = {Q_ext:.4f}, Q_sca = {Q_sca:.4f}")
```

### Dielectric Sphere (PMCHWT)

```python
import numpy as np
from bem_core import (icosphere, build_rwg, assemble_pmchwt,
                      compute_rhs_planewave, compute_cross_sections)

radius = 1.0; k_ext = 2*np.pi; eta_ext = 377.0
m_rel = 1.5  # relative refractive index
k_int = k_ext * m_rel; eta_int = eta_ext / m_rel

verts, tris = icosphere(radius, refinements=3)
rwg = build_rwg(verts, tris)
N = rwg['N']

# Assemble PMCHWT system
Z = assemble_pmchwt(rwg, verts, tris, k_ext, k_int, eta_ext, eta_int)
b = compute_rhs_planewave(rwg, verts, tris, k_ext, eta_ext)

# Solve
coeffs = np.linalg.solve(Z, b)
J = coeffs[:N]; M = coeffs[N:]

# Cross sections (sM=-1 for PMCHWT convention)
Q_ext, Q_sca = compute_cross_sections(rwg, verts, tris, J, M, k_ext, eta_ext, radius, sM=-1)
print(f"Q_ext = {Q_ext:.4f}, Q_sca = {Q_sca:.4f}")
```

## Physics Conventions

- Time convention: e^{+iωt}
- Green's function: G(R) = e^{ikR} / (4πR)
- L operator: L_mn = ik ∫∫ f_m·f_n G dS dS' − (i/k) ∫∫ (∇·f_m)(∇·f_n) G dS dS'
- K operator: K_mn = ∫∫ f_m · (∇G × f_n) dS dS'
- PMCHWT matrix: `[ηL, −K; +K, L/η] · [J; M] = +b`
- Far field: F = −ik/(4π) [η J̃_⊥ + sM (r̂ × M̃)], with sM = −1 for PMCHWT
- Optical theorem: Q_ext = (4π/k) Im(F_θ(θ=0)) / (πR²)

## API Reference

### Mesh Loading

#### `load_mesh(filename)`

Load a triangular surface mesh from file. The mesh must be closed (no boundary edges).

| Parameter | Type | Description |
|---|---|---|
| `filename` | str | Path to mesh file (.stl, .obj, or .msh) |

**Supported formats:**
- **STL** — binary and ASCII
- **OBJ** — triangular and quad faces (quads split into 2 triangles)
- **Gmsh .msh** — version 2 and 4, extracts triangular elements

**Returns:** `(verts, tris)` — vertex array (V×3) and triangle index array (T×3).

### Mesh Generation

#### `icosphere(radius=1.0, refinements=2)`

Generate a triangulated sphere (icosphere) by recursive subdivision of an icosahedron.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `radius` | float | 1.0 | Sphere radius |
| `refinements` | int | 2 | Number of subdivision levels. 0→20 tris, 1→80, 2→320, 3→1280, 4→5120 |

**Returns:** `(verts, tris)` — vertex array (V×3) and triangle index array (T×3).

### Basis Functions

#### `build_rwg(verts, tris)`

Build RWG (Rao–Wilton–Glisson) basis function data from a triangle mesh. Each interior edge defines one RWG basis function spanning its two adjacent triangles (T+ and T−).

| Parameter | Type | Description |
|---|---|---|
| `verts` | ndarray (V×3) | Vertex coordinates |
| `tris` | ndarray (T×3) | Triangle vertex indices |

**Returns:** dict with keys:
| Key | Shape | Description |
|---|---|---|
| `N` | int | Number of RWG basis functions (interior edges) |
| `tri_p` | (N,) | T+ triangle index for each basis |
| `tri_m` | (N,) | T− triangle index for each basis |
| `free_p` | (N,3) | Free vertex position in T+ |
| `free_m` | (N,3) | Free vertex position in T− |
| `area_p` | (N,) | Area of T+ |
| `area_m` | (N,) | Area of T− |
| `length` | (N,) | Edge length |

### Quadrature

#### `tri_quadrature(order=7)`

Symmetric triangle quadrature rule.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `order` | int | 7 | Quadrature order. Supported: 1 (1 pt), 3 (3 pts), 7 (7 pts) |

**Returns:** `(pts, wts)` — quadrature points in barycentric (ξ₁,ξ₂) coordinates and weights (sum to 0.5).

### Analytical Integrals

#### `potential_integral_triangle(r_obs, v0, v1, v2)`

Compute ∫_T 1/|r_obs − r'| dS' analytically (Graglia 1993). Works for observation points on or off the triangle plane.

| Parameter | Type | Description |
|---|---|---|
| `r_obs` | ndarray (3,) | Observation point |
| `v0, v1, v2` | ndarray (3,) | Triangle vertices |

**Returns:** scalar value of the integral.

#### `gradient_potential_integral_triangle(r_obs, v0, v1, v2)`

Compute ∇_r ∫_T 1/|r − r'| dS' analytically.

**Returns:** 3-vector (gradient w.r.t. observation point).

#### `vector_potential_integral_triangle(r_obs, v0, v1, v2)`

Compute ∫_T r'/|r_obs − r'| dS' using identity: ∫ r'/R dS' = r_obs·P(r_obs) − ∇W(r_obs), where P = ∫1/R, W = ∫R.

**Returns:** 3-vector.

### Operator Assembly

#### `assemble_L_K(rwg, verts, tris, k, quad_order=7)`

Assemble L and K operator matrices with singularity extraction.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `rwg` | dict | — | RWG data from `build_rwg` |
| `verts` | ndarray | — | Vertex coordinates |
| `tris` | ndarray | — | Triangle indices |
| `k` | float | — | Wavenumber |
| `quad_order` | int | 7 | Quadrature order for source integration |

**Returns:** `(L, K)` — complex matrices of size (N×N).

Singularity extraction: decomposes G = G_0 + G_smooth where G_0 = 1/(4πR) is integrated analytically (Graglia) and G_smooth = (e^{ikR}−1)/(4πR) is smooth and handled by standard quadrature.

#### `assemble_L_K_snc(rwg, verts, tris, k, quad_order=7)`

Same as `assemble_L_K` but with SNC (n̂×RWG) test functions. SNC testing eliminates the div-div term since ∇·(n̂×f) = 0 on flat triangles, giving a better-conditioned L operator.

**Returns:** `(L, K)` — complex matrices (N×N).

### PMCHWT System

#### `assemble_pmchwt(rwg, verts, tris, k_ext, k_int, eta_ext, eta_int, quad_order=7)`

Assemble the full PMCHWT system matrix `[ηL, −K; +K, L/η]` of size (2N×2N).

| Parameter | Type | Description |
|---|---|---|
| `rwg` | dict | RWG data |
| `verts, tris` | ndarray | Mesh |
| `k_ext` | float | Exterior wavenumber |
| `k_int` | float | Interior wavenumber |
| `eta_ext` | float | Exterior impedance |
| `eta_int` | float | Interior impedance |
| `quad_order` | int | Quadrature order (default 7) |

**Returns:** `Z` — complex matrix (2N×2N).

#### `assemble_pmchwt_snc(rwg, verts, tris, k_ext, k_int, eta_ext, eta_int, quad_order=7)`

Same as `assemble_pmchwt` but with SNC testing.

### Right-Hand Side

#### `compute_rhs_planewave(rwg, verts, tris, k_ext, eta_ext, E0, k_hat, quad_order=7)`

Compute PMCHWT RHS for x-polarized plane wave incidence.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `E0` | ndarray (3,) | [1,0,0] | Electric field polarization vector |
| `k_hat` | ndarray (3,) | [0,0,1] | Wave propagation direction |
| `quad_order` | int | 7 | Quadrature order |

**Returns:** `b` — complex vector (2N,), with b[:N] = ⟨f, E_inc⟩ and b[N:] = ⟨f, H_inc⟩.

#### `compute_rhs_planewave_snc(rwg, verts, tris, k_ext, eta_ext, E0, k_hat, quad_order=7)`

Same as `compute_rhs_planewave` but with SNC testing.

### Far-Field and Cross Sections

#### `compute_far_field(rwg, verts, tris, coeffs_J, coeffs_M, k_ext, eta_ext, theta_arr, phi=0.0, sM=-1, quad_order=7)`

Compute far-field scattering amplitude F(θ,φ).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `coeffs_J` | ndarray (N,) | — | Electric current RWG coefficients |
| `coeffs_M` | ndarray (N,) | — | Magnetic current RWG coefficients (zeros for PEC) |
| `k_ext` | float | — | Exterior wavenumber |
| `eta_ext` | float | — | Exterior impedance |
| `theta_arr` | ndarray | — | Polar angles (radians) |
| `phi` | float | 0.0 | Azimuthal angle (radians) |
| `sM` | int | −1 | Sign for magnetic current term. −1 for PMCHWT, +1 for PEC (irrelevant when M=0) |
| `quad_order` | int | 7 | Quadrature order |

**Returns:** `(F_theta, F_phi)` — complex arrays, same length as `theta_arr`.

#### `compute_cross_sections(rwg, verts, tris, coeffs_J, coeffs_M, k_ext, eta_ext, radius, sM=-1, ntheta=181, nphi=36, quad_order=7)`

Compute extinction and scattering efficiency factors Q_ext and Q_sca.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `radius` | float | — | Object radius (for normalization by geometric cross section πR²) |
| `sM` | int | −1 | Sign convention (see `compute_far_field`) |
| `ntheta` | int | 181 | Number of θ points for integration |
| `nphi` | int | 36 | Number of φ points for integration |
| `quad_order` | int | 7 | Quadrature order |

**Returns:** `(Q_ext, Q_sca)` — extinction and scattering efficiencies.

- Q_ext via optical theorem: Q_ext = (4π/k) Im(F_θ(0)) / (πR²)
- Q_sca via numerical integration of |F|² over the sphere

### Mueller Matrix

#### `compute_amplitude_matrix(rwg, verts, tris, k_ext, eta_ext, theta_arr, Z_lu=None, Z=None, k_hat=..., sM=-1, quad_order=7)`

Compute the 2×2 amplitude scattering matrix S(θ) by solving for two orthogonal incident polarizations. Convention follows Bohren & Huffman.

| Parameter | Type | Description |
|---|---|---|
| `Z_lu` | tuple | Pre-computed LU factorization (`scipy.linalg.lu_factor(Z)`) |
| `Z` | ndarray | System matrix (used if `Z_lu` not provided) |
| `k_hat` | ndarray (3,) | Incident wave direction (default [0,0,1]) |

**Returns:** dict with keys `'S1','S2','S3','S4'` (complex arrays) and `'theta'`.

#### `amplitude_to_mueller(S1, S2, S3, S4)`

Convert amplitude matrix elements to 4×4 Mueller matrix (Bohren & Huffman, eq. 3.16).

**Returns:** `M` — array of shape (4, 4, N_theta).

#### `compute_mueller_matrix(rwg, verts, tris, k_ext, eta_ext, theta_arr, Z_lu=None, Z=None, ...)`

Compute normalized Mueller matrix M(θ) for a fixed particle orientation. M[0,0] = dσ/dΩ for unpolarized light.

**Returns:** `M` — array of shape (4, 4, N_theta), normalized by 1/k².

### Orientation Averaging

#### `orientation_average_mueller(rwg, verts, tris, k_ext, eta_ext, theta_arr, Z_lu=None, Z=None, sM=-1, n_alpha=8, n_beta=8, n_gamma=1)`

Compute orientation-averaged Mueller matrix ⟨M(θ)⟩ over Euler angles (α, β, γ). The system matrix Z is factorized once and reused for all orientations (only the RHS changes).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_alpha` | int | 8 | Number of α quadrature points (rotation around z) |
| `n_beta` | int | 8 | Number of β points (Gauss-Legendre on [0,π]) |
| `n_gamma` | int | 1 | Number of γ points (1 suffices for axially symmetric particles) |

Total number of orientations: n_alpha × n_beta × n_gamma. Each orientation requires 2 back-substitutions (parallel and perpendicular polarization).

**Returns:** `M_avg` — array of shape (4, 4, N_theta).

**Example:**
```python
from scipy.linalg import lu_factor
Z, _, _ = assemble_pmchwt(rwg, verts, tris, k_ext, k_int, eta_ext, eta_int)
Z_lu = lu_factor(Z)

theta = np.linspace(0.01, np.pi-0.01, 181)
M_avg = orientation_average_mueller(rwg, verts, tris, k_ext, eta_ext, theta,
                                     Z_lu=Z_lu, sM=-1, n_alpha=16, n_beta=8, n_gamma=8)
# M_avg[0,0,:] = <dσ/dΩ> for unpolarized light
# M_avg[0,1,:] / M_avg[0,0,:] = degree of linear polarization
```

## OpenCL Acceleration

The optional module `bem_opencl.py` provides GPU/CPU-accelerated assembly and GMRES solver using OpenCL with float32 kernels (optimal for consumer GPUs like AMD RDNA 4 with 1/32 fp64 rate).

### Installation

```bash
pip install pyopencl
```

**CPU OpenCL runtime** (works without a GPU):
```bash
# Ubuntu/Debian
sudo apt-get install pocl-opencl-icd

# or via conda
conda install -c conda-forge pocl
```

**AMD GPU (ROCm)**:
```bash
sudo apt-get install rocm-opencl-runtime
```

**NVIDIA GPU**:
```bash
sudo apt-get install nvidia-opencl-icd
```

Verify installation:
```bash
python -c "import pyopencl; print(pyopencl.get_platforms())"
```

### Usage

Drop-in replacements for the CPU functions:

```python
from bem_opencl import assemble_pmchwt_ocl, solve_gmres_ocl

# Assemble on GPU (float32 kernels, singular corrections on CPU in float64)
Z = assemble_pmchwt_ocl(rwg, verts, tris, k_ext, k_int, eta_ext, eta_int)

# RHS (same as CPU version)
from bem_core import compute_rhs_planewave
b = compute_rhs_planewave(rwg, verts, tris, k_ext, eta_ext)

# GMRES with GPU matvec + block-diagonal preconditioner
coeffs = solve_gmres_ocl(Z, b, tol=1e-6, maxiter=200)
```

### Performance (ref=3, 1920 RWG basis functions)

| Backend | Assembly | GMRES | Total |
|---|---|---|---|
| NumPy (CPU) | 87 s | 7 s | 94 s |
| POCL (CPU OpenCL) | 12.6 s | 2.5 s | 15 s |
| GPU (estimated) | ~1 s | ~0.3 s | ~2 s |

## Validation

Tested against Mie theory for:
- **PEC sphere** (ka=1): Q_sca converges to analytical value (1.2% error at refinement=3 / 1280 triangles)
- **Dielectric sphere** (m=1.5, ka=1): Q_ext and Q_sca converge (1.4% error at refinement=3)
- **Optical theorem** satisfied: Q_ext = Q_sca for PEC (lossless) scatterers

## License

MIT
