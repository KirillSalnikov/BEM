# BEM-CUDA: GPU-Accelerated Boundary Element Method for Light Scattering

CUDA/C++ implementation of the Boundary Element Method (BEM) with PMCHWT formulation for electromagnetic scattering by dielectric particles.

## Features

- **Dense solver** (LU factorization via cuSOLVER) for small problems (N < 10000)
- **FMM+GMRES solver** (plane-wave MLFMA) for large problems (N up to 100000+)
  - Multilevel Fast Multipole Algorithm with GPU-accelerated kernels
  - P2P near-field with float32 transcendentals + double accumulation
  - CSR-optimized M2L translations with shared memory transfer reuse
  - Batched evaluation: two charge vectors in a single tree traversal
  - P2P/FMM pipeline overlap via CUDA streams
- **Preconditioners**: diagonal, ILU(0), near-field LU
- **GMRES variants**: standard, paired (two RHS), GCRO-DR (deflated restarting)
- **Orientation averaging** with Gauss-Legendre quadrature
- **Mueller matrix** computation from far-field amplitudes
- **Icosphere meshes** with configurable refinement

## Requirements

- CUDA Toolkit 11.0+ (tested with 12.8)
- GPU with compute capability 7.0+ (tested on RTX 3080 Ti, sm_86)
- g++ with C++11 support

## Build

```bash
make -j$(nproc)
```

Set GPU architecture in `Makefile` (default: `sm_86`):
```makefile
ARCH = -arch=sm_86
```

## Usage

### Dense solver (small N)
```bash
bin/bem_cuda --ka 5 --ref 3 --ri 1.3116 0 --single --out result.json
```

### FMM+GMRES solver (large N)
```bash
bin/bem_cuda --ka 10 --ref 5 --ri 1.3116 0 --fmm --fmm-digits 3 --max-leaf 32 --prec diag --single --out result.json
```

### Full orientation averaging
```bash
bin/bem_cuda --ka 5 --ref 4 --ri 1.3116 0 --fmm --prec diag --orient 8 8 1 --out result.json
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--ka F` | Size parameter (required) | — |
| `--ri RE IM` | Complex refractive index | 1.3116 0 |
| `--ref N` | Icosphere refinement level | 3 |
| `--single` | Single orientation (no averaging) | off |
| `--orient NA NB NG` | Orientation quadrature grid | 8 8 1 |
| `--fmm` | Use FMM+GMRES instead of dense LU | off |
| `--fmm-digits N` | FMM accuracy (digits) | 3 |
| `--max-leaf N` | Max particles per octree leaf | 64 |
| `--prec TYPE` | Preconditioner: `diag`, `ilu0`, `nearlu` | none |
| `--gmres-restart N` | GMRES restart parameter | 100 |
| `--gmres-tol F` | GMRES relative tolerance | 1e-4 |
| `--ntheta N` | Number of scattering angles | 181 |
| `--quad N` | Triangle quadrature order: 4, 7, 13 | 7 |
| `--out FILE` | Output JSON file | result.json |

## Output

JSON file containing:
- `mueller`: 4×4×Nθ Mueller matrix array
- `theta`: scattering angles (degrees)
- `ka`, `ri_re`, `ri_im`: input parameters

## Mesh sizes

| Refinement | Triangles | RWG basis (N) | Suitable ka |
|-----------|-----------|---------------|-------------|
| 2 | 320 | 480 | 1–2 |
| 3 | 1280 | 1920 | 2–5 |
| 4 | 5120 | 7680 | 5–10 |
| 5 | 20480 | 30720 | 10–20 |
| 6 | 81920 | 122880 | 20–40 |

Rule of thumb: ~10 elements per wavelength → N ≈ 8·ka².

## Architecture

```
src/
├── main.cpp          # CLI entry point
├── mesh.cpp/h        # Icosphere mesh generation
├── rwg.cpp/h         # RWG basis functions
├── quadrature.h      # Dunavant triangle quadrature
├── graglia.h         # Singular integrals (Graglia)
├── rhs.cpp/h         # Plane-wave RHS
├── assembly.cu/h     # Dense Z-matrix assembly (GPU)
├── pmchwt.cu/h       # PMCHWT system operators
├── solver.cu/h       # Dense LU solver
├── octree.h          # Adaptive octree (CPU)
├── sphere_quad.h     # Sphere quadrature for FMM
├── fmm.cu/h          # FMM engine (P2M, M2M, M2L, L2L, L2P)
├── p2p.cu/h          # P2P near-field CUDA kernels
├── bem_fmm.cu/h      # BEM-FMM coupling (L/K operators, matvec)
├── gmres.cu/h        # GMRES(m) solver
├── block_gmres.cu/h  # Paired GMRES (two RHS)
├── gmres_dr.cu/h     # GCRO-DR (deflated restarting)
├── precond.cu/h      # Preconditioners
├── farfield.cu/h     # Far-field + Mueller matrix (GPU)
├── orient.cpp/h      # Orientation averaging
├── output.cpp/h      # JSON output
└── types.h           # Common types (cdouble, fmm_real)

python/               # Python reference implementations
├── bem_core.py       # Dense BEM solver
├── bem_fmm.py        # FMM-accelerated BEM
├── bem_opencl.py     # OpenCL acceleration
└── test_fmm_bem.py   # Validation tests
```

## Performance (RTX 3080 Ti, sphere m=1.3116)

| ka | N | Mode | Solve time | ms/matvec |
|----|------|------|-----------|-----------|
| 1 | 480 | Dense LU | 0.1s | — |
| 2 | 1920 | Dense LU | 0.5s | — |
| 5 | 7680 | FMM+GMRES | 84s | 342 |
| 10 | 30720 | FMM+GMRES | ~600s | ~400 |

## References

- PMCHWT formulation: Rao, Wilton, Glisson (1982)
- Plane-wave MLFMA: Chew, Jin, Michielssen, Song (2001)
- Graglia singular integrals: Graglia (1993)
