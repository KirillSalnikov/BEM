#ifndef BEM_PRECOND_H
#define BEM_PRECOND_H

#include "types.h"
#include <vector>
#include <complex>

struct BemFmmOperator;

enum PrecondMode {
    PREC_NONE = 0,
    PREC_DIAG = 1,      // Diagonal scaling (Z_diag^{-1})
    PREC_ILU0 = 2,      // ILU(0) on near-field sparse matrix
    PREC_NEARLU = 3      // Full LU on near-field sparse matrix (small N only)
};

// Preconditioner for PMCHWT BEM system.
//
// The 2N×2N system has structure:
//   [ eta_e*L_ext + eta_i*L_int    -(K_ext + K_int)        ] [J]
//   [  K_ext + K_int           L_ext/eta_e + L_int/eta_i   ] [M]
//
// Right-preconditioning in GMRES: solve Z*M^{-1} * (M*x) = b.
struct NearFieldPrecond {
    int N;      // RWG count
    int N2;     // 2*N (system size)
    PrecondMode mode;

    // Diagonal preconditioner (PREC_DIAG)
    std::vector<cdouble> diag_val;         // (2N) diagonal entries of near-field Z

    // Sparse 2N×2N matrix in CSR format
    std::vector<int> csr_row_ptr;       // (2N+1)
    std::vector<int> csr_col_idx;       // (nnz_total)
    std::vector<cdouble> csr_val;       // (nnz_total)

    // For each row i, index into csr_col_idx where the diagonal element is
    std::vector<int> diag_ptr;          // (2N)

    // Full LU factorization (PREC_NEARLU, small N only)
    std::vector<cdouble> lu_dense;      // (2N × 2N) column-major
    std::vector<int> lu_piv;            // (2N) pivot indices

    // Build preconditioner from near-field BEM entries
    // radius_mult: cell_size = radius_mult * avg_extent (default 2.0)
    void build(BemFmmOperator& op, PrecondMode mode, double radius_mult = 2.0);

    // Apply: z = M^{-1} * r
    void apply(const cdouble* r, cdouble* z) const;
};

#endif // BEM_PRECOND_H
