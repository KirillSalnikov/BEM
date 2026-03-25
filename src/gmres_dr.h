#ifndef GMRES_DR_H
#define GMRES_DR_H

#include <complex>
typedef std::complex<double> cdouble;

class BemFmmOperator;
class NearFieldPrecond;

// Persistent context for GCRO-DR recycling across solves
// (same operator, different RHS — e.g. orientation averaging)
struct GcroDrContext;
GcroDrContext* gcro_dr_create(int n, int k);
void gcro_dr_destroy(GcroDrContext* ctx);

// GCRO-DR: Deflated restarting GMRES for paired systems
// If ctx != nullptr, reuses/updates recycling vectors across calls.
// Returns total number of matvec evaluations
int gmres_dr_paired(BemFmmOperator& op,
    const cdouble* b1, const cdouble* b2,
    cdouble* x1, cdouble* x2,
    int restart = 100, int ndefl = 20,
    double tol = 1e-4, int maxiter = 300,
    bool verbose = true, NearFieldPrecond* precond = nullptr,
    GcroDrContext* ctx = nullptr);

#endif
