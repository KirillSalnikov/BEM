#include "gmres_dr.h"
#include "bem_fmm.h"
#include "precond.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <complex>
#include <vector>
#include <algorithm>
#include <numeric>

// Schur decomposition of mm×mm upper Hessenberg (column-major stride mm)
static void hessenberg_schur(cdouble* H, int mm, cdouble* Q, cdouble* eig)
{
    memset(Q, 0, mm*mm*sizeof(cdouble));
    for (int i = 0; i < mm; i++) Q[i+i*mm] = 1.0;
    int nn = mm;
    for (int iter = 0; iter < 60*mm && nn > 0; iter++) {
        int l = nn-1;
        while (l > 0) {
            double sub = std::abs(H[l+(l-1)*mm]);
            double diag = std::abs(H[(l-1)+(l-1)*mm]) + std::abs(H[l+l*mm]);
            if (diag < 1e-30) diag = 1e-30;
            if (sub < 1e-14 * diag) { H[l+(l-1)*mm] = 0; break; }
            l--;
        }
        if (l == nn-1) { eig[nn-1] = H[(nn-1)+(nn-1)*mm]; nn--; continue; }
        cdouble a=H[(nn-2)+(nn-2)*mm], b=H[(nn-2)+(nn-1)*mm];
        cdouble c_=H[(nn-1)+(nn-2)*mm], d=H[(nn-1)+(nn-1)*mm];
        cdouble disc = std::sqrt((a-d)*(a-d) + 4.0*b*c_);
        cdouble e1 = (a+d+disc)/2.0, e2 = (a+d-disc)/2.0;
        cdouble shift = (std::abs(e1-d) < std::abs(e2-d)) ? e1 : e2;
        for (int i = l; i < nn-1; i++) {
            cdouble x = H[i+i*mm]-shift, y = H[(i+1)+i*mm];
            double r = std::sqrt(std::norm(x)+std::norm(y));
            if (r < 1e-30) continue;
            cdouble cs = x/r, sn = y/r;
            for (int j = i; j < mm; j++) {
                cdouble t1=H[i+j*mm], t2=H[(i+1)+j*mm];
                H[i+j*mm] = std::conj(cs)*t1 + std::conj(sn)*t2;
                H[(i+1)+j*mm] = -sn*t1 + cs*t2;
            }
            for (int j = l; j <= std::min(i+2, nn-1); j++) {
                cdouble t1=H[j+i*mm], t2=H[j+(i+1)*mm];
                H[j+i*mm] = t1*cs + t2*sn;
                H[j+(i+1)*mm] = -t1*std::conj(sn) + t2*std::conj(cs);
            }
            for (int j = 0; j < mm; j++) {
                cdouble t1=Q[j+i*mm], t2=Q[j+(i+1)*mm];
                Q[j+i*mm] = t1*cs + t2*sn;
                Q[j+(i+1)*mm] = -t1*std::conj(sn) + t2*std::conj(cs);
            }
        }
    }
    for (int i = 0; i < nn; i++) eig[i] = H[i+i*mm];
}

// Householder reduction to upper Hessenberg form (column-major, nn×nn)
static void householder_hessenberg(cdouble* A, int nn, cdouble* Q)
{
    memset(Q, 0, nn*nn*sizeof(cdouble));
    for (int i = 0; i < nn; i++) Q[i+i*nn] = 1.0;

    for (int kk = 0; kk < nn-2; kk++) {
        int len = nn - kk - 1;
        std::vector<cdouble> v(len);
        for (int i = 0; i < len; i++) v[i] = A[(kk+1+i) + kk*nn];

        double sigma = 0;
        for (int i = 0; i < len; i++) sigma += std::norm(v[i]);
        sigma = std::sqrt(sigma);
        if (sigma < 1e-30) continue;

        cdouble alpha = v[0];
        double alpha_abs = std::abs(alpha);
        cdouble phase = (alpha_abs > 1e-30) ? alpha/alpha_abs : cdouble(1.0);
        v[0] += phase * sigma;

        double vnorm = 0;
        for (int i = 0; i < len; i++) vnorm += std::norm(v[i]);
        vnorm = std::sqrt(vnorm);
        if (vnorm < 1e-30) continue;
        for (int i = 0; i < len; i++) v[i] /= vnorm;

        for (int j = kk; j < nn; j++) {
            cdouble dot = 0;
            for (int i = 0; i < len; i++) dot += std::conj(v[i]) * A[(kk+1+i) + j*nn];
            for (int i = 0; i < len; i++) A[(kk+1+i) + j*nn] -= 2.0 * v[i] * dot;
        }
        for (int i = 0; i < nn; i++) {
            cdouble dot = 0;
            for (int j = 0; j < len; j++) dot += A[i + (kk+1+j)*nn] * v[j];
            for (int j = 0; j < len; j++) A[i + (kk+1+j)*nn] -= 2.0 * dot * std::conj(v[j]);
        }
        for (int i = 0; i < nn; i++) {
            cdouble dot = 0;
            for (int j = 0; j < len; j++) dot += Q[i + (kk+1+j)*nn] * v[j];
            for (int j = 0; j < len; j++) Q[i + (kk+1+j)*nn] -= 2.0 * dot * std::conj(v[j]);
        }
    }
}

// ============================================================
// GCRO-DR: Deflated restarting GMRES (Parks et al. 2006)
//
// Maintains C = A*U (orthonormal) across restarts.
// At each cycle: deflate r via C, run standard GMRES(m) with
// projected matvec w -= C*(C^H*w). No free steps, no
// approximate Hessenberg — exact Arnoldi relation throughout.
// ============================================================

struct GcroDrContext {
    cuDoubleComplex *d_C1, *d_C2, *d_U1, *d_U2;
    int k_have1, k_have2;
    int n, k;
};

GcroDrContext* gcro_dr_create(int n, int k)
{
    GcroDrContext* ctx = new GcroDrContext;
    ctx->n = n; ctx->k = k;
    ctx->k_have1 = 0; ctx->k_have2 = 0;
    ctx->d_C1 = 0; ctx->d_C2 = 0;
    ctx->d_U1 = 0; ctx->d_U2 = 0;
    if (k > 0) {
        CUDA_CHECK(cudaMalloc(&ctx->d_C1, (size_t)n*k*sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMalloc(&ctx->d_C2, (size_t)n*k*sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMalloc(&ctx->d_U1, (size_t)n*k*sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMalloc(&ctx->d_U2, (size_t)n*k*sizeof(cuDoubleComplex)));
    }
    return ctx;
}

void gcro_dr_destroy(GcroDrContext* ctx)
{
    if (!ctx) return;
    if (ctx->d_C1) cudaFree(ctx->d_C1);
    if (ctx->d_C2) cudaFree(ctx->d_C2);
    if (ctx->d_U1) cudaFree(ctx->d_U1);
    if (ctx->d_U2) cudaFree(ctx->d_U2);
    delete ctx;
}

struct GRot { cdouble c, s; int r1, r2; };

static inline void apply_grot(const GRot& g, cdouble& a, cdouble& b)
{
    cdouble t1 = a, t2 = b;
    a = std::conj(g.c)*t1 + std::conj(g.s)*t2;
    b = -g.s*t1 + g.c*t2;
}

static inline GRot make_grot(cdouble a, cdouble b, int r1, int r2)
{
    double den = std::sqrt(std::norm(a) + std::norm(b));
    GRot g;
    g.r1 = r1; g.r2 = r2;
    if (den > 1e-30) { g.c = a/den; g.s = b/den; }
    else { g.c = 1; g.s = 0; }
    return g;
}

int gmres_dr_paired(BemFmmOperator& op,
                    const cdouble* b1, const cdouble* b2,
                    cdouble* x1, cdouble* x2,
                    int restart, int ndefl,
                    double tol, int maxiter,
                    bool verbose, NearFieldPrecond* precond,
                    GcroDrContext* ctx)
{
    int n = op.system_size;
    int m = restart;
    int k = ndefl;
    if (k >= m - 2) k = m / 2;
    if (k < 0) k = 0;

    int mtot = m + 1;

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // GPU: Krylov basis V (n × mtot), workspace
    cuDoubleComplex *d_V1, *d_V2, *d_w1, *d_w2;
    cuDoubleComplex *d_htmp1, *d_htmp2;
    CUDA_CHECK(cudaMalloc(&d_V1, (size_t)n*mtot*sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_V2, (size_t)n*mtot*sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_w1, n*sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_w2, n*sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_htmp1, mtot*sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_htmp2, mtot*sizeof(cuDoubleComplex)));

    // GCRO-DR: C = orthonormal A*U (n × k), U = recycled vectors (n × k)
    cuDoubleComplex *d_C1 = 0, *d_C2 = 0;   // C = orthonormal A*U
    cuDoubleComplex *d_U1 = 0, *d_U2 = 0;   // U = recycled vectors
    cuDoubleComplex *d_Vinp1 = 0, *d_Vinp2 = 0;  // gather buffer for recycling
    bool own_CU = (ctx == nullptr);  // true = we manage C,U memory
    int k_have1 = 0, k_have2 = 0;
    if (ctx && k > 0) {
        d_C1 = ctx->d_C1; d_C2 = ctx->d_C2;
        d_U1 = ctx->d_U1; d_U2 = ctx->d_U2;
        k_have1 = ctx->k_have1; k_have2 = ctx->k_have2;
        CUDA_CHECK(cudaMalloc(&d_Vinp1, (size_t)n*m*sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMalloc(&d_Vinp2, (size_t)n*m*sizeof(cuDoubleComplex)));
    } else if (k > 0) {
        CUDA_CHECK(cudaMalloc(&d_C1, (size_t)n*k*sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMalloc(&d_C2, (size_t)n*k*sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMalloc(&d_U1, (size_t)n*k*sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMalloc(&d_U2, (size_t)n*k*sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMalloc(&d_Vinp1, (size_t)n*m*sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMalloc(&d_Vinp2, (size_t)n*m*sizeof(cuDoubleComplex)));
    }

    // Host arrays
    std::vector<cdouble> h_r1(n), h_r2(n), h_w1(n), h_w2(n);
    std::vector<cdouble> h_v1(n), h_v2(n);
    std::vector<cdouble> h_x1(n), h_x2(n);

    // Hessenberg: mtot rows × m cols, row-major H[row*m + col]
    std::vector<cdouble> H1(mtot*m, 0), H2(mtot*m, 0);
    std::vector<cdouble> H_raw1(mtot*m, 0), H_raw2(mtot*m, 0);
    std::vector<cdouble> s1(mtot, 0), s2(mtot, 0);
    std::vector<cdouble> y1(m), y2(m);
    std::vector<GRot> giv1, giv2;

    // B = C^H * A * V (k × m, row-major B[i*m+j]): projection coefficients
    // needed for solution correction x -= U*(B*y)
    std::vector<cdouble> B1(k*m, 0), B2(k*m, 0);

    memcpy(h_x1.data(), x1, n*sizeof(cdouble));
    memcpy(h_x2.data(), x2, n*sizeof(cdouble));

    bool has_x0 = false;
    for (int i = 0; i < n && !has_x0; i++)
        if (std::abs(x1[i]) > 1e-30 || std::abs(x2[i]) > 1e-30) has_x0 = true;

    int init_matvecs = 0;
    if (has_x0) {
        op.matvec_batch2(h_x1.data(), h_x2.data(), h_r1.data(), h_r2.data());
        init_matvecs = 1;
        for (int i=0; i<n; i++) { h_r1[i]=b1[i]-h_r1[i]; h_r2[i]=b2[i]-h_r2[i]; }
    } else {
        memcpy(h_r1.data(), b1, n*sizeof(cdouble));
        memcpy(h_r2.data(), b2, n*sizeof(cdouble));
    }

    double bnorm1=0, bnorm2=0;
    for (int i=0; i<n; i++) { bnorm1+=std::norm(b1[i]); bnorm2+=std::norm(b2[i]); }
    bnorm1=std::sqrt(bnorm1); bnorm2=std::sqrt(bnorm2);
    if (bnorm1<1e-30) bnorm1=1; if (bnorm2<1e-30) bnorm2=1;

    double rnorm1=0, rnorm2=0;
    for (int i=0; i<n; i++) { rnorm1+=std::norm(h_r1[i]); rnorm2+=std::norm(h_r2[i]); }
    rnorm1=std::sqrt(rnorm1); rnorm2=std::sqrt(rnorm2);

    if (verbose)
        printf("  [GCRO-DR] start: rel1=%.2e rel2=%.2e (k=%d, m=%d)\n",
               rnorm1/bnorm1, rnorm2/bnorm2, k, m);

    int total_matvecs = init_matvecs;
    bool converged1 = false, converged2 = false;

    for (int cycle = 0; cycle < maxiter; cycle++) {
        int nv1 = 0, nv2 = 0;

        // ====== INIT PHASE ======
        memset(H1.data(), 0, mtot*m*sizeof(cdouble));
        memset(H_raw1.data(), 0, mtot*m*sizeof(cdouble));
        memset(H2.data(), 0, mtot*m*sizeof(cdouble));
        memset(H_raw2.data(), 0, mtot*m*sizeof(cdouble));
        memset(s1.data(), 0, mtot*sizeof(cdouble));
        memset(s2.data(), 0, mtot*sizeof(cdouble));
        if (k > 0) { memset(B1.data(), 0, k*m*sizeof(cdouble)); memset(B2.data(), 0, k*m*sizeof(cdouble)); }
        giv1.clear(); giv2.clear();

        int k_use = std::min(k_have1, k_have2);
        if (converged1 || converged2) k_use = 0;

        // GCRO-DR deflation: x += U*(C^H*r), r -= C*(C^H*r)
        if (!converged1 && k_use > 0) {
            CUDA_CHECK(cudaMemcpy(d_w1, h_r1.data(), n*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
            cuDoubleComplex one_c={1,0}, zer_c={0,0}, neg_c={-1,0};
            // t = C^H * r
            CUBLAS_CHECK(cublasZgemv(handle, CUBLAS_OP_C, n, k_use, &one_c, d_C1, n, d_w1, 1, &zer_c, d_htmp1, 1));
            // x += U * t (download t, download U columns, accumulate on host)
            std::vector<cdouble> h_t(k_use);
            CUDA_CHECK(cudaMemcpy(h_t.data(), d_htmp1, k_use*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
            for (int j = 0; j < k_use; j++) {
                CUDA_CHECK(cudaMemcpy(h_v1.data(), (cuDoubleComplex*)d_U1 + (size_t)j*n,
                    n*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
                for (int i = 0; i < n; i++) h_x1[i] += h_t[j] * h_v1[i];
            }
            // r -= C * t
            CUBLAS_CHECK(cublasZgemv(handle, CUBLAS_OP_N, n, k_use, &neg_c, d_C1, n, d_htmp1, 1, &one_c, d_w1, 1));
            CUDA_CHECK(cudaMemcpy(h_r1.data(), d_w1, n*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
            rnorm1 = 0; for (int i=0; i<n; i++) rnorm1 += std::norm(h_r1[i]); rnorm1 = std::sqrt(rnorm1);
            if (rnorm1/bnorm1 < tol) { converged1 = true; }
        }
        if (!converged2 && k_use > 0) {
            CUDA_CHECK(cudaMemcpy(d_w2, h_r2.data(), n*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
            cuDoubleComplex one_c={1,0}, zer_c={0,0}, neg_c={-1,0};
            CUBLAS_CHECK(cublasZgemv(handle, CUBLAS_OP_C, n, k_use, &one_c, d_C2, n, d_w2, 1, &zer_c, d_htmp2, 1));
            std::vector<cdouble> h_t(k_use);
            CUDA_CHECK(cudaMemcpy(h_t.data(), d_htmp2, k_use*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
            for (int j = 0; j < k_use; j++) {
                CUDA_CHECK(cudaMemcpy(h_v2.data(), (cuDoubleComplex*)d_U2 + (size_t)j*n,
                    n*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
                for (int i = 0; i < n; i++) h_x2[i] += h_t[j] * h_v2[i];
            }
            CUBLAS_CHECK(cublasZgemv(handle, CUBLAS_OP_N, n, k_use, &neg_c, d_C2, n, d_htmp2, 1, &one_c, d_w2, 1));
            CUDA_CHECK(cudaMemcpy(h_r2.data(), d_w2, n*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
            rnorm2 = 0; for (int i=0; i<n; i++) rnorm2 += std::norm(h_r2[i]); rnorm2 = std::sqrt(rnorm2);
            if (rnorm2/bnorm2 < tol) { converged2 = true; }
        }

        if (converged1 && converged2) break;

        // V[:,0] = r / ||r|| (standard GMRES start)
        if (!converged1) {
            double inv = 1.0/rnorm1;
            for (int i=0; i<n; i++) h_v1[i] = h_r1[i]*inv;
            CUDA_CHECK(cudaMemcpy(d_V1, h_v1.data(), n*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
            s1[0] = cdouble(rnorm1);
            nv1 = 1;
        }
        if (!converged2) {
            double inv = 1.0/rnorm2;
            for (int i=0; i<n; i++) h_v2[i] = h_r2[i]*inv;
            CUDA_CHECK(cudaMemcpy(d_V2, h_v2.data(), n*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
            s2[0] = cdouble(rnorm2);
            nv2 = 1;
        }

        // ====== ARNOLDI LOOP (standard GMRES with projected matvec) ======
        int m1 = 0, m2 = 0;
        for (int j = 0; j < m; j++) {
            // Read input V[:,j] from GPU
            if (!converged1)
                CUDA_CHECK(cudaMemcpy(h_v1.data(), (cuDoubleComplex*)d_V1 + (size_t)j*n, n*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
            if (!converged2)
                CUDA_CHECK(cudaMemcpy(h_v2.data(), (cuDoubleComplex*)d_V2 + (size_t)j*n, n*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

            // Matvec
            if (!converged1 && !converged2) {
                op.matvec_batch2(h_v1.data(), h_v2.data(), h_w1.data(), h_w2.data());
                total_matvecs++;
            } else if (!converged1) {
                op.matvec(h_v1.data(), h_w1.data()); total_matvecs++;
            } else {
                op.matvec(h_v2.data(), h_w2.data()); total_matvecs++;
            }

            if (!converged1)
                CUDA_CHECK(cudaMemcpy(d_w1, h_w1.data(), n*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
            if (!converged2)
                CUDA_CHECK(cudaMemcpy(d_w2, h_w2.data(), n*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

            // GCRO-DR projection: w -= C*(C^H*w), save B[:,j] = C^H*A*v_j
            if (k_use > 0) {
                cuDoubleComplex one={1,0}, zer={0,0}, neg={-1,0};
                if (!converged1) {
                    CUBLAS_CHECK(cublasZgemv(handle, CUBLAS_OP_C, n, k_use, &one, d_C1, n, d_w1, 1, &zer, d_htmp1, 1));
                    std::vector<cdouble> bcol(k_use);
                    CUDA_CHECK(cudaMemcpy(bcol.data(), d_htmp1, k_use*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
                    for (int i = 0; i < k_use; i++) B1[i*m + j] = bcol[i];
                    CUBLAS_CHECK(cublasZgemv(handle, CUBLAS_OP_N, n, k_use, &neg, d_C1, n, d_htmp1, 1, &one, d_w1, 1));
                }
                if (!converged2) {
                    CUBLAS_CHECK(cublasZgemv(handle, CUBLAS_OP_C, n, k_use, &one, d_C2, n, d_w2, 1, &zer, d_htmp2, 1));
                    std::vector<cdouble> bcol(k_use);
                    CUDA_CHECK(cudaMemcpy(bcol.data(), d_htmp2, k_use*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
                    for (int i = 0; i < k_use; i++) B2[i*m + j] = bcol[i];
                    CUBLAS_CHECK(cublasZgemv(handle, CUBLAS_OP_N, n, k_use, &neg, d_C2, n, d_htmp2, 1, &one, d_w2, 1));
                }
            }

            // ---- Arnoldi system 1 ----
            if (!converged1) {
                int nv = nv1;
                cuDoubleComplex one={1,0}, zer={0,0}, neg={-1,0};
                CUBLAS_CHECK(cublasZgemv(handle, CUBLAS_OP_C, n, nv, &one, d_V1, n, d_w1, 1, &zer, d_htmp1, 1));
                std::vector<cdouble> hcol(nv);
                CUDA_CHECK(cudaMemcpy(hcol.data(), d_htmp1, nv*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
                for (int i = 0; i < nv; i++) { H1[i*m+j] = hcol[i]; H_raw1[i*m+j] = hcol[i]; }
                CUBLAS_CHECK(cublasZgemv(handle, CUBLAS_OP_N, n, nv, &neg, d_V1, n, d_htmp1, 1, &one, d_w1, 1));
                // Reorthogonalize
                CUBLAS_CHECK(cublasZgemv(handle, CUBLAS_OP_C, n, nv, &one, d_V1, n, d_w1, 1, &zer, d_htmp1, 1));
                CUDA_CHECK(cudaMemcpy(hcol.data(), d_htmp1, nv*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
                for (int i = 0; i < nv; i++) { H1[i*m+j] += hcol[i]; H_raw1[i*m+j] += hcol[i]; }
                CUBLAS_CHECK(cublasZgemv(handle, CUBLAS_OP_N, n, nv, &neg, d_V1, n, d_htmp1, 1, &one, d_w1, 1));

                double wn; CUBLAS_CHECK(cublasDznrm2(handle, n, d_w1, 1, &wn));
                H1[nv*m+j] = cdouble(wn);
                H_raw1[nv*m+j] = cdouble(wn);
                if (wn > 1e-30) { cuDoubleComplex sc={1.0/wn,0}; CUBLAS_CHECK(cublasZscal(handle, n, &sc, d_w1, 1)); }
                CUDA_CHECK(cudaMemcpy((cuDoubleComplex*)d_V1 + (size_t)nv*n, d_w1, n*sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));
                nv1 = nv + 1;

                for (size_t g = 0; g < giv1.size(); g++)
                    apply_grot(giv1[g], H1[giv1[g].r1*m+j], H1[giv1[g].r2*m+j]);

                // Standard Givens: zero row nv (= j+1)
                if (std::abs(H1[nv*m+j]) > 1e-30) {
                    GRot g = make_grot(H1[(nv-1)*m+j], H1[nv*m+j], nv-1, nv);
                    apply_grot(g, H1[(nv-1)*m+j], H1[nv*m+j]);
                    apply_grot(g, s1[nv-1], s1[nv]);
                    giv1.push_back(g);
                }
                m1 = j + 1;

                double res1 = std::abs(s1[j+1]);
                if (res1/bnorm1 < tol) converged1 = true;
            }

            // ---- Arnoldi system 2 ----
            if (!converged2) {
                int nv = nv2;
                cuDoubleComplex one={1,0}, zer={0,0}, neg={-1,0};
                CUBLAS_CHECK(cublasZgemv(handle, CUBLAS_OP_C, n, nv, &one, d_V2, n, d_w2, 1, &zer, d_htmp2, 1));
                std::vector<cdouble> hcol(nv);
                CUDA_CHECK(cudaMemcpy(hcol.data(), d_htmp2, nv*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
                for (int i = 0; i < nv; i++) { H2[i*m+j] = hcol[i]; H_raw2[i*m+j] = hcol[i]; }
                CUBLAS_CHECK(cublasZgemv(handle, CUBLAS_OP_N, n, nv, &neg, d_V2, n, d_htmp2, 1, &one, d_w2, 1));
                CUBLAS_CHECK(cublasZgemv(handle, CUBLAS_OP_C, n, nv, &one, d_V2, n, d_w2, 1, &zer, d_htmp2, 1));
                CUDA_CHECK(cudaMemcpy(hcol.data(), d_htmp2, nv*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
                for (int i = 0; i < nv; i++) { H2[i*m+j] += hcol[i]; H_raw2[i*m+j] += hcol[i]; }
                CUBLAS_CHECK(cublasZgemv(handle, CUBLAS_OP_N, n, nv, &neg, d_V2, n, d_htmp2, 1, &one, d_w2, 1));

                double wn; CUBLAS_CHECK(cublasDznrm2(handle, n, d_w2, 1, &wn));
                H2[nv*m+j] = cdouble(wn);
                H_raw2[nv*m+j] = cdouble(wn);
                if (wn > 1e-30) { cuDoubleComplex sc={1.0/wn,0}; CUBLAS_CHECK(cublasZscal(handle, n, &sc, d_w2, 1)); }
                CUDA_CHECK(cudaMemcpy((cuDoubleComplex*)d_V2 + (size_t)nv*n, d_w2, n*sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));
                nv2 = nv + 1;

                for (size_t g = 0; g < giv2.size(); g++)
                    apply_grot(giv2[g], H2[giv2[g].r1*m+j], H2[giv2[g].r2*m+j]);

                if (std::abs(H2[nv*m+j]) > 1e-30) {
                    GRot g = make_grot(H2[(nv-1)*m+j], H2[nv*m+j], nv-1, nv);
                    apply_grot(g, H2[(nv-1)*m+j], H2[nv*m+j]);
                    apply_grot(g, s2[nv-1], s2[nv]);
                    giv2.push_back(g);
                }
                m2 = j + 1;

                double res2 = std::abs(s2[j+1]);
                if (res2/bnorm2 < tol) converged2 = true;
            }

            if (verbose && (total_matvecs <= 3 || total_matvecs % 10 == 0)) {
                double r1 = 0, r2 = 0;
                if (!converged1) r1 = std::abs(s1[m1]);
                if (!converged2) r2 = std::abs(s2[m2]);
                printf("    GCRO-DR iter %d: res1=%.2e(%.2e) res2=%.2e(%.2e)%s%s\n",
                    total_matvecs,
                    r1, r1/bnorm1, r2, r2/bnorm2,
                    converged1?" [1]":"", converged2?" [2]":"");
            }
            if (converged1 && converged2) break;
        }

        // ====== SOLUTION UPDATE ======
        if (m1 > 0) {
            for (int i=m1-1; i>=0; i--) {
                y1[i] = s1[i];
                for (int p=i+1; p<m1; p++) y1[i] -= H1[i*m+p]*y1[p];
                y1[i] /= H1[i*m+i];
            }
            for (int j=0; j<m1; j++) {
                CUDA_CHECK(cudaMemcpy(h_v1.data(), (cuDoubleComplex*)d_V1+(size_t)j*n,
                    n*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
                for (int i=0; i<n; i++) h_x1[i] += y1[j]*h_v1[i];
            }
        }
        if (m2 > 0) {
            for (int i=m2-1; i>=0; i--) {
                y2[i] = s2[i];
                for (int p=i+1; p<m2; p++) y2[i] -= H2[i*m+p]*y2[p];
                y2[i] /= H2[i*m+i];
            }
            for (int j=0; j<m2; j++) {
                CUDA_CHECK(cudaMemcpy(h_v2.data(), (cuDoubleComplex*)d_V2+(size_t)j*n,
                    n*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
                for (int i=0; i<n; i++) h_x2[i] += y2[j]*h_v2[i];
            }
        }

        // GCRO-DR correction: x -= U*(B*y) to cancel C-component leak
        if (k_use > 0) {
            if (m1 > 0) {
                std::vector<cdouble> d_corr(k_use, 0);
                for (int i = 0; i < k_use; i++)
                    for (int jj = 0; jj < m1; jj++)
                        d_corr[i] += B1[i*m + jj] * y1[jj];
                for (int jj = 0; jj < k_use; jj++) {
                    CUDA_CHECK(cudaMemcpy(h_v1.data(), (cuDoubleComplex*)d_U1 + (size_t)jj*n,
                        n*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
                    for (int i = 0; i < n; i++) h_x1[i] -= d_corr[jj] * h_v1[i];
                }
            }
            if (m2 > 0) {
                std::vector<cdouble> d_corr(k_use, 0);
                for (int i = 0; i < k_use; i++)
                    for (int jj = 0; jj < m2; jj++)
                        d_corr[i] += B2[i*m + jj] * y2[jj];
                for (int jj = 0; jj < k_use; jj++) {
                    CUDA_CHECK(cudaMemcpy(h_v2.data(), (cuDoubleComplex*)d_U2 + (size_t)jj*n,
                        n*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
                    for (int i = 0; i < n; i++) h_x2[i] -= d_corr[jj] * h_v2[i];
                }
            }
        }

        if (converged1 && converged2) break;

        // ====== EXTRACT RECYCLING VECTORS (GCRO-DR style) ======
        int k_new = k;
        for (int sys = 0; sys < 2; sys++) {
            if (sys == 0 && converged1) continue;
            if (sys == 1 && converged2) continue;
            int m_done = (sys == 0) ? m1 : m2;
            if (m_done < k_new + 2) { if (sys==0) k_have1=0; else k_have2=0; continue; }

            cdouble* H_raw_p = (sys == 0) ? H_raw1.data() : H_raw2.data();
            cuDoubleComplex* d_V = (sys == 0) ? d_V1 : d_V2;
            cuDoubleComplex* d_C = (sys == 0) ? d_C1 : d_C2;
            cuDoubleComplex* d_U = (sys == 0) ? d_U1 : d_U2;
            cuDoubleComplex* d_Vinp = (sys == 0) ? d_Vinp1 : d_Vinp2;

            // 1. Extract m_done × m_done square Hessenberg (column-major)
            std::vector<cdouble> Hm(m_done*m_done, 0);
            for (int col = 0; col < m_done; col++)
                for (int row = 0; row < m_done; row++)
                    Hm[row + col*m_done] = H_raw_p[row*m + col];

            // 2. Householder → upper Hessenberg
            std::vector<cdouble> Q_h(m_done*m_done);
            householder_hessenberg(Hm.data(), m_done, Q_h.data());

            // 3. Schur decomposition
            std::vector<cdouble> Q_s(m_done*m_done), eig(m_done);
            hessenberg_schur(Hm.data(), m_done, Q_s.data(), eig.data());

            // 4. Q_total = Q_h * Q_s
            std::vector<cdouble> Q_total(m_done*m_done, 0);
            for (int j = 0; j < m_done; j++)
                for (int i = 0; i < m_done; i++)
                    for (int l = 0; l < m_done; l++)
                        Q_total[i + j*m_done] += Q_h[i + l*m_done] * Q_s[l + j*m_done];

            // 5. Select k smallest eigenvalues
            std::vector<int> idx(m_done);
            std::iota(idx.begin(), idx.end(), 0);
            std::sort(idx.begin(), idx.end(), [&](int a, int b) {
                return std::abs(eig[a]) < std::abs(eig[b]);
            });

            int kk = std::min(k_new, m_done - 2);
            if (kk <= 0) { if (sys==0) k_have1=0; else k_have2=0; continue; }

            // P_k
            std::vector<cdouble> Pk(m_done * kk);
            for (int j = 0; j < kk; j++)
                for (int i = 0; i < m_done; i++)
                    Pk[i + j*m_done] = Q_total[i + idx[j]*m_done];

            // 6. Gather V[:,0:m_done] into d_Vinp (contiguous)
            CUDA_CHECK(cudaMemcpy(d_Vinp, d_V, (size_t)m_done*n*sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));

            // 7. Compute AŨ = V_all * H_raw * P_k → d_C (will be orthonormalized)
            int nv_total = (sys == 0) ? nv1 : nv2;
            std::vector<cdouble> Gk(nv_total * kk, 0);
            for (int jj = 0; jj < kk; jj++)
                for (int i = 0; i < nv_total; i++)
                    for (int l = 0; l < m_done; l++)
                        Gk[i + jj*nv_total] += H_raw_p[i*m + l] * Pk[l + jj*m_done];

            cuDoubleComplex* d_Gk;
            CUDA_CHECK(cudaMalloc(&d_Gk, nv_total*kk*sizeof(cuDoubleComplex)));
            CUDA_CHECK(cudaMemcpy(d_Gk, Gk.data(), nv_total*kk*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
            cuDoubleComplex one_c={1,0}, zer_c={0,0};
            CUBLAS_CHECK(cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                n, kk, nv_total, &one_c, d_V, n, d_Gk, nv_total, &zer_c, d_C, n));
            cudaFree(d_Gk);

            // 8. Compute Ũ = d_Vinp * P_k → d_U
            cuDoubleComplex* d_Pk;
            CUDA_CHECK(cudaMalloc(&d_Pk, m_done*kk*sizeof(cuDoubleComplex)));
            CUDA_CHECK(cudaMemcpy(d_Pk, Pk.data(), m_done*kk*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
            CUBLAS_CHECK(cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                n, kk, m_done, &one_c, d_Vinp, n, d_Pk, m_done, &zer_c, d_U, n));
            cudaFree(d_Pk);

            // 9. GCRO-DR orthonormalization: orthonormalize C=AŨ, adjust U consistently
            //    Inner products and norms in C-space (not U-space!)
            //    After: C^H C = I, A*U = C
            for (int j = 0; j < kk; j++) {
                cuDoubleComplex* cj = (cuDoubleComplex*)d_C + (size_t)j*n;
                cuDoubleComplex* uj = (cuDoubleComplex*)d_U + (size_t)j*n;
                for (int i = 0; i < j; i++) {
                    cuDoubleComplex* ci = (cuDoubleComplex*)d_C + (size_t)i*n;
                    cuDoubleComplex* ui = (cuDoubleComplex*)d_U + (size_t)i*n;
                    cuDoubleComplex dot;
                    CUBLAS_CHECK(cublasZdotc(handle, n, ci, 1, cj, 1, &dot));
                    cuDoubleComplex neg_dot = {-dot.x, -dot.y};
                    CUBLAS_CHECK(cublasZaxpy(handle, n, &neg_dot, ci, 1, cj, 1));
                    CUBLAS_CHECK(cublasZaxpy(handle, n, &neg_dot, ui, 1, uj, 1));
                }
                double nrm;
                CUBLAS_CHECK(cublasDznrm2(handle, n, cj, 1, &nrm));
                if (nrm > 1e-30) {
                    cuDoubleComplex sc = {1.0/nrm, 0};
                    CUBLAS_CHECK(cublasZscal(handle, n, &sc, cj, 1));
                    CUBLAS_CHECK(cublasZscal(handle, n, &sc, uj, 1));
                }
            }

            if (sys == 0) k_have1 = kk;
            else k_have2 = kk;

            if (verbose) printf("    sys%d: %d recycling vectors (eig min=%.2e)\n",
                sys+1, kk, std::abs(eig[idx[0]]));
        }

        // ====== COMPUTE TRUE RESIDUAL ======
        if (!converged1 && !converged2) {
            op.matvec_batch2(h_x1.data(), h_x2.data(), h_r1.data(), h_r2.data()); total_matvecs++;
            for (int i=0; i<n; i++) { h_r1[i]=b1[i]-h_r1[i]; h_r2[i]=b2[i]-h_r2[i]; }
        } else if (!converged1) {
            op.matvec(h_x1.data(), h_r1.data()); total_matvecs++;
            for (int i=0; i<n; i++) h_r1[i]=b1[i]-h_r1[i];
        } else if (!converged2) {
            op.matvec(h_x2.data(), h_r2.data()); total_matvecs++;
            for (int i=0; i<n; i++) h_r2[i]=b2[i]-h_r2[i];
        }

        if (!converged1) { rnorm1=0; for (int i=0; i<n; i++) rnorm1+=std::norm(h_r1[i]); rnorm1=std::sqrt(rnorm1); }
        if (!converged2) { rnorm2=0; for (int i=0; i<n; i++) rnorm2+=std::norm(h_r2[i]); rnorm2=std::sqrt(rnorm2); }

        if (verbose)
            printf("  [GCRO-DR] restart %d: rel1=%.2e rel2=%.2e (k=%d)\n",
                cycle+1, converged1?0:rnorm1/bnorm1, converged2?0:rnorm2/bnorm2, k_use);

        if (!converged1 && rnorm1/bnorm1 < tol) converged1 = true;
        if (!converged2 && rnorm2/bnorm2 < tol) converged2 = true;
        if (converged1 && converged2) break;
    }

    memcpy(x1, h_x1.data(), n*sizeof(cdouble));
    memcpy(x2, h_x2.data(), n*sizeof(cdouble));

    if (verbose) {
        if (converged1 && converged2)
            printf("  [GCRO-DR] Both converged, %d matvec evaluations\n", total_matvecs);
        else
            printf("  [GCRO-DR] NOT converged, %d matvecs, res1=%.2e res2=%.2e\n",
                total_matvecs, rnorm1/bnorm1, rnorm2/bnorm2);
    }

    // Save recycling state to context
    if (ctx) {
        ctx->k_have1 = k_have1;
        ctx->k_have2 = k_have2;
    }

    cudaFree(d_V1); cudaFree(d_V2); cudaFree(d_w1); cudaFree(d_w2);
    cudaFree(d_htmp1); cudaFree(d_htmp2);
    if (own_CU) {
        if (d_C1) cudaFree(d_C1); if (d_C2) cudaFree(d_C2);
        if (d_U1) cudaFree(d_U1); if (d_U2) cudaFree(d_U2);
    }
    if (d_Vinp1) cudaFree(d_Vinp1); if (d_Vinp2) cudaFree(d_Vinp2);
    cublasDestroy(handle);
    return total_matvecs;
}
