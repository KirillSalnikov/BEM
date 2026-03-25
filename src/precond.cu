#include "precond.h"
#include "bem_fmm.h"
#include <cstdio>
#include <cmath>
#include <complex>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <numeric>

// ============================================================
// Spatial hash key for 3D integer cell coordinates
// ============================================================
struct CellKey {
    int ix, iy, iz;
    bool operator==(const CellKey& o) const {
        return ix == o.ix && iy == o.iy && iz == o.iz;
    }
};

struct CellKeyHash {
    size_t operator()(const CellKey& c) const {
        size_t h = (size_t)c.ix * 73856093u ^ (size_t)c.iy * 19349663u ^ (size_t)c.iz * 83492791u;
        return h;
    }
};

// ============================================================
// Assemble near-field sparse matrix (shared by ILU0, NEARLU, DIAG)
// ============================================================
static void assemble_near_field(NearFieldPrecond& P, BemFmmOperator& op, double radius_mult)
{
    Timer timer;
    int N = P.N;
    int N2 = P.N2;
    int Nq = op.Nq;

    // Step 1: Compute RWG centers and average extent
    std::vector<double> centers(N * 3);
    double avg_extent = 0;

    for (int m = 0; m < N; m++) {
        double cx_val = 0, cy_val = 0, cz_val = 0;
        double max_x = -1e30, min_x = 1e30;
        double max_y = -1e30, min_y = 1e30;
        double max_z = -1e30, min_z = 1e30;
        for (int q = 0; q < Nq; q++) {
            double px = op.qpts_p[m*Nq*3 + q*3];
            double py = op.qpts_p[m*Nq*3 + q*3+1];
            double pz = op.qpts_p[m*Nq*3 + q*3+2];
            double mx = op.qpts_m[m*Nq*3 + q*3];
            double my = op.qpts_m[m*Nq*3 + q*3+1];
            double mz = op.qpts_m[m*Nq*3 + q*3+2];
            cx_val += px + mx;
            cy_val += py + my;
            cz_val += pz + mz;
            max_x = std::max(max_x, std::max(px, mx));
            min_x = std::min(min_x, std::min(px, mx));
            max_y = std::max(max_y, std::max(py, my));
            min_y = std::min(min_y, std::min(py, my));
            max_z = std::max(max_z, std::max(pz, mz));
            min_z = std::min(min_z, std::min(pz, mz));
        }
        double inv2Nq = 1.0 / (2 * Nq);
        centers[m*3]   = cx_val * inv2Nq;
        centers[m*3+1] = cy_val * inv2Nq;
        centers[m*3+2] = cz_val * inv2Nq;
        double ext = std::max({max_x - min_x, max_y - min_y, max_z - min_z});
        avg_extent += ext;
    }
    avg_extent /= N;

    double cell_size = radius_mult * avg_extent;
    if (cell_size < 1e-15) cell_size = 1e-15;
    double inv_cell = 1.0 / cell_size;

    printf("  [Precond] Avg RWG extent=%.4e, cell_size=%.4e\n", avg_extent, cell_size);

    // Step 2: Spatial hashing
    std::unordered_map<CellKey, std::vector<int>, CellKeyHash> cell_map;
    cell_map.reserve(N);

    std::vector<CellKey> rwg_cell(N);
    for (int m = 0; m < N; m++) {
        CellKey key;
        key.ix = (int)std::floor(centers[m*3]   * inv_cell);
        key.iy = (int)std::floor(centers[m*3+1] * inv_cell);
        key.iz = (int)std::floor(centers[m*3+2] * inv_cell);
        rwg_cell[m] = key;
        cell_map[key].push_back(m);
    }

    printf("  [Precond] %d spatial cells\n", (int)cell_map.size());

    // Step 3: Build near-field pair list
    std::vector<std::vector<int>> nf_lists(N);

    for (int m = 0; m < N; m++) {
        std::unordered_set<int> nf_set;
        nf_set.insert(m);

        CellKey& ck = rwg_cell[m];
        for (int dix = -1; dix <= 1; dix++)
            for (int diy = -1; diy <= 1; diy++)
                for (int diz = -1; diz <= 1; diz++) {
                    CellKey nk = {ck.ix + dix, ck.iy + diy, ck.iz + diz};
                    auto it = cell_map.find(nk);
                    if (it != cell_map.end())
                        for (int n_idx : it->second)
                            nf_set.insert(n_idx);
                }

        for (int jc = op.corr_row_ptr[m]; jc < op.corr_row_ptr[m + 1]; jc++)
            nf_set.insert(op.corr_col_idx[jc]);

        nf_lists[m].assign(nf_set.begin(), nf_set.end());
        std::sort(nf_lists[m].begin(), nf_lists[m].end());
    }

    // Stats
    {
        long long total_nf = 0;
        int min_nf = N, max_nf = 0;
        for (int m = 0; m < N; m++) {
            int sz = (int)nf_lists[m].size();
            total_nf += sz;
            min_nf = std::min(min_nf, sz);
            max_nf = std::max(max_nf, sz);
        }
        printf("  [Precond] Near-field pairs per RWG: min=%d, max=%d, avg=%.1f\n",
               min_nf, max_nf, (double)total_nf / N);
        printf("  [Precond] Total NxN near-field entries: %lld\n", total_nf);
        printf("  [Precond] Coverage: %.1f%% of full N×N matrix\n",
               100.0 * total_nf / ((long long)N * N));
    }

    printf("  [Precond] Near-field detection: %.2fs\n", timer.elapsed_s());

    // Step 4: Build 2N×2N CSR
    Timer t_assemble;

    long long nnz_total = 0;
    for (int m = 0; m < N; m++)
        nnz_total += 2 * (long long)nf_lists[m].size();
    nnz_total *= 2;

    P.csr_row_ptr.resize(N2 + 1);
    P.csr_col_idx.resize(nnz_total);
    P.csr_val.assign(nnz_total, cdouble(0));
    P.diag_ptr.resize(N2);

    P.csr_row_ptr[0] = 0;
    for (int i = 0; i < N2; i++) {
        int m = (i < N) ? i : i - N;
        int nnz_row = 2 * (int)nf_lists[m].size();
        P.csr_row_ptr[i + 1] = P.csr_row_ptr[i] + nnz_row;
    }

    for (int i = 0; i < N2; i++) {
        int m = (i < N) ? i : i - N;
        int base = P.csr_row_ptr[i];
        int nnz_half = (int)nf_lists[m].size();
        for (int j = 0; j < nnz_half; j++)
            P.csr_col_idx[base + j] = nf_lists[m][j];
        for (int j = 0; j < nnz_half; j++)
            P.csr_col_idx[base + nnz_half + j] = nf_lists[m][j] + N;

        P.diag_ptr[i] = -1;
        for (int j = P.csr_row_ptr[i]; j < P.csr_row_ptr[i + 1]; j++) {
            if (P.csr_col_idx[j] == i) {
                P.diag_ptr[i] = j;
                break;
            }
        }
        if (P.diag_ptr[i] < 0)
            fprintf(stderr, "  [Precond] ERROR: diagonal not found for row %d\n", i);
    }

    printf("  [Precond] 2N×2N CSR: %lld nonzeros (%.1f nnz/row avg)\n",
           nnz_total, (double)nnz_total / N2);

    // Step 5: Assemble entries via quadrature
    double inv4pi = 1.0 / (4.0 * M_PI);
    cdouble k_vals[2] = {op.k_ext, op.k_int};
    cdouble eta_e = op.eta_ext, eta_i = op.eta_int;

    // Lookup: nf_lists[m] -> position
    std::vector<std::unordered_map<int,int>> nf_pos(N);
    for (int m = 0; m < N; m++) {
        nf_pos[m].reserve(nf_lists[m].size());
        for (int j = 0; j < (int)nf_lists[m].size(); j++)
            nf_pos[m][nf_lists[m][j]] = j;
    }

    auto csr_idx = [&](int row, int col) -> int {
        int m = (row < N) ? row : row - N;
        int base = P.csr_row_ptr[row];
        int nnz_half = (int)nf_lists[m].size();

        if (col < N) {
            auto it = nf_pos[m].find(col);
            if (it != nf_pos[m].end()) return base + it->second;
        } else {
            auto it = nf_pos[m].find(col - N);
            if (it != nf_pos[m].end()) return base + nnz_half + it->second;
        }
        return -1;
    };

    for (int m = 0; m < N; m++) {
        for (int jn = 0; jn < (int)nf_lists[m].size(); jn++) {
            int n_idx = nf_lists[m][jn];

            cdouble L_vals_k[2] = {0, 0};
            cdouble K_vals_k[2] = {0, 0};

            for (int hm = 0; hm < 2; hm++) {
                const double* qm = (hm == 0) ? &op.qpts_p[m * Nq * 3] : &op.qpts_m[m * Nq * 3];
                const double* fm = (hm == 0) ? &op.f_p[m * Nq * 3] : &op.f_m[m * Nq * 3];
                double dm = (hm == 0) ? op.div_p[m] : op.div_m[m];
                const double* jwm = (hm == 0) ? &op.jw_p[m * Nq] : &op.jw_m[m * Nq];

                for (int hn = 0; hn < 2; hn++) {
                    const double* qn = (hn == 0) ? &op.qpts_p[n_idx * Nq * 3] : &op.qpts_m[n_idx * Nq * 3];
                    const double* fn = (hn == 0) ? &op.f_p[n_idx * Nq * 3] : &op.f_m[n_idx * Nq * 3];
                    double dn = (hn == 0) ? op.div_p[n_idx] : op.div_m[n_idx];
                    const double* jwn = (hn == 0) ? &op.jw_p[n_idx * Nq] : &op.jw_m[n_idx * Nq];

                    for (int qi = 0; qi < Nq; qi++) {
                        double rx = qm[qi*3], ry = qm[qi*3+1], rz = qm[qi*3+2];
                        double fxm = fm[qi*3], fym = fm[qi*3+1], fzm = fm[qi*3+2];
                        double wm_val = jwm[qi];

                        for (int qj = 0; qj < Nq; qj++) {
                            double dx = rx - qn[qj*3];
                            double dy = ry - qn[qj*3+1];
                            double dz = rz - qn[qj*3+2];
                            double R = std::sqrt(dx*dx + dy*dy + dz*dz);
                            double wn_val = jwn[qj];
                            double ww = wm_val * wn_val;

                            double fxn = fn[qj*3], fyn = fn[qj*3+1], fzn = fn[qj*3+2];
                            double f_dot = fxm*fxn + fym*fyn + fzm*fzn;

                            for (int ki = 0; ki < 2; ki++) {
                                cdouble kv = k_vals[ki];
                                cdouble ik = cdouble(0, 1) * kv;
                                cdouble iok = cdouble(0, 1) / kv;

                                if (R > 1e-12) {
                                    cdouble G = std::exp(ik * R) * inv4pi / R;
                                    L_vals_k[ki] += (ik * f_dot - iok * dm * dn) * G * ww;

                                    cdouble gG = G * (ik - 1.0/R) / R;
                                    double cross_x = dy*fzn - dz*fyn;
                                    double cross_y = dz*fxn - dx*fzn;
                                    double cross_z = dx*fyn - dy*fxn;
                                    K_vals_k[ki] += gG * (fxm*cross_x + fym*cross_y + fzm*cross_z) * ww;
                                } else {
                                    cdouble G0 = ik * inv4pi;
                                    L_vals_k[ki] += (ik * f_dot - iok * dm * dn) * G0 * ww;
                                }
                            }
                        }
                    }
                }
            }

            // Add singular corrections
            for (int jc = op.corr_row_ptr[m]; jc < op.corr_row_ptr[m + 1]; jc++) {
                if (op.corr_col_idx[jc] == n_idx) {
                    L_vals_k[0] += op.corr_L_ext_val[jc];
                    K_vals_k[0] += op.corr_K_ext_val[jc];
                    L_vals_k[1] += op.corr_L_int_val[jc];
                    K_vals_k[1] += op.corr_K_int_val[jc];
                }
            }

            // PMCHWT 2x2 block entries
            cdouble A_mn = eta_e * L_vals_k[0] + eta_i * L_vals_k[1];     // JJ
            cdouble B_mn = -(K_vals_k[0] + K_vals_k[1]);                  // JM
            cdouble C_mn = K_vals_k[0] + K_vals_k[1];                     // MJ
            cdouble D_mn = L_vals_k[0] / eta_e + L_vals_k[1] / eta_i;    // MM

            int idx_jj = csr_idx(m, n_idx);
            if (idx_jj >= 0) P.csr_val[idx_jj] = A_mn;

            int idx_jm = csr_idx(m, n_idx + N);
            if (idx_jm >= 0) P.csr_val[idx_jm] = B_mn;

            int idx_mj = csr_idx(m + N, n_idx);
            if (idx_mj >= 0) P.csr_val[idx_mj] = C_mn;

            int idx_mm = csr_idx(m + N, n_idx + N);
            if (idx_mm >= 0) P.csr_val[idx_mm] = D_mn;
        }

        if (m > 0 && m % (N / 10 + 1) == 0)
            printf("  [Precond] Assembly: %d/%d RWG rows (%.0f%%)\n",
                   m, N, 100.0 * m / N);
    }

    printf("  [Precond] Assembly done: %.2fs\n", t_assemble.elapsed_s());

    // Diagnostic: print diagonal statistics
    P.diag_val.resize(N2);
    double dmin = 1e30, dmax = 0, dmean = 0;
    for (int i = 0; i < N2; i++) {
        P.diag_val[i] = P.csr_val[P.diag_ptr[i]];
        double d = std::abs(P.diag_val[i]);
        dmin = std::min(dmin, d);
        dmax = std::max(dmax, d);
        dmean += d;
    }
    dmean /= N2;
    printf("  [Precond] Diagonal |Z_ii|: min=%.4e, max=%.4e, mean=%.4e, ratio=%.1f\n",
           dmin, dmax, dmean, dmax/dmin);

    // Print first few diagonal entries
    printf("  [Precond] First 5 JJ diag: ");
    for (int i = 0; i < std::min(5, N); i++)
        printf("(%.3e,%.3e) ", P.diag_val[i].real(), P.diag_val[i].imag());
    printf("\n");
    printf("  [Precond] First 5 MM diag: ");
    for (int i = N; i < std::min(N+5, N2); i++)
        printf("(%.3e,%.3e) ", P.diag_val[i].real(), P.diag_val[i].imag());
    printf("\n");
}

// ============================================================
// ILU(0) factorization
// ============================================================
static void do_ilu0(NearFieldPrecond& P)
{
    Timer t_ilu;
    int N2 = P.N2;

    // Build per-row hash maps for column lookup
    std::vector<std::unordered_map<int,int>> row_col_map(N2);
    for (int i = 0; i < N2; i++) {
        int rs = P.csr_row_ptr[i], re = P.csr_row_ptr[i + 1];
        row_col_map[i].reserve(re - rs);
        for (int j = rs; j < re; j++)
            row_col_map[i][P.csr_col_idx[j]] = j;
    }

    printf("  [Precond] Starting ILU(0) factorization...\n");

    int zero_diag_count = 0;

    for (int i = 0; i < N2; i++) {
        int rs = P.csr_row_ptr[i];
        int re = P.csr_row_ptr[i + 1];

        for (int p = rs; p < re; p++) {
            int k = P.csr_col_idx[p];
            if (k >= i) break;

            cdouble akk = P.csr_val[P.diag_ptr[k]];
            if (std::abs(akk) < 1e-30) {
                zero_diag_count++;
                akk = cdouble(1e-15);
            }
            cdouble a_ik = P.csr_val[p] / akk;
            P.csr_val[p] = a_ik;

            for (int q = p + 1; q < re; q++) {
                int j = P.csr_col_idx[q];
                auto it = row_col_map[k].find(j);
                if (it != row_col_map[k].end())
                    P.csr_val[q] -= a_ik * P.csr_val[it->second];
            }
        }

        if (std::abs(P.csr_val[P.diag_ptr[i]]) < 1e-30) {
            zero_diag_count++;
            P.csr_val[P.diag_ptr[i]] = cdouble(1e-15);
        }

        if (i > 0 && i % (N2 / 10 + 1) == 0)
            printf("  [Precond] ILU(0): %d/%d rows (%.0f%%)\n",
                   i, N2, 100.0 * i / N2);
    }

    if (zero_diag_count > 0)
        printf("  [Precond] WARNING: %d near-zero diagonal entries\n", zero_diag_count);

    printf("  [Precond] ILU(0) factorization done: %.2fs\n", t_ilu.elapsed_s());
}

// ============================================================
// Full dense LU factorization of near-field matrix (small N)
// ============================================================
static void do_near_lu(NearFieldPrecond& P)
{
    Timer t_lu;
    int N2 = P.N2;

    if (N2 > 8000) {
        printf("  [Precond] ERROR: NEARLU requires N2 <= 8000 (got %d)\n", N2);
        P.mode = PREC_DIAG;  // fallback
        return;
    }

    printf("  [Precond] Building dense LU from sparse near-field (%d×%d, %.1f MB)...\n",
           N2, N2, (double)N2 * N2 * 16 / 1e6);

    // Convert CSR to dense column-major
    P.lu_dense.assign((size_t)N2 * N2, cdouble(0));
    for (int i = 0; i < N2; i++) {
        for (int p = P.csr_row_ptr[i]; p < P.csr_row_ptr[i + 1]; p++) {
            int j = P.csr_col_idx[p];
            P.lu_dense[(size_t)j * N2 + i] = P.csr_val[p];  // column-major
        }
    }

    // LU factorization with partial pivoting (LAPACK-style, manual)
    P.lu_piv.resize(N2);
    for (int k = 0; k < N2; k++) {
        // Find pivot
        int max_idx = k;
        double max_val = std::abs(P.lu_dense[(size_t)k * N2 + k]);
        for (int i = k + 1; i < N2; i++) {
            double v = std::abs(P.lu_dense[(size_t)k * N2 + i]);
            if (v > max_val) { max_val = v; max_idx = i; }
        }
        P.lu_piv[k] = max_idx;

        // Swap rows k and max_idx
        if (max_idx != k) {
            for (int j = 0; j < N2; j++) {
                std::swap(P.lu_dense[(size_t)j * N2 + k],
                          P.lu_dense[(size_t)j * N2 + max_idx]);
            }
        }

        cdouble akk = P.lu_dense[(size_t)k * N2 + k];
        if (std::abs(akk) < 1e-30) akk = cdouble(1e-15);

        // Eliminate below
        for (int i = k + 1; i < N2; i++) {
            cdouble factor = P.lu_dense[(size_t)k * N2 + i] / akk;
            P.lu_dense[(size_t)k * N2 + i] = factor;  // store L
            for (int j = k + 1; j < N2; j++)
                P.lu_dense[(size_t)j * N2 + i] -= factor * P.lu_dense[(size_t)j * N2 + k];
        }

        if (k > 0 && k % (N2 / 5 + 1) == 0)
            printf("  [Precond] LU: %d/%d (%.0f%%)\n", k, N2, 100.0 * k / N2);
    }

    printf("  [Precond] Dense LU done: %.2fs\n", t_lu.elapsed_s());

    // Free sparse data not needed for NEARLU
    P.csr_val.clear();
    P.csr_val.shrink_to_fit();
}

// ============================================================
// Build preconditioner
// ============================================================
void NearFieldPrecond::build(BemFmmOperator& op, PrecondMode pm, double radius_mult)
{
    Timer timer;
    N = op.N;
    N2 = 2 * N;
    mode = pm;

    const char* mode_name[] = {"NONE", "DIAG", "ILU0", "NEARLU"};
    printf("  [Precond] Building %s preconditioner (N=%d, system_size=%d, radius=%.1f)...\n",
           mode_name[mode], N, N2, radius_mult);

    if (mode == PREC_NONE) return;

    // Assemble near-field sparse matrix (needed for all modes)
    assemble_near_field(*this, op, radius_mult);

    // Mode-specific factorization
    switch (mode) {
        case PREC_DIAG:
            // diag_val already filled in assemble_near_field
            // Free sparse data not needed
            csr_val.clear();
            csr_val.shrink_to_fit();
            break;

        case PREC_ILU0:
            do_ilu0(*this);
            break;

        case PREC_NEARLU:
            do_near_lu(*this);
            break;

        default:
            break;
    }

    printf("  [Precond] %s preconditioner built: %.2fs total\n",
           mode_name[mode], timer.elapsed_s());
}

// ============================================================
// Apply preconditioner: z = M^{-1} * r
// ============================================================
void NearFieldPrecond::apply(const cdouble* r, cdouble* z) const
{
    switch (mode) {
        case PREC_NONE:
            // Identity: z = r
            for (int i = 0; i < N2; i++)
                z[i] = r[i];
            break;

        case PREC_DIAG:
            // Diagonal scaling: z[i] = r[i] / diag[i]
            for (int i = 0; i < N2; i++) {
                cdouble d = diag_val[i];
                if (std::abs(d) > 1e-30)
                    z[i] = r[i] / d;
                else
                    z[i] = r[i];
            }
            break;

        case PREC_ILU0:
            // ILU forward/backward solve
            for (int i = 0; i < N2; i++)
                z[i] = r[i];

            // Forward solve: L * y = r (L has unit diagonal)
            for (int i = 0; i < N2; i++) {
                int rs = csr_row_ptr[i];
                int dp = diag_ptr[i];
                for (int p = rs; p < dp; p++)
                    z[i] -= csr_val[p] * z[csr_col_idx[p]];
            }

            // Backward solve: U * z = y
            for (int i = N2 - 1; i >= 0; i--) {
                int dp = diag_ptr[i];
                int re = csr_row_ptr[i + 1];
                for (int p = dp + 1; p < re; p++)
                    z[i] -= csr_val[p] * z[csr_col_idx[p]];
                z[i] /= csr_val[dp];
            }
            break;

        case PREC_NEARLU:
        {
            // Dense LU forward/backward solve
            // Copy r with pivot permutation (forward)
            std::vector<cdouble> tmp(N2);
            for (int i = 0; i < N2; i++)
                tmp[i] = r[i];

            // Apply permutation
            for (int i = 0; i < N2; i++) {
                if (lu_piv[i] != i)
                    std::swap(tmp[i], tmp[lu_piv[i]]);
            }

            // Forward: L * y = Pr
            for (int i = 1; i < N2; i++) {
                for (int j = 0; j < i; j++)
                    tmp[i] -= lu_dense[(size_t)j * N2 + i] * tmp[j];
            }

            // Backward: U * z = y
            for (int i = N2 - 1; i >= 0; i--) {
                for (int j = i + 1; j < N2; j++)
                    tmp[i] -= lu_dense[(size_t)j * N2 + i] * tmp[j];
                tmp[i] /= lu_dense[(size_t)i * N2 + i];
            }

            for (int i = 0; i < N2; i++)
                z[i] = tmp[i];
            break;
        }
    }
}
