#ifndef P2P_H
#define P2P_H

#include <cuda_runtime.h>

// P2P near-field kernel launch functions (definitions in p2p.cu)
// All accept an optional CUDA stream (default: 0 = default stream)

// Scalar potential only: phi_i = sum_j G(r_i, r_j) * q_j
void launch_p2p_potential(
    const double* d_tgt, const double* d_src,
    const double* d_q_re, const double* d_q_im,
    const int* d_offsets, const int* d_indices,
    double k_re, double k_im,
    double* d_out_re, double* d_out_im, int Nt,
    cudaStream_t stream = 0);

// Gradient only: grad_phi_i = sum_j nabla_G(r_i, r_j) * q_j
void launch_p2p_gradient(
    const double* d_tgt, const double* d_src,
    const double* d_q_re, const double* d_q_im,
    const int* d_offsets, const int* d_indices,
    double k_re, double k_im,
    double* d_gx_re, double* d_gx_im,
    double* d_gy_re, double* d_gy_im,
    double* d_gz_re, double* d_gz_im, int Nt,
    cudaStream_t stream = 0);

// Combined potential + gradient in a single pass (avoids redundant work)
void launch_p2p_pot_grad(
    int Nt,
    const double* d_tgt, const double* d_src,
    const double* d_q_re, const double* d_q_im,
    const int* d_offsets, const int* d_indices,
    double k_re, double k_im,
    double* d_pot_re, double* d_pot_im,
    double* d_gx_re, double* d_gx_im,
    double* d_gy_re, double* d_gy_im,
    double* d_gz_re, double* d_gz_im,
    cudaStream_t stream = 0);

// Batch-2 potential: process two charge vectors in a single CSR traversal.
// Green's function exp(ikR)/(4piR) computed once, multiplied by both q1 and q2.
void launch_p2p_potential_batch2(
    const double* d_tgt, const double* d_src,
    const double* d_q1_re, const double* d_q1_im,
    const double* d_q2_re, const double* d_q2_im,
    const int* d_offsets, const int* d_indices,
    double k_re, double k_im,
    double* d_out1_re, double* d_out1_im,
    double* d_out2_re, double* d_out2_im, int Nt,
    cudaStream_t stream = 0);

// Batch-2 combined potential + gradient: two charge vectors, single CSR traversal.
void launch_p2p_pot_grad_batch2(
    int Nt,
    const double* d_tgt, const double* d_src,
    const double* d_q1_re, const double* d_q1_im,
    const double* d_q2_re, const double* d_q2_im,
    const int* d_offsets, const int* d_indices,
    double k_re, double k_im,
    double* d_pot1_re, double* d_pot1_im,
    double* d_pot2_re, double* d_pot2_im,
    double* d_gx1_re, double* d_gx1_im,
    double* d_gy1_re, double* d_gy1_im,
    double* d_gz1_re, double* d_gz1_im,
    double* d_gx2_re, double* d_gx2_im,
    double* d_gy2_re, double* d_gy2_im,
    double* d_gz2_re, double* d_gz2_im,
    cudaStream_t stream = 0);

#endif // P2P_H
