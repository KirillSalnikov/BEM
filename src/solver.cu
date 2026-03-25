#include "solver.h"
#include <cusolverDn.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

int lu_factorize_cuda(std::complex<double>* Z, int n, int* ipiv) {
    Timer timer;
    cusolverDnHandle_t handle;
    CUSOLVER_CHECK(cusolverDnCreate(&handle));

    // Allocate device memory
    cuDoubleComplex* d_Z;
    int* d_ipiv;
    int* d_info;
    CUDA_CHECK(cudaMalloc(&d_Z, n * n * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_ipiv, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

    // Copy Z to device
    CUDA_CHECK(cudaMemcpy(d_Z, Z, n * n * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

    // Query workspace
    int lwork;
    CUSOLVER_CHECK(cusolverDnZgetrf_bufferSize(handle, n, n, d_Z, n, &lwork));

    cuDoubleComplex* d_work;
    CUDA_CHECK(cudaMalloc(&d_work, lwork * sizeof(cuDoubleComplex)));

    // Factorize
    CUSOLVER_CHECK(cusolverDnZgetrf(handle, n, n, d_Z, n, d_work, d_ipiv, d_info));

    int h_info;
    CUDA_CHECK(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0) {
        fprintf(stderr, "  LU factorization failed: info=%d\n", h_info);
    }

    // Copy back
    CUDA_CHECK(cudaMemcpy(Z, d_Z, n * n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ipiv, d_ipiv, n * sizeof(int), cudaMemcpyDeviceToHost));

    cudaFree(d_Z); cudaFree(d_ipiv); cudaFree(d_info); cudaFree(d_work);
    cusolverDnDestroy(handle);

    printf("  LU factorization (%dx%d): %.1fs\n", n, n, timer.elapsed_s());
    return h_info;
}


int lu_solve_cuda(const std::complex<double>* Z, const int* ipiv,
                  int n, std::complex<double>* B, int nrhs) {
    Timer timer;
    cusolverDnHandle_t handle;
    CUSOLVER_CHECK(cusolverDnCreate(&handle));

    cuDoubleComplex* d_Z;
    cuDoubleComplex* d_B;
    int* d_ipiv;
    int* d_info;
    CUDA_CHECK(cudaMalloc(&d_Z, n * n * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_B, n * nrhs * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_ipiv, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_Z, Z, n * n * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, n * nrhs * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ipiv, ipiv, n * sizeof(int), cudaMemcpyHostToDevice));

    // Z is row-major on host; cuSOLVER sees it as Z^T (column-major).
    // Use CUBLAS_OP_T to solve (Z^T)^T * X = Z * X = B.
    CUSOLVER_CHECK(cusolverDnZgetrs(handle, CUBLAS_OP_T, n, nrhs,
                                     d_Z, n, d_ipiv, d_B, n, d_info));

    int h_info;
    CUDA_CHECK(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(B, d_B, n * nrhs * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

    cudaFree(d_Z); cudaFree(d_B); cudaFree(d_ipiv); cudaFree(d_info);
    cusolverDnDestroy(handle);

    printf("  LU solve (%dx%d, %d RHS): %.2fs\n", n, n, nrhs, timer.elapsed_s());
    return h_info;
}


int lu_solve_full(std::complex<double>* Z, int n,
                  std::complex<double>* B, int nrhs) {
    Timer timer;
    cusolverDnHandle_t handle;
    CUSOLVER_CHECK(cusolverDnCreate(&handle));

    cuDoubleComplex* d_Z;
    cuDoubleComplex* d_B;
    int* d_ipiv;
    int* d_info;
    cuDoubleComplex* d_work;

    CUDA_CHECK(cudaMalloc(&d_Z, n * n * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_B, n * nrhs * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_ipiv, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_Z, Z, n * n * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, n * nrhs * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

    // Query workspace
    int lwork;
    CUSOLVER_CHECK(cusolverDnZgetrf_bufferSize(handle, n, n, d_Z, n, &lwork));
    CUDA_CHECK(cudaMalloc(&d_work, lwork * sizeof(cuDoubleComplex)));

    // Factorize
    CUSOLVER_CHECK(cusolverDnZgetrf(handle, n, n, d_Z, n, d_work, d_ipiv, d_info));

    int h_info;
    CUDA_CHECK(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0) {
        fprintf(stderr, "  LU factorization failed: info=%d\n", h_info);
        cudaFree(d_Z); cudaFree(d_B); cudaFree(d_ipiv); cudaFree(d_info); cudaFree(d_work);
        cusolverDnDestroy(handle);
        return h_info;
    }
    printf("  LU factorization: %.1fs\n", timer.elapsed_s());

    // Solve: Z is row-major, cuSOLVER sees Z^T → use CUBLAS_OP_T
    Timer t_solve;
    CUSOLVER_CHECK(cusolverDnZgetrs(handle, CUBLAS_OP_T, n, nrhs,
                                     d_Z, n, d_ipiv, d_B, n, d_info));

    CUDA_CHECK(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(B, d_B, n * nrhs * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
    // Also copy back LU and pivots for potential reuse
    CUDA_CHECK(cudaMemcpy(Z, d_Z, n * n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

    printf("  LU solve (%d RHS): %.2fs\n", nrhs, t_solve.elapsed_s());

    cudaFree(d_Z); cudaFree(d_B); cudaFree(d_ipiv); cudaFree(d_info); cudaFree(d_work);
    cusolverDnDestroy(handle);

    printf("  Total factorize+solve: %.1fs\n", timer.elapsed_s());
    return h_info;
}
