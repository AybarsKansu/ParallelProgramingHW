//nvcc managed.cu -arch=sm_86 -o managed


#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>

// ---------------------------------------------------------
// CUDA error checking
// ---------------------------------------------------------
#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t err = call;                               \
        if (err != cudaSuccess) {                              \
            std::cerr << "CUDA Error: "                        \
                      << cudaGetErrorString(err) << std::endl; \
            std::exit(EXIT_FAILURE);                          \
        }                                                     \
    } while (0)


// ---------------------------------------------------------
// Common initialization
// ---------------------------------------------------------
void initializeInputs(float* A, float* B, size_t N)
{
    for (size_t i = 0; i < N; i++) {
        A[i] = static_cast<float>(i);
        B[i] = static_cast<float>(2 * i);
    }
}


// ---------------------------------------------------------
// Common result verification
// ---------------------------------------------------------
bool verifyResult(const float* C, size_t N)
{
    for (size_t i = 0; i < N; i++) {
        float expected = 3.0f * static_cast<float>(i);

        if (std::fabs(C[i] - expected) > 1e-5f) {
            std::cout << "Verification failed at index "
                      << i << std::endl;
            return false;
        }
    }

    std::cout << "Verification successful!" << std::endl;
    return true;
}


// ---------------------------------------------------------
// GPU kernel: Vector addition
// ---------------------------------------------------------
__global__
void vectorAddKernel(const float* A,
                     const float* B,
                     float* C,
                     size_t N)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}


// =========================================================
// 1. CPU VECTOR ADDITION
// =========================================================
void runCPU(size_t N)
{
    std::cout << "\n===== CPU VECTOR ADDITION =====\n";

    std::vector<float> A(N);
    std::vector<float> B(N);
    std::vector<float> C(N);

    // Initialize input vectors
    initializeInputs(A.data(), B.data(), N);

    // Perform vector addition on CPU
    for (size_t i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }

    verifyResult(C.data(), N);
}


// =========================================================
// 2. GPU VECTOR ADDITION - NON-MANAGED MEMORY
// =========================================================
void runGPUExplicit(size_t N)
{
    std::cout << "\n===== GPU: EXPLICIT MEMORY MANAGEMENT =====\n";

    size_t bytes = N * sizeof(float);

    // Host memory
    std::vector<float> h_A(N);
    std::vector<float> h_B(N);
    std::vector<float> h_C(N);

    initializeInputs(h_A.data(), h_B.data(), N);

    // Device memory pointers
    float *d_A, *d_B, *d_C;

    // Allocate GPU memory
    CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));

    // Copy input vectors: CPU -> GPU
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes,
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes,
                          cudaMemcpyHostToDevice));

    // Configure kernel execution
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // Execute vector addition on GPU
    vectorAddKernel<<<blocks, threads>>>(d_A, d_B, d_C, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result: GPU -> CPU
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, bytes,
                          cudaMemcpyDeviceToHost));

    // Verify results on CPU
    verifyResult(h_C.data(), N);

    // Release GPU memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}


// =========================================================
// 3. GPU VECTOR ADDITION - MANAGED MEMORY
// =========================================================
void runGPUManaged(size_t N)
{
    std::cout << "\n===== GPU: MANAGED MEMORY =====\n";

    size_t bytes = N * sizeof(float);

    // Managed memory pointers
    float *A, *B, *C;

    // Allocate Unified Memory
    CUDA_CHECK(cudaMallocManaged((void**)&A, bytes));
    CUDA_CHECK(cudaMallocManaged((void**)&B, bytes));
    CUDA_CHECK(cudaMallocManaged((void**)&C, bytes));

    // Initialize input vectors on CPU
    initializeInputs(A, B, N);

    // Configure kernel execution
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // Execute vector addition on GPU
    vectorAddKernel<<<blocks, threads>>>(A, B, C, N);

    CUDA_CHECK(cudaGetLastError());

    // Wait for GPU execution to complete
    CUDA_CHECK(cudaDeviceSynchronize());

    // CPU directly accesses managed memory
    verifyResult(C, N);

    // Release Unified Memory allocations
    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(B));
    CUDA_CHECK(cudaFree(C));
}


// =========================================================
// MAIN
// =========================================================
int main()
{
    const size_t N = 1 << 20;  // 1M elements

    std::cout << "Vector size: " << N << std::endl;

    // 1. CPU implementation
    runCPU(N);

    // 2. GPU with explicit memory management
    runGPUExplicit(N);

    // 3. GPU with Unified Memory
    runGPUManaged(N);

    std::cout << "\nAll implementations completed.\n";

    return 0;
}
