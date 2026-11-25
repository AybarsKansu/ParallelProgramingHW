#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define RADIUS 3       // kernel size = 2*RADIUS + 1

// -------------------------------------------------------------
// VALID PADDED GPU KERNEL 1 — Shared memory version
// -------------------------------------------------------------
__global__ void conv1D_shared(int *in, int *out, int n) {
    __shared__ int tempBuf[BLOCK_SIZE + 2*RADIUS];

    int gloIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int locIdx = threadIdx.x + RADIUS;

    if (gloIdx >= n) 
        return;

    tempBuf[locIdx] = in[gloIdx];
    if (threadIdx.x < RADIUS) {
        int leftIndex  = gloIdx - RADIUS;
        int rightIndex = gloIdx + BLOCK_SIZE;

        tempBuf[locIdx - RADIUS]  = (leftIndex >= 0) ? in[leftIndex] : 0;
        tempBuf[locIdx + BLOCK_SIZE] = (rightIndex < n) ? in[rightIndex] : 0;
    }

    __syncthreads();

    if (gloIdx >= RADIUS && gloIdx < n - RADIUS) {
        int acc = 0;
        for (int k = -RADIUS; k <= RADIUS; k++)
            acc += tempBuf[locIdx + k];
        out[gloIdx] = acc;
    }
}


// -------------------------------------------------------------
// VALID PADDED GPU KERNEL 2 — GLOBAL MEMORY version (NO SHARED MEMORY)
// -------------------------------------------------------------
__global__ void conv1D_global(int *in, int *out, int n) {
    int gloIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (gloIdx < RADIUS || gloIdx >= n - RADIUS) return;

    int sum = 0;
    for (int k = -RADIUS; k <= RADIUS; k++)
        sum += in[gloIdx + k];

    out[gloIdx] = sum;
}


// -------------------------------------------------------------
// VALID PADDED CPU convolution 
// -------------------------------------------------------------
void cpu_convolve(const int *in, int *out, int n) {
    for (int i = RADIUS; i < n - RADIUS; i++) {
        int sum = 0;
        for (int k = -RADIUS; k <= RADIUS; k++)
            sum += in[i + k];
        out[i] = sum;
    }
}


// -------------------------------------------------------------
// MAIN PROGRAM
// -------------------------------------------------------------
int main() {
    const int N = 1 << 20;   // 1M elements

    int *h_in  = new int[N];
    int *h_cpu = new int[N];
    int *h_gpu_shared = new int[N];
    int *h_gpu_global = new int[N];

    // ---------------------------------------------------------
    // 1. Create random data
    // ---------------------------------------------------------
    for (int i = 0; i < N; i++)
        h_in[i] = rand() % 10;

    // ---------------------------------------------------------
    // 2. CPU timing
    // ---------------------------------------------------------
    double cpu_total = 0;
    for (int t = 0; t < 100; t++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        cpu_convolve(h_in, h_cpu, N);
        auto t1 = std::chrono::high_resolution_clock::now();
        cpu_total += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    // ---------------------------------------------------------
    // 3. CUDA memory setup
    // ---------------------------------------------------------
    int *d_in, *d_out;
    cudaMalloc(&d_in,  N * sizeof(int));
    cudaMalloc(&d_out, N * sizeof(int));

    cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    //cudaFuncSetCacheConfig(conv1D_shared, cudaFuncCachePreferShared);
    //cudaFuncSetCacheConfig(conv1D_global, cudaFuncCachePreferShared);
    
    /*int carveout = 100; //100% shared
    cudaFuncSetAttribute(
        (const void*)conv1D_shared,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        carveout
    );*/

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ---------------------------------------------------------
    // 4. GPU timing — shared memory kernel
    // ---------------------------------------------------------
    float shared_total = 0;

    for (int t = 0; t < 100; t++) {
        cudaEventRecord(start);
        conv1D_shared<<<grid, block>>>(d_in, d_out, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        shared_total += ms;
    }

    cudaMemcpy(h_gpu_shared, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);

    // ---------------------------------------------------------
    // 5. GPU timing — global memory kernel
    // ---------------------------------------------------------
    float global_total = 0;

    for (int t = 0; t < 100; t++) {
        cudaEventRecord(start);
        conv1D_global<<<grid, block>>>(d_in, d_out, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        global_total += ms;
    }

    cudaMemcpy(h_gpu_global, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);

    // ---------------------------------------------------------
    // 6. Correctness check: CPU vs GPU (shared)
    // ---------------------------------------------------------
    bool ok_shared = true;
    for (int i = RADIUS; i < N - RADIUS; i++) {
        if (h_cpu[i] != h_gpu_shared[i]) {
            printf("Shared mismatch at %d: CPU=%d GPU=%d\n",
                   i, h_cpu[i], h_gpu_shared[i]);
            ok_shared = false;
            break;
        }
    }

    // ---------------------------------------------------------
    // 7. Correctness check: CPU vs GPU (global)
    // ---------------------------------------------------------
    bool ok_global = true;
    for (int i = RADIUS; i < N - RADIUS; i++) {
        if (h_cpu[i] != h_gpu_global[i]) {
            printf("Global mismatch at %d: CPU=%d GPU=%d\n",
                   i, h_cpu[i], h_gpu_global[i]);
            ok_global = false;
            break;
        }
    }

    // ---------------------------------------------------------
    // 8. Print results
    // ---------------------------------------------------------
    printf("\nCorrectness:\n");
    printf("  Shared memory kernel: %s\n", ok_shared ? "OK" : "FAIL");
    printf("  Global memory kernel: %s\n", ok_global ? "OK" : "FAIL");

    printf("\nAverage execution times over 100 runs:\n");
    printf("  CPU:              %.4f ms\n", cpu_total / 100.0);
    printf("  GPU Shared:       %.4f ms\n", shared_total / 100.0);
    printf("  GPU Global:       %.4f ms\n", global_total / 100.0);

    // Cleanup
    delete[] h_in;
    delete[] h_cpu;
    delete[] h_gpu_shared;
    delete[] h_gpu_global;
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}

