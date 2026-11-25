#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define BLOCK_DIM 512
#define COARSE_FACTOR 4         // Each thread reads COARSE_FACTOR*2 elements
#define COARSE_FACTOR_PRIM 2    // Each thread reads COARSE_FACTOR*2 elements
#define N (1 << 20)             // 1 million elements
#define RUN_COUNT 100000

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (at %s:%d)\n", \
            cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    }

// Segmented sum reduction
__global__ void SegmentedSumReductionKernel(float* input, float* output)
{
    __shared__ float input_s[BLOCK_DIM];

    unsigned int segment = 2 * blockDim.x * blockIdx.x;
    unsigned int i = segment + threadIdx.x;
    unsigned int t = threadIdx.x;

    input_s[t] = input[i] + input[i + BLOCK_DIM];

    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride >>= 1) {
        __syncthreads();
        if (t < stride) {
            input_s[t] += input_s[t + stride];
        }
    }

    if (t == 0) {
        atomicAdd(output, input_s[0]);
    }
}

// Coarsened sum reduction
__global__ void CoarsenedSumReductionKernel(float* input, float* output)
{
    __shared__ float input_s[BLOCK_DIM];

    unsigned int segment = COARSE_FACTOR * 2 * blockDim.x * blockIdx.x;
    unsigned int i = segment + threadIdx.x;
    unsigned int t = threadIdx.x;

    float sum = input[i];

    for (unsigned int tile = 1; tile < COARSE_FACTOR * 2; ++tile) {
        sum += input[i + tile * BLOCK_DIM];
    }

    input_s[t] = sum;

    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride >>= 1) {
        __syncthreads();
        if (t < stride) {
            input_s[t] += input_s[t + stride];
        }
    }

    if (t == 0) {
        atomicAdd(output, input_s[0]);
    }
}

//alternative reduction
__global__ void alt_reduce(float *gdata, float *out){
    __shared__ float sdata[BLOCK_DIM];
    int tid = threadIdx.x;
    sdata[tid] = 0.0f;
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;

    while (idx < N) { // grid stride loop to load data
        sdata[tid] += gdata[idx];
        idx += gridDim.x * blockDim.x;
    }

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        __syncthreads();
        if (tid < s) // parallel sweep reduction
            sdata[tid] += sdata[tid + s];
    }
    if (tid == 0) atomicAdd(out, sdata[0]);
}


__global__ void CoarsenedSumReductionKernel_shuffle(float* input, float* output){ 
    __shared__ float input_s[BLOCK_DIM];

    unsigned int segment = COARSE_FACTOR_PRIM * 2 * blockDim.x * blockIdx.x;
    unsigned int i = segment + threadIdx.x;
    unsigned int t = threadIdx.x;
    float val = 0.0f;
    unsigned mask = 0xFFFFFFFFU;
    int warpID = threadIdx.x / warpSize;

    float sum = input[i];

    for (unsigned int tile = 1; tile < COARSE_FACTOR_PRIM * 2; ++tile) {
        sum += input[i + tile * BLOCK_DIM];
    }

    input_s[t] = sum;

    for (unsigned int stride = blockDim.x / 2; stride >= 32; stride >>= 1) {
        __syncthreads();
        if (t < stride) {
            input_s[t] += input_s[t + stride];
        }
    }
    
    if (warpID == 0){
        val = input_s[t]; //(tid < blockDim.x/warpSize)?sdata[lane]:0;
        for (int offset = warpSize/2; offset > 0; offset >>= 1)
            val += __shfl_down_sync(mask, val, offset);
        if (t == 0) 
            atomicAdd(output, val);
    }
}

// Benchmark helper
template <typename Kernel, typename... Args>
float benchmarkKernel(Kernel kernel, dim3 blocks, dim3 threads, Args... args)
{
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup
    for (int i = 0; i < 10; i++)
        kernel<<<blocks, threads>>>(args...);

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < RUN_COUNT; i++)
        kernel<<<blocks, threads>>>(args...);
    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaDeviceSynchronize());

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return (ms * 1000.0f) / static_cast<float>(RUN_COUNT); // microseconds per run
}

int main()
{
    const size_t bytes = N * sizeof(float);

    printf("N = %d elements\n", N);

    float* h_in  = (float*)malloc(bytes);
    float* h_ref = (float*)malloc(sizeof(float));

    // Populate input
    for (int i = 0; i < N; i++)
        h_in[i] = 1.0f;  // simple predictable value

    // CPU reference sum
    float ref_sum = 0.0f;
    for (int i = 0; i < N; i++)
        ref_sum += h_in[i];

    *h_ref = ref_sum;

    // Device memory
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // SegmentedSumReductionKernel *************************************************
    int elements_per_block = BLOCK_DIM * 2;
    int blocks = N / elements_per_block;
    dim3 threads(BLOCK_DIM);

    printf("\nBlocks = %d, Threads/block = %d\n", blocks, BLOCK_DIM);

    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));

    SegmentedSumReductionKernel<<<blocks, threads>>>(d_in, d_out);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    float h_sum1;
    CUDA_CHECK(cudaMemcpy(&h_sum1, d_out, sizeof(float), cudaMemcpyDeviceToHost));

    printf("SegmentedSumReductionKernel correctness: %s (GPU=%f CPU=%f)\n",
           (fabs(h_sum1 - ref_sum) < 1e-5) ? "PASS" : "FAIL",
           h_sum1, ref_sum);

    float t1 = benchmarkKernel(SegmentedSumReductionKernel,
                               blocks, threads, d_in, d_out);

    printf("Segmented kernel average time: %.2f microseconds\n", t1);

    // CoarsenedSumReductionKernel *************************************************
    int elements_per_block_coarse = COARSE_FACTOR * 2 * BLOCK_DIM;
    int blocks_coarse = N / elements_per_block_coarse;
    
    printf("\nBlocks = %d, Threads/block = %d\n", blocks_coarse, BLOCK_DIM);

    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));

    CoarsenedSumReductionKernel<<<blocks_coarse, threads>>>(d_in, d_out);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    float h_sum2;
    CUDA_CHECK(cudaMemcpy(&h_sum2, d_out, sizeof(float), cudaMemcpyDeviceToHost));

    printf("CoarsenedSumReductionKernel correctness: %s (GPU=%f CPU=%f)\n",
           (fabs(h_sum2 - ref_sum) < 1e-5) ? "PASS" : "FAIL",
           h_sum2, ref_sum);

    float t2 = benchmarkKernel(CoarsenedSumReductionKernel,
                               blocks_coarse, threads, d_in, d_out);

    printf("Coarsened kernel average time: %.2f microseconds\n", t2);
    
    // alternative reduce *************************************************
    dim3 block(BLOCK_DIM);
    int grid((N + BLOCK_DIM - 1) / BLOCK_DIM);
    
    printf("\nBlocks = %d, Threads/block = %d\n", grid, BLOCK_DIM);
    
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
    
    // Launch kernel once to check correctness
    alt_reduce<<<grid, block>>>(d_in, d_out);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    float h_sum3;
    CUDA_CHECK(cudaMemcpy(&h_sum3, d_out, sizeof(float), cudaMemcpyDeviceToHost));

    printf("alt_reduce correctness: %s (GPU=%f CPU=%f)\n",
           (fabs(h_sum3 - ref_sum) < 1e-5) ? "PASS" : "FAIL",
           h_sum3, ref_sum);

    float t3 = benchmarkKernel(alt_reduce, grid, block, d_in, d_out);

    printf("alternative kernel average time: %.2f microseconds\n", t3);
    
    // reduce_warpShuffle *************************************************
    elements_per_block_coarse = COARSE_FACTOR_PRIM * 2 * BLOCK_DIM;
    blocks_coarse = N / elements_per_block_coarse;
    
    printf("\nBlocks = %d, Threads/block = %d\n", blocks_coarse, BLOCK_DIM);
    
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
    
    // Launch kernel once to check correctness
    CoarsenedSumReductionKernel_shuffle<<<blocks_coarse, threads>>>(d_in, d_out);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    float h_sum4;
    CUDA_CHECK(cudaMemcpy(&h_sum4, d_out, sizeof(float), cudaMemcpyDeviceToHost));

    printf("alt_reduce correctness: %s (GPU=%f CPU=%f)\n",
           (fabs(h_sum4 - ref_sum) < 1e-5) ? "PASS" : "FAIL",
           h_sum4, ref_sum);

    float t4 = benchmarkKernel(CoarsenedSumReductionKernel_shuffle, blocks_coarse, threads, d_in, d_out);

    printf("alternative kernel average time: %.2f microseconds\n", t4);
    
    // Cleanup
    free(h_in);
    free(h_ref);
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
