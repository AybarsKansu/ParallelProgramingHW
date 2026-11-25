
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>

#define CUDA_CHECK(call)                                      \
do {                                                          \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
        std::cerr << "CUDA Error: "                           \
                  << cudaGetErrorString(err)                  \
                  << " at line " << __LINE__ << std::endl;    \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while (0)

// Simple CUDA kernel
__global__ void scaleKernel(const float* input, float* output, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
        output[idx] = input[idx] * 2.0f;
}

int main()
{
    const int N = 1 << 24; // 16M elements
    const int CHUNK_SIZE = 1 << 20; // 1M elements
    const int NUM_STREAMS = 4;

    const size_t totalBytes = N * sizeof(float);
    const size_t chunkBytes = CHUNK_SIZE * sizeof(float);

    // Pinned host memory
    float *h_input, *h_output;
    CUDA_CHECK(cudaMallocHost(&h_input, totalBytes));
    CUDA_CHECK(cudaMallocHost(&h_output, totalBytes));

    // Initialize input
    for (int i = 0; i < N; i++)
        h_input[i] = static_cast<float>(i);

    // Device memory: separate buffers for each stream
    float *d_input[NUM_STREAMS];
    float *d_output[NUM_STREAMS];

    cudaStream_t streams[NUM_STREAMS];

    for (int s = 0; s < NUM_STREAMS; s++)
    {
        CUDA_CHECK(cudaMalloc(&d_input[s], chunkBytes));
        CUDA_CHECK(cudaMalloc(&d_output[s], chunkBytes));

        CUDA_CHECK(cudaStreamCreate(&streams[s]));
    }

    const int threads = 256;
    const int blocks = (CHUNK_SIZE + threads - 1) / threads;

    // Process chunks using multiple streams
    for (int chunk = 0; chunk < N / CHUNK_SIZE; chunk++)
    {
        int s = chunk % NUM_STREAMS;
        int offset = chunk * CHUNK_SIZE;

        //Host -> Device
        CUDA_CHECK(cudaMemcpyAsync(d_input[s], h_input + offset, chunkBytes, cudaMemcpyHostToDevice, streams[s]));

        // 2. Kernel execution
        scaleKernel<<<blocks, threads, 0, streams[s]>>>(d_input[s], d_output[s], CHUNK_SIZE);

        CUDA_CHECK(cudaGetLastError());

        // 3. Device -> Host transfer
        CUDA_CHECK(cudaMemcpyAsync(h_output + offset, d_output[s], chunkBytes, cudaMemcpyDeviceToHost, streams[s]));
    }

    // Wait for all streams to finish
    for (int s = 0; s < NUM_STREAMS; s++)
        CUDA_CHECK(cudaStreamSynchronize(streams[s]));

    // Verify results
    bool correct = true;

    for (int i = 0; i < N; i++)
    {
        if (h_output[i] != h_input[i] * 2.0f)
        {
            correct = false;
            break;
        }
    }

    std::cout << "Result: " << (correct ? "CORRECT" : "INCORRECT") << std::endl;

    // Cleanup
    for (int s = 0; s < NUM_STREAMS; s++)
    {
        CUDA_CHECK(cudaStreamDestroy(streams[s]));
        CUDA_CHECK(cudaFree(d_input[s]));
        CUDA_CHECK(cudaFree(d_output[s]));
    }

    CUDA_CHECK(cudaFreeHost(h_input));
    CUDA_CHECK(cudaFreeHost(h_output));

    return 0;
}
