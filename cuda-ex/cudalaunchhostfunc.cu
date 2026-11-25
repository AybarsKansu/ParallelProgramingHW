
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>

// CUDA error checking
#define CUDA_CHECK(call)                                      \
do {                                                          \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
        std::cerr << "CUDA Error: "                           \
                  << cudaGetErrorString(err) << std::endl;    \
        std::exit(EXIT_FAILURE);                              \
    }                                                         \
} while (0)

// Simple GPU kernel
__global__ void squareKernel(const float* input,
                             float* output,
                             int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
        output[idx] = input[idx] * input[idx];
}

// CPU callback executed after preceding stream operations
void CUDART_CB hostCallback(void* userData)
{
    int* value = static_cast<int*>(userData);

    std::cout << "\n[Host Callback] GPU operations completed!" << std::endl;

    std::cout << "[Host Callback] Callback data = " << *value << std::endl;
}

int main()
{
    const int N = 1024;
    const size_t bytes = N * sizeof(float);

    // Host memory
    std::vector<float> h_input(N);
    std::vector<float> h_output(N);

    for (int i = 0; i < N; i++)
        h_input[i] = static_cast<float>(i);

    // Device memory
    float *d_input, *d_output;

    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));

    // Create CUDA stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    int callbackData = 1234;

    std::cout << "[CPU] Submitting GPU operations..." << std::endl;

    // 1. Asynchronous H2D copy
    CUDA_CHECK(cudaMemcpyAsync(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice, stream));

    // 2. Launch GPU kernel
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    squareKernel<<<blocks, threads, 0, stream>>>(d_input, d_output, N);

    CUDA_CHECK(cudaGetLastError());

    // 3. Asynchronous D2H copy
    CUDA_CHECK(cudaMemcpyAsync(h_output.data(), d_output, bytes, cudaMemcpyDeviceToHost, stream));

    // 4. Schedule CPU callback
    // Callback executes after all preceding stream operations.
    CUDA_CHECK(cudaLaunchHostFunc(stream, hostCallback, &callbackData));

    std::cout << "[CPU] GPU operations submitted." << std::endl;

    // 5. Synchronize the stream
    // CPU blocks here until all stream operations,
    // including the host callback, have completed.
    std::cout << "[CPU] Waiting at cudaStreamSynchronize..." << std::endl;

    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::cout << "[CPU] Stream synchronization completed!" << std::endl;

    // Verify results
    std::cout << "\nFirst 10 results:" << std::endl;

    for (int i = 0; i < 10; i++)
    {
        std::cout << h_input[i] << " -> " << h_output[i] << std::endl;
    }

    // Cleanup
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}
