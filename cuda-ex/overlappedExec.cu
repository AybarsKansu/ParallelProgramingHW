#include <cuda_runtime.h>
#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <mutex>

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

using Clock = std::chrono::steady_clock;

std::mutex printMutex;

double elapsedMs(Clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// GPU producer: square every input element
__global__ void producerKernel(const float* input, float* output, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
        output[idx] = input[idx] * input[idx];
}

// Callback data
struct CallbackData
{
    const float* output;
    int N;

    double checksum;
    float minValue;
    float maxValue;

    Clock::time_point start;
};

// CPU consumer: process GPU results
void CUDART_CB consumerCallback(void* userData)
{
    CallbackData* data = static_cast<CallbackData*>(userData);

    {
        std::lock_guard<std::mutex> lock(printMutex);

        std::cout << "\n[" << elapsedMs(data->start) << " ms] CALLBACK: Started processing GPU results" << std::endl;
    }

    // CPU-side processing of GPU results
    double sum = 0.0;
    float minVal = data->output[0];
    float maxVal = data->output[0];

    for (int i = 0; i < data->N; i++)
    {
        float value = data->output[i];

        sum += value;

        if (value < minVal)
            minVal = value;

        if (value > maxVal)
            maxVal = value;
    }

    // Store results for the main CPU thread
    data->checksum = sum;
    data->minValue = minVal;
    data->maxValue = maxVal;

    {
        std::lock_guard<std::mutex> lock(printMutex);

        std::cout << "[" << elapsedMs(data->start) << " ms] CALLBACK: Finished processing results \n" << std::endl;
    }

    // IMPORTANT:
    // Do not call CUDA APIs from this callback.
}

int main()
{
    const int N = 1 << 22;  // Approximately 4 million elements
    const size_t bytes = N * sizeof(float);

    Clock::time_point start = Clock::now();

    // Allocate pinned host memory
    float* h_input;
    float* h_output;
    CUDA_CHECK(cudaHostAlloc(&h_input, bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_output, bytes, cudaHostAllocDefault));

    // Initialize input data
    for (int i = 0; i < N; i++)
        h_input[i] = static_cast<float>(i % 1000);

    // Allocate GPU memory
    float *d_input;
    float *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));

    // Create CUDA stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Callback data must remain valid until callback finishes.
    CallbackData callbackData;

    callbackData.output = h_output;
    callbackData.N = N;
    callbackData.checksum = 0.0;
    callbackData.minValue = 0.0f;
    callbackData.maxValue = 0.0f;
    callbackData.start = start;

    // Submit GPU producer operations
    {
        std::lock_guard<std::mutex> lock(printMutex);

        std::cout << "[" << elapsedMs(start) << " ms] CPU: Submitting GPU operations" << std::endl;
    }

    // H2D transfer
    CUDA_CHECK(cudaMemcpyAsync(d_input, h_input, bytes, cudaMemcpyHostToDevice, stream));

    // GPU computation
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    producerKernel<<<blocks, threads, 0, stream>>>(d_input, d_output, N);
    CUDA_CHECK(cudaGetLastError());

    // D2H transfer
    CUDA_CHECK(cudaMemcpyAsync(h_output, d_output, bytes, cudaMemcpyDeviceToHost, stream));

    // Schedule CPU consumer callback
    CUDA_CHECK(cudaLaunchHostFunc(stream, consumerCallback, &callbackData));

    {
        std::lock_guard<std::mutex> lock(printMutex);

        std::cout << "[" << elapsedMs(start) << " ms] CPU: GPU work submitted." << std::endl;
    }

    // CPU performs unrelated work
    {
        std::lock_guard<std::mutex> lock(printMutex);

        std::cout << "[" << elapsedMs(start) << " ms] CPU: Starting unrelated CPU work" << std::endl;
    }

    // Independent CPU computation. This does not depend on GPU results.
    uint64_t cpuChecksum = 0;

    // Completely arbitrary arithmetic on the main CPU thread
    for (uint64_t i = 0; i < 50000000ULL; i++)
        cpuChecksum = cpuChecksum * 1664525ULL + i + 1013904223ULL;

    {
        std::lock_guard<std::mutex> lock(printMutex);

        std::cout << "[" << elapsedMs(start) << " ms] CPU: Unrelated work completed." << std::endl;

        std::cout << "CPU checksum = " << cpuChecksum << std::endl;
    }

    // Synchronize only when GPU results are needed
    {
        std::lock_guard<std::mutex> lock(printMutex);

        std::cout << "[" << elapsedMs(start) << " ms] CPU: Waiting for GPU consumer..." << std::endl;
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    {
        std::lock_guard<std::mutex> lock(printMutex);

        std::cout << "[" << elapsedMs(start) << " ms] CPU: Stream synchronization completed." << std::endl;
    }

    // Consume callback results
    std::cout << "\n========== FINAL RESULTS ==========" << std::endl;
    std::cout << "GPU output checksum = " << callbackData.checksum << std::endl;
    std::cout << "Minimum output = " << callbackData.minValue << std::endl;
    std::cout << "Maximum output = " << callbackData.maxValue << std::endl;
    std::cout << "\nFirst 10 GPU results:" << std::endl;

    for (int i = 0; i < 10; i++)
    {
        std::cout << h_input[i] << " -> " << h_output[i] << std::endl;
    }

    // Cleanup
    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    CUDA_CHECK(cudaFreeHost(h_input));
    CUDA_CHECK(cudaFreeHost(h_output));

    return 0;
}
