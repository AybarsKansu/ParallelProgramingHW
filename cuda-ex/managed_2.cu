
#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>

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
// GPU kernel: Initialize the entire array
// ---------------------------------------------------------
__global__
void initializeKernel(float* data, size_t N)
{
    size_t idx = blockIdx.x * (size_t)blockDim.x
               + threadIdx.x;

    if (idx < N) {
        data[idx] = static_cast<float>(idx % 1000);
    }
}


// ---------------------------------------------------------
// Main wrapper
// ---------------------------------------------------------
void runManagedOversubscription()
{
    std::cout << "===== MANAGED MEMORY OVERSUBSCRIPTION =====\n";

    // -----------------------------------------------------
    // 1. Query GPU properties
    // -----------------------------------------------------
    int device;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: "
              << prop.major << "." << prop.minor << std::endl;

    if (prop.major < 6) {
        std::cerr << "Pascal or newer GPU required.\n";
        return;
    }

    // Total physical GPU DRAM
    size_t gpuMemory = prop.totalGlobalMem;

    // -----------------------------------------------------
    // 2. Calculate allocation size (2x GPU DRAM)
    // -----------------------------------------------------
    size_t allocationSize = 2 * gpuMemory;

    size_t N = allocationSize / sizeof(float);

    std::cout << "\nGPU physical DRAM: "
              << gpuMemory / (1024ULL * 1024 * 1024)
              << " GiB" << std::endl;

    std::cout << "Requested allocation: "
              << allocationSize / (1024ULL * 1024 * 1024)
              << " GiB" << std::endl;

    std::cout << "Number of float elements: "
              << N << std::endl;

    // -----------------------------------------------------
    // 3. Allocate Unified Memory
    // -----------------------------------------------------
    float* data = nullptr;

    std::cout << "\nAllocating managed memory...\n";

    CUDA_CHECK(cudaMallocManaged(
        (void**)&data, allocationSize));

    std::cout << "Managed allocation successful!\n";

    // -----------------------------------------------------
    // 4. Initialize entire allocation on GPU
    // -----------------------------------------------------
    int threads = 256;

    int blocks = (int)((N + threads - 1) / threads);

    std::cout << "\nLaunching GPU kernel...\n";
    std::cout << "Blocks: " << blocks << std::endl;
    std::cout << "Threads per block: " << threads << std::endl;

    initializeKernel<<<blocks, threads>>>(data, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "GPU initialization completed!\n";

    // -----------------------------------------------------
    // 5. Read selected elements on CPU
    // -----------------------------------------------------
    std::cout << "\nReading selected elements on CPU:\n";

    std::cout << "data[0]     = " << data[0] << std::endl;
    std::cout << "data[N/2]   = " << data[N/2] << std::endl;
    std::cout << "data[N-1]   = " << data[N-1] << std::endl;

    // -----------------------------------------------------
    // 6. Release managed memory
    // -----------------------------------------------------
    CUDA_CHECK(cudaFree(data));

    std::cout << "\nManaged memory released.\n";
}


// ---------------------------------------------------------
// MAIN
// ---------------------------------------------------------
int main()
{
    runManagedOversubscription();

    return 0;
}
