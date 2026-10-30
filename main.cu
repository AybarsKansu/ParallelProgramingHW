#include <cuda_runtime.h>
#include <iostream>

using namespace std;

void displayGpuConfig() {
     int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        cout << "GPU: " << prop.name << "\n";
        cout << "Compute Capability: "
             << prop.major << "." << prop.minor << "\n";

        cout << "Global Memory: "
             << prop.totalGlobalMem / (1024 * 1024)
             << " MB\n";

        cout << "SM Count: "
             << prop.multiProcessorCount << "\n";

        cout << "Warp Size: "
             << prop.warpSize << "\n";

        cout << "Max Threads per Block: "
             << prop.maxThreadsPerBlock << "\n";

        cout << "Max Threads per SM: "
             << prop.maxThreadsPerMultiProcessor << "\n";

        cout << "Max Block Dimensions: "
             << prop.maxThreadsDim[0] << ", "
             << prop.maxThreadsDim[1] << ", "
             << prop.maxThreadsDim[2] << "\n";

        cout << "Max Grid Dimensions: "
             << prop.maxGridSize[0] << ", "
             << prop.maxGridSize[1] << ", "
             << prop.maxGridSize[2] << "\n";
    }
}

__global__ void matadd(float* A, float* B, float* C) {
     
}

int main() {
     // displayGpuConfig();
      
     cudaDeviceSynchronize();  
}