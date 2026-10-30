#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
using namespace std;

__global__ void matAddGPU(const float* A, const float* B,
                          float* C, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        int index = row * cols + col;
        C[index] = A[index] + B[index];
    }
}

int main() {
    int rows = 1000, cols = 2000;
    int size = rows * cols;
    size_t bytes = size * sizeof(float);

    float* A = new float[size];
    float* B = new float[size];
    float* C = new float[size];

    for (int i = 0; i < size; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }

    float *A_d, *B_d, *C_d;

    cudaMalloc((void**)&A_d, bytes);
    cudaMalloc((void**)&B_d, bytes);
    cudaMalloc((void**)&C_d, bytes);

    cudaMemcpy(A_d, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, bytes, cudaMemcpyHostToDevice);

    // 2D block: 16 x 16 = 256 threads
    dim3 block(16, 16);

    // 2D grid
    dim3 grid(
        (cols + block.x - 1) / block.x,
        (rows + block.y - 1) / block.y
    );

    matAddGPU<<<grid, block>>>(A_d, B_d, C_d, rows, cols);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cerr << cudaGetErrorString(err) << endl;
        return 1;
    }

    cudaMemcpy(C, C_d, bytes, cudaMemcpyDeviceToHost);

    bool correct = true;
    for (int i = 0; i < size; i++) {
        if (fabs(C[i] - 3.0f) > 0.001f) {
            correct = false;
            break;
        }
    }

    cout << (correct ? "Correct!" : "Wrong!") << endl;

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
