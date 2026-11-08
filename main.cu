#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <cstdlib>

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

void fillMatrix(float* m, int n) {
     for (int i = 0; i < n; i++) {
          *(m+i)=((float)rand()/RAND_MAX)*10;
     }
}

void fillMatrixSeq(float* m, int n, float t) {
     for (int i = 0; i < n; i++) {
          *(m+i)=t++;
     }
}

void fillMatrix(float* m, int n, float t) {
     for (int i = 0; i < n; i++) {
          *(m+i)=t;
     }
}

void display(float* m, int row, int col){
     for (int i = 0; i < row; i++) {
          for (int j = 0; j < col; j++) {
               int index = i * col + j;
               printf("%.02f ", *(m+index));
          }
          std::cout << std::endl;
     }
}

void matmulCpu(const float* A, const float* B, float* C, int N, int tileSize) {
     int TILESIZE = tileSize;
     for (int jj = 0; jj < N; jj+=TILESIZE){
          for (int kk = 0; kk < min(N, N); kk+=TILESIZE){
               for (int ii = 0; ii < N; ii+=TILESIZE){
                    for (int j = jj; j < min(jj+TILESIZE, N); j++){
                         for (int k = kk; k < min(kk + TILESIZE, min(N, N)); k++) {
                              for (int i = ii; i < min(ii+TILESIZE, N); i++) {
                                   C[j * N + i] += A[j * N + k] * B[k * N + j];
                              }
                         }
                    }
               }
          }
     }
}

bool check(const float* m1, const float* m2, int n) {
     for (int i = 0; i < n; i++) {
          if (m1[i] - m2[i] != 0) {
               return false;
          }
     }
     return true;
}

__global__ void matmul(const float* A, const float* B, float* C, int N) {
     int row = blockIdx.y * blockDim.y + threadIdx.y;
     int col = blockIdx.x * blockDim.x + threadIdx.x;
     if (row >= N || col >= N) return;

     float sum = 0.0f;
     for (int n = 0; n < N; n++) {
          sum += A[row * N + n] * B[n * N + col];
     }
     C[row * N + col] = sum;
}

void cudaCheck(cudaError_t err) {
    if (err != cudaSuccess) {
        cerr << "CUDA error: " << cudaGetErrorString(err) << '\n';
        std::exit(EXIT_FAILURE);
    }
}

__global__ void transpose(const float* A, float* B, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // B row
    int col = blockIdx.x * blockDim.x + threadIdx.x; // B col
    int index = row * cols + col; // B index
    int A_index = col * rows + row;
    //printf("%d %d => index: %d, A index: %d\n", row, col, index, A_index);
    for (int j = 0; j < blockDim.y; j++){
        for (int i = 0; i < blockDim.x; i++) {
            int B_index = index + i + j * cols; int temp_index = A_index + i * rows + j;
            if (B_index < rows * cols && temp_index < rows*cols){
                //printf("B index: %d, A index: %d\n", B_index, temp_index);
                B[B_index] = A[temp_index];
            }
        }   
    }
}

void transposeExample() {
    int size = 16;
    int bytes = sizeof(float) * size * size;

    float* A_h = new float[size*size];
    float* B_h = new float[size*size];
    fillMatrixSeq(A_h, size*size, 10.0);
    //display(A_h, size, size);

    //transpose_h(A_h, size, size, B_h);
    //display(B_h, size, size);
    float* B_d = nullptr;
    float* A_d = nullptr; 

    cudaCheck(cudaMalloc(&A_d, bytes));
    cudaCheck(cudaMalloc(&B_d, bytes));

    cudaCheck(cudaMemcpy(A_d, A_h, bytes, cudaMemcpyHostToDevice));
    
    dim3 threadsPerBlock(8, 2);
    dim3 blocksPerGrid((size + threadsPerBlock.x - 1)/threadsPerBlock.x, (size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    transpose<<<blocksPerGrid, threadsPerBlock>>>(A_d, B_d, size, size);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());

    cudaCheck(cudaMemcpy(B_h, B_d, bytes, cudaMemcpyDeviceToHost));
    display(A_h, size, size);
    display(B_h, size, size);

    cudaFree(A_d); cudaFree(B_d);
    delete[] A_h; delete[] B_h;
}

void matmulExample() {
     int size = 512;
     size_t bytes = sizeof(float) * size * size;
     float* A_h = new float[size*size];
     float* B_h = new float[size*size];
     float* C_h = new float[size*size];
     float* C = new float[size*size];
     for (int i = 0; i < size*size; i++) {
          A_h[i] = 1.25; B_h[i] = 2.56; C_h[i] = 0;
     }

     matmulCpu(A_h, B_h, C, size, 32);
     //display(C_h, size, size);
     float* A_d = nullptr;
     float* B_d = nullptr;
     float* C_d = nullptr;

     cudaCheck(cudaMalloc((void**)&A_d, bytes));
     cudaCheck(cudaMalloc((void**)&B_d, bytes));
     cudaCheck(cudaMalloc((void**)&C_d, bytes));

     cudaCheck(cudaMemcpy(A_d, A_h, bytes, cudaMemcpyHostToDevice));
     cudaCheck(cudaMemcpy(B_d, B_h, bytes, cudaMemcpyHostToDevice));

     dim3 threadsPerBlock(16, 16);
     dim3 blocksPerGrid((size + threadsPerBlock.x - 1) / threadsPerBlock.x, (size + threadsPerBlock.y - 1) / threadsPerBlock.y);

     matmul<<<blocksPerGrid, threadsPerBlock>>>(A_d, B_d, C_d, size);

     cudaCheck(cudaGetLastError());
     cudaCheck(cudaDeviceSynchronize());
     cudaCheck(cudaMemcpy(C_h, C_d, bytes, cudaMemcpyDeviceToHost));

     //display(C_h, size, size);
     std::cout << check(C, C_h, size*size);

     cudaFree(A_d);cudaFree(B_d);cudaFree(C_d);
     delete[] A_h; delete[] B_h; delete[] C_h; delete[] C;
}

__global__ void convolve(const float* A, const float* kernel, float* B, int outRows, int outCols, int kernelSize) {
     int row = blockDim.y * blockIdx.y + threadIdx.y;
     int col = blockIdx.x * blockDim.x + threadIdx.x;

     if (row >= outRows || col >= outCols) return;

     int inRows = outRows + kernelSize - 1;
     int inCols = outCols + kernelSize - 1;
     
     int index = row * outCols + col;
     int k = kernelSize / 2;
     int centerRow = row + k;
     int centerCol = col + k;

     float sum = 0.0f;
     for (int ky = -k; ky <= k; ++ky) {
          for (int kx = -k; kx <= k; ++kx) {
               int aIndex = (centerRow + ky) * inCols + (centerCol + kx);
               int kernelIndex = (k - ky) * kernelSize + (k - kx);
               sum += A[aIndex] * kernel[kernelIndex];
          }
     }
     B[row * outCols + col] = sum;
     }

void convolutionExample() {
          int config = 16;
     int row = config; int col = config;

     int kernelSize = 3; // 3x3

     float* kernel_h = new float[kernelSize * kernelSize];
     float* A_h = new float[row * col];

     fillMatrixSeq(A_h, row*col, 10.0);
     fillMatrix(kernel_h, kernelSize * kernelSize);

     display(A_h, row, col);
     display(kernel_h, kernelSize, kernelSize);

     float* B_h = new float[(row - kernelSize + 1)*(col - kernelSize + 1)];
     fillMatrix(B_h, (row - kernelSize + 1)*(col - kernelSize + 1), 0);

     float* kernel_d = nullptr;
     float* A_d = nullptr;
     float* B_d = nullptr;

     cudaCheck(cudaMalloc((void**)&A_d, sizeof(float)*row*col));
     cudaCheck(cudaMalloc((void**)&B_d, sizeof(float)*(row - kernelSize + 1)*(col - kernelSize + 1)));
     cudaCheck(cudaMalloc((void**)&kernel_d, sizeof(float)*kernelSize*kernelSize));
     
     cudaCheck(cudaMemcpy(A_d, A_h, sizeof(float)*row*col, cudaMemcpyHostToDevice));
     cudaCheck(cudaMemcpy(kernel_d, kernel_h, sizeof(float)*kernelSize*kernelSize, cudaMemcpyHostToDevice));

     dim3 threadsPerBlock(4, 4);
     dim3 blocksPerGrid((threadsPerBlock.x + row - kernelSize) / threadsPerBlock.x, (threadsPerBlock.y + col - kernelSize) / threadsPerBlock.y);
     convolve<<<blocksPerGrid, threadsPerBlock>>>(A_d, kernel_d, B_d, (row - kernelSize + 1), (col - kernelSize + 1), kernelSize);
     cudaCheck(cudaGetLastError());
     cudaCheck(cudaDeviceSynchronize());

     cudaCheck(cudaMemcpy(B_h, B_d, (row - kernelSize + 1)*(col - kernelSize + 1), cudaMemcpyDeviceToHost));
     cudaFree(A_d); cudaFree(kernel_d); cudaFree(B_d);
     
     display(B_h, (row - kernelSize + 1), (col - kernelSize + 1));
     delete[] A_h; delete[] kernel_h; delete[] B_h;
}

int main() {
     //displayGpuConfig();
     matmulExample();
}