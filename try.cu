#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdlib>

using namespace std;

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
     std::cout << std::endl << std::endl;
}

void cudaCheck(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << '\n';
        std::exit(EXIT_FAILURE);
    }
}

bool checkMatMul(const float* A, const float* B, int n) {
     for (int i = 0; i < n; i++) {
          if (abs(A[i] - B[i]) > 0.5f) {
               return false;
          }
     }
     return true;
}

__global__ void matmul(const float* A, const float* B, float* C, int K, int N, int M){
     int row = blockDim.y * blockIdx.y + threadIdx.y;
     int col = blockDim.x * blockIdx.x + threadIdx.x;

     int index = row * K + col;

     if (row >= M || col >= K) return;

     float sum = 0.0f;
     for (int i = 0; i < N; i++) {
          sum += A[row * N + i] * B[i * K + col];
     }
     C[index] = sum;
}

void tileMatMul(const float* A, const float* B, float* C, int TILE_SIZE, int M, int N, int K) {
     for (int jj = 0; jj < M; jj+=TILE_SIZE) {
          for (int ii = 0; ii < K; ii+=TILE_SIZE) {
               for (int kk = 0; kk < N; kk+=TILE_SIZE) {
                    
                    for (int j = jj; j < min(jj+TILE_SIZE, M); j++) {
                         for (int k = kk; k < min(kk+TILE_SIZE, N); k++) {
                              for (int i = ii; i < min(ii + TILE_SIZE, K); i++) {
                                   C[j * K + i] += A[j * N + k] * B[k * K + i];
                              }
                         }
                    }

               }
          }         
     }
}

int main() {
     int M = 30; int N = 40; int K = 50;
     int TILE_SIZE = 8;
     int size_A = M * N; int size_B = N * K; int size_C = M*K;
     
     size_t bytes_A = sizeof(float) * size_A;
     size_t bytes_B = sizeof(float) * size_B;
     size_t bytes_C = sizeof(float) * size_C;
     
     float* A_h = new float[size_A];
     float* B_h = new float[size_B];
     float* C_h = new float[size_C];
     float* C = new float[size_C];

     fillMatrix(A_h, size_A, 5);
     fillMatrix(B_h, size_B, 10);
     fillMatrix(C_h, size_C, 0);
     fillMatrix(C, size_C, 0);

     tileMatMul(A_h, B_h, C_h, TILE_SIZE, M, N, K);

     display(C_h, M, K);

     float* A_d = nullptr;
     float* B_d = nullptr;
     float* C_d = nullptr;

     cudaCheck(cudaMalloc(&A_d, bytes_A));
     cudaCheck(cudaMalloc(&B_d, bytes_B));
     cudaCheck(cudaMalloc(&C_d, bytes_C));

     cudaCheck(cudaMemcpy(A_d, A_h, bytes_A, cudaMemcpyHostToDevice));
     cudaCheck(cudaMemcpy(B_d, B_h, bytes_B, cudaMemcpyHostToDevice));

     dim3 blockDim(32, 32);
     dim3 gridDim((K + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);
     matmul<<<gridDim, blockDim>>>(A_d, B_d, C_d, K, N, M);

     cudaDeviceSynchronize();
     cudaCheck(cudaGetLastError());

     cudaCheck(cudaMemcpy(C, C_d, bytes_C, cudaMemcpyDeviceToHost));
     cudaCheck(cudaFree(A_d));cudaCheck(cudaFree(B_d));cudaCheck(cudaFree(C_d));
     display(C, M, K);
     printf("%d\n", checkMatMul(C, C_h, size_C));

     delete[] A_h; delete[] B_h; delete[] C; delete[] C_h;
}