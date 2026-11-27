/*
vector addition

streams
callback
managed mem

*/

#include <iostream>
#include <stdio.h>

#define NUM_STREAMS 4
#define N (1<<20)

void f(const float* a, const float* b, float* c) {
    for (int i = 0; i < N; i++) {
        *(c+i)=*(a+i)+*(b+i);
    }
}

__global__ void sumUp(const float* a, const float* b, float* c, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < n) {
        *(c+i)=*(a+i)+*(b+i);
    }
}

__host__ void cudaCheck(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cout << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void fill(float* a, float* b, float c) {
    for (int i = 0; i < N; i++) {
        *(a+i) = c++;
        *(b+i) = N-c;
    }
}

void display(float* ar) {
    for (int i = 0; i < 5; i++) {
        printf("%.2f ", *(ar+i));
    }
    printf("\n");
}

void CUDART_CB callback(void* data) {
    const float* result = static_cast<const float*>(data);
    printf("Chunk completed: %.2f\n", result[0]);
}

void managedMem() {
        int chunkSize = 1 << 19;
    size_t bytes = N * sizeof(float);
    size_t chunkBytes = chunkSize * sizeof(float);

    float* A; float* B; float* C;
    cudaCheck(cudaMallocManaged((void**)&A, bytes));
    cudaCheck(cudaMallocManaged((void**)&B, bytes));
    cudaCheck(cudaMallocManaged((void**)&C, bytes));
    fill(A, B, 0);

    dim3 block(256);
    dim3 grid((chunkSize + block.x - 1) / block.x);

    float* stream_in[NUM_STREAMS];
    float* stream_out[NUM_STREAMS];
    cudaStream_t stream[NUM_STREAMS];

    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaCheck(cudaMalloc(&stream_in[i], chunkBytes));
        cudaCheck(cudaMalloc(&stream_out[i], chunkBytes));
        cudaCheck(cudaStreamCreate(&stream[i]));
    }

    for (int i = 0; i < N / chunkSize; i++) {
        int index = i % NUM_STREAMS;
        size_t offset = i * chunkSize;

        // memcpyasync 
        sumUp<<<grid, block, 0, stream[index]>>>(A + offset, B + offset, C + offset, chunkSize);
        // memcpyasync
    }

    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaGetLastError());
    display(C);
}

int main() {
    int chunkSize = 1 << 19;
    size_t bytes = N * sizeof(float);
    size_t chunkBytes = chunkSize * sizeof(float);

    float* h_A; float* h_B; float* h_C;
    cudaCheck(cudaMallocHost(&h_A, bytes));
    cudaCheck(cudaMallocHost(&h_B, bytes));
    cudaCheck(cudaMallocHost(&h_C, bytes));
    fill(h_A, h_B, 0);

    dim3 block(256);
    dim3 grid((chunkSize + block.x - 1) / block.x);

    float* stream_inA[NUM_STREAMS];
    float* stream_inB[NUM_STREAMS];
    float* stream_out[NUM_STREAMS];
    cudaStream_t stream[NUM_STREAMS];

    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaCheck(cudaMalloc(&stream_inA[i], chunkBytes));
        cudaCheck(cudaMalloc(&stream_inB[i], chunkBytes));
        cudaCheck(cudaMalloc(&stream_out[i], chunkBytes));
        cudaCheck(cudaStreamCreate(&stream[i]));
    }

    for (int i = 0; i < N / chunkSize; i++) {
        int index = i % NUM_STREAMS;
        size_t offset = i * chunkSize;

        cudaCheck(cudaMemcpyAsync(stream_inA[index], h_A + offset, chunkBytes, cudaMemcpyHostToDevice, stream[index]));
        cudaCheck(cudaMemcpyAsync(stream_inB[index], h_B + offset, chunkBytes, cudaMemcpyHostToDevice, stream[index]));

        sumUp<<<grid, block, 0, stream[index]>>>(stream_inA[index], stream_inB[index], stream_out[index], chunkSize);
        
        cudaCheck(cudaGetLastError());

        cudaCheck(cudaMemcpyAsync(h_C+offset, stream_out[index], chunkBytes, cudaMemcpyDeviceToHost, stream[index]));
        cudaCheck(cudaLaunchHostFunc(stream[index], callback, h_C + offset));
    }

    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaGetLastError());
    display(h_C);

    return 0;
}