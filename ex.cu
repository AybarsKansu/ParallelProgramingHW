#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

#define ar_size 20000
#define blocksize 512

#define float double

void fill(float* ar, float c) {
    for (int i = 0; i < ar_size; i++) {
        *(ar+i) = c++;
    }
}

void display(float* ar) {
    for (int i = 0; i < ar_size; i++) {
        printf("%.2f ", *(ar+i));
    }
    printf("\n");
}

void integral_array(float* res) {
    for (int i = 1; i < ar_size; i++) {
        *(res+i) += *(res + i - 1);
    }
}

__global__ void integral_array_gpu(float* ar, float* res) {
    __shared__ float mem[blocksize];

    int t = threadIdx.x;
    int index = blockIdx.x * blockDim.x + t;

    mem[t] = (index < ar_size) ? ar[index] : 0.0f;
    __syncthreads();

    for (int off = 1; off < blockDim.x; off *= 2) {
        float val = 0.0f;

        if (t >= off)
            val = mem[t - off];

        __syncthreads();

        mem[t] += val;

        __syncthreads();
    }

    if (index < ar_size)
        res[index] = mem[t];
}

bool check(float* ar, float* c) {
    for (int i = 0; i < ar_size; i++) {
        if (abs(*(ar+i) - *(c+i)) >= 1)
            return false;
    }
    return true;
}

__host__ void cudaCheck(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cout << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void fix(float* ar) {
    float sum_value = 0.0f;
    for (int i = blocksize; i < ar_size; i++) {
        if (i % blocksize == 0)
            sum_value = ar[i - 1];
        ar[i] += sum_value;
    }
}

int main() {
    float *ar = new float[ar_size];
    float *res = new float[ar_size];
    float *res2 = new float[ar_size];

    size_t bytes = ar_size * sizeof(float);

    fill(ar, 10.0f); fill(res, 10.0f);
    integral_array(res);
    //display(ar);
    //display(res);

    float *d_ar = nullptr; float* d_res = nullptr;
    cudaCheck(cudaMalloc(&d_ar, bytes));
    cudaCheck(cudaMalloc(&d_res, bytes));

    cudaCheck(cudaMemcpy(d_ar, ar, bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(d_res, 0, bytes));

    dim3 block(blocksize);
    dim3 grid((ar_size + blocksize - 1) / blocksize);
    integral_array_gpu<<<grid, block>>>(d_ar, d_res);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());

    cudaCheck(cudaMemcpy(res2, d_res, bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_ar); cudaFree(d_res);

    //display(res2);
    fix(res2);
    //display(res2);

    if (check(res, res2)) {
        std::cout << "result is correct";
    } else {
        std::cout << "results is wrong";
    }
    delete[] ar; delete[] res; delete[] res2;

    return 0;
}